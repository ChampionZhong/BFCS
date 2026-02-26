"""
Regenerated Google-style docstrings for module 'selfies'.
README source: others/readme/selfies/README.md
Generated at: 2025-12-02T00:51:57.206366Z

Total functions: 13
"""


from typing import List

################################################################################
# Source: selfies.bond_constraints.get_preset_constraints
# File: selfies/bond_constraints.py
# Category: valid
################################################################################

def selfies_bond_constraints_get_preset_constraints(name: str):
    """Returns the preset semantic constraints with the given name. This function is part of the SELFIES library for molecular string representations and is used to obtain a predefined mapping of atom tokens to their allowed bonding capacities (integer maximum bonds). These preset constraints capture common chemical semantics used by SELFIES to enforce or relax bonding rules during encoding, decoding, and generation of molecular graphs for machine learning and cheminformatics workflows. Calling this function lets downstream code (for example, selfies.set_semantic_constraints, encoder, decoder, or any generator that constructs SELFIES strings) apply a consistent, read-only policy for how many bonds each atom type may form. The returned mapping is a shallow copy of an internal preset and can be safely inspected or modified by callers without mutating the global presets stored in the library.
    
    Args:
        name (str): The preset name to retrieve. Accepted values are exactly "default", "octet_rule", and "hypervalent". This parameter is case-sensitive and selects one of the built-in semantic constraint sets: "default" (the library default bonding capacities used for typical molecules), "octet_rule" (constraints chosen to enforce the classical octet rule for main-group elements), and "hypervalent" (constraints that allow higher bonding capacities to accommodate hypervalent species). Provide one of these literal strings; passing any other string will trigger a ValueError.
    
    Returns:
        Dict[str, int]: A dictionary mapping atom tokens (keys) to their integer bonding capacities (values). Atom tokens are the same atom labels used by SELFIES symbols and may include plain element tokens like "C", "N", "O", "Cl", "Br", "I" and also variant tokens used to indicate alternative valence states such as "P+1", "P-1", "S+1", and "S-1". The mapping represents the maximum number of bonds that an atom token is allowed to form under the chosen preset semantic policy. The function returns a new dict copy of the preset so callers can inspect or locally modify the mapping without affecting the global preset definitions.
    
    Behavior and defaults:
        The three built-in presets differ in how permissive they are for certain atoms. For example, for typical keys shown in the library documentation:
        - "default" permits halogens Cl, Br, I to form 1 bond; N to form 3 bonds; P to form 5 bonds with variants P+1 = 4 and P-1 = 6; S to form 6 bonds with variants S+1 = 5 and S-1 = 5.
        - "octet_rule" enforces lower capacities consistent with the octet rule: Cl, Br, I = 1; N = 3; P = 3 with P+1 = 4 and P-1 = 2; S = 2 with S+1 = 3 and S-1 = 1.
        - "hypervalent" is more permissive for atoms that can exhibit hypervalence: Cl, Br, I = 7; N = 5; P = 5 with P+1 = 6 and P-1 = 4; S = 6 with S+1 = 7 and S-1 = 5.
        These example capacities mirror the documented presets in the SELFIES repository and illustrate how choosing a preset affects the allowed bonding patterns during SELFIES operations. The function itself applies no chemistry validation beyond returning the selected mapping.
    
    Side effects:
        This function has no side effects on global state; it returns a new dictionary copy of the requested preset. It does not change the active semantic constraints used by other functions in the library. To set the active constraints globally, call selfies.set_semantic_constraints with an appropriate name after retrieving or inspecting presets.
    
    Failure modes:
        If name is not one of the recognized preset keys ("default", "octet_rule", "hypervalent"), the function raises ValueError with a message indicating the unrecognized preset name. No other exceptions are raised by this function under normal usage.
    """
    from selfies.bond_constraints import get_preset_constraints
    return get_preset_constraints(name)


################################################################################
# Source: selfies.compatibility.modernize_symbol
# File: selfies/compatibility.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def selfies_compatibility_modernize_symbol(symbol: str):
    """Converts a SELFIES symbol from a pre-v2 representation to its latest equivalent.
    
    This function is part of the compatibility utilities in the selfies package and is used when ingesting or transforming SELFIES strings produced by older versions of the library. It attempts to map legacy symbols to their modern equivalents so that downstream SELFIES processing (encoding, decoding, alphabet construction, model input preparation) operates on the current symbol set. The function first consults an internal mapping table of explicit symbol updates. If no direct mapping exists, it applies a second heuristic transform for legacy "expl" symbols (symbols that end with the substring "expl]"), handling an optional bond prefix character and normalizing the atom symbol via the package's SMILES/atom helpers when possible. If no update applies, the original symbol is returned unchanged.
    
    Args:
        symbol (str): A single SELFIES symbol string to modernize. The value is expected to be a SELFIES token such as "[C]" or legacy forms like "[XXXexpl]". This argument is required and must be a Python str. The function examines this string to determine if a newer equivalent exists in the internal _SYMBOL_UPDATE_TABLE or if it matches the legacy "expl" pattern.
    
    Returns:
        str: A SELFIES symbol string representing the modernized equivalent of the input. If the input symbol appears as a key in the internal _SYMBOL_UPDATE_TABLE, the mapped (up-to-date) symbol is returned. If the input ends with "expl]" (for example "[XXXexpl]"), the function will attempt to split off an initial bond character if present (one of '=', '#', '/', '\\') and parse the remaining atom token using the package helpers smiles_to_atom and atom_to_smiles to obtain a standardized, bracket-free atom representation for non-aromatic atoms; in that case the function returns a reconstructed symbol that combines the optional bond prefix with the standardized atom (for example producing "[=Si]" or "[Si]" depending on input). If parsing fails, the atom is aromatic, or no mapping rule applies, the function returns the original input symbol unchanged.
    
    Behavioral notes, side effects, and failure modes:
        - The function performs a pure string-to-string transformation and does not mutate global SELFIES strings; the only external lookups are the internal mapping table _SYMBOL_UPDATE_TABLE and the helper routines smiles_to_atom and atom_to_smiles used for standardization.
        - For symbols that match the "expl" suffix handling, the function checks symbol[1] to detect an initial bond character; this assumes the input is a non-empty string formatted as a SELFIES token (leading '['). If the input is malformed (not a SELFIES-like token), the function will not explicitly validate format beyond the string operations and will typically return the original string.
        - The helper smiles_to_atom may return None for an unrecognized atom string; in that case, no normalization occurs and the original symbol (or the mapping table result) is returned.
        - If smiles_to_atom returns an atom object and atom.is_aromatic is False, atom_to_smiles is called with brackets=False to produce a standardized atom symbol; aromatic atoms are left unchanged by that normalization step.
        - The function does not introduce new symbol types beyond returning a str. Any exceptions raised by smiles_to_atom or atom_to_smiles (for example, due to malformed SMILES/atom inputs) are propagated; callers should handle such exceptions if they pass untrusted inputs.
        - This function is intended to be used prior to decoding or other semantic processing of SELFIES strings so that legacy tokens do not prevent successful interpretation by newer library versions.
    """
    from selfies.compatibility import modernize_symbol
    return modernize_symbol(symbol)


################################################################################
# Source: selfies.decoder.decoder
# File: selfies/decoder.py
# Category: valid
################################################################################

def selfies_decoder_decoder(selfies: str, compatible: bool = False, attribute: bool = False):
    """selfies.decoder.decoder: Translate a SELFIES string into its corresponding SMILES string.
    
    Translates a SELFIES molecular representation into a SMILES string deterministically under the library's current semantic constraints (the constraints configured via selfies.set_semantic_constraints). This function is used throughout the SELFIES package and in downstream machine-learning workflows (for example, generative models and random-molecule generation) to convert the robust SELFIES token sequence back into a conventional SMILES string that can be consumed by cheminformatics tools. The output SMILES is guaranteed by the decoder implementation to be syntactically valid and to represent a molecule that obeys the active semantic constraints. The decoder processes dot-separated fragments (SELFIES segments separated by ".") and assembles rings and branches according to those constraints. The implementation constructs an internal MolecularGraph, forms rings bilocally, and then serializes the graph to SMILES; when attribution is requested the function also tracks which input SELFIES tokens contributed to each output SMILES token.
    
    Args:
        selfies (str): The input SELFIES string to be translated. This must be a sequence of SELFIES symbols (for example, "[C][=O][Branch1]...") possibly containing multiple fragments separated by ".". SELFIES is a robust molecular string representation designed to be used directly as input to machine-learning models; providing a valid SELFIES string ensures deterministic decoding. The function will tokenize the provided string and derive a MolecularGraph from those tokens. Common practical uses include translating SELFIES produced by generative models back into standard SMILES for downstream validation, visualization, or property prediction.
        compatible (bool = False): If True, accept deprecated SELFIES symbols from previous releases when tokenizing and deriving the molecular graph. This is a permissive compatibility mode intended to help migrate older SELFIES strings, but it does not guarantee exact backward compatibility across major releases and the decoded result may differ from historical behavior. When this flag is True the function emits a runtime warning advising the user to update SELFIES to the current alphabet. Default is False.
        attribute (bool = False): If True, produce an attribution mapping in addition to the SMILES string that explains which input SELFIES tokens contributed to each output SMILES token. This is useful for explainability and debugging (for example, to trace how branched tokens or ring tokens in SELFIES map to specific SMILES characters). When False (the default), only the SMILES string is returned and no attribution structures are created; when True, the decoder builds and returns attribution information while deriving the MolecularGraph, which increases memory/work but enables downstream inspection.
    
    Returns:
        str: When attribute is False, returns a single SMILES string that is the decoded representation of the input SELFIES. The returned SMILES is syntactically valid and respects the current semantic constraints.
        Tuple[str, List[Tuple[str, List[Tuple[int, str]]]]]: When attribute is True, returns a tuple (smiles, attribution_list). smiles is the decoded SMILES string (as above). attribution_list is a list where each element is a tuple (smiles_token, contributors). smiles_token is a string representing one output SMILES token (for example, an atom symbol or punctuation in the SMILES tokenization used by the decoder). contributors is a list of pairs (index, selfies_symbol) where index is an int indicating the index position of the contributing SELFIES token in the tokenized input (indexing is the order used by the decoder during derivation) and selfies_symbol is the exact SELFIES symbol string (type str) at that index. This attribution structure allows a caller to map output SMILES tokens back to the specific input SELFIES tokens that produced them; types and nesting match the function signature exactly.
    
    Raises:
        DecoderError: If the input SELFIES string is malformed in a way that prevents tokenization or molecular derivation (for example, invalid symbol syntax, impossible local derivation under the active semantic constraints, or other structural errors), a DecoderError is raised. Callers should catch DecoderError when decoding arbitrary or externally sourced SELFIES.
        RuntimeWarning: If compatible is True, the function issues a runtime warning advising that decoding behavior may differ from previous major releases and recommending updating SELFIES symbols.
    
    Behavior and side effects:
        - The translation is deterministic for a given set of semantic constraints; changing global semantic constraints (via selfies.set_semantic_constraints) will change decoding results for the same SELFIES input.
        - The function internally constructs a MolecularGraph object and mutates it while deriving bonds, rings, and atom properties. The MolecularGraph construction is a necessary side effect to produce the SMILES string and, when attribute is True, to gather attribution information from the derivation process.
        - The decoder treats the special "[nop]" SELFIES symbol as a no-operation/padding symbol that is ignored during derivation (useful when SELFIES strings are padded for fixed-length encodings).
        - The function handles multiple fragments separated by "." in the input SELFIES and attempts to form rings bilocally before serializing to SMILES.
        - Enabling attribute increases memory usage and computation because the function maintains attribution stacks and indices during derivation.
        - When compatible is True the decoder will attempt to accept deprecated symbols but may not reproduce historical outputs exactly; callers should update inputs to the current SELFIES alphabet where precise reproducibility is required.
    
    Practical significance:
        - In machine-learning pipelines, selfies.decoder.decoder is the canonical utility for converting model outputs in the SELFIES language back to SMILES for evaluation with cheminformatics toolkits, property calculators, or human inspection.
        - The attribution mode helps interpret model outputs and debug token-to-token correspondences (for example, to understand how SELFIES branch or ring tokens contribute to particular SMILES substructures).
    """
    from selfies.decoder import decoder
    return decoder(selfies, compatible, attribute)


################################################################################
# Source: selfies.encoder.encoder
# File: selfies/encoder.py
# Category: valid
################################################################################

def selfies_encoder_encoder(smiles: str, strict: bool = True, attribute: bool = False):
    """selfies.encoder.encoder translates a SMILES string into its corresponding SELFIES string. This function is the encoder entry point in the selfies package and is used in the chemical string-processing workflow described in the README: it produces a deterministic, machine-learning-friendly SELFIES representation (robust molecular string representation) from a SMILES input, preserving the input atom order so randomized SMILES yield randomized SELFIES.
    
    Args:
        smiles (str): The input SMILES string to be translated into SELFIES. In practice this string should be a chemically valid SMILES (for example validated with RDKit) because the function first parses the SMILES via smiles_to_mol(smiles, attributable=attribute). The parser may fail for syntactically invalid SMILES or unsupported constructs; such failures are converted to a selfies.EncoderError with a message that includes the failing SMILES. The function supports aromatic SMILES by internally kekulizing them prior to translation.
        strict (bool): If True (default), the function checks that the parsed molecule obeys the current semantic constraints configured for selfies (for example via selfies.set_semantic_constraints). This check is performed by calling the internal _check_bond_constraints(mol, smiles). If the check fails, the function raises selfies.EncoderError. If False, the function will not raise an error for semantic-constraint violations; however, SELFIES strings produced with strict=False are not guaranteed to decode back to a SMILES representing the original molecule. The translation itself is deterministic and does not otherwise depend on the current semantic constraints.
        attribute (bool): If False (default), the function returns only the SELFIES string. If True, the function constructs and returns an attribution list alongside the SELFIES string; the internal parser is invoked with attributable=attribute so token-level attribution information is collected while constructing the SELFIES. Attribution maps are trimmed of any entries with empty tokens before being returned.
    
    Returns:
        str: When attribute is False, returns a SELFIES string derived from the input SMILES. The SELFIES string is produced by iterating over molecular fragments (mol.get_roots()), converting each fragment with the internal _fragment_to_selfies routine, and joining fragment SELFIES with '.' for disconnected components.
        tuple: When attribute is True, returns a tuple (selfies_string, attribution_maps). selfies_string is as above. attribution_maps is a list of attribution entries (AttributionMap objects as used by the decoder/attribution machinery in this package) that record which input tokens contributed to output tokens; empty-token entries are removed prior to return.
    
    Raises:
        EncoderError: Raised when the input SMILES cannot be parsed, when kekulization fails, or when the molecule violates semantic constraints and strict is True. Parser errors (SMILESParserError) are caught and re-raised as EncoderError with a message of the form "failed to parse input\tSMILES: {smiles}". Kekulization failures raise EncoderError with a message indicating kekulization failed for the input SMILES.
    
    Behavior, side effects, and failure modes:
        The function first parses the SMILES into an internal molecular representation via smiles_to_mol. It then attempts to kekulize the molecule; kekulization is required for the encoder and the function raises EncoderError if kekulization fails. If strict is True, bond/valence rules described by the current semantic constraints are checked and violations raise EncoderError. To preserve stereochemical information across encode/decode, the encoder will invert the chirality of certain atoms in the internal molecule representation (atoms with atom.chirality not None where mol.has_out_ring_bond(atom.index) and _should_invert_chirality(mol, atom) evaluate True) so that decoding the produced SELFIES will restore original chirality. The function constructs SELFIES fragment strings for each connected component/root and joins them with '.'; when attribute=True it also accumulates per-token attribution maps and trims empty tokens before returning. Limitations inherited from the implementation: this function does not support the wildcard '*' symbol, the quadruple bond symbol '$', chirality specifications other than '@' and '@@', ring bonds across a dot (e.g., fragments joined by '.' that share a ring), or ring bonds between atoms more than 4000 atoms apart. Aromatic SMILES are supported by internal kekulization prior to translation.
    """
    from selfies.encoder import encoder
    return encoder(smiles, strict, attribute)


################################################################################
# Source: selfies.utils.encoding_utils.batch_flat_hot_to_selfies
# File: selfies/utils/encoding_utils.py
# Category: valid
################################################################################

def selfies_utils_encoding_utils_batch_flat_hot_to_selfies(
    one_hot_batch: List[List[int]],
    vocab_itos: Dict[int, str]
):
    """selfies.utils.encoding_utils.batch_flat_hot_to_selfies converts a batch of flattened one-hot encodings produced by machine learning models into their corresponding SELFIES strings.
    
    Args:
        one_hot_batch (List[List[int]]): A batch (list) of flattened one-hot encodings. Each element is a flat list of integers representing a sequence of one-hot vectors concatenated into a single list. The length of each inner list must be divisible by len(vocab_itos); the function will reshape each flat vector into L rows and M columns where M = len(vocab_itos) and L = len(flat_vector) // M. In the SELFIES/cheminformatics context (see README), these encodings typically come from neural network outputs that represent positions in a SELFIES sequence over a fixed vocabulary (for example, [nop] padding and atom/token symbols).
        vocab_itos (Dict[int, str]): A vocabulary mapping from integer indices to SELFIES symbols (index-to-string), for example {0: "[nop]", 1: "[C]"}. The length of this dictionary (M) defines the number of columns used when unflattening each flat one-hot vector. This mapping is used by encoding_to_selfies to translate per-position one-hot vectors into the corresponding SELFIES tokens.
    
    Returns:
        List[str]: A list of SELFIES strings, one per input flattened encoding, in the same order as one_hot_batch. Each returned string is produced by unflattening the corresponding input vector into an L x M one-hot matrix and decoding it with encoding_to_selfies(enc_type="one_hot", vocab_itos=vocab_itos). Returned SELFIES strings may include special symbols such as [nop] used for padding; per README, [nop] is ignored by the decoder and is useful for fixed-length model outputs.
    
    Behavior, side effects, and failure modes:
        The function performs no external side effects (no I/O); it purely transforms inputs to outputs. For each flat_one_hot in one_hot_batch, it computes M = len(vocab_itos) and checks divisibility: if len(flat_one_hot) % M != 0 a ValueError is raised with the message "size of vector in one_hot_batch not divisible by the length of the vocabulary." After reshaping into a list of L rows (each row a sublist of length M), the function calls encoding_to_selfies(one_hot, vocab_itos, enc_type="one_hot") to obtain the SELFIES string and appends it to the output list. Any exceptions raised by encoding_to_selfies (for example, due to invalid one-hot vectors that cannot be decoded or malformed vocabulary mappings) will propagate to the caller. The function expects the inner lists to contain integer entries (typically 0 or 1 for strict one-hot encodings); supplying other types or shapes may cause decoding errors. This utility is intended for downstream conversion of batched model outputs into SELFIES for subsequent decoding to SMILES or downstream cheminformatics tasks described in the project README.
    """
    from selfies.utils.encoding_utils import batch_flat_hot_to_selfies
    return batch_flat_hot_to_selfies(one_hot_batch, vocab_itos)


################################################################################
# Source: selfies.utils.encoding_utils.batch_selfies_to_flat_hot
# File: selfies/utils/encoding_utils.py
# Category: valid
################################################################################

def selfies_utils_encoding_utils_batch_selfies_to_flat_hot(
    selfies_batch: List[str],
    vocab_stoi: Dict[str, int],
    pad_to_len: int = -1
):
    """Converts a batch of SELFIES strings into flattened one-hot encodings.
    
    This function is a convenience wrapper used in the SELFIES project to prepare batches of SELFIES strings as fixed-length flat feature vectors for machine learning workflows (for example, input to generative models, VAEs, or classifiers that expect 1D vectors). For each SELFIES string in selfies_batch, this function calls selfies.selfies_to_encoding(selfies, vocab_stoi, pad_to_len, enc_type="one_hot") to obtain a sequence of one-hot vectors and then flattens that sequence into a single list of integers (0/1). The mapping from SELFIES symbols to one-hot positions is determined by vocab_stoi. The pad_to_len argument is forwarded to selfies_to_encoding and controls the padding behavior used when converting each SELFIES string to a fixed-length sequence.
    
    Args:
        selfies_batch (List[str]): A list of SELFIES strings to be encoded. Each element is a SELFIES molecular string (for example, "[C][O][C]"). The order of strings is preserved in the returned list so that the i-th output corresponds to the i-th input. This argument is used when constructing datasets or minibatches for model training or evaluation in cheminformatics tasks described in the README.
        vocab_stoi (Dict[str, int]): A vocabulary mapping from SELFIES symbol (string) to integer index. The index values define the column positions in each one-hot vector produced by selfies_to_encoding; the length of each per-symbol one-hot vector is len(vocab_stoi). vocab_stoi must contain every symbol that appears in the SELFIES strings in selfies_batch or the underlying encoding call may raise an error.
        pad_to_len (int): The length to which each SELFIES string is padded when converted to a sequence of one-hot vectors. The value is forwarded unchanged to selfies_to_encoding. Defaults to -1. When using a non-negative value, each encoded (and flattened) output will contain pad_to_len one-hot vectors (one per sequence position) each of length len(vocab_stoi), producing a flattened length of pad_to_len * len(vocab_stoi). When pad_to_len is left at the default or when its semantics are handled by selfies_to_encoding, the actual per-sequence length is determined by selfies_to_encoding; consult that function for exact behavior.
    
    Behavior and failure modes:
    This function performs no in-place mutation of inputs and has no side effects beyond calling selfies_to_encoding for each string in selfies_batch. If a SELFIES symbol in selfies_batch is not present in vocab_stoi, the underlying selfies_to_encoding call is expected to raise an error (such as KeyError or a library-specific encoding exception). The caller is responsible for ensuring vocab_stoi covers the alphabet of selfies_batch and for selecting an appropriate pad_to_len for consistent vector sizes across the batch if required by downstream models.
    
    Returns:
        List[List[int]]: A list with the same length as selfies_batch. Each element is a flattened one-hot encoding represented as a list of integers (0 or 1). For each input SELFIES string, the corresponding output list contains the concatenation of its per-symbol one-hot vectors in sequence order (row-major flattening). If pad_to_len is non-negative, each inner list will have length pad_to_len * len(vocab_stoi); otherwise the inner list length equals the sequence length used by selfies_to_encoding multiplied by len(vocab_stoi).
    """
    from selfies.utils.encoding_utils import batch_selfies_to_flat_hot
    return batch_selfies_to_flat_hot(selfies_batch, vocab_stoi, pad_to_len)


################################################################################
# Source: selfies.utils.encoding_utils.selfies_to_encoding
# File: selfies/utils/encoding_utils.py
# Category: valid
################################################################################

def selfies_utils_encoding_utils_selfies_to_encoding(
    selfies: str,
    vocab_stoi: Dict[str, int],
    pad_to_len: int = -1,
    enc_type: str = "both"
):
    """selfies.utils.encoding_utils.selfies_to_encoding converts a SELFIES string into an integer label encoding and/or a one-hot encoding suitable for machine learning workflows that operate on sequence representations of molecules. This function is used in the SELFIES library to transform the SELFIES token sequence (a robust molecular string representation described in the README) into numeric encodings for use as model inputs, dataset storage, batching, and downstream tasks such as training generative or predictive models. The function uses the helper functions len_selfies and split_selfies to determine symbol length and to tokenize the SELFIES string.
    
    Args:
        selfies (str): The SELFIES string to encode. SELFIES is a sequence of bracketed symbols (for example, "[C][=O][Ring1]") representing molecular graphs. The function treats the input as an immutable string; internally it may concatenate the padding symbol "[nop]" to a local copy of this string when pad_to_len requires padding, but the caller's variable is not mutated. The symbol length L used for encoding is computed with len_selfies(selfies), and tokenization is performed with split_selfies(selfies).
        vocab_stoi (Dict[str, int]): A mapping from SELFIES symbols to integer indices. Indices must be integers in the range 0..(len(vocab_stoi)-1), because one-hot vectors are created with length equal to len(vocab_stoi) and the integer indices are used to set a single 1 in those vectors. The mapping is expected (by convention and practical use) to be non-negative and contiguous starting at 0, so that models and indexing behave predictably; if pad_to_len is used, the special padding symbol "[nop]" must be present as a key in this dictionary. The function does not modify this dictionary.
        pad_to_len (int = -1): If greater than the symbol length L of the input (computed by len_selfies), the function pads the SELFIES string on the right with the padding symbol "[nop]" repeated (pad_to_len - L) times before encoding. If pad_to_len is less than or equal to L (including the default -1), no padding is added. Padding is commonly used in batch processing and sequence models to produce fixed-length encodings across examples. Side effect: only a local copy of the SELFIES string is padded; no external state is changed.
        enc_type (str = 'both'): Determines the encoding(s) returned. Accepted values are "label", "one_hot", and "both". "label" returns the integer label encoding (a list of integers of length L or pad_to_len if padded). "one_hot" returns the one-hot encoding (a list of L lists, each inner list of length len(vocab_stoi) with a single 1 at the index given by vocab_stoi). "both" returns a tuple (label_encoding, one_hot_encoding). Default is "both". If an invalid value is supplied, the function raises ValueError.
    
    Returns:
        List[int]: When enc_type is "label", returns a list of integers of length equal to the symbol length L (or pad_to_len if padding was applied). Each integer is vocab_stoi[symbol] for the corresponding SELFIES symbol. This encoding is suitable for models or pipelines that expect categorical integer labels.
        List[List[int]]: When enc_type is "one_hot", returns a two-dimensional list of shape (L, len(vocab_stoi)). Each inner list is a one-hot vector with a single 1 at the index given by the integer encoding and 0s elsewhere. This encoding is suitable for models that take explicit binary indicator vectors per time-step or token.
        Tuple[List[int], List[List[int]]]: When enc_type is "both", returns a tuple where the first element is the label encoding (List[int]) and the second element is the one-hot encoding (List[List[int]]). The label and one-hot encodings correspond elementwise.
        The function never returns None. It returns one of the above types exactly according to enc_type.
    
    Behavior and failure modes:
        - The symbol length L is determined by len_selfies(selfies). Tokenization is performed with split_selfies(selfies). If pad_to_len > L, the function appends "[nop]" repeated (pad_to_len - L) times to a local copy of the selfies string before tokenization and encoding.
        - If enc_type is not one of "label", "one_hot", or "both", the function raises ValueError with the message "enc_type must be in ('label', 'one_hot', 'both')".
        - If the SELFIES string contains the separator character "." (indicating multiple disconnected molecules) and vocab_stoi does not contain the "." key, the function raises KeyError with guidance that the "." key should be added to the vocabulary or the molecules separated. Other missing symbol keys in vocab_stoi will also raise KeyError at lookup time; users must ensure all symbols produced by split_selfies(selfies) (including "[nop]" if padding is requested) are present in vocab_stoi.
        - The function assumes vocab_stoi indices are valid positions in a one-hot vector of length len(vocab_stoi). If indices are out of range or non-integer, behavior will be an indexing error or incorrect one-hot vectors; therefore indices should be integers in 0..len(vocab_stoi)-1.
        - The function does not perform chemical validation or semantic checks beyond token lookup. It merely converts tokens to numeric encodings for use in machine learning or dataset operations.
    
    Practical significance in the SELFIES domain:
        - Converting SELFIES to integer or one-hot encodings is a standard preprocessing step when using SELFIES as direct input to neural networks (for example, sequence models or generative models described in the README). The label encoding is compact and efficient for embedding layers, while the one-hot encoding is explicit and useful for models that require binary token features or for visualization and debugging.
        - Padding with "[nop]" enables fixed-length representations required for mini-batching and consistent tensor shapes during training. The caller must ensure the vocabulary includes "[nop]" if padding will be applied.
        - This function is deterministic and side-effect free with respect to external structures (it returns new lists and does not mutate the provided vocabulary or input variables).
    """
    from selfies.utils.encoding_utils import selfies_to_encoding
    return selfies_to_encoding(selfies, vocab_stoi, pad_to_len, enc_type)


################################################################################
# Source: selfies.utils.matching_utils.find_perfect_matching
# File: selfies/utils/matching_utils.py
# Category: valid
################################################################################

def selfies_utils_matching_utils_find_perfect_matching(graph: List[List[int]]):
    """Finds a perfect matching for an undirected graph (without self-loops) and returns a node-to-partner mapping or None if no perfect matching exists.
    
    This utility is part of the selfies.utils.matching_utils module and is intended for use inside the SELFIES codebase where algorithms operate on graph representations of molecules (for example, molecular connectivity graphs used when manipulating rings, branches, or during translation between SMILES and SELFIES). The function accepts an adjacency-list representation of an undirected graph and attempts to pair every vertex with exactly one neighbor so that each vertex appears in exactly one matched pair. The implementation first constructs a maximal greedy matching for efficiency and then repeatedly searches for augmenting paths and flips them until either a perfect matching covering all vertices is found or no augmenting path exists (in which case the function reports failure). The input graph must not contain self-loops; edges are expected to be represented by integer indices in the adjacency lists. The returned matching is a compact representation suitable for downstream graph algorithms in SELFIES that require disjoint pairings of nodes.
    
    Args:
        graph (List[List[int]]): An adjacency list representing an undirected graph without self-loops. The outer list has length n, where n is the number of nodes; each inner list contains integer indices of neighbors of that node. Node indices are integers in the range [0, n-1]. Because the graph is undirected, each edge should appear in both endpoints' adjacency lists (for example, if j appears in graph[i], then i should appear in graph[j]). The function does not modify the input adjacency lists.
    
    Returns:
        Optional[List[int]]: If a perfect matching exists, returns a list of length n where the i-th element is an integer j indicating that node i is matched with node j. For a valid perfect matching returned by this function, every element is an integer index in [0, n-1], matching[i] == j implies matching[j] == i, and every node appears exactly once in a matched pair. If no perfect matching exists (for example, when the graph has an odd number of vertices or when some component cannot be fully matched and no augmenting path can be found), returns None.
    """
    from selfies.utils.matching_utils import find_perfect_matching
    return find_perfect_matching(graph)


################################################################################
# Source: selfies.utils.selfies_utils.len_selfies
# File: selfies/utils/selfies_utils.py
# Category: valid
################################################################################

def selfies_utils_selfies_utils_len_selfies(selfies: str):
    """Returns the number of symbols in a given SELFIES string. In the SELFIES molecular-string domain used throughout this repository (see README), each atomic or structural token is represented as a bracketed symbol such as "[C]" or "[=C]", and the period character "." is used as an explicit symbol to separate disconnected fragments. This function therefore computes the symbol length that is commonly required when constructing alphabets, computing padding lengths for encodings, or sizing input layers for machine learning models that operate on SELFIES.
    
    Args:
        selfies (str): A SELFIES string to measure. The function treats each occurrence of the character "[" as the start of one bracketed SELFIES symbol and each literal period character "." as an extra symbol (for disconnected fragments). The argument must be a Python str; if a non-str is supplied, attribute access will fail (e.g., an AttributeError) because the implementation calls str.count on the input.
    
    Returns:
        int: The symbol count of the provided SELFIES string. Concretely, this is computed as the number of "[" characters plus the number of "." characters in the input string. The returned integer corresponds to the number of SELFIES tokens one would expect when tokenizing the string into bracketed symbols and period separators, and it is suitable for use as a padding length or expected token sequence length in SELFIES-to-encoding and encoding-to-SELFIES workflows.
    
    Behavior and failure modes:
        This function performs only a simple structural count and does not validate SELFIES syntax or chemical semantics; malformed bracketings, unexpected characters, or logically invalid SELFIES will still be counted based solely on occurrences of "[" and ".". The implementation uses two calls to str.count and thus runs in linear time with respect to the length of the input string (two scans of the string, i.e., O(n)). There are no side effects; the function is pure and deterministic.
    """
    from selfies.utils.selfies_utils import len_selfies
    return len_selfies(selfies)


################################################################################
# Source: selfies.utils.selfies_utils.split_selfies
# File: selfies/utils/selfies_utils.py
# Category: fix_docstring
# Reason: Schema parsing failed: ("Couldn't parse this type hint, likely due to a custom class or object: ", typing.Iterator[str])
################################################################################

def selfies_utils_selfies_utils_split_selfies(selfies: str):
    """selfies.utils.selfies_utils.split_selfies tokenizes a SELFIES string into its individual symbols and yields them in the original order. This function implements the lexical splitting used across the SELFIES library to convert a SELFIES string (the bracketed-symbol molecular string representation described in the README) into discrete tokens suitable for building alphabets, creating integer or one-hot encodings, and feeding generative or discriminative machine learning models.
    
    Args:
        selfies (str): A SELFIES string to tokenize. The string is expected to contain bracketed symbols such as "[C]", "[=O]" etc., possibly separated by the dot character "." which denotes a molecular separator. This function does not perform semantic validation of symbols or chemical valence rules; it only performs lexical tokenization.
    
    Returns:
        Iterator[str]: An iterator that yields each symbol of the input SELFIES string one-by-one, in left-to-right order. Bracketed symbols are yielded including their surrounding brackets (for example, "[C]" or "[=C]"); a "." that immediately follows a bracketed symbol is yielded as a separate token ".". The iterator is lazy (generator-style), so tokens are produced on demand and the function does not construct a full list unless the caller forces it (for example, via list()).
    
    Behavior and side effects:
        The function searches for the first "[" and then repeatedly finds the matching "]" to extract bracketed symbols. After yielding a bracketed symbol, if the next character is a dot ".", that dot is yielded as a separate token and iteration continues. There are no external side effects; the function does not modify its input and does not depend on or change global state.
    
    Failure modes and edge cases:
        If a "[" is found with no corresponding closing "]" later in the string, the function raises ValueError("malformed SELFIES string, hanging '[' bracket") to signal a malformed SELFIES string. If the input contains no "[" characters, the returned iterator yields no tokens (i.e., iterates to completion immediately). Note that dots that appear before the first bracketed symbol or dots not immediately following a bracketed symbol are not emitted by this routine. The function performs lexical splitting only and does not guarantee that the yielded sequence represents a semantically valid molecule under SELFIES semantic constraints.
    """
    from selfies.utils.selfies_utils import split_selfies
    return split_selfies(selfies)


################################################################################
# Source: selfies.utils.smiles_utils.smiles_to_atom
# File: selfies/utils/smiles_utils.py
# Category: valid
################################################################################

def selfies_utils_smiles_utils_smiles_to_atom(atom_symbol: str):
    """selfies.utils.smiles_utils.smiles_to_atom parses a single SMILES atom token and returns the corresponding selfies.mol_graph.Atom object used by the SELFIES translation pipeline.
    
    Args:
        atom_symbol (str): A SMILES atom symbol token to parse. This is a single SMILES atom string such as an unbracketed organic atom ("C", "n", "O"), an unbracketed aromatic atom (lowercase like "c", "n"), or a bracketed atom token (for example "[13CH@H+]", "[O-]", "[C@@H]"). In the SELFIES/SMILES translation context (see the SELFIES README), this function is used by encoder/decoder utilities to convert SMILES atom tokens into the internal Atom representation for constructing or analyzing molecular graphs. The function expects a non-empty Python str containing exactly one SMILES atom token.
    
    Returns:
        Optional[selfies.mol_graph.Atom]: An Atom instance representing the parsed atom when the input is recognized and maps to a known chemical element. The returned Atom contains the following semantic fields used throughout SELFIES: element (capitalized chemical symbol, e.g., "C", "O"), is_aromatic (bool), isotope (Optional[int], None if unspecified), chirality (Optional[str], None if unspecified), h_count (int, number of implicit/explicit H atoms, default 0), and charge (int, signed formal charge, default 0). Practical behaviors:
        - For unbracketed organic subset tokens (e.g., "C", "N", "O") this returns Atom(element=atom_symbol, is_aromatic=False) with other Atom attributes left at their defaults used by the SELFIES mol_graph.
        - For unbracketed aromatic subset tokens (lowercase like "c", "n", "o") this returns Atom(element=atom_symbol.capitalize(), is_aromatic=True).
        - For bracketed atom tokens the function parses isotope, element symbol, chirality descriptor, hydrogen count (H, H2, ...), and charge using the module's SMILES_BRACKETED_ATOM_PATTERN. Parsed values are converted to the Atom fields as follows: isotope -> int or None; element -> capitalized string and validated against ELEMENTS; chirality -> string or None; h_count -> integer (0 if absent, 1 if "H" present without a number); charge -> integer (0 if absent; numeric form like "+2" or "-1" parsed as signed integer; repeated signs "+++" or "---" interpreted as magnitude equal to the number of signs and sign determined by the first character).
        - If the bracketed pattern does not match, the element is not in the recognized ELEMENTS set, or the token is otherwise unrecognized, the function returns None to signal an unparsable or unsupported SMILES atom token. This return value is used by higher-level SELFIES functions to detect invalid tokens during encoding/decoding.
    
    Behavioral notes, defaults, and failure modes:
        - The function performs no I/O and has no side effects; it is pure and deterministic for a given input string.
        - The function expects a non-empty string. Supplying an empty string will raise an IndexError because the implementation inspects atom_symbol[0] and atom_symbol[-1].
        - The function does not itself raise parsing exceptions for malformed bracketed atoms; instead it returns None when the input does not match the expected SMILES atom token patterns or when the parsed element is not recognized in the module's ELEMENTS set.
        - Charge parsing follows SMILES conventions implemented in the code: numeric signed charges like "+2" are parsed as integers; repeated signs like "+++" are parsed as the count of signs with the appropriate sign.
        - Hydrogen counts follow bracketed SMILES conventions: absent -> 0, "H" -> 1, "H2" -> 2, etc.
        - Aromaticity for bracketed atoms is inferred from the element symbol being lowercase and present in the module's AROMATIC_SUBSET.
    
    Examples of practical significance in the SELFIES project:
        - During SMILES-to-SELFIES encoding and SELFIES-to-SMILES decoding (see README overview), this function is a low-level utility that maps SMILES atom tokens to the Atom objects used to build molecular graphs, enforce semantic constraints, and ensure that generated SELFIES map back to chemically valid structures.
        - The function is relied upon by tokenization and validation routines in the SELFIES library and by downstream generative-model pipelines that require accurate atom-level interpretation of SMILES strings.
    """
    from selfies.utils.smiles_utils import smiles_to_atom
    return smiles_to_atom(atom_symbol)


################################################################################
# Source: selfies.utils.smiles_utils.smiles_to_mol
# File: selfies/utils/smiles_utils.py
# Category: valid
################################################################################

def selfies_utils_smiles_utils_smiles_to_mol(smiles: str, attributable: bool):
    """Reads a molecular graph representation from a SMILES string and returns a MolecularGraph suitable for downstream SELFIES operations (encoding, decoding, attribution). This function is a deterministic parser that tokenizes the input SMILES, iteratively derives atoms/bonds/structural features by repeatedly calling internal parsing logic, and constructs a selfies.mol_graph.MolecularGraph object that represents the molecular graph implied by the SMILES. In the SELFIES/SMILES workflow described in the project README, this MolecularGraph is the canonical in-memory graph representation used by higher-level functions (for example, the SELFIES encoder/decoder and attribution utilities).
    
    Args:
        smiles (str): The input SMILES string to parse. This is the standard linear textual representation of a molecule (symbols, branches, ring closures, bond orders, etc.). The function does not modify this string. An empty string is considered invalid and will trigger a SMILESParserError with a message indicating "empty SMILES" and a position of 0. The SMILES must be syntactically valid for the tokenizer and the parsing routines to succeed.
        attributable (bool): Flag indicating whether the returned MolecularGraph should include attribution bookkeeping structures. When True, the MolecularGraph will be constructed with attribution-capable internals (to record mappings from input tokens to graph elements), which enables downstream attribution queries (for example, tracing which input tokens led to particular atoms or SMILES tokens in translation). When False, the MolecularGraph will omit attribution information to reduce memory/overhead. This parameter corresponds directly to the attributable argument passed to the MolecularGraph constructor and has no default in the function signature.
    
    Returns:
        selfies.mol_graph.MolecularGraph: A newly constructed MolecularGraph instance that encodes the atoms, bonds, ring/branch structure, and any associated metadata parsed from the input SMILES. The returned graph is ready for use by other SELFIES utilities (encoder/decoder, attribution inspection, graph-to-string conversion). If attributable was True, the MolecularGraph will contain attribution mappings to relate input tokens to graph elements; if False, those attribution mappings will be absent.
    
    Raises:
        SMILESParserError: If the input SMILES is invalid or cannot be tokenized/parsed. Examples of failure modes include an empty input SMILES (explicitly raised with message "empty SMILES" at position 0), invalid token sequences emitted by the tokenizer, unbalanced branch or ring specifications, or other syntactic problems detected by the internal _derive_mol_from_tokens routine. The SMILESParserError includes context (the offending SMILES and a message and position) to aid debugging.
    
    Behaviour and side effects:
        The function creates and returns a new MolecularGraph object and does not mutate external state. Internally it calls tokenize_smiles(smiles) to obtain a deque of tokens and then repeatedly invokes _derive_mol_from_tokens(mol, smiles, tokens, i) to consume tokens and build the graph until no tokens remain. The integer index i is advanced by the parsing routine to track token-to-graph correspondence for attribution when requested. Because the function relies on the internal tokenizer and parse routine, any exceptions from those helpers are propagated as SMILESParserError on parse failure. The function is deterministic and has no random side effects.
    """
    from selfies.utils.smiles_utils import smiles_to_mol
    return smiles_to_mol(smiles, attributable)


################################################################################
# Source: selfies.utils.smiles_utils.tokenize_smiles
# File: selfies/utils/smiles_utils.py
# Category: fix_docstring
# Reason: Schema parsing failed: ("Couldn't parse this type hint, likely due to a custom class or object: ", typing.Iterator[selfies.utils.smiles_utils.SMILESToken])
################################################################################

def selfies_utils_smiles_utils_tokenize_smiles(smiles: str):
    """Splits a SMILES string into a stream of SMILESToken objects, yielding tokens in the original left-to-right order for use by higher-level SMILES→SELFIES translation and SMILES parsing utilities.
    
    This generator implements the low-level lexical analysis used throughout the selfies package to convert a SMILES textual representation of a molecule into token objects that record the token text, its span in the original SMILES string, its token type (the SMILESTokenTypes enum), and an optional index of a preceding bond symbol when present. The function is intended for downstream parsers, attribution mapping (see README examples showing how tokens map to output SMILES), and any component that needs precise token spans for error reporting or semantic processing.
    
    Args:
        smiles (str): The input SMILES string to tokenize. This must be the SMILES string exactly as produced or consumed by other parts of the library (no implicit trimming is performed). The function reads characters from this string and yields tokens whose start and end indices are Python-style indices into this same string.
    
    Behavior and token semantics:
        The function scans the input from left to right and yields SMILESToken objects for the following constructs, preserving order and providing the original substring and start/end indices for each token.
        - Dot separator: A literal '.' is recognized as a DOT token and yielded immediately as SMILESToken(None, start_idx, end_idx, SMILESTokenTypes.DOT, '.').
        - Bond prefix: If a bond character from SMILES_BOND_ORDERS appears immediately before another token, that bond character is not yielded as a standalone token; instead its index is recorded as the bond_idx value passed into the subsequently yielded token's SMILESToken. If a bond character appears at the end of the string with no following token, a SMILESParserError for a "hanging bond" is raised.
        - Organic subset elements: Alphabetic characters are parsed as atom tokens. Two-letter organic elements "Br" and "Cl" are parsed as a single ATOM token; single-letter alphabetic atoms (e.g., "C", "N", "O") are parsed as single-character ATOM tokens.
        - Bracketed atoms: A '[' opens a bracketed atom token; the function looks for the next ']' and yields the substring from '[' through ']' as an ATOM token. If the closing ']' is missing, a SMILESParserError indicating a "hanging bracket [" is raised with the index of the '['.
        - Branch parentheses: '(' and ')' are yielded as BRANCH tokens. If a bond prefix was recorded immediately before a branch parenthesis (i.e., a bond index is pending), a SMILESParserError is raised for a "hanging_bond" at the bond position because a bond cannot directly precede a branch delimiter in this tokenizer.
        - Ring closures: A single digit character is parsed as a one-digit RING token. A '%' character introduces a two-digit ring number; the two following characters must be numeric digits and exactly two in length (e.g., '%12'), otherwise a SMILESParserError is raised with an "invalid ring number" message and the index of the '%' character.
        - Unrecognized characters: Any character that does not match the above categories causes a SMILESParserError with an "unrecognized symbol" message and the index of that character.
    
    Side effects, performance, and error modes:
        The function has no external side effects and does not modify the input string. It is a lazy generator: tokens are produced on demand and the function maintains only the current scanning index. Errors are reported via SMILESParserError exceptions to allow callers to locate and report syntax issues precisely (index and message). Because bond characters are attached to the following token by index, callers interested in explicit bond tokens must inspect the bond index field on each SMILESToken. The tokenizer relies on the module-level constants and enums (SMILES_BOND_ORDERS, SMILESTokenTypes, SMILESToken, SMILESParserError) defined in selfies.utils.smiles_utils and will behave according to their definitions.
    
    Returns:
        Iterator[SMILESToken]: A generator that yields SMILESToken objects one-by-one in the same order they appear in the input SMILES. Each SMILESToken records the optional bond_idx (index of a preceding bond character, or None), the start and end indices (Python slice indices) delimiting the token in the original smiles string, the token type (SMILESTokenTypes enum: DOT, ATOM, BRANCH, RING, etc.), and the raw substring corresponding to the token.
    """
    from selfies.utils.smiles_utils import tokenize_smiles
    return tokenize_smiles(smiles)


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
