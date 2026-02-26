"""
Regenerated Google-style docstrings for module 'chai_lab'.
README source: others/readme/chai_lab/README.md
Generated at: 2025-12-02T00:21:08.090906Z

Total functions: 12
"""


################################################################################
# Source: chai_lab.data.collate.utils.pad_size
# File: chai_lab/data/collate/utils.py
# Category: fix_docstring
# Reason: Schema parsing failed: Cannot generate JSON schema for pad_size because the docstring has no description for the argument 'max_in_batch'
################################################################################

def chai_lab_data_collate_utils_pad_size(max_in_batch: int, allowed_sizes: list[int]):
    """chai_lab.data.collate.utils.pad_size returns the chosen padding length for a batch by selecting the smallest allowed size that is greater than or equal to the largest item in the batch. In the Chai-1 codebase this helper is used by collate utilities to decide the tensor length to pad sequences/embeddings to when constructing batches for model inference or training, which helps control GPU memory layout and ensures batch tensors conform to sizes supported by downstream kernels and model code.
    
    This function is pure and has no side effects. It expects allowed_sizes to enumerate the candidate padding lengths that are acceptable in the collate pipeline (for example, powers-of-two or predefined bucket sizes used to limit memory fragmentation). The last element of allowed_sizes is treated as the maximum allowed padding size. The function performs a strict capacity check and will raise an error if the requested maximum in the batch exceeds that maximum allowed padding.
    
    Args:
        max_in_batch (int): The largest required unpadded size in the current batch (for example, the length of the longest sequence, number of residues, or number of tokens among all items in the batch). This value is compared against allowed_sizes to determine the padding target. Providing a value larger than the maximum element of allowed_sizes triggers an error because no allowed bucket can accommodate the batch without truncation or re-bucketing.
        allowed_sizes (list[int]): A non-empty list of candidate padding sizes (integers) used by the collate code path. This list is expected to contain sizes in non-decreasing order so that the final element represents the maximum permitted padding size; the function uses allowed_sizes[-1] as the maximum allowed value and then selects the first element n in allowed_sizes with n >= max_in_batch. Duplicate sizes are permitted but unnecessary. If allowed_sizes is empty, indexing allowed_sizes[-1] will raise an IndexError; callers should ensure the list is populated.
    
    Returns:
        int: The selected padding size: the smallest element n from allowed_sizes such that n >= max_in_batch. This is the length to which batch items should be padded so they fit into a single tensor with a size drawn from the allowed buckets. If max_in_batch is greater than allowed_sizes[-1], a ValueError is raised describing the overflow; if elements of allowed_sizes are not comparable to max_in_batch (e.g., non-integers), comparisons will raise a TypeError.
    """
    from chai_lab.data.collate.utils import pad_size
    return pad_size(max_in_batch, allowed_sizes)


################################################################################
# Source: chai_lab.data.features.feature_utils.get_entry_for_key
# File: jaxtyping/_decorator.py
# Category: valid
################################################################################

def chai_lab_data_features_feature_utils_get_entry_for_key(data: dict, key: str):
    """chai_lab.data.features.feature_utils.get_entry_for_key returns a nested entry from a mapping by interpreting a slash-separated key path and walking the mapping one level at a time. This utility is used in the Chai-1 feature-processing pipeline to extract nested features (for example embeddings, MSA or template sub-objects) from dictionaries produced or consumed by functions such as run_folding_on_context and AllAtomFeatureContext construction.
    
    Args:
        data (dict): The dictionary (mapping) to search. In the Chai-1 codebase this is typically a features dictionary that may contain nested dicts representing structured inputs (for example per-chain features, embedding blocks, or restraint specifications). The function does not modify this mapping; it only reads from it.
        key (str): A slash-separated key path that identifies the nested entry to retrieve. Each segment between slashes denotes one dictionary lookup level (for example "foo/bar" first indexes data["foo"] then indexes ["bar"] on the resulting value). An empty string or segments that do not exist will lead to the failure modes described below.
    
    Returns:
        The value stored at the nested key path: the object retrieved after performing the sequence of dictionary lookups. This may be a dict, scalar, list, or any Python object that was stored at that location in the input mapping. In the Chai-1 domain this is commonly a feature dict (when requesting a top-level key) or a scalar/value within a feature (when requesting a deeper path).
    
    Behavior and practical details:
        The function splits the provided key on "/" and performs successive dictionary lookups using each segment in order. It performs no type coercion, no deep copying, and no validation of the returned value beyond the lookups themselves. There are no side effects; the input dictionary is not modified.
    
    Failure modes and exceptions:
        If any segment of the key path is not present as a key in the current mapping, a KeyError will be raised. If the code attempts to index into a value that is not a dict (for example if an intermediate lookup returns a list or scalar but further segments remain), a TypeError or the object's native indexing error may be raised. These exceptions are produced directly by the underlying Python operations (no custom exceptions are raised). Callers in the Chai-1 pipeline should catch KeyError/TypeError if keys may be missing or if the structure of feature dictionaries is uncertain.
    """
    from chai_lab.data.features.feature_utils import get_entry_for_key
    return get_entry_for_key(data, key)


################################################################################
# Source: chai_lab.data.io.cif_utils.get_chain_letter
# File: chai_lab/data/io/cif_utils.py
# Category: fix_docstring
# Reason: Schema parsing failed: Cannot generate JSON schema for get_chain_letter because the docstring has no description for the argument 'asym_id'
################################################################################

def chai_lab_data_io_cif_utils_get_chain_letter(asym_id: int):
    """Get the single-character chain letter corresponding to a one-indexed asym_id.
    
    This function maps a one-indexed asym_id (as used when parsing mmCIF/structure files or when interpreting template hit tables in the Chai-1 template-loading pipeline described in the README) to the canonical chain letter used by the codebase. In the Chai-1 project this mapping is used when loading template structures and parsing chains from CIF files (for example, when reading hits from an m8 file and then selecting a chain within the downloaded CIF), so callers can convert numeric asym unit identifiers into human-readable chain identifiers for downstream file naming, indexing, and matching chains to sequence records.
    
    Args:
        asym_id (int): A one-indexed asymmetry unit identifier. This is an integer where 1 corresponds to the first chain, 2 to the second, etc. The function uses this index to look up the corresponding chain letter in the module-level _CHAIN_VOCAB by subtracting one to convert to a zero-based index (vocab_index = asym_id - 1). The asym_id must be greater than 0 and less than or equal to the length of _CHAIN_VOCAB; this enforces the maximum number of distinct chains supported by the current chain vocabulary.
    
    Returns:
        str: A single-character string taken from _CHAIN_VOCAB that represents the chain letter for the provided asym_id (for example, asym_id 1 -> 'A', asym_id 2 -> 'B'). This function has no side effects; it only performs a deterministic lookup and returns the corresponding character.
    
    Raises:
        AssertionError: If asym_id is not in the valid range (asym_id <= 0 or asym_id > len(_CHAIN_VOCAB)), an AssertionError is raised. This indicates the requested asymmetry unit index is outside the supported chain vocabulary and callers must handle or prevent out-of-range indices before calling.
    """
    from chai_lab.data.io.cif_utils import get_chain_letter
    return get_chain_letter(asym_id)


################################################################################
# Source: chai_lab.data.parsing.input_validation.constituents_of_modified_fasta
# File: chai_lab/data/parsing/input_validation.py
# Category: fix_docstring
# Reason: Schema parsing failed: ("Couldn't parse this type hint, likely due to a custom class or object: ", list[str] | None)
################################################################################

def chai_lab_data_parsing_input_validation_constituents_of_modified_fasta(x: str):
    """chai_lab.data.parsing.input_validation.constituents_of_modified_fasta parses and validates a single-line modified FASTA-style sequence string used as input to Chai-1 folding workflows, returning the sequence constituents (single-letter residues and bracketed multi-character modifications) or None if the input is invalid.
    
    This function is used in the Chai-1 codepath that accepts FASTA inputs containing modified residues (for example, RNA/DNA/protein sequences with modifications expressed in parentheses) as described in the repository README. It is intended to detect and extract the atomic constituents that the folding pipeline will interpret as residues or modification blocks, and to reject strings that do not conform to the accepted modified-FASTA format (for example, SMILES strings for ligands should not be passed to this parser).
    
    Args:
        x (str): A sequence string potentially containing modifications indicated with parentheses, e.g. "agtc", "AGT(ASP)TG", or " A G T ( N H 2 ) " (whitespace will be stripped). The function strips leading/trailing whitespace and converts the input to uppercase before validation. This parameter represents a single FASTA sequence line (not a multi-line FASTA file) intended for use as an input sequence to the Chai-1 model.
    
    Returns:
        list[str] | None: If valid, returns a list of sequence constituents where each unmodified residue is represented as a single-character uppercase string (e.g., "A", "G", "T") and each modification block originally given in parentheses is returned as a multi-character uppercase string (e.g., "ASP", "NH2"). The returned list preserves the original sequential order of residues and modifications. If the input is invalid according to the parser rules, returns None. Invalid cases include but are not limited to: characters outside ASCII letters, digits, or parentheses; nested or double-open parentheses; a closing parenthesis without a matching open; an unclosed open parenthesis at end of string; a modification block that is empty or only a single character (e.g., "()" or "(K)"); and a top-level token that is not an ASCII letter (digits are permitted only inside parenthesized modification blocks). No side effects occur; the function does not modify external state and performs only in-memory validation and parsing.
    """
    from chai_lab.data.parsing.input_validation import constituents_of_modified_fasta
    return constituents_of_modified_fasta(x)


################################################################################
# Source: chai_lab.data.parsing.input_validation.identify_potential_entity_types
# File: chai_lab/data/parsing/input_validation.py
# Category: fix_docstring
# Reason: Schema parsing failed: Cannot generate JSON schema for identify_potential_entity_types because the docstring has no description for the argument 'sequence'
################################################################################

def chai_lab_data_parsing_input_validation_identify_potential_entity_types(sequence: str):
    """Identify which high-level molecular entity types a single input string could represent for Chai-1 input parsing and validation.
    
    This function inspects a provided FASTA-like sequence or a SMILES-like ligand string and returns a conservative list of possible entity types that the string could represent. It is used by Chai-1 input parsing and validation code to choose how an input should be interpreted before constructing embedding/MSA/template contexts or before attempting folding. The function applies two syntactic checks: (1) it calls constituents_of_modified_fasta(...) to parse modified FASTA tokens and examine one-letter residue constituents; (2) it checks whether the sequence only contains a restricted set of ASCII symbols commonly used in SMILES and manual glycan notations. The function is intentionally permissive: it may return multiple EntityType values when the input is ambiguous (for example, short sequences that could be DNA, RNA, or protein). It does not perform full biochemical or SMILES validation, does not query external services, and does not perform any I/O or mutate program state.
    
    Args:
        sequence (str): A single input sequence string to classify. In practice this is the raw sequence token provided by the user in a FASTA entry (which may include modified-residue tokens) or a SMILES-like ligand/glycan string. The function first strips leading/trailing whitespace from this value and treats an empty string (after stripping) as invalid, returning an empty list. The parameter must be a Python str; callers should supply the exact FASTA/SMILES string used for folding or input validation.
    
    Returns:
        list[chai_lab.data.parsing.structure.entity_type.EntityType]: A list of candidate EntityType members that the input sequence could represent for downstream Chai-1 processing. Possible members produced by the current implementation include:
        - EntityType.DNA when the parsed one-letter constituents are a subset of the set {'A','G','T','C'}.
        - EntityType.RNA when the parsed one-letter constituents are a subset of the set {'A','G','U','C'}.
        - EntityType.PROTEIN when the parsed one-letter constituents are consistent with amino-acid one-letter codes and do not include 'U' (the function treats presence of 'U' as excluding a protein interpretation).
        - EntityType.LIGAND and EntityType.MANUAL_GLYCAN when the entire sequence (converted to uppercase for matching) consists only of characters from the ASCII symbol set string.ascii_letters + string.digits + ".-+=#$%:/\\[]()<>@" (the heuristic used for common SMILES and manual glycan notation).
        The function returns an empty list when the stripped sequence is empty or when none of the syntactic checks match. Note that these are syntactic, not semantic, checks: the function does not validate chemical connectivity, full SMILES syntax, or structural correctness, so ambiguous or malformed inputs can lead to multiple candidates or false positives. The function is deterministic and has no side effects.
    """
    from chai_lab.data.parsing.input_validation import identify_potential_entity_types
    return identify_potential_entity_types(sequence)


################################################################################
# Source: chai_lab.data.parsing.msas.a3m.tokenize_sequences_to_arrays
# File: chai_lab/data/parsing/msas/a3m.py
# Category: fix_docstring
# Reason: Schema parsing failed: Cannot generate JSON schema for tokenize_sequences_to_arrays because the docstring has no description for the argument 'seqs_str'
################################################################################

def chai_lab_data_parsing_msas_a3m_tokenize_sequences_to_arrays(seqs_str: list[str]):
    """Tokenize a list of aligned sequences in a3m format into two NumPy arrays that are ready for downstream MSA embedding and folding in the Chai-1 pipeline.
    
    This function is used by the Chai-1 codepath that accepts multiple sequence alignments (MSAs) in a3m-like format (see README section "How can MSAs be provided to Chai-1?"). It converts a list of aligned sequence strings into compact uint8 arrays: one containing token identifiers for each alignment column per sequence (used as the primary encoded MSA input to the model) and one containing per-sequence, per-column deletion information (used by MSA-processing helpers). The function derives the alignment column count from the first sequence and relies on the tokenization mapping returned by the internal _get_tokenization_mapping() helper and the parsing implemented in _parse_seqs_to_ndarrays().
    
    Args:
        seqs_str (list[str]): A non-empty list of alignment strings in a3m-style format. Each element must be a single aligned sequence (as a Python str). The first sequence in this list is used to determine the alignment length (seq_len) by counting ASCII uppercase letters and the '-' character as alignment columns. Typical a3m conventions (uppercase letters for match columns, lowercase for insertions, '-' for gaps) are expected because the internal parser interprets these categories; providing an empty list will trigger an assertion error. The function concatenates these strings with newline separators and passes them to the internal parser, so sequences should be valid text strings containing the alignment rows.
    
    Behavior, side effects, and failure modes:
        The function asserts that seqs_str is non-empty and will raise an AssertionError otherwise. It computes seq_len = sum(c in string.ascii_uppercase or c == "-" for c in seqs_str[0]) and sets n_seqs = len(seqs_str). It allocates two NumPy arrays: out_sequences and out_deletions with shapes (n_seqs, seq_len) and dtype uint8, then calls the internal helper _parse_seqs_to_ndarrays(...) with a byte-concatenated representation of the sequences and a tokenization mapping from _get_tokenization_mapping(). The internal parser populates the two arrays in-place. No external files are read or written by this function. If the sequences are inconsistent with the expectations of the parser (for example, if the first sequence's counting logic does not reflect the actual alignment columns, or if unexpected characters are present), the internal parsing routine may raise errors (e.g., ValueError or IndexError) or produce arrays that are not meaningful to downstream code. The function itself does not perform normalization beyond the described counting rule, so callers must ensure their input follows a3m alignment conventions used by Chai-1.
    
    Returns:
        tuple[numpy.ndarray, numpy.ndarray]: A 2-tuple (out_sequences, out_deletions).
        out_sequences (numpy.ndarray): A uint8 array of shape (n_seqs, seq_len) containing token identifiers for each sequence and alignment column. Token ids follow the mapping supplied by _get_tokenization_mapping() and are intended as the encoded MSA sequence input for Chai-1 model components.
        out_deletions (numpy.ndarray): A uint8 array of shape (n_seqs, seq_len) containing deletion information per sequence and alignment column as produced by _parse_seqs_to_ndarrays(). This array encodes deletion counts/flags used by downstream MSA processing and embedding routines in the Chai-1 pipeline.
    """
    from chai_lab.data.parsing.msas.a3m import tokenize_sequences_to_arrays
    return tokenize_sequences_to_arrays(seqs_str)


################################################################################
# Source: chai_lab.data.parsing.msas.aligned_pqt.expected_basename
# File: chai_lab/data/parsing/msas/aligned_pqt.py
# Category: fix_docstring
# Reason: Schema parsing failed: Cannot generate JSON schema for expected_basename because the docstring has no description for the argument 'query_sequence'
################################################################################

def chai_lab_data_parsing_msas_aligned_pqt_expected_basename(query_sequence: str):
    """chai_lab.data.parsing.msas.aligned_pqt.expected_basename returns the expected filename (basename) for an aligned.pqt MSA file corresponding to a provided query sequence. This function is used by the Chai-1 MSA handling code to compute the canonical filename under which an MSA (aligned.pqt, a parquet-style table that can include metadata like source database and pairing keys) should be stored or looked up for a given sequence.
    
    Args:
        query_sequence (str): The input sequence string for which to compute the expected aligned.pqt filename. In the Chai-1 workflow this should be the sequence identifier or sequence string as provided in FASTA inputs (it may represent standard amino-acid sequences, modified residues, nucleotides, or ligand SMILES as used in the project). The function normalizes this value by calling query_sequence.upper() to ensure case-insensitive mapping: sequences that differ only by letter case map to the same filename. This parameter must be a Python str; passing a non-str (for example None or a bytes object) will raise an exception when the method attempts to call .upper().
    
    Returns:
        str: A filename string of the form "<seqhash>.aligned.pqt", where <seqhash> is the result of internal hash_sequence(query_sequence.upper()). The returned basename is intended to be the canonical filename under which Chai-1 expects an aligned.pqt file for the given (uppercased) query sequence. The function performs no filesystem I/O or validation of file existence; it only constructs the expected name. Note that the exact contents and format of <seqhash> are determined by the hash_sequence implementation; hash collisions, while unlikely, would cause distinct sequences to map to the same basename and are a possible failure mode outside this function's control.
    """
    from chai_lab.data.parsing.msas.aligned_pqt import expected_basename
    return expected_basename(query_sequence)


################################################################################
# Source: chai_lab.data.parsing.structure.sequence.fasta_one_letter_sequence
# File: chai_lab/data/parsing/structure/sequence.py
# Category: fix_docstring
# Reason: Schema parsing failed: Cannot generate JSON schema for fasta_one_letter_sequence because the docstring has no description for the argument 'residue_codes'
################################################################################

def chai_lab_data_parsing_structure_sequence_fasta_one_letter_sequence(
    residue_codes: list[str]
):
    """chai_lab.data.parsing.structure.sequence.fasta_one_letter_sequence: Convert a list of residue identifiers into a concatenated one-letter FASTA sequence suitable for Chai-1 inputs and downstream folding.
    
    Converts each string in residue_codes by looking up the corresponding tabulated residue via gemmi.find_tabulated_residue(res) and calling its fasta_code() method, then concatenates the results in the same order as the input list. This utility is used to produce the one-letter sequence lines that are commonly required by the Chai-1 folding pipeline (for example, to create FASTA files consumed by the CLI command `chai-lab fold` or by the programmatic inference entrypoints such as chai_lab.chai1.run_inference). The function performs no I/O, does not mutate its inputs, and is deterministic for a given gemmi residue table.
    
    Args:
        residue_codes (list[str]): An ordered list of residue identifiers (strings) to convert. Each entry is passed verbatim to gemmi.find_tabulated_residue(res) and therefore must correspond to a residue name/code present in gemmi's tabulated residue data (typical examples are three-letter amino-acid codes or nucleotide residue codes). The order of elements in residue_codes determines the order of characters in the returned one-letter sequence. If a residue corresponds to a nonstandard chemical entity (for example, certain ligands or modified residues), ensure that gemmi's tabulated residues include an appropriate mapping for that identifier before calling this function.
    
    Returns:
        str: A single string formed by concatenating the fasta_code() result for each residue in residue_codes, in the same order as the input list. This returned string is the one-letter FASTA sequence representation intended for use as model input to Chai-1 (e.g., written as the sequence line in an input FASTA file or passed directly to downstream functions). No other side effects occur.
    
    Raises:
        Exception: Any exception raised by gemmi.find_tabulated_residue or by the fasta_code() call will propagate. Common failure modes include a missing tabulated residue for a provided identifier or other gemmi lookup/attribute errors; callers should validate that residue_codes entries are recognized by gemmi before invoking this function if silent failures are not acceptable.
    """
    from chai_lab.data.parsing.structure.sequence import fasta_one_letter_sequence
    return fasta_one_letter_sequence(residue_codes)


################################################################################
# Source: chai_lab.data.parsing.structure.sequence.protein_one_letter_sequence
# File: chai_lab/data/parsing/structure/sequence.py
# Category: fix_docstring
# Reason: Schema parsing failed: Cannot generate JSON schema for protein_one_letter_sequence because the docstring has no description for the argument 'residue_codes'
################################################################################

def chai_lab_data_parsing_structure_sequence_protein_one_letter_sequence(
    residue_codes: list[str]
):
    """chai_lab.data.parsing.structure.sequence.protein_one_letter_sequence converts a list of protein residue names into a contiguous one-letter amino-acid sequence string used by Chai-1 for sequence inputs such as FASTA lines, MSA entries, template chain sequences, and embedding contexts. This function provides a stable, explicit mapping of residue names to single-character amino-acid codes (similar in effect to gemmi.fasta_code()), and it intentionally encodes non-standard or unknown residue names as the placeholder "X" so downstream folding, MSA, and template-processing code receive a well-formed sequence.
    
    Args:
        residue_codes (list[str]): The ordered list of residue identifiers for a polypeptide chain. Each item is expected to be a string representing a residue name or code (for example, three-letter residue names like "ALA" or other repository-specific residue tokens). The order of items in this list corresponds to the N-to-C terminal order in the biological sequence and is preserved in the returned string. In the Chai-1 workflow this list is typically produced by parsing FASTA inputs, PDB/CIF chains, or preprocessed embedding contexts; the resulting one-letter string is used when constructing inputs to run_inference, building MSAs, or generating templates. The function assumes residue_codes is a list of str; if a residue identifier is not a recognized standard protein residue it will be converted to the single-letter "X". Supplying a non-iterable or a list containing non-str types may raise a TypeError or cause the underlying token-mapping helper to fail.
    
    Returns:
        str: A single string containing the concatenated one-letter amino-acid codes corresponding to residue_codes, in the same order. The return value is deterministic and has no side effects. An empty input list yields an empty string. Non-standard or unknown residues are represented as "X" in the output. If residue_codes or its elements are of the wrong type (for example, not a list of str), the function will raise a runtime error originating from iteration or from the internal residue-token mapping helper.
    """
    from chai_lab.data.parsing.structure.sequence import protein_one_letter_sequence
    return protein_one_letter_sequence(residue_codes)


################################################################################
# Source: chai_lab.data.parsing.structure.sequence.protein_one_letter_sequence_with_mods
# File: chai_lab/data/parsing/structure/sequence.py
# Category: fix_docstring
# Reason: Schema parsing failed: Cannot generate JSON schema for protein_one_letter_sequence_with_mods because the docstring has no description for the argument 'residue_codes'
################################################################################

def chai_lab_data_parsing_structure_sequence_protein_one_letter_sequence_with_mods(
    residue_codes: list[str]
):
    """Convert a list of protein residue codes into a single one-letter amino-acid sequence string, emitting non-standard or modified residues in bracketed form so the sequence preserves modification identity for downstream processing in Chai-1.
    
    This function is used by Chai-1 input parsing and folding utilities to produce a compact, human- and machine-readable linear sequence from per-residue annotations (for example, sequences parsed from FASTA files or PDB residue lists) that may contain chemically modified residues. Non-standard residues are preserved as bracketed tokens so the model input retains modification information required for accurate folding of modified proteins and complexes. The implementation delegates token mapping to the internal helper _get_protein_only_residue_token with mods_in_brackets=True, producing a deterministic, side-effect-free transformation.
    
    Args:
        residue_codes (list[str]): A list of residue code strings, in sequence order, representing each residue of the protein chain. Each element should be the residue identifier as produced by upstream parsers (for example conventional three-letter amino-acid codes like "ALA" or modified-residue codes like "HIP"). The function will map recognized canonical residues to their single-letter amino-acid codes and will convert non-standard or otherwise unmapped residue codes into bracketed tokens of the form "[CODE]" where CODE is the original residue code string. An empty list yields an empty string.
    
    Returns:
        str: A single concatenated string representing the one-letter sequence for canonical residues and bracketed tokens for non-standard residues. For example, given residue_codes corresponding to a chain containing a histidine-like modification coded "HIP", the returned sequence might contain "...APNGL[HIP]TRP...". This return value is intended to be used as model input or for human inspection; the function has no side effects and does not perform I/O.
    
    Behavior and failure modes:
        - The mapping behavior is deterministic and relies on the internal _get_protein_only_residue_token mapping with mods_in_brackets=True.
        - Non-standard or unmapped residue codes are emitted as bracketed tokens "[CODE]" rather than being silently replaced or dropped, preserving modification identity for downstream tools.
        - If residue_codes contains non-string elements, a TypeError will be raised by the list iteration or by the helper mapping function (the function expects list[str] per the signature).
        - The function does not validate chemical correctness beyond token mapping; it does not check chain connectivity, stereochemistry, or presence of ligands/nucleotides (those should be handled by higher-level parsing functions).
        - No external resources are accessed and there are no side effects; this is a pure transformation suitable for preparing sequence strings for use with chai_lab folding workflows (for example, feeding into chai_lab.chai1.run_inference or writing FASTA inputs).
    """
    from chai_lab.data.parsing.structure.sequence import protein_one_letter_sequence_with_mods
    return protein_one_letter_sequence_with_mods(residue_codes)


################################################################################
# Source: chai_lab.utils.paths.chai1_component
# File: chai_lab/utils/paths.py
# Category: fix_docstring
# Reason: Schema parsing failed: Cannot generate JSON schema for chai1_component because the docstring has no description for the argument 'comp_key'
################################################################################

def chai_lab_utils_paths_chai1_component(comp_key: str):
    """chai_lab.utils.paths.chai1_component retrieves and returns the local filesystem path to a Chai-1 exported model component, downloading the component from the official component store if it is not already present.
    
    Args:
        comp_key (str): Filename key identifying the model component to fetch, e.g. 'trunk.pt'. This string must be the exact basename of a released PyTorch model file and is used to format the remote download URL via the module-level COMPONENT_URL template. In the Chai-1 workflow, comp_key selects which exported model weight file (a component of the Chai-1 multimodal model) to ensure is available locally for downstream inference functions such as chai_lab.chai1.run_inference.
    
    Returns:
        pathlib.Path: Absolute or relative Path object pointing to the downloaded (or already-existing) model file stored under the package downloads directory in the subfolder "models_v2" (typically <package_root>/downloads/models_v2/{comp_key} unless overridden by the CHAI_DOWNLOADS_DIR environment variable). This Path can be used by model-loading code to open the file for loading weights into the Chai-1 model.
    
    Behavior and side effects:
    - The function asserts that comp_key ends with the literal suffix ".pt"; if this condition is not met an AssertionError is raised immediately.
    - It constructs a download URL by formatting the module-level COMPONENT_URL with the provided comp_key, and it resolves the destination path as downloads_path / "models_v2" / comp_key.
    - The function calls download_if_not_exists(url, result) which attempts to download the file to the destination only if it is not already present. On success the local file is created (or left unchanged if already present) and the Path to that file is returned.
    - The function has the practical effect of ensuring required Chai-1 model weight files are available on disk for folding/inference pipelines; callers rely on this to obtain components needed by chai_lab.chai1.run_inference and related routines.
    
    Failure modes and exceptions:
    - AssertionError if comp_key does not end with ".pt".
    - Errors raised by download_if_not_exists propagate (for example network errors, HTTP errors, permission errors, or filesystem I/O errors) if the download fails or the destination cannot be written.
    - If downloads_path or the target directory is misconfigured or not writable, an OSError or similar I/O exception may be raised when attempting to create or write the file.
    """
    from chai_lab.utils.paths import chai1_component
    return chai1_component(comp_key)


################################################################################
# Source: chai_lab.utils.tensor_utils.set_seed
# File: chai_lab/utils/tensor_utils.py
# Category: fix_docstring
# Reason: Schema parsing failed: Cannot generate JSON schema for set_seed because the docstring has no description for the argument 'seed_sequence'
################################################################################

def chai_lab_utils_tensor_utils_set_seed(seed_sequence: list[int]):
    """Seeds numpy, torch, and the Python stdlib random module using a reproducible, hierarchical SeedSequence strategy so that Chai-1 workloads (for example folding runs, MSA sampling, and any stochastic model components described in the project README) have well-defined, independent pseudo-random number generator (PRNG) streams. This function is heavily inspired by Lightning's pl_worker_init_function and is intended to be called during process or worker initialization to avoid correlated RNG streams across libraries.
    
    Args:
        seed_sequence (list[int]): A sequence of integers used as the entropy source to initialize numpy.random.SeedSequence. In practice this list is passed verbatim to numpy.random.SeedSequence(seed_sequence) and establishes the root entropy for all downstream PRNGs used by Chai-1. The function uses this SeedSequence to spawn two distinct child SeedSequences: one for PyTorch and one for the Python stdlib random module, ensuring that numpy, torch, and random each receive independent, non-overlapping streams derived from the same root seed material. The caller should provide a deterministic list of integers when reproducible model behavior is required (for example, to reproduce folding outputs, sampled restraints, or training/benchmarking behaviors).
    
    Returns:
        None: This function does not return a value. Its effect is purely side-effectful: it sets global RNG states for numpy (via np.random.seed), for PyTorch (via torch.manual_seed), and for the Python stdlib random module (via random.seed). After calling this function, calls to numpy.random.*, torch.* sampling functions, and random.* will draw from the seeded streams described above.
    
    Behavior, side effects, and failure modes:
        The function constructs a numpy.random.SeedSequence from the provided seed_sequence, then spawns two child SeedSequences. It seeds numpy with 128 bits (generated as four 32-bit words via np_ss.generate_state(4)), seeds PyTorch by extracting a single 64-bit unsigned integer from the first child SeedSequence (torch_ss.generate_state(1, dtype=np.uint64)[0]) and passing it to torch.manual_seed, and seeds the Python stdlib random module by generating two 64-bit words from the second child SeedSequence, concatenating them into a single 128-bit integer, and passing that integer to random.seed. These steps produce independent but reproducible RNG states across the three libraries, which is important for ensuring consistent behavior in Chai-1 inference and utility code that relies on stochasticity.
    
        Side effects include changing global RNG state visible to any subsequent code that uses numpy.random, torch random functions, or random. If the same seed_sequence is used in different processes without additional process-specific variation, PRNG streams will be identical across those processes; to avoid that in multi-process setups, construct per-process seed_sequence values (for example by including the process rank).
    
        Errors raised by underlying libraries will propagate: numpy.random.SeedSequence may raise a TypeError or ValueError if seed_sequence is not a valid sequence of integers as expected by NumPy, and torch.manual_seed will raise errors if PyTorch is not available or if the provided seed value is outside the range accepted by torch. If the module environment lacks the required libraries (numpy, torch, or the Python stdlib random is unavailable, which is uncommon), import-time or runtime exceptions may occur.
    """
    from chai_lab.utils.tensor_utils import set_seed
    return set_seed(seed_sequence)


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
