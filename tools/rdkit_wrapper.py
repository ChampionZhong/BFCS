"""
Regenerated Google-style docstrings for module 'rdkit'.
README source: others/readme/rdkit/README.md
Generated at: 2025-12-02T04:04:43.645523Z

Total functions: 110
"""


import numpy

################################################################################
# Source: rdkit.Chem.CanonSmiles
# File: rdkit/Chem/__init__.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def rdkit_Chem_CanonSmiles(smi: str, useChiral: bool = 1):
    """rdkit.Chem.CanonSmiles: Return a canonical SMILES string for a molecule represented by an input SMILES.
    
    This convenience function is part of the RDKit cheminformatics toolkit and is used to produce a deterministic, canonical SMILES string for a molecule given an input SMILES representation. Canonical SMILES are useful in cheminformatics workflows (for example: deduplication of compound lists, database keys, fingerprinting and descriptor generation for machine learning, and reproducible serialization of molecular structures). The function parses the input SMILES into an RDKit Mol object using MolFromSmiles and then generates a canonical SMILES with MolToSmiles. The useChiral parameter controls whether stereochemical/chiral information is preserved in the output canonicalization.
    
    Args:
        smi (str): The input SMILES string representing a molecule. This is the textual molecular representation to be parsed by RDKit's MolFromSmiles. The string should follow standard SMILES syntax; if it cannot be parsed, canonicalization cannot proceed and the function will typically propagate an error from the underlying RDKit parsing/serialization functions.
        useChiral (bool): Optional flag (default True, expressed in the original signature as 1) that determines whether chiral and stereochemical information is included in the canonicalization and in the returned SMILES. When True, stereochemical markers (e.g., @, @@, /, \) are considered and emitted in the canonical SMILES where appropriate. When False, stereochemical information is ignored in determining the canonical form and is not included in the returned SMILES.
    
    Returns:
        str: The canonical SMILES string produced by serializing the parsed RDKit Mol with MolToSmiles. The returned string is a deterministic SMILES representation for the parsed molecule according to RDKit's canonicalization algorithm and the useChiral setting. If the input SMILES cannot be parsed, the function will not return a valid canonical SMILES and will instead propagate an error from the underlying RDKit functions (for example, a parsing or serialization exception). The function has no other side effects and does not modify external state; it does not return an RDKit Mol object, only the canonical SMILES text.
    """
    from rdkit.Chem import CanonSmiles
    return CanonSmiles(smi, useChiral)


################################################################################
# Source: rdkit.Chem.QuickSmartsMatch
# File: rdkit/Chem/__init__.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def rdkit_Chem_QuickSmartsMatch(
    smi: str,
    sma: str,
    unique: bool = True,
    display: bool = False
):
    """rdkit.Chem.QuickSmartsMatch quickly matches a SMARTS substructure pattern against a SMILES molecule string and returns the atom index matches found in the molecule.
    
    Args:
        smi (str): The input molecule specified as a SMILES string. In the RDKit cheminformatics context (see README), SMILES is a compact, ASCII representation of a molecule used as input to RDKit's parser (MolFromSmiles). This function parses smi to an RDKit Mol internally; if parsing fails (invalid SMILES), the function may raise an exception when attempting to perform the substructure search. The returned atom indices refer to the parsed molecule's atom ordering.
        sma (str): The SMARTS substructure pattern to match against the molecule, expressed as a SMARTS string. SMARTS is the RDKit-supported pattern language for specifying substructures used in substructure searches (MolFromSmarts). If sma cannot be parsed as a valid SMARTS pattern, parsing will fail and the function may raise an exception when attempting to match.
        unique (bool): Optional; defaults to True. When True, only unique matches are returned: matches that are equivalent under symmetry or automorphism of the query/molecule are filtered so duplicate-equivalent mappings are omitted. When False, all distinct mappings found by RDKit's internal substructure matching engine are returned, which can include symmetry-related duplicates. Use False when you need every mapping instance; use True for a concise set of representative matches.
        display (bool): Optional; defaults to False. This parameter is accepted for API compatibility but is ignored by the implementation. It has no effect on parsing, matching behavior, return values, or side effects.
    
    Returns:
        list of list of int: A collection of matches; each match is a list of integer atom indices identifying the atoms in the parsed molecule that correspond to the SMARTS query. Indices are zero-based and refer to the atom ordering in the RDKit Mol produced from smi. If no matches are found, an empty list is returned. Note that the original implementation constructs the RDKit Mol objects via MolFromSmiles(smi) and MolFromSmarts(sma) and calls Mol.GetSubstructMatches(...), so behavior follows RDKit's matching semantics. If input parsing fails (invalid SMILES or SMARTS), the function will not return a valid list of matches and an exception may be raised when the code attempts to call GetSubstructMatches on a failed parse result.
    
    Behavior and side effects:
        This is a convenience, stateless helper that performs in-memory parsing of smi and sma and runs a substructure search; it does not modify files or global RDKit state and does not modify the input strings. The function returns the raw match mappings produced by RDKit; callers who require molecule objects or more detailed match information should parse the SMILES/SMARTS separately (using MolFromSmiles and MolFromSmarts) and use RDKit's richer API. The display parameter is present for backward compatibility only and produces no side effects.
    """
    from rdkit.Chem import QuickSmartsMatch
    return QuickSmartsMatch(smi, sma, unique, display)


################################################################################
# Source: rdkit.Chem.SupplierFromFilename
# File: rdkit/Chem/__init__.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def rdkit_Chem_SupplierFromFilename(fileN: str, delim: str = "", **kwargs):
    """rdkit.Chem.SupplierFromFilename creates and returns an RDKit molecule supplier object selected by the input filename extension. This convenience factory inspects the filename extension and constructs one of the RDKit supplier classes (SDMolSupplier, SmilesMolSupplier, or TDTMolSupplier) so callers can iterate over or access molecules stored in common cheminformatics file formats for downstream tasks such as descriptor and fingerprint generation, substructure/similarity searching, or bulk database loading.
    
    Args:
        fileN (str): Path to the input file containing molecular data. The function uses the file extension (the substring after the last '.') to select the supplier implementation: '.sdf' -> SDMolSupplier, '.csv' -> SmilesMolSupplier, '.txt' -> SmilesMolSupplier, and '.tdt' -> TDTMolSupplier. The value should be a filesystem-accessible filename or path string. The practical significance in RDKit workflows is that this filename determines both the parser used and the expected file record format (SDF records, SMILES lists, or TDT-tagged entries).
        delim (str): Optional delimiter string passed to suppliers that accept a delimiter (CSV/TXT/TDT). If empty (the default), SupplierFromFilename applies sensible defaults for CSV and TXT: a missing delim becomes ',' for '.csv' files and '\t' for '.txt' files. For '.tdt' files the provided delim is forwarded unchanged (an empty string will be passed if not set). This parameter does not affect SDMolSupplier (SDF readers), which do not use a delimiter.
        kwargs (dict): Additional keyword arguments forwarded directly to the chosen supplier constructor. These keyword arguments are implementation-specific options recognized by SDMolSupplier, SmilesMolSupplier, or TDTMolSupplier (for example, parser or sanitization flags supported by those classes). The function does not validate or reinterpret these keys; they are passed through as-is to the underlying RDKit supplier.
    
    Returns:
        object: An instance of an RDKit molecule supplier appropriate to the file extension: SDMolSupplier for '.sdf', SmilesMolSupplier for '.csv' and '.txt', or TDTMolSupplier for '.tdt'. The returned supplier implements RDKit's supplier protocol (typically lazy iteration and indexed access over RDKit Mol objects) and is intended for use in cheminformatics pipelines (e.g., generating descriptors or fingerprints, filtering by substructure). Construction may succeed even if the file contains parse issues; some parsing errors may be raised at construction time by the underlying supplier, while others may surface later during iteration.
    
    Raises:
        ValueError: If the filename extension is not one of the supported types ('sdf', 'csv', 'txt', 'tdt'), a ValueError is raised indicating an unrecognized extension.
        IOError or OSError: Any filesystem or low-level I/O errors raised by the underlying supplier constructors when opening or reading the file are propagated. Parsing errors or format-specific exceptions raised by the underlying suppliers may also propagate either at construction or when iterating over the returned supplier.
    
    Behavior notes and side effects:
        - The function selects the supplier solely based on the filename extension; it does not inspect file contents to verify format correctness.
        - For CSV and TXT inputs, if delim is an empty string the function substitutes defaults (',' for CSV, '\t' for TXT) before constructing SmilesMolSupplier.
        - For TDT inputs the delim argument is forwarded unchanged; if left empty the underlying TDT supplier receives an empty delimiter.
        - The returned supplier typically performs lazy parsing: molecule objects may be constructed on-demand during iteration, which can defer format-related errors until access time.
        - The function is a convenience wrapper used in RDKit-based cheminformatics workflows to simplify creating the correct supplier type from a filename.
    """
    from rdkit.Chem import SupplierFromFilename
    return SupplierFromFilename(fileN, delim, **kwargs)


################################################################################
# Source: rdkit.Chem.AtomPairs.Pairs.ExplainPairScore
# File: rdkit/Chem/AtomPairs/Pairs.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def rdkit_Chem_AtomPairs_Pairs_ExplainPairScore(score: int, includeChirality: bool = False):
    """rdkit.Chem.AtomPairs.Pairs.ExplainPairScore decodes an integer "score" value produced by RDKit Atom Pairs encoding into the three logical components used by atom-pair fingerprints: the first atom code, the topological distance (path length) between the two atoms, and the second atom code. This function is used in cheminformatics workflows to interpret and human-readably explain the compact integer representation generated by pyScorePair and AtomPairs fingerprint encoders so that users can see which atom environments and what inter-atomic distance the integer encodes.
    
    The function interprets the integer by applying bit masks and shifts. It computes a code size from rdMolDescriptors.AtomPairsParameters.codeSize and, if includeChirality is True, adds rdMolDescriptors.AtomPairsParameters.numChiralBits to account for chiral information. A path mask based on the module-level numPathBits is used to extract the distance (path length) encoded in the low-order bits. The remaining bits are split into two atom codes; each atom code is decoded by calling Utils.ExplainAtomCode(..., includeChirality=includeChirality). No mutation of inputs or global state occurs.
    
    Args:
        score (int): An integer atom-pair score produced by RDKit's Atom Pairs encoding (for example by pyScorePair). This integer is expected to contain, in its low-order bits, an encoded path (topological distance) and, in the higher-order bits, two packed atom codes. The function treats score as an unsigned bit field; if score was not produced by Atom Pairs encoding or has been truncated, the decoded atom codes and distance may be meaningless but no exception is raised by this function.
        includeChirality (bool): If True, include chiral bits when decoding atom codes. When True, the function increases the per-atom code bit width by rdMolDescriptors.AtomPairsParameters.numChiralBits so that the returned atom-code tuples include chirality information (strings such as 'R' or 'S' or an empty string). Default is False, meaning chirality bits are not considered and the returned atom-code tuples will not contain the chirality field.
    
    Returns:
        tuple: A 3-tuple (atomCode1, dist, atomCode2) where:
            atomCode1 (tuple): The decoded representation of the first atom as returned by Utils.ExplainAtomCode(code1, includeChirality=includeChirality). In practice this is a small tuple whose first element is the element symbol (str) and whose later elements are small integers encoding local atom features; if includeChirality is True an additional string element representing chirality (e.g. 'R'/'S' or '') will appear.
            dist (int): The topological distance (path length) between the two atoms as extracted from the low-order path bits of the input score.
            atomCode2 (tuple): The decoded representation of the second atom, analogous to atomCode1, obtained from Utils.ExplainAtomCode(code2, includeChirality=includeChirality).
    
    Behavioral notes and failure modes: The function performs pure decoding and has no side effects. It relies on the module-level constant numPathBits and rdMolDescriptors.AtomPairsParameters.{codeSize,numChiralBits} to determine bit widths; changes to those parameters elsewhere will change decoding behavior. If score does not conform to the Atom Pairs encoding (for example negative values, truncated integers, or integers generated by a different scheme), the returned atom-code tuples and dist may not correspond to real atoms or distances; the function will not raise an error in such cases but will return the bitwise-decoded values.
    """
    from rdkit.Chem.AtomPairs.Pairs import ExplainPairScore
    return ExplainPairScore(score, includeChirality)


################################################################################
# Source: rdkit.Chem.AtomPairs.Torsions.ExplainPathScore
# File: rdkit/Chem/AtomPairs/Torsions.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def rdkit_Chem_AtomPairs_Torsions_ExplainPathScore(score: int, size: int = 4):
    """rdkit.Chem.AtomPairs.Torsions.ExplainPathScore decodes an encoded integer path score (the integer produced by the AtomPairs torsion scoring code such as pyScorePath) into a per-atom explanation useful for human inspection and cheminformatics debugging. The function is used in RDKit to recover, for each atom along a scored path, the atom symbol, an adjusted branch/substituent count used by the torsion scoring routine, and the number of pi-electrons/features that were encoded for that atom.
    
    Args:
        score (int): The encoded path score integer produced by pyScorePath or other AtomPairs torsion scoring routines. This integer contains bit-packed per-atom codes (each codeSize bits wide where codeSize is taken from rdMolDescriptors.AtomPairsParameters.codeSize). The function extracts these per-atom codes from the least-significant bits upward and decodes them with Utils.ExplainAtomCode. The value must be a non-negative integer that was produced by the corresponding scoring routine; negative values or integers encoded with a different codeSize will produce incorrect or undefined decoded results.
        size (int): The number of atoms expected in the path (the path length). This defaults to 4. The function will decode exactly size atom codes from the integer (producing a tuple of length size). If size does not match the length used when the score was produced, the decoded information will not correspond to the original path (higher-order codes beyond the encoded bit-length will appear as zeros). size must be an integer.
    
    Behavior and implementation details: The function reads the bit width per atom (codeSize) from rdMolDescriptors.AtomPairsParameters.codeSize and computes a mask to extract each atom code. For i from 0 to size-1 it extracts the next code by masking the low codeSize bits of score and right-shifting score by codeSize bits. It decodes each extracted code with Utils.ExplainAtomCode(code) to obtain a tuple (symb, nBranch, nPi) where symb is the atom symbol (e.g., 'C', 'O'), nBranch is the branching/neighbor count contribution encoded for that atom, and nPi is the encoded count of pi-electron features. The function then increments the decoded nBranch by an additional value sub that encodes whether the atom is terminal in the path (sub = 1 for the first and last atom, sub = 2 for internal atoms) and returns the adjusted triple (symb, nBranch + sub, nPi) for each position. The returned tuple is ordered by the sequence of codes as they are extracted from the integer; in the AtomPairs torsion encoding used by RDKit this results in an order-independent multiset for paths that are reversed before encoding, so reversing the atom order in the original path typically yields the same set of decoded triplets (as demonstrated in the examples shown in the source).
    
    Side effects: The function has no side effects on user data structures. It reads module-level configuration values (rdMolDescriptors.AtomPairsParameters.codeSize) and relies on Utils.ExplainAtomCode to perform the per-atom decoding. It does not modify the input integer.
    
    Failure modes and error conditions: If rdMolDescriptors.AtomPairsParameters.codeSize is missing or changed incompatibly, the bit extraction will be incorrect and the decoded values will be meaningless. If Utils.ExplainAtomCode is not available or raises an exception for an extracted code, that exception will propagate. Passing a negative score can lead to incorrect decoding due to Python's right-shift behavior on negative integers. Providing a size that does not match the number of atoms encoded in score will produce decoded elements that do not correspond to the original path (missing or zero-valued codes).
    
    Returns:
        tuple: A tuple of length size where each element is a 3-tuple (symbol, adjusted_branch_count, nPi). symbol is the atom symbol string decoded for that path position (for example 'C' or 'O'), adjusted_branch_count is an integer equal to the decoded branching count plus an additional 1 for terminal atoms or 2 for internal atoms (this mirrors how the AtomPairs torsion scorer accounts for path connectivity), and nPi is an integer representing the decoded pi-electron / pi-feature count for that atom. The returned structure is intended for human-readable explanation and debugging of an encoded path score; it does not modify the original score.
    """
    from rdkit.Chem.AtomPairs.Torsions import ExplainPathScore
    return ExplainPathScore(score, size)


################################################################################
# Source: rdkit.Chem.AtomPairs.Utils.BitsInCommon
# File: rdkit/Chem/AtomPairs/Utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def rdkit_Chem_AtomPairs_Utils_BitsInCommon(v1: tuple, v2: tuple):
    """Compute the number of bit identifiers that occur in common between two sorted bit‑id vectors used as molecular fingerprints.
    
    This function is part of RDKit's AtomPairs.Utils utilities and is used in cheminformatics tasks that compare fingerprint-like representations of molecules. Each input vector represents the positions (IDs) of set bits in a fingerprint; the function counts how many bit IDs are shared between the two vectors. In practical use within RDKit this count can serve as the numerator for similarity measures (for example, the intersection used when computing Tanimoto similarity) or as a fast overlap check between atom-pair/fingerprint bit lists.
    
    Args:
        v1 (tuple): A tuple of bit identifiers (integers) representing set bits in the first fingerprint vector. The tuple must be sorted in non-decreasing (ascending) order. Duplicate bit IDs are allowed and are significant: each duplicate occurrence in v1 contributes separately to the count if matched in v2. The function iterates over v1 and advances a pointer through v2 to identify matches efficiently.
        v2 (tuple): A tuple of bit identifiers (integers) representing set bits in the second fingerprint vector. The tuple must be sorted in non-decreasing (ascending) order. Duplicate bit IDs are allowed and are counted multiple times when matched against duplicates in v1. The function assumes v2 supports len() and index access for forward scanning.
    
    Returns:
        int: The number of matching bit identifiers between v1 and v2. Each matching pair of entries increments the count by one; duplicate IDs are counted for each occurrence (for example, v1=(2,2) and v2=(2,2) yields 2). If there are no matches the function returns 0. This integer is suitable for use in downstream cheminformatics calculations (e.g., as the intersection size when computing fingerprint similarities).
    
    Behavior and implementation notes:
        - The algorithm performs a single forward scan through v1 while advancing a pointer through v2, so its time complexity is O(len(v1) + len(v2)) and it uses O(1) additional memory.
        - Inputs must be sorted ascending. If either v1 or v2 is not sorted, the result is undefined and matches may be missed.
        - The function does not modify v1 or v2; it has no side effects on the input tuples or external state.
        - The function treats duplicate IDs as distinct occurrences; each matched occurrence increases the count.
    
    Failure modes:
        - If elements of v1 and v2 are not mutually comparable with the '<' and '==' operators (for example, mixing incomparable types), Python will raise a TypeError during comparisons.
        - If v1 or v2 do not implement the expected tuple-like behaviors (iteration, len(), indexable access), a TypeError or AttributeError may be raised by the Python runtime. The signature documents v1 and v2 as tuples; callers should provide tuples of integers to avoid such errors.
    """
    from rdkit.Chem.AtomPairs.Utils import BitsInCommon
    return BitsInCommon(v1, v2)


################################################################################
# Source: rdkit.Chem.AtomPairs.Utils.CosineSimilarity
# File: rdkit/Chem/AtomPairs/Utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def rdkit_Chem_AtomPairs_Utils_CosineSimilarity(v1: tuple, v2: tuple):
    """rdkit.Chem.AtomPairs.Utils.CosineSimilarity computes the cosine similarity between two fingerprint vectors represented as sorted tuples of bit identifiers. This function implements the metric recommended in the LaSSI paper and is intended for comparing molecular fingerprints or other non-negative integer bit-id encodings used in RDKit for similarity searching, clustering, and descriptor-based machine learning.
    
    Args:
        v1 (tuple): A tuple of integers representing the first fingerprint vector as sorted bit identifiers. Each element corresponds to a set or multiset occurrence of a fingerprint bit; duplicates are allowed and are treated according to the Dot() implementation used internally. The tuple must be sorted in non-decreasing order for the expected behavior; the function does not sort its inputs and may produce incorrect results if the inputs are not pre-sorted.
        v2 (tuple): A tuple of integers representing the second fingerprint vector as sorted bit identifiers. Same conventions and requirements as v1 apply.
    
    Returns:
        float: The cosine similarity score between v1 and v2 computed as Dot(v1, v2) / sqrt(Dot(v1, v1) * Dot(v2, v2)), where Dot refers to the AtomPairs.Utils.Dot implementation used in this module. For non-negative integer bit counts (the typical RDKit fingerprint usage), the result lies between 0.0 and 1.0 inclusive. If either vector has zero magnitude (so the denominator is zero), the function returns 0.0. This function has no side effects and does not modify v1 or v2.
    
    Behavior and failure modes:
        The function computes squared magnitudes using Dot(v1, v1) and Dot(v2, v2) and forms the denominator as the square root of their product. If the denominator evaluates to zero (for example, when one or both tuples are empty or contain only zero-magnitude representations), the function returns 0.0 rather than raising an exception. Inputs that are not tuples or do not contain integer bit identifiers may lead to TypeError or incorrect results depending on the behavior of the underlying Dot implementation; callers should pass tuples of integers as used elsewhere in RDKit AtomPairs utilities.
    """
    from rdkit.Chem.AtomPairs.Utils import CosineSimilarity
    return CosineSimilarity(v1, v2)


################################################################################
# Source: rdkit.Chem.AtomPairs.Utils.DiceSimilarity
# File: rdkit/Chem/AtomPairs/Utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def rdkit_Chem_AtomPairs_Utils_DiceSimilarity(v1: tuple, v2: tuple, bounds: float = None):
    """rdkit.Chem.AtomPairs.Utils.DiceSimilarity computes the DICE similarity coefficient for two fingerprint-like vectors of bit identifiers used in RDKit atom-pair and topological-torsion fingerprinting. This function implements the metric recommended in the Topological Torsions and Atom Pairs papers and is intended for comparing molecular fingerprints for similarity searching, clustering, and descriptor-based machine-learning workflows in cheminformatics.
    
    Args:
        v1 (tuple): A sequence (tuple) of integer bit IDs representing the first fingerprint vector. In RDKit usage these are typically the bit identifiers produced by Atom Pairs or Topological Torsions fingerprint generators. The sequence must be sorted prior to calling this function, because the implementation (via BitsInCommon) expects sorted inputs for correct and efficient computation. Duplicate bit IDs are allowed in v1 and are interpreted as multiple occurrences (multiplicity) of that bit in the fingerprint.
        v2 (tuple): A sequence (tuple) of integer bit IDs representing the second fingerprint vector. As with v1, this should be a sorted tuple of bit identifiers from an RDKit fingerprint. Duplicate IDs in v2 are allowed and contribute multiplicity. Only multiplicities that occur in both v1 and v2 are counted toward the intersection (i.e., the intersection uses the minimum multiplicity per bit ID).
        bounds (float): Optional threshold that short-circuits computation when the smaller vector is too small relative to the combined lengths. If bounds is None (the default) no early rejection is applied. If bounds is a float, the function computes min(len(v1), len(v2)) / (len(v1) + len(v2)) and, if that ratio is strictly less than bounds, treats the number of bits in common as zero and returns 0.0. This parameter is provided to allow inexpensive early rejection of pairs that cannot meet a specified minimum overlap criterion. Passing None disables this behavior.
    
    Returns:
        float: The DICE similarity coefficient computed as 2 * |intersection| / (len(v1) + len(v2)), returned as a floating-point number. The intersection is computed by BitsInCommon(v1, v2), which counts common bit IDs with multiplicity equal to the minimum occurrence in each vector (multiset intersection). If both v1 and v2 are empty (len(v1) + len(v2) == 0) the function returns 0.0. If bounds is provided and the early-rejection condition is met, the function returns 0.0. The returned value is intended to indicate the normalized overlap between two fingerprint vectors for similarity comparisons in cheminformatics. No side effects occur; the function does not modify its inputs.
    """
    from rdkit.Chem.AtomPairs.Utils import DiceSimilarity
    return DiceSimilarity(v1, v2, bounds)


################################################################################
# Source: rdkit.Chem.AtomPairs.Utils.Dot
# File: rdkit/Chem/AtomPairs/Utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def rdkit_Chem_AtomPairs_Utils_Dot(v1: tuple, v2: tuple):
    """rdkit.Chem.AtomPairs.Utils.Dot computes the dot product between two sorted integer vectors of bit identifiers used in RDKit fingerprint-like representations.
    
    Computes a specialized integer "dot product" between two sequences of bit IDs (sparse fingerprint vectors) that are stored as tuples of integers. In the AtomPairs/fingerprints domain of RDKit (cheminformatics and molecular descriptors), these tuples represent the indices of bits set in a fingerprint; this function determines a weighted count of shared bit indices while accounting for duplicate occurrences. The implementation uses a single-pass, merge-like algorithm over the two sorted input tuples so it runs in linear time relative to the sum of the input lengths.
    
    Args:
        v1 (tuple): A tuple of integers representing bit IDs (indices) in a fingerprint-like sparse vector for the first item. The tuple must be sorted in non-decreasing order and may contain repeated integers; repeated bit IDs are meaningful and counted according to their multiplicity. This parameter plays the role of the left operand of the dot-product calculation in atom-pair/fingerprint comparisons and is typically produced by routines that extract or enumerate fingerprint bits in RDKit.
        v2 (tuple): A tuple of integers representing bit IDs (indices) in a fingerprint-like sparse vector for the second item. The tuple must be sorted in non-decreasing order and may contain repeated integers; repeated bit IDs are meaningful and counted according to their multiplicity. This parameter is the right operand of the dot-product calculation and is used when comparing or scoring similarity between two fingerprint encodings in cheminformatics workflows.
    
    Behavior and important details:
        - The function expects both inputs to be sorted tuples of comparable values (typically integers). If the inputs are not sorted, the algorithm may miss matches and produce an incorrect result.
        - Duplicates are handled explicitly: for each bit ID present in both inputs, the contribution to the result is min(count_in_v1, count_in_v2) squared (common_count * common_count). For example, if bit ID 2 appears twice in both v1 and v2, the contribution is 2*2 = 4.
        - The algorithm advances pointers through v1 and v2 in a merge-like fashion, so its time complexity is O(len(v1) + len(v2)) and it uses O(1) additional memory.
        - If either tuple is empty, the result is 0.
        - If elements are not comparable with the < and == operators (for example, mixed incompatible types), the behavior is undefined and the call may raise an exception from those comparisons.
        - There are no side effects: the input tuples are not modified.
    
    Returns:
        int: The integer dot-product-like score computed as the sum, over all bit IDs present in both tuples, of min(count_in_v1, count_in_v2) squared. This return value quantifies the weighted overlap of the two fingerprint-like vectors in RDKit's atom-pair/fingerprint domain and can be used directly in similarity calculations or as an intermediate scoring measure.
    """
    from rdkit.Chem.AtomPairs.Utils import Dot
    return Dot(v1, v2)


################################################################################
# Source: rdkit.Chem.AtomPairs.Utils.ExplainAtomCode
# File: rdkit/Chem/AtomPairs/Utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def rdkit_Chem_AtomPairs_Utils_ExplainAtomCode(
    code: int,
    branchSubtract: int = 0,
    includeChirality: bool = False
):
    """ExplainAtomCode returns a human-readable breakdown of an integer atom code used by RDKit's Atom Pairs / topological torsions encoding. It decodes the packed bitfields (type index, branch count, pi-related count, and optionally chirality) using the bit widths and type table defined in rdMolDescriptors.AtomPairsParameters, and maps the atom type index to an element symbol using RDKit's periodic table. This function is intended for diagnostics and interpretation of atom codes produced by GetAtomCode or equivalent Atom Pairs encoding routines in cheminformatics workflows (for example when inspecting atom features used in fingerprint or descriptor calculations).
    
    Args:
        code (int): The packed atom code to be decoded. This must be an integer produced by RDKit's atom-coding routines (for example GetAtomCode) where fields were packed according to rdMolDescriptors.AtomPairsParameters (numTypeBits, numBranchBits, numPiBits, numChiralBits and atomTypes). The function extracts bitfields in the order branch bits, pi bits, type bits, and optionally chiral bits. If an integer not produced by the expected encoding is provided, the returned values will be the bitwise interpretation and may be meaningless.
        branchSubtract (int): (optional) The constant that was subtracted from the true neighbor count when the atom code was generated. This parameter documents how the branch count field was derived during encoding (used by topological torsions code) and defaults to 0. Note: ExplainAtomCode does not modify the stored branch field; it returns the raw encoded branch value (the value present in the code). To recover the original neighbor count used at encoding time add branchSubtract to the returned branch value. This parameter is accepted for compatibility/documentation and has no side effects on global state.
        includeChirality (bool): (optional) If False (the default), the function decodes and returns only the element symbol, the encoded branch count, and the encoded pi-related count. If True, the function also reads the chiral bits from the code (using rdMolDescriptors.AtomPairsParameters.numChiralBits) and returns an additional string indicating chirality: 'R' for the R configuration, 'S' for the S configuration, or the empty string '' for non-chiral atoms. If includeChirality is False but the code contains chiral bits, those bits are ignored and the returned tuple contains only three elements.
    
    Returns:
        tuple: If includeChirality is False, returns a 3-tuple (atomSymbol, nBranch, nPi) where atomSymbol is a string element symbol derived by mapping the decoded type index to an atomic number via rdMolDescriptors.AtomPairsParameters.atomTypes and Chem.GetPeriodicTable().GetElementSymbol; nBranch is the integer value stored in the branch bitfield (to get the neighbor count used at encoding time add branchSubtract); nPi is the integer value stored in the pi-related bitfield (pi-related information as encoded by GetAtomCode). If includeChirality is True, returns a 4-tuple (atomSymbol, nBranch, nPi, chirality) where chirality is one of 'R', 'S', or '' for non-chiral atoms. The function performs no mutations and is purely informational.
    
    Behavior and failure modes:
        This function relies on the bit widths and atomTypes table in rdMolDescriptors.AtomPairsParameters; if those attributes are missing or malformed an AttributeError or related exception may be raised. The mapping from type index to element symbol returns 'X' when the decoded type index is out of range of atomTypes (this indicates an unknown or out-of-range type index in the provided code). The function expects the code to follow RDKit's Atom Pairs encoding; providing arbitrary integers will yield the bitwise-decoded fields but they may not correspond to physically meaningful atom properties. There are no side effects or global state changes performed by this function. Default values are branchSubtract=0 and includeChirality=False.
    """
    from rdkit.Chem.AtomPairs.Utils import ExplainAtomCode
    return ExplainAtomCode(code, branchSubtract, includeChirality)


################################################################################
# Source: rdkit.Chem.BRICS.BRICSBuild
# File: rdkit/Chem/BRICS.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def rdkit_Chem_BRICS_BRICSBuild(
    fragments: list,
    onlyCompleteMols: bool = True,
    seeds: list = None,
    uniquify: bool = True,
    scrambleReagents: bool = True,
    maxDepth: int = 3
):
    """rdkit.Chem.BRICS.BRICSBuild builds new RDKit molecules by iteratively reconnecting BRICS fragments using the BRICS reverse reactions. This function is part of RDKit's cheminformatics toolkit and is used to enumerate candidate molecules by matching fragment attachment-point patterns, running the corresponding reverse BRICS reactions to join fragments, and optionally recursing to extend assemblies up to a specified depth.
    
    Args:
        fragments (list): A list of BRICS fragment objects to use as building blocks. In practice these are rdkit.Chem.Mol instances created from BRICS fragment SMILES or other molecule constructors; each fragment is inspected for substructure matches against the BRICS reverse reaction matchers in order to determine valid connection points. This parameter is required and the function will attempt to call HasSubstructMatch and other RDKit molecule methods on each element; providing objects that do not implement the expected RDKit Mol API will result in AttributeError or TypeError.
        onlyCompleteMols (bool): If True (default), the generator yields only "complete" molecules that have no remaining BRICS attachment-point dummy atoms (determined by matching a module-level dummyPattern). If False, intermediate assemblies that still contain attachment points are also yielded as they are discovered (these intermediate molecules are also accumulated as seeds for further growth when recursion is enabled). This controls whether partial assemblies are visible to the caller or only fully closed molecules are returned.
        seeds (list): Optional list of seed molecules to use as starting points for the enumeration. If None (the default), the function copies the fragments list and uses that as the initial set of seeds. Seeds should be a list of rdkit.Chem.Mol objects; the function will iterate over seeds and attempt substructure matches with the BRICS reverse reactions. Supplying a seeds list lets the caller bias or restrict the starting pool of building blocks.
        uniquify (bool): If True (default), the function attempts to avoid yielding duplicate molecules by tracking canonical SMILES strings computed with Chem.MolToSmiles(..., True) in an internal seen set. Uniquification is applied locally within each BRICSBuild invocation: intermediate products generated in the current call are filtered on the fly, and when recursion produces additional molecules the parent call filters those returned molecules against its own seen set before yielding. This reduces repeated identical outputs during a single enumeration run but does not globally deduplicate across independent BRICSBuild calls unless the caller enforces that externally.
        scrambleReagents (bool): If True (default) the order of reagents (the seeds list) is randomized before enumeration and the list of reverse reactions (module-level reverseReactions) is also randomized for this invocation by calling random.shuffle with random=random.random. This introduces non-determinism to enumeration order and can affect the sequence of yielded molecules; setting this to False yields a deterministic ordering governed by the supplied seeds and the module-level reverseReactions sequence. Note that reproducibility when scrambleReagents is True depends on the state of the global Python random module.
        maxDepth (int): The maximum recursive depth for building multi-fragment assemblies. The function performs a depth-first style recursion: when intermediate assemblies with remaining attachment points are found they are collected as next-step seeds and BRICSBuild is invoked recursively with maxDepth decreased by one. The default is 3. When maxDepth is 0 or there are no next-step seeds, recursion stops. This limits combinatorial explosion by bounding how many sequential connection steps are performed.
    
    Returns:
        generator: A generator that yields rdkit.Chem.Mol objects produced by combining the provided fragments using the BRICS reverse reactions. Each yielded molecule is an RDKit molecule object representing an assembled structure; if uniquify is enabled, yielded molecules are unique within the scope of the caller's BRICSBuild invocation as described above. If onlyCompleteMols is True, only molecules without BRICS dummy attachment atoms (complete molecules) are yielded; if False, intermediate assemblies containing attachment points may also be yielded. Side effects include possible modification of the module-level random order when scrambleReagents is True (via random.shuffle) and use of module-level data structures reverseReactions and dummyPattern to identify valid joins and attachment points.
    
    Raises:
        AttributeError: If elements of fragments or seeds do not implement the RDKit Mol API methods used (for example HasSubstructMatch or RunReactants) or if module-level reaction objects do not expose expected attributes (such as _matchers and RunReactants).
        TypeError: If arguments are provided with incorrect types (for example non-list values for fragments or seeds).
    """
    from rdkit.Chem.BRICS import BRICSBuild
    return BRICSBuild(
        fragments,
        onlyCompleteMols,
        seeds,
        uniquify,
        scrambleReagents,
        maxDepth
    )


################################################################################
# Source: rdkit.Chem.BuildFragmentCatalog.CalcGainsFromFps
# File: rdkit/Chem/BuildFragmentCatalog.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def rdkit_Chem_BuildFragmentCatalog_CalcGainsFromFps(
    suppl: list,
    fps: list,
    topN: int = -1,
    actName: str = "",
    acts: list = None,
    nActs: int = 2,
    reportFreq: int = 10,
    biasList: list = None
):
    """rdkit.Chem.BuildFragmentCatalog.CalcGainsFromFps calculates information gains for fingerprint bits from a collection of molecules and their corresponding fingerprint vectors, using RDKit's InfoTheory.InfoBitRanker. This function is used in cheminformatics workflows (for example, when building fragment catalogs or selecting discriminative fingerprint bits/descriptors for machine learning) to rank fingerprint bit positions by how informative they are with respect to an activity/label.
    
    Args:
        suppl (list): A sequence (typically a Python list) of RDKit molecule objects (Chem.Mol) that supply activity/label values used as class labels for information-gain calculation. Each molecule is expected to expose per-molecule properties accessible via mol.GetProp(name). The function will attempt to read the activity label from each molecule using actName unless an explicit acts list is provided. The length of suppl is used for progress reporting when available.
        fps (list): A list of fingerprint vectors corresponding to the molecules in suppl. Each entry is a fingerprint representation (for example an RDKit fingerprint bit vector or any sequence/object where bits can be indexed by the underlying InfoBitRanker implementation). The function reads the first entry fps[0] to determine the number of bits (nBits) and assumes all fingerprint entries have the same number of bits.
        topN (int): The number of top-ranking fingerprint bits (by information gain) to return. If topN is negative (the default -1), it is set to nBits (the total number of bits determined from fps[0]), returning gains for all bits. This controls the size of the returned ranking and therefore how many candidate bits/fragments are considered for downstream tasks such as feature selection.
        actName (str): The property name (string) to read from each molecule in suppl to obtain its activity/label when acts is not provided. Default is the empty string, which causes the function to select the last property name in suppl[0].GetPropNames() if acts is not supplied. The value read is converted to int for use as a label. If the named property is missing on any molecule, a KeyError is raised.
        acts (list): Optional list of numeric activity/label values provided externally. If provided, acts[i] is used as the activity/label for suppl[i] instead of reading a property from the molecule. The list must be aligned (same ordering) with suppl and fps; mismatched lengths may result in IndexError when the function indexes into fps or acts.
        nActs (int): The number of distinct activity/label states expected by the InfoBitRanker (default 2). This parameter configures the underlying InfoTheory.InfoBitRanker to consider the given number of classes when computing entropies and information gains; it is relevant when working with multi-class classification labels rather than binary activity.
        reportFreq (int): How often (in number of molecules processed) to emit progress messages via RDKit's message() function (default 10). When suppl has a determinable length, progress messages include counts of molecules processed out of the total. When suppl has no length (no __len__), progress messages report only the number processed so far. Setting reportFreq <= 0 may suppress periodic reporting depending on the environment.
        biasList (list): Optional bias list passed to the InfoBitRanker. If provided, the function constructs the ranker using InfoTheory.InfoType.BIASENTROPY and calls ranker.SetBiasList(biasList) before accumulating votes. This alters how information gain is computed to account for bit-specific biases (useful when certain bits should be weighted or adjusted due to external knowledge).
    
    Behavior, defaults, side effects, and failure modes:
        The function determines nBits from len(fps[0]) and, if topN < 0, sets topN to nBits so that all bits will be considered by default. If biasList is provided, a bias-aware ranking (BIASENTROPY) is used; otherwise plain ENTROPY ranking is used. For each molecule index i, the function obtains the activity label either from acts[i] (if acts is provided) or by reading the molecule property named actName (or the last property name of the first molecule if actName is empty). The activity value read from a molecule is converted to int; if the property is absent a KeyError is raised and processing stops. The function increments the InfoBitRanker via ranker.AccumulateVotes(fp, act) for each fingerprint fp and activity act. Every reportFreq molecules processed, the function emits a progress message using RDKit's message() function; if suppl has a length, the message includes the total count.
        The function assumes fps is indexable and aligned with suppl; if fps has fewer entries than suppl, indexing fps[i] will raise IndexError. The underlying InfoTheory.InfoBitRanker methods (constructor, SetBiasList, AccumulateVotes, GetTopN) may raise their own exceptions for invalid inputs or internal errors; these propagate to the caller. No molecular objects are modified by this function itself; its primary side effects are progress messages and potential exceptions.
    
    Returns:
        list: The result of ranker.GetTopN(topN) containing the topN information-gain entries for fingerprint bits as computed by RDKit's InfoTheory.InfoBitRanker. This returned sequence represents the ranking of fingerprint bit positions by their information contribution relative to the provided activity labels and is intended for use in fragment selection, feature ranking, or downstream descriptor/ML workflows. The exact element structure is defined by InfoBitRanker.GetTopN but the returned value is a sequence of entries ordered by decreasing information gain.
    """
    from rdkit.Chem.BuildFragmentCatalog import CalcGainsFromFps
    return CalcGainsFromFps(
        suppl,
        fps,
        topN,
        actName,
        acts,
        nActs,
        reportFreq,
        biasList
    )


################################################################################
# Source: rdkit.Chem.ChemUtils.DescriptorUtilities.setDescriptorVersion
# File: rdkit/Chem/ChemUtils/DescriptorUtilities.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def rdkit_Chem_ChemUtils_DescriptorUtilities_setDescriptorVersion(version: str = "1.0.0"):
    """rdkit.Chem.ChemUtils.DescriptorUtilities.setDescriptorVersion sets a version string on a descriptor function and returns a decorator that applies that metadata to the function. This is intended for use in the RDKit cheminformatics library to annotate descriptor-calculation callables (functions used to compute molecular descriptors and features for machine learning and database storage) with a stable version identifier that can be inspected by downstream tooling, descriptor registries, or serialization/database cartridges.
    
    Args:
        version (str): The version string to attach to a descriptor function. This argument is the metadata value that will be assigned to the decorated function as its .version attribute. The default value is "1.0.0". In RDKit workflows this string is typically used to indicate the descriptor implementation or schema version (for example, following semantic versioning) so consumers of descriptor values (machine-learning pipelines, descriptor registries, or database cartridges) can detect changes in descriptor calculation logic or provenance.
    
    Returns:
        function: A decorator function (wrapper) that accepts a single argument func (a Python callable implementing a descriptor) and performs an in-place side effect by setting func.version = version, then returns func. The returned decorator does not create a new function object copy; it mutates the original callable object so the version metadata is available at runtime. Failures can occur if the target callable does not support attribute assignment (for example certain built-in or extension-callables implemented in C), in which case an AttributeError will be raised when the decorator is applied. Use as:
            @setDescriptorVersion("1.2.3")
            def my_descriptor(mol):
                ...
        which results in my_descriptor.version == "1.2.3".
    """
    from rdkit.Chem.ChemUtils.DescriptorUtilities import setDescriptorVersion
    return setDescriptorVersion(version)


################################################################################
# Source: rdkit.Chem.ChemUtils.SDFToCSV.existingFile
# File: rdkit/Chem/ChemUtils/SDFToCSV.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def rdkit_Chem_ChemUtils_SDFToCSV_existingFile(filename: str):
    """rdkit.Chem.ChemUtils.SDFToCSV.existingFile: Validate that a filesystem path refers to an existing filesystem entry and return it unchanged. This function is intended to be used as the "type" callable for argparse when validating command-line filenames in RDKit utilities (for example SDF-to-CSV conversion scripts); it ensures a provided filename exists before downstream RDKit file-processing code is invoked.
    
    Args:
        filename (str): A filesystem path provided as a string (typically from command-line input). This is the candidate path to validate for existence. In the RDKit SDF-to-CSV command-line context, this should be the path to an input SDF (Structure-Data File) or other file the tool will open. The function checks existence using os.path.exists and therefore accepts relative or absolute paths; it does not open the file, does not verify readability or writability, and does not verify that the path is a regular file (directories and symlinks that exist will also pass). The practical significance is to fail fast during argument parsing if the named file is missing, preventing later file-not-found errors during RDKit molecular processing.
    
    Returns:
        str: The same filename string that was passed in, returned unchanged so argparse and subsequent RDKit processing code receive the validated path. Returning the original string allows it to be stored as the parsed argument value and used directly by SDF-to-CSV conversion routines.
    
    Raises:
        argparse.ArgumentTypeError: If os.path.exists(filename) is False, an argparse.ArgumentTypeError is raised with a message of the form "<filename> does not exist". This integrates with argparse to present a validation error to the user at command-line parsing time. Note that existence is checked at the time of parsing only; a race condition is possible if the file is removed or changed after parsing and before actual file access.
    """
    from rdkit.Chem.ChemUtils.SDFToCSV import existingFile
    return existingFile(filename)


################################################################################
# Source: rdkit.Chem.Draw.MolsMatrixToGridImage
# File: rdkit/Chem/Draw/__init__.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def rdkit_Chem_Draw_MolsMatrixToGridImage(
    molsMatrix: list,
    subImgSize: tuple = (200, 200),
    legendsMatrix: list = None,
    highlightAtomListsMatrix: list = None,
    highlightBondListsMatrix: list = None,
    useSVG: bool = False,
    returnPNG: bool = False,
    **kwargs
):
    """Creates a grid image of molecules from a nested (matrix-like) Python list, where each inner list represents a row. This function is part of RDKit's cheminformatics drawing utilities and is used to assemble per-row molecular layouts (for example, generations in a fragmentation tree) into a single image by padding rows to the length of the longest row and delegating rendering to MolsToGridImage.
    
    Args:
        molsMatrix (list): A two-deep nested data structure (list of lists) containing RDKit molecule objects or None for blank cells. Each inner list represents a row in the resulting grid. This parameter is the primary input: molecules are drawn in row-major order and None entries produce blank cells. Rows are automatically padded with blank cells at the end so all rows have the same length; to insert a blank cell in the middle of a row explicitly supply None at that position.
        subImgSize (tuple): Size of each cell in the grid (width, height) in pixels. This value is passed through to MolsToGridImage and controls the rendered size of each molecule cell. Default is (200, 200). Practical significance: increasing these dimensions makes each molecular depiction larger and may change overall image dimensions and layout.
        legendsMatrix (list): A two-deep nested data structure (list of lists) of strings matching the layout of molsMatrix used to label molecules. Each inner list corresponds to a row of legend strings. None or missing entries produce no legend for that cell. Legends are forwarded to MolsToGridImage after the nested structure is linearized, so their ordering and row correspondence are preserved when rendering.
        highlightAtomListsMatrix (list): A three-deep nested data structure (list of lists of lists) of integers specifying atom indices to highlight per molecule cell. The outer two depths match molsMatrix (rows and columns); the innermost lists contain integer atom indices for the corresponding molecule. None entries indicate no highlighting for that cell. These atom indices are passed to MolsToGridImage and are interpreted as RDKit atom indices for highlighting; invalid indices or non-integer values will cause downstream errors in the renderer.
        highlightBondListsMatrix (list): A three-deep nested data structure (list of lists of lists) of integers specifying bond indices to highlight per molecule cell. Structure and semantics mirror highlightAtomListsMatrix but for bond indices. Bond indices must correspond to RDKit bond indices in the molecule; invalid indices or types will result in rendering errors raised by the underlying drawing code.
        useSVG (bool): If True, return an SVG string. If False, return a PNG representation (either as PNG binary data or as a PIL image object depending on returnPNG). This flag is passed through to MolsToGridImage. Note that when useSVG is True, returnPNG is ignored because SVG output is not PNG.
        returnPNG (bool): When useSVG is False, controls whether the function returns raw PNG binary data (True) or a PIL.Image object for a PNG image (False). Default is False (PIL.Image). This parameter has no effect when useSVG is True.
        kwargs (dict): Additional keyword arguments forwarded unchanged to MolsToGridImage. Typical examples (documented in RDKit) include drawOptions or drawOptions-like objects such as an rdMolDraw2D.MolDrawOptions instance supplied as drawOptions in kwargs. Any unsupported kwargs will be handled (or rejected) by MolsToGridImage and may raise exceptions.
    
    Returns:
        str or bytes or PIL.Image.Image: Depending on useSVG and returnPNG:
            - If useSVG is True: returns an SVG string (str) containing the drawn grid in SVG format.
            - If useSVG is False and returnPNG is True: returns PNG binary data (bytes) representing the rasterized grid image.
            - If useSVG is False and returnPNG is False (default): returns a PIL.Image.Image object for a PNG image file.
        Practical significance: the returned object can be saved to disk, embedded in documents, or displayed in Jupyter notebooks. If a PIL object is returned, call its save() method to write a PNG file. If PNG bytes are returned, write the bytes to a file opened in binary mode. If an SVG string is returned, write it as text to an .svg file or embed it in HTML.
    
    Behavior, side effects, defaults, and failure modes:
        - The function first linearizes the nested inputs (molsMatrix and the optional legends and highlight matrices) via an internal helper so that each molecule, legend, and highlight list is placed in row-major order; it also computes molsPerRow to preserve original row boundaries when drawing.
        - Rows shorter than the longest row are padded at the end with blank cells; explicit None entries in molsMatrix create blank cells in arbitrary positions.
        - Highlight lists (atom and bond) must be lists of integers referring to RDKit atom/bond indices; providing mismatched nesting depths, non-iterable rows, or non-integer indices can raise exceptions from the internal conversion helper or from MolsToGridImage during rendering.
        - kwargs are forwarded to MolsToGridImage; any validation, interpretation, or side effects of those arguments are the responsibility of MolsToGridImage (for example, a drawOptions object affects styling and atom index display).
        - This function is intended for use cases where each row has semantic meaning (e.g., generations in a mass spectrometry fragmentation tree). If rows do not carry semantic meaning and you simply want a uniform grid, prefer MolsToGridImage with a flat list of molecules.
        - The function does not modify the input molecule objects; it only reads them to create the image. Any errors related to molecule validity (e.g., malformed RDKit Mol objects) will be raised by RDKit rendering code when attempting to draw those molecules.
    """
    from rdkit.Chem.Draw import MolsMatrixToGridImage
    return MolsMatrixToGridImage(
        molsMatrix,
        subImgSize,
        legendsMatrix,
        highlightAtomListsMatrix,
        highlightBondListsMatrix,
        useSVG,
        returnPNG,
        **kwargs
    )


################################################################################
# Source: rdkit.Chem.Draw.MolsToImage
# File: rdkit/Chem/Draw/__init__.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def rdkit_Chem_Draw_MolsToImage(
    mols: list,
    subImgSize: tuple = (200, 200),
    legends: list = None,
    **kwargs
):
    """rdkit.Chem.Draw.MolsToImage renders a sequence of RDKit molecule objects into a single horizontal composite image by creating a fixed-size subimage for each molecule and pasting them side-by-side into an RGBA canvas. This function is typically used in cheminformatics workflows (see RDKit) to create quick visual summaries of multiple molecules (for example, for reports, notebooks, or GUI displays) by delegating per-molecule rendering to MolToImage and composing the results.
    
    Args:
        mols (list): A list of objects representing molecules to be drawn. In the RDKit context these are expected to be RDKit molecule objects (e.g., Chem.Mol). Each entry in this list will be rendered independently by MolToImage and pasted into the output image in left-to-right order. If an element cannot be rendered by MolToImage (for example, because it is not an RDKit molecule or required atom coordinates are missing), the underlying MolToImage call will raise an exception (TypeError, ValueError, or other), which is propagated to the caller.
        subImgSize (tuple): A two-element tuple (width, height) specifying the pixel size for each per-molecule subimage. The first element is the subimage width in pixels and the second element is the subimage height in pixels. The function creates a new PIL RGBA image whose width is subImgSize[0] * len(mols) and whose height is subImgSize[1], then pastes each rendered molecule image into successive horizontal slots. Both elements of subImgSize should be integers; non-integer or negative values will typically cause Pillow (PIL) to raise an exception (TypeError or ValueError) when creating the new image or when pasting.
        legends (list): A list of legend values (typically strings) to be shown beneath or alongside each molecule when MolToImage renders them. The list is indexed in parallel with mols so legends[i] is passed as the legend for mols[i]. If legends is None (the default), a new list of the same length as mols filled with None is created so no legend text is rendered. If a provided legends list is shorter than mols, attempting to access missing entries will raise an IndexError; if it is longer, extra legend entries are ignored. Individual legend items may be None to indicate no legend for that molecule.
        kwargs (dict): Additional keyword arguments forwarded directly to MolToImage for each molecule. These arguments control per-molecule rendering behavior supported by MolToImage (for example drawing options, atom/bond highlighting, or font settings) and are applied identically to every molecule in mols. Any invalid or unexpected keyword arguments will be passed to MolToImage and may cause it to raise TypeError or other exceptions; such exceptions are not caught by MolsToImage and will propagate to the caller.
    
    Returns:
        PIL.Image.Image: A Pillow Image (mode "RGBA") containing the rendered molecules arranged horizontally. The image width equals subImgSize[0] * len(mols) and the height equals subImgSize[1]. The function has the side effect of creating and returning this composite image but does not modify the input mol objects. If len(mols) is zero, behavior depends on the underlying Pillow implementation (creating an image with zero width may raise an error); callers should ensure mols is non-empty if they require a valid non-zero-size image. Exceptions raised by Pillow (e.g., ValueError for invalid image sizes) or by MolToImage are propagated to the caller.
    """
    from rdkit.Chem.Draw import MolsToImage
    return MolsToImage(mols, subImgSize, legends, **kwargs)


################################################################################
# Source: rdkit.Chem.Draw.SimilarityMaps.GetStandardizedWeights
# File: rdkit/Chem/Draw/SimilarityMaps.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def rdkit_Chem_Draw_SimilarityMaps_GetStandardizedWeights(weights: list):
    """GetStandardizedWeights normalizes a sequence of per-atom weights so that the largest absolute weight becomes 1.0.
    
    This function is used in the cheminformatics context of rdkit.Chem.Draw.SimilarityMaps to prepare atomic contribution values (weights) for visualization and comparison across molecules or views. Given a list of numeric atomic weights (for example, per-atom similarity contributions or feature importances used when drawing similarity maps), the function rescales every weight by the maximum absolute weight found in the list so that the absolute maximum after scaling equals 1.0. This makes color/size mappings and cross-molecule comparisons meaningful and stable when rendering 2D similarity maps or other atom-level visual representations.
    
    Args:
        weights (list): A list of numeric per-atom weights. Each element represents an atomic contribution (e.g., similarity contribution, feature importance) that will be used in drawing or comparing molecular similarity maps. The function expects a non-empty list of numeric values; elements must be compatible with math.fabs (typically int or float). The order of elements is significant because the returned normalized list preserves the original atom ordering.
    
    Returns:
        tuple: A 2-tuple (standardized_weights, max_abs_weight).
            standardized_weights (list): A list of weights with the same length and ordering as the input 'weights'. If the maximum absolute weight in the input is greater than zero, each returned weight is the input weight divided by that maximum absolute value (resulting in at least one value having absolute value 1.0). If the maximum absolute weight is zero (all input weights are exactly zero), the original input list is returned unchanged.
            max_abs_weight (float): The maximum absolute weight found in the input list (i.e., max(abs(w) for w in weights)). This value is returned so callers can inspect the original scaling factor or undo the normalization if needed.
    
    Behavior, side effects, and failure modes:
        This function is pure (no side effects) and returns new values derived from the input list. It does not modify the input list in-place. If 'weights' is empty, calling max(...) on the sequence raises a ValueError; callers should ensure the list is non-empty or handle that exception. If any element of 'weights' is not numeric or not compatible with math.fabs (for example, a string or None), a TypeError or ValueError may be raised during computation. The function preserves the ordering and length of the input list and is intended for use in preparing atomic weights for RDKit similarity-map visualizations and related atom-level analyses.
    """
    from rdkit.Chem.Draw.SimilarityMaps import GetStandardizedWeights
    return GetStandardizedWeights(weights)


################################################################################
# Source: rdkit.Chem.EState.AtomTypes.BuildPatts
# File: rdkit/Chem/EState/AtomTypes.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def rdkit_Chem_EState_AtomTypes_BuildPatts(rawV: list = None):
    """BuildPatts compiles SMARTS patterns into RDKit Mol pattern objects and stores them in the module-level esPatterns list used by the EState atom-typing routines.
    
    This function is part of RDKit's cheminformatics EState (electrotopological state) machinery and is intended for internal use: it translates a sequence of named SMARTS pattern specifications into RDKit molecule pattern objects (Chem.Mol) that other EState code uses to match atom types. If rawV is omitted, the function reads the pattern specifications from the module-level _rawD. The function has no return value; instead it updates the global esPatterns variable to a list with the same length as rawV where each element is either None (if compilation failed or the pattern was skipped) or a tuple (name, patt) with patt being the RDKit Mol object produced by Chem.MolFromSmarts(sma). On SMARTS compilation failures the function writes a warning message to sys.stderr and leaves the corresponding esPatterns entry as None. Typical callers are other EState descriptor/atom-typing routines that expect esPatterns to contain compiled patterns for matching against molecule atoms.
    
    Args:
        rawV (list): A list of pattern specifications. Each element must be a two-element sequence (name, sma) where name is a textual identifier (usually a string) for the pattern and sma is a SMARTS pattern string that describes the atom or substructure to match. If rawV is None (the default), the function reads the pattern list from the module-level _rawD. The function uses len(rawV) to size the esPatterns list and iterates over rawV with enumerate, so rawV must be a sequence with a well-defined length and must yield 2-tuples on iteration.
    
    Returns:
        None: This function does not return a value. Side effects: it sets the module-level esPatterns variable to a list of length len(rawV) in which each populated entry is a tuple (name, patt) where patt is an RDKit Mol object created by Chem.MolFromSmarts(sma). Entries corresponding to SMARTS strings that fail to compile are left as None and a warning is written to sys.stderr. If rawV is None and _rawD is not defined, a NameError will be raised; if rawV is not a sequence of 2-tuples, unpacking or length operations may raise TypeError or ValueError. The function relies on RDKit's Chem.MolFromSmarts to compile SMARTS and will propagate exceptions from that call if they occur.
    """
    from rdkit.Chem.EState.AtomTypes import BuildPatts
    return BuildPatts(rawV)


################################################################################
# Source: rdkit.Chem.EState.EState.GetPrincipleQuantumNumber
# File: rdkit/Chem/EState/EState.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def rdkit_Chem_EState_EState_GetPrincipleQuantumNumber(atNum: int):
    """rdkit.Chem.EState.EState.GetPrincipleQuantumNumber maps an atomic number to the corresponding principal quantum number used by RDKit's EState (electrotopological state) calculations. This function is a small utility in the EState module that converts an element's atomic number (Z) into the principal quantum number (n) that represents the electron shell level used in descriptor calculations.
    
    Args:
        atNum (int): The atomic number (Z) of the element for which the principal quantum number is required. In the RDKit EState context this is the integer element number used when computing electrotopological state descriptors. The function expects an integer input; typical inputs are positive integers corresponding to chemical elements (e.g., 1 for hydrogen, 6 for carbon). The implementation does not perform explicit type validation, so passing a non-integer or a type that does not support integer comparisons may raise a TypeError in normal Python execution. Values less than or equal to 2 are treated the same as atomic numbers 1 and 2 (see behavior below).
    
    Returns:
        int: The principal quantum number (n) corresponding to the provided atomic number. The mapping implemented (and used by RDKit EState code) is: atNum <= 2 returns 1; atNum <= 10 returns 2; atNum <= 18 returns 3; atNum <= 36 returns 4; atNum <= 54 returns 5; atNum <= 86 returns 6; all larger integer atomic numbers return 7. The function is pure and deterministic with no side effects. For inputs outside the usual chemical element range, the function still returns an integer according to the mapping above (for example, any atNum > 86 yields 7, and atNum <= 2 yields 1).
    """
    from rdkit.Chem.EState.EState import GetPrincipleQuantumNumber
    return GetPrincipleQuantumNumber(atNum)


################################################################################
# Source: rdkit.Chem.FeatFinderCLI.existingFile
# File: rdkit/Chem/FeatFinderCLI.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def rdkit_Chem_FeatFinderCLI_existingFile(filename: str):
    """rdkit.Chem.FeatFinderCLI.existingFile: Type callable intended for use with argparse that verifies a provided filesystem path exists before the RDKit FeatFinder CLI proceeds to use it. In the RDKit cheminformatics CLI context this function is used to validate command-line arguments that are expected to be paths to existing files (for example input molecule files or feature definition files) so the program can fail early with a clear error instead of later encountering IO errors when opening or reading the file.
    
    Args:
        filename (str): A filesystem path provided as a string (the value passed by argparse for a command-line filename argument). This function uses os.path.exists to determine presence on the filesystem; it does not open, read, or otherwise access the file contents, nor does it check permissions or readability beyond what os.path.exists reports.
    
    Returns:
        str: The same filename string passed in, returned unmodified when the path exists on the filesystem. Returning the original string makes this function suitable as an argparse 'type' callable so the validated value can be used directly by subsequent CLI code.
    
    Raises:
        argparse.ArgumentTypeError: If os.path.exists(filename) returns False, this function raises argparse.ArgumentTypeError with the message "<filename> does not exist", which causes argparse to report the argument as invalid and display a usage/help message. This is the primary failure mode; callers should expect this exception when the provided path is missing.
    """
    from rdkit.Chem.FeatFinderCLI import existingFile
    return existingFile(filename)


################################################################################
# Source: rdkit.Chem.Features.FeatDirUtilsRD.findNeighbors
# File: rdkit/Chem/Features/FeatDirUtilsRD.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def rdkit_Chem_Features_FeatDirUtilsRD_findNeighbors(atomId: int, adjMat: list):
    """Find the IDs of atoms that are directly bonded to a given atom in a molecular adjacency matrix.
    
    This function is used in RDKit's cheminformatics context to obtain the neighbor atom indices for graph-based operations (for example, building neighbor lists for traversals, computing local descriptors, feature detection, or preparing inputs for fingerprint and descriptor calculations). It inspects the adjacency matrix row for the atom of interest and returns the indices of entries that indicate a bond (any numeric entry >= 1 is treated as a bond presence). The function performs no modification of its inputs and returns a new list.
    
    Args:
        atomId (int): The integer index of the atom of interest in the molecule's atom ordering. In practical RDKit usage this corresponds to the zero-based atom index used throughout molecule representations; the function uses this index to select the corresponding row in adjMat and find which other atom indices are connected (bonded) to this atom.
        adjMat (list): The adjacency matrix for the compound represented as a Python list (typically a list of lists where each sublist is indexable by atom index). Each element adjMat[i][j] is a numeric value representing the presence/strength/number of bonds between atom i and atom j; entries with value >= 1 are treated as indicating a bonded neighbor. The adjacency matrix is expected to be consistent with the molecule's atom ordering and sized so that adjMat[atomId] is a valid row.
    
    Returns:
        list: A list of int values representing the atom indices (neighbor IDs) that are directly connected to the atom identified by atomId. The returned indices are in ascending order because they are produced by iterating over the row for atomId; multiple bonds or bond-order counts in adjMat are treated as neighbour-presence (any value >= 1) and do not cause duplicated indices. If no neighbors are found, an empty list is returned.
    
    Raises:
        IndexError: If atomId is outside the range of indices for adjMat or if adjMat rows are shorter than expected, indicating a malformed adjacency matrix.
        TypeError: If adjMat is not indexable as a list of sequences or if elements of adjMat[atomId] do not support comparison with integer 1 (e.g., non-numeric types), causing the >= operation to fail.
        ValueError: If the structure of adjMat is inconsistent (for example, non-square when a square matrix is required by the caller) and this inconsistency is detected by higher-level code; this function itself will typically raise IndexError/TypeError in such cases.
    
    Behavior and notes:
        - The function treats any adjMat[atomId][j] value >= 1 as indicating that atom j is a neighbor (bonded) to atomId. This means single, double, and higher-order bonds represented by integer counts (1, 2, ...) will all be reported as neighbors.
        - No mutation: the function does not modify adjMat or atomId; it returns a newly constructed list.
        - Common failure modes include passing an adjacency structure that is not a list of indexable rows, supplying an out-of-range atomId, or using non-numeric matrix entries; callers should validate inputs or handle the raised exceptions when integrating this function into molecular-processing pipelines.
    """
    from rdkit.Chem.Features.FeatDirUtilsRD import findNeighbors
    return findNeighbors(atomId, adjMat)


################################################################################
# Source: rdkit.Chem.MCS.FindMCS
# File: rdkit/Chem/MCS.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def rdkit_Chem_MCS_FindMCS(
    mols: list,
    minNumAtoms: int = 2,
    maximize: str = "bonds",
    atomCompare: str = "elements",
    bondCompare: str = "bondtypes",
    matchValences: bool = False,
    ringMatchesRingOnly: bool = False,
    completeRingsOnly: bool = False,
    timeout: float = None,
    threshold: float = None
):
    """Find the maximum common substructure (MCS) of a set of molecules and return an MCSResult describing that substructure. This function is part of RDKit's cheminformatics toolkit and implements an exhaustive MCS search over the provided molecules. Note that this module is deprecated: a DeprecationWarning is issued and users are directed to use rdkit.Chem.rdFMCS for new code. The function is typically used in cheminformatics workflows to identify common scaffolds or subgraphs among multiple rdkit.Chem.Mol objects, for tasks such as scaffold analysis, clustering, or feature extraction for machine learning.
    
    Args:
        mols (list): A Python list of RDKit molecule objects (rdkit.Chem.Mol) to be compared. Each element should be a valid RDKit Mol representing a chemical structure. The MCS search will attempt to find a substructure common to the set of molecules in this list. If invalid objects are provided, the underlying FMCS implementation will raise an exception.
        minNumAtoms (int): The minimum number of atoms required for a common substructure to be considered a valid result. Default is 2. If no common substructure of at least this many atoms is found, the returned MCSResult will indicate failure by having numAtoms and numBonds set to -1 and smarts set to None.
        maximize (str): Which measure the search should maximize when choosing an MCS. The argument must be a string such as "atoms" (maximize the number of atoms in the MCS) or "bonds" (maximize the number of bonds in the MCS). Default is "bonds". Choosing "bonds" tends to favor substructures with more ring connectivity; choosing "atoms" favors larger atom counts. The choice affects which equivalent-size substructure is reported when multiple maxima exist.
        atomCompare (str): How atoms are considered equivalent during matching. Typical values are "elements" (atoms match when they are the same element), "any" (any atom matches any other atom), and "isotopes" (atoms match when their isotope labels are equal). Default is "elements". When matchValences is True this atom comparison is further constrained to require matching valences as well.
        bondCompare (str): How bonds are compared for equivalence. Typical values are "bondtypes" (bonds match only if they have the same bond type, e.g., single/double) and "any" (any bond matches any other bond). Default is "bondtypes". This setting controls whether bond order and type are enforced in the MCS.
        matchValences (bool): If False (default), valence information is ignored when comparing atoms. If True, the atom comparison is augmented to require that matching atoms have the same valence (for example, distinguishing a 3-valent nitrogen from a 5-valent nitrogen). When enabled, the SMARTS for atoms in the result may include valence indicators (for example, "v4").
        ringMatchesRingOnly (bool): If False (default), ring bonds in one molecule may match non-ring bonds in another (a linear chain can match a ring). If True, ring bonds are only allowed to match other ring bonds. Use this to avoid matching ring fragments to acyclic fragments.
        completeRingsOnly (bool): If False (default) partial rings may be included in the MCS. If True, the search requires that any atom that is in a ring in the original molecule must also be in a ring in the reported MCS; this prevents partial-ring matches. Setting completeRingsOnly to True also enables ringMatchesRingOnly behavior.
        timeout (float): Maximum wall-clock time in seconds allowed for the MCS search. Default is None (no timeout). The underlying algorithm performs an exhaustive search which can be computationally expensive for difficult instances; specify a timeout to abort the search and return the best solution found so far. If the timeout is reached the returned MCSResult will have its completed property set to 0 to indicate the search did not finish exhaustively.
        threshold (float): Optional fraction (float) specifying the minimum fraction of input molecules that must contain the reported common substructure. Default is None, which means the MCS must be present in all molecules in the provided mols list. If provided, this should be a float value (for example, 0.8) indicating that the reported MCS needs to appear in at least that fraction of the input molecules. The interpretation and allowed range are governed by the underlying FMCS implementation; invalid values will be handled by the underlying routine.
    
    Returns:
        MCSResult: An object describing the best common substructure found. The MCSResult contains at least the following fields commonly used in cheminformatics:
            numAtoms (int): The number of atoms in the reported MCS. If no MCS meeting minNumAtoms was found, this will be -1.
            numBonds (int): The number of bonds in the reported MCS. If no MCS meeting minNumAtoms was found, this will be -1.
            smarts (str or None): A SMARTS string representing the common substructure (atom and bond queries using RDKit SMARTS syntax). If no valid MCS was found this will be None.
            completed (int): 1 if the exhaustive search completed within the allotted time and the result is guaranteed to be a global maximum under the given parameters, or 0 if the search was terminated early (for example due to timeout) and the returned result is the best found so far.
        The returned MCSResult is constructed from the internal FMCS computation performed by rdkit.Chem.MCS; it contains the practical information needed to inspect, re-use, or apply the MCS (for example, substructure searches or scaffold extraction). Side effects: calling this function emits a DeprecationWarning advising users to prefer rdkit.Chem.rdFMCS.
    """
    from rdkit.Chem.MCS import FindMCS
    return FindMCS(
        mols,
        minNumAtoms,
        maximize,
        atomCompare,
        bondCompare,
        matchValences,
        ringMatchesRingOnly,
        completeRingsOnly,
        timeout,
        threshold
    )


################################################################################
# Source: rdkit.Chem.MolKey.MolKey.ErrorBitsToText
# File: rdkit/Chem/MolKey/MolKey.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def rdkit_Chem_MolKey_MolKey_ErrorBitsToText(err: int):
    """Concise translator of MolKey error-bit integer codes into human-readable error labels used by RDKit MolKey routines.
    
    Args:
        err (int): An integer error code produced by MolKey-related functions in RDKit that encodes one or more error conditions as bit flags. Each bit in this integer corresponds to an entry in the module-level ERROR_DICT mapping (a mapping from textual error description keys to integer bit masks). The function inspects the bits set in err using a bitwise AND against the values in ERROR_DICT and selects the textual keys whose associated bit masks are present in err. This parameter must be an integer; non-integer inputs will typically raise a TypeError when a bitwise operation is attempted. Typical practical use is to pass the error value returned by MolKey generation or validation routines so a developer or a diagnostic routine can obtain readable explanations of what went wrong.
    
    Returns:
        list: A list of textual error descriptions (the keys from ERROR_DICT) corresponding to the bit flags set in err. If no known bits are set in err (for example err == 0 or err contains bit positions not present in ERROR_DICT), an empty list is returned. The function has no side effects: it does not modify ERR OR ERROR_DICT and only reads ERROR_DICT to produce the list. If ERROR_DICT is not defined in the module namespace, a NameError will occur.
    """
    from rdkit.Chem.MolKey.MolKey import ErrorBitsToText
    return ErrorBitsToText(err)


################################################################################
# Source: rdkit.Chem.MolKey.MolKey.GetInchiForCTAB
# File: rdkit/Chem/MolKey/MolKey.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def rdkit_Chem_MolKey_MolKey_GetInchiForCTAB(ctab: str):
    """Get an InChI string and status information for a molecule provided as a CTAB (MolBlock) string.
    
    This function is part of RDKit's MolKey utilities and is used in cheminformatics workflows to convert a CTAB/MolBlock representation of a molecule into its standardized InChI identifier while validating and, where possible, fixing structural problems in the input CTAB. The routine performs a structural check using CheckCTAB(ctab, False), attempts to parse the (possibly fixed) MolBlock into an RDKit Mol without full sanitization, and then requests InChI generation using Chem.MolToInchi with options '/FixedH /SUU'. The returned result preserves any warnings or errors discovered during structural checking and combines them with any conversion or InChI-related errors into a single status bitmask.
    
    Args:
        ctab (str): A CTAB/MolBlock string containing the molecular structure to convert. In practice this is a text block conforming to MDL Molfile/CTAB format produced by molecular editors or coordinate generators (for example the output of pyAvalonTools.Generate2DCoords). The function does not modify the caller's input; instead it returns a possibly corrected MolBlock (see Returns). The input is passed to CheckCTAB for structural validation and potential fixing; if CheckCTAB identifies irrecoverable structural errors (the BAD_SET condition), the function returns immediately with no InChI computed.
    
    Returns:
        InchiResult: A 3-field result (positionally constructed as InchiResult(status_mask, inchi, fixed_mol)) where:
            - status_mask (int): An integer bitmask encoding the outcome of structural checking and conversion. This is the bitwise OR of the structuring-check flags returned by CheckCTAB and any conversion or InChI generation error flags set by this function. Known flags that may appear in the mask (as used in the implementation) include BAD_SET (structural errors that prevent conversion), RDKIT_CONVERSION_ERROR (failure to create an RDKit Mol from the MolBlock), INCHI_COMPUTATION_ERROR (MolToInchi returned no InChI), and INCHI_READWRITE_ERROR (an exception was raised by the InChI read/write layer). Callers should test bits in this integer to detect and distinguish warning versus fatal conditions.
            - inchi (str or None): The generated InChI string for the validated/fixed MolBlock when conversion and InChI generation succeed. If conversion fails or InChI generation fails or is not attempted (for example because BAD_SET was detected), this field is None. The InChI is produced by Chem.MolToInchi called with options '/FixedH /SUU' (the function passes these options directly to the InChI writer).
            - fixed_mol (str): The MolBlock (CTAB) string returned by CheckCTAB after validation and any automated fixes. This is the MolBlock that was actually passed to RDKit for conversion; it may be identical to the input ctab or a corrected version. Even when inchi is None, fixed_mol is provided so callers can inspect or persist the corrected structure.
    
    Behavior and failure modes:
        - The function first calls CheckCTAB(ctab, False) to validate the input CTAB and obtain a corrected MolBlock. If the structural-check result includes BAD_SET, the function returns immediately with inchi set to None and fixed_mol set to the corrected MolBlock; the status_mask will include the BAD_SET bit and any warning bits emitted by CheckCTAB.
        - The function attempts to create an RDKit Mol from the corrected MolBlock using Chem.MolFromMolBlock(..., sanitize=False). The sanitize=False choice means the parser avoids full sanitization; this reduces the chance of sanitization exceptions but may allow chemically inconsistent molecules to be represented, so conversion to InChI can still fail.
        - If MolFromMolBlock returns a valid Mol, the function calls Chem.MolToInchi(r_mol, '/FixedH /SUU') to produce an InChI; if this call returns an empty/false value, the INCHI_COMPUTATION_ERROR flag is set in the status_mask. If MolFromMolBlock fails, the RDKIT_CONVERSION_ERROR flag is set.
        - If Chem.MolToInchi raises Chem.InchiReadWriteError, the INCHI_READWRITE_ERROR flag is set.
        - Warnings and non-fatal issues from CheckCTAB are preserved and combined (bitwise OR) with any conversion/InChI errors in the returned status_mask; callers should inspect that mask to determine whether the returned inchi is reliable.
        - No exceptions are propagated for the handled conversion and InChI read/write errors; instead the function encodes failure modes in the status_mask and returns an appropriate inchi (or None) and the fixed MolBlock for downstream inspection.
    """
    from rdkit.Chem.MolKey.MolKey import GetInchiForCTAB
    return GetInchiForCTAB(ctab)


################################################################################
# Source: rdkit.Chem.PandasTools.RGroupDecompositionToFrame
# File: rdkit/Chem/PandasTools.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def rdkit_Chem_PandasTools_RGroupDecompositionToFrame(
    groups: dict,
    mols: list,
    include_core: bool = False,
    redraw_sidechains: bool = False
):
    """rdkit.Chem.PandasTools.RGroupDecompositionToFrame returns a pandas.DataFrame that organizes the results of an RDKit R-group decomposition into a table suitable for analysis and notebook display. It is used in cheminformatics workflows (see RDKit) to take the output of rdRGroupDecomposition.RGroupDecompose and produce a DataFrame whose rows correspond to input molecules and whose columns contain the original molecule and the extracted R-group fragments (and optionally the decomposition core).
    
    Args:
        groups (dict): A mapping from group names (strings such as 'Core', 'R1', 'R2', ...) to sequences (typically lists) of RDKit molecule objects (Chem.Mol). This dict is expected to contain an entry for each R-group with a list of fragment molecules of the same length as the mols list. The function makes a shallow copy of this dict internally (groups = groups.copy()) before modifying it; however, the molecules inside the lists are not deep-copied and may be modified in place.
        mols (list): A list of RDKit molecule objects (Chem.Mol) corresponding to the original input molecules. These are placed in the 'Mol' column of the returned DataFrame and must have the same length as the lists in groups. The mols list is used directly (not copied) as the 'Mol' column values.
        include_core (bool): If True, include the 'Core' column in the output DataFrame, preserving the 'Core' entry from groups; if False (default), the function removes the 'Core' key and column. If groups does not contain a 'Core' key and include_core is False, the attempt to remove 'Core' will raise a ValueError because 'Core' will not be found in the inferred columns list. This parameter controls whether the decomposition scaffold (core) is returned alongside R-group fragments.
        redraw_sidechains (bool): If True, recompute 2D coordinates for the sidechain fragments before placing them in the DataFrame. When enabled, the function imports rdkit.Chem.rdDepictor, removes explicit H atoms from each sidechain fragment via Chem.RemoveHs, and calls rdDepictor.Compute2DCoords on each fragment. This alters the Mol objects in place; because groups is only shallow-copied, these changes will affect the original fragment molecules referenced by the caller. Default is False.
    
    Behavior and side effects:
        - The function constructs column names as ['Mol'] followed by the keys from groups in their iteration/insertion order. If include_core is False, 'Core' is removed from that column list and the groups dict copy has its 'Core' key deleted.
        - If redraw_sidechains is True, the function will modify fragment molecules by removing hydrogens and computing 2D coordinates; these modifications affect the Mol objects themselves (no deep copy is performed).
        - The function calls ChangeMoleculeRendering(frame) before returning. This call configures RDKit-specific rendering metadata on the returned DataFrame (for example, molecule images in Jupyter/IPython environments) and is a side effect on the DataFrame used for interactive display.
        - The function relies on pandas (pd.DataFrame) to assemble the table; pandas will raise its usual errors (for example, ValueError) if the provided lists in groups and mols differ in length or otherwise cannot form a rectangular table.
        - The function does not serialize molecules to SMILES; it places RDKit Mol objects into the DataFrame (consistent with the typical use of rdRGroupDecomposition when called with asSmiles=False).
    
    Failure modes and constraints:
        - If the lists in groups (values) are not the same length as mols, pandas.DataFrame construction will fail (ValueError).
        - If include_core is False but 'Core' is not present in groups, cols.remove('Core') will raise a ValueError.
        - If rdDepictor is unavailable or Compute2DCoords raises an error, setting redraw_sidechains to True may raise an ImportError or runtime error.
        - Because the groups dict is shallow-copied, in-place modifications to Molecule objects will be visible to the caller.
    
    Returns:
        pandas.DataFrame: A DataFrame with one row per input molecule. Columns are ordered as 'Mol' followed by the group names in the order of groups.keys(); if include_core is False the 'Core' column is omitted. Each cell in the R-group columns contains the corresponding RDKit molecule fragment (Chem.Mol) for that row. The returned DataFrame has been passed to ChangeMoleculeRendering(frame) to set RDKit-specific display/rendering metadata.
    """
    from rdkit.Chem.PandasTools import RGroupDecompositionToFrame
    return RGroupDecompositionToFrame(groups, mols, include_core, redraw_sidechains)


################################################################################
# Source: rdkit.Chem.PandasTools.RenderImagesInAllDataFrames
# File: rdkit/Chem/PandasTools.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def rdkit_Chem_PandasTools_RenderImagesInAllDataFrames(images: bool = True):
    """rdkit.Chem.PandasTools.RenderImagesInAllDataFrames changes the global pandas DataFrame HTML rendering behavior so that HTML is not escaped and inline images (for example RDKit-generated molecule depictions) can be displayed in every DataFrame rendered in the current Python session.
    
    Args:
        images (bool): Flag that enables or disables rendering of images in all pandas DataFrames. When True (the default), the function instructs the RDKit pandas patcher to stop escaping HTML characters in DataFrame HTML output so that embedded <img> tags or other HTML produced by RDKit rendering routines are rendered as images in environments that display HTML (for example Jupyter notebooks). When False, the function requests the patcher to restore escaping of HTML characters so that raw HTML is not rendered. This parameter must be a Python bool; other types are not accepted by the underlying patcher.
    
    Returns:
        None: This function does not return a value. Instead it has a global side effect on pandas display behavior for the entire Python process: it calls PandasPatcher.renderImagesInAllDataFrames(images) to change pandas' HTML escaping policy. If the PandasPatcher object is not available in the runtime (NameError), the function catches that error and emits a warning via the RDKit logger and leaves pandas behavior unchanged. Because the change is global, it affects all subsequently rendered DataFrames in the session; to change rendering for a single DataFrame only, use ChangeMoleculeRendering instead.
    """
    from rdkit.Chem.PandasTools import RenderImagesInAllDataFrames
    return RenderImagesInAllDataFrames(images)


################################################################################
# Source: rdkit.Chem.Pharm2D.Utils.BinsTriangleInequality
# File: rdkit/Chem/Pharm2D/Utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def rdkit_Chem_Pharm2D_Utils_BinsTriangleInequality(d1: tuple, d2: tuple, d3: tuple):
    """rdkit.Chem.Pharm2D.Utils.BinsTriangleInequality checks whether three binned distance intervals satisfy a conservative form of the triangle inequality used in Pharm2D pharmacophore/fingerprint calculations.
    
    This function is used when working with binned distance representations (distance bins) of pairwise feature separations in 2D pharmacophore or fingerprint generation. Each input tuple represents a bin as a (lower, upper) interval for a single pairwise distance. The check is conservative: it uses the upper bounds of two bins and the lower bound of the third to ensure that no combination of values drawn from the bins can violate the standard triangle inequality. This conservative test is useful for pruning impossible triplets of feature distances early in descriptor/fingerprint generation, improving performance and correctness of downstream cheminformatics algorithms.
    
    Args:
        d1 (tuple): A binned distance interval for the first pairwise distance. The tuple is expected to contain two elements where d1[0] is the lower bound and d1[1] is the upper bound of the bin. These bounds represent the inclusive or otherwise-applicable numerical limits for that binned distance used by Pharm2D computations.
        d2 (tuple): A binned distance interval for the second pairwise distance. The tuple is expected to contain two elements where d2[0] is the lower bound and d2[1] is the upper bound of the bin. This parameter plays the role of the second side length interval in the triangle-inequality check for pharmacophore triplets.
        d3 (tuple): A binned distance interval for the third pairwise distance. The tuple is expected to contain two elements where d3[0] is the lower bound and d3[1] is the upper bound of the bin. This parameter plays the role of the third side length interval in the triangle-inequality check for pharmacophore triplets.
    
    Behavior:
        The function performs a conservative binned triangle-inequality test by evaluating all three permutations of the inequality in the following form:
           d1_upper + d2_upper >= d3_lower
           d2_upper + d3_upper >= d1_lower
           d3_upper + d1_upper >= d2_lower
        where dX_lower corresponds to dX[0] and dX_upper corresponds to dX[1]. Only if all three conditions hold does the function consider the three bins mutually compatible under the triangle inequality. There are no side effects: the function does not modify its inputs or any external state.
    
    Failure modes and errors:
        If any input tuple does not contain at least two indexable elements, accessing dX[0] or dX[1] will raise an IndexError. If tuple elements are not comparable with the '+' and '<' operators (for example, non-numeric types), a TypeError or other comparison/operation error may be raised. The function does not perform explicit type checking or shape validation beyond tuple indexing; callers should ensure bins are provided as two-element tuples of comparable numeric-like bounds.
    
    Returns:
        bool: True if the conservative binned triangle inequality holds for all three permutations (i.e., no combination of values from the provided bins can violate the triangle inequality), False otherwise. A return value of False indicates that at least one permutation fails the conservative check and the three binned distances cannot all correspond to a valid triangle under the conservative binning assumption.
    """
    from rdkit.Chem.Pharm2D.Utils import BinsTriangleInequality
    return BinsTriangleInequality(d1, d2, d3)


################################################################################
# Source: rdkit.Chem.Pharm2D.Utils.CountUpTo
# File: rdkit/Chem/Pharm2D/Utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def rdkit_Chem_Pharm2D_Utils_CountUpTo(
    nItems: int,
    nSlots: int,
    vs: list,
    idx: int = 0,
    startAt: int = 0
):
    """CountUpTo computes the zero-based rank (the count of preceding combinations) of a given combination of slot indices within the combinatorial enumeration produced by the Pharm2D index-combination generator (_GetIndexCombinations_). This function is part of RDKit's Chem.Pharm2D utilities used when enumerating combinations for descriptor/fingerprint generation in cheminformatics workflows (see RDKit documentation). It traverses the combination space using recursion and combinatorial counts (via NumCombinations) and uses a global cache to speed repeated queries.
    
    Args:
        nItems (int): The total number of indistinguishable items to distribute across slots. In the Pharm2D context this represents the pool of discrete units (for example, positions or counts) whose distributions are enumerated. This value is used by the combinatorial counting routine (NumCombinations) to compute how many combinations lie below a given partial selection.
        nSlots (int): The number of slots (positions, levels, or places) among which nItems may be distributed. In Pharm2D enumeration, each slot corresponds to a level in the nested combination representation. The function expects to count combinations across these nSlots.
        vs (list): A list containing the target combination values whose position is sought. Each element of vs is expected to be an integer used as an index/value for the corresponding slot; the function reads vs[idx] during recursion. Practically, vs represents one particular combination (one tuple of slot indices) produced by the index-combination generator.
        idx (int, optional): The current recursion depth / slot index being processed. Default is 0. Callers normally invoke CountUpTo with the default (idx=0); internal recursive calls advance idx+1. This parameter must be a non-negative integer and is used only to traverse vs and compute partial counts for subsequent slots.
        startAt (int, optional): The minimum value permitted at the current slot during counting. Default is 0. In the combinatorial enumeration used by _GetIndexCombinations_, slots are generated with nondecreasing values and startAt enforces that lower bound as recursion proceeds; callers should not need to set this when initiating a top-level query.
    
    Returns:
        int: The zero-based number of combinations that occur before the provided combination vs in the enumeration order generated by _GetIndexCombinations_. This integer is computed by summing counts of combinations with smaller values at each slot (using NumCombinations for the remaining items and slots) and then recursing into the next slot. The returned value can be used as a linear index for mapping a combination to a unique position in a flattened enumeration (for example, when indexing descriptor or fingerprint entries).
    
    Behavior, side effects, and failure modes:
        - The function computes its result recursively. For idx >= nSlots the function treats the contribution as zero; when idx == nSlots - 1 it returns vs[idx] - startAt as the terminal count contribution for the final slot.
        - Results for top-level calls (idx == 0) are stored in a global cache (_countCache) to accelerate repeated queries with the same (nItems, nSlots, tuple(vs)). This is a side effect: the cache is updated with the computed integer when idx == 0.
        - If a global debug flag (_verbose) is set, the function will print diagnostic messages to standard output during execution; this is a side effect used only for debugging and does not affect return values.
        - The function assumes vs contains at least nSlots elements accessible at indices 0..nSlots-1; if vs is shorter, an IndexError will be raised. If elements of vs are not integers or violate the implied nondecreasing constraint expected by the enumeration logic, the computed rank will be incorrect or may lead to negative or otherwise invalid intermediate counts. Such incorrect inputs represent undefined or erroneous usage rather than handled exceptions.
        - No new types are introduced by this function; callers should pass Python ints for nItems, nSlots, idx, startAt and a list for vs as in the original signature.
    """
    from rdkit.Chem.Pharm2D.Utils import CountUpTo
    return CountUpTo(nItems, nSlots, vs, idx, startAt)


################################################################################
# Source: rdkit.Chem.Pharm2D.Utils.GetAllCombinations
# File: rdkit/Chem/Pharm2D/Utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def rdkit_Chem_Pharm2D_Utils_GetAllCombinations(
    choices: list,
    noDups: int = 1,
    which: int = 0
):
    """rdkit.Chem.Pharm2D.Utils.GetAllCombinations enumerates all possible combinations formed by picking one element from each sequence in a list of sequences. In the context of RDKit (a cheminformatics library) and the Pharm2D utilities, this function is used to generate all possible tuples of pharmacophoric/feature choices when constructing 2D pharmacophore descriptors or fingerprints; these enumerated combinations are the raw candidate feature vectors that downstream code will score or encode into descriptors.
    
    Args:
        choices (list): A list (sequence) of sequences; each element of choices is itself a sequence (for example, a tuple or list) containing alternative values/options for that position. The function returns combinations that pick exactly one element from each sequence in choices, preserving the original ordering of positions. Practical significance: in Pharm2D generation this models the alternative features or atom/group choices at successive positions when building feature tuples.
        noDups (int): If nonzero, combinations that would contain duplicate values (for example, a constructed combination like [1, 1, 0] where the same value appears in more than one selected position) are omitted from the result. If zero, duplicates are allowed. Default is 1 (duplicates suppressed). This parameter enables callers to avoid degenerate or redundant tuples when elements represent identical chemical features.
        which (int): Internal recursion index used to track the current position in choices; callers should normally leave this as the default 0. The function recurses by incrementing which until it reaches the last sequence. Practical significance: callers may pass a different value only for advanced recursive use; passing values >= len(choices) causes the function to return an empty list. Negative or out-of-range values are allowed by the implementation but are intended for internal recursion control and may lead to empty results or other non-useful outputs.
    
    Returns:
        list: A list of lists where each inner list is one combination constructed by selecting one element from each sequence in choices in order. If choices is empty or which >= len(choices), an empty list is returned. Each inner list preserves the ordering of positions from choices. Notes on behavior and failure modes: the function performs no mutation of its inputs (pure function) and uses recursion; it can produce a combinatorial explosion in the number of returned combinations when choices contains many sequences or sequences with many alternatives, which may lead to high memory use or long runtimes. The membership test used to suppress duplicates (when noDups is nonzero) relies on Python equality semantics for the elements; elements that do not implement meaningful equality may yield unexpected duplicate suppression behavior.
    """
    from rdkit.Chem.Pharm2D.Utils import GetAllCombinations
    return GetAllCombinations(choices, noDups, which)


################################################################################
# Source: rdkit.Chem.Pharm2D.Utils.GetIndexCombinations
# File: rdkit/Chem/Pharm2D/Utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def rdkit_Chem_Pharm2D_Utils_GetIndexCombinations(
    nItems: int,
    nSlots: int,
    slot: int = 0,
    lastItemVal: int = 0
):
    """rdkit.Chem.Pharm2D.Utils.GetIndexCombinations generates all ordered index combinations used by Pharm2D utilities in RDKit for distributing a fixed number of items into a fixed number of slots without producing duplicate permutations. It is commonly used in RDKit pharmacophore/descriptor and fingerprint construction code to enumerate index tuples (for example, atom or feature indices) that represent combinations with repetition where the order of slots is fixed and permutations are suppressed.
    
    Args:
        nItems (int): The number of distinct items (indices) available to place into slots. In the Pharm2D/fingerprint context this typically corresponds to the number of distinct features, atoms, or bins that can be selected. Values are used as exclusive upper bounds for index values (valid indices are 0 .. nItems-1). Supplying non-positive or otherwise small nItems will result in empty combination lists when no valid indices exist.
        nSlots (int): The number of slots (positions) to fill in each generated combination. In practical use this is the tuple length for a descriptor/fingerprint element (for example, the arity of a pharmacophore feature tuple). If nSlots <= 0, the function returns an empty list; otherwise the recursion depth is proportional to nSlots.
        slot (int): Internal recursion parameter indicating the current slot index being filled. Callers constructing combinations externally should normally leave this at the default 0. When used internally, slot increments from 0 up to nSlots-1; if slot >= nSlots the function returns an empty list. Supplying a non-default slot value alters recursion and bypasses the memoization used when slot == 0.
        lastItemVal (int): Internal recursion parameter that defines the minimum item index allowed for the current slot. It enforces non-decreasing order of indices across slots to avoid duplicate permutations. Callers should normally leave this at the default 0; providing a different value restricts the allowed indices for the current and subsequent slots.
    
    Returns:
        list[list[int]]: A list of combinations, where each inner list is an ordered sequence of length nSlots of integers in the range [0, nItems-1]. Each inner list is non-decreasing (i.e., each element is >= the previous), which implements combinations with replacement and avoids duplicate permutations. If no valid combinations exist for the provided arguments (for example, nItems <= 0 or nSlots <= 0), an empty list is returned.
    
    Behavior and side effects:
        - The function performs a recursive depth-first enumeration: for each allowed value x at the current slot it recursively enumerates combinations for the remaining slots with x as the new minimum (lastItemVal), then prefixes x to each returned suffix.
        - To avoid recomputation at top-level calls, the function uses and updates a global cache named _indexCombinations: when called with slot == 0 and a previously computed (nItems, nSlots) key exists in the cache, the cached result is returned; when slot == 0 and the result is newly computed it is stored in the cache. Recursive calls with slot != 0 do not use or populate the cache.
        - The function does not perform explicit type validation; arguments are used as integers. Passing very large nSlots may cause deep recursion and raise a RecursionError in Python; the number of returned combinations grows combinatorially with nItems and nSlots and may consume large amounts of memory/time.
        - The defaults slot=0 and lastItemVal=0 configure the routine for normal external use; modifying them is intended for internal/recursive control and affects caching and output.
    """
    from rdkit.Chem.Pharm2D.Utils import GetIndexCombinations
    return GetIndexCombinations(nItems, nSlots, slot, lastItemVal)


################################################################################
# Source: rdkit.Chem.Pharm2D.Utils.GetPossibleScaffolds
# File: rdkit/Chem/Pharm2D/Utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def rdkit_Chem_Pharm2D_Utils_GetPossibleScaffolds(
    nPts: int,
    bins: list,
    useTriangleInequality: bool = True
):
    """GetPossibleScaffolds returns all candidate pharmacophore scaffolds (distance-bin assignments)
    for a specified number of pharmacophore points, optionally filtering them to only those
    that satisfy the triangle inequality. In the RDKit Pharm2D context this function is used
    to enumerate all realizable assignments of inter-point distance bins (as used by Pharm2D
    fingerprinting and scaffold matching) given the discretized distance bins available.
    
    Args:
        nPts (int): The number of pharmacophore points (points in the scaffold) to consider.
            This determines the number of pairwise distances for the scaffold via the global
            mapping nPointDistDict[nPts]. Practical significance: for pharmacophore-based
            matching and fingerprints, nPts selects the scaffold size (e.g., 2, 3, 4 point
            pharmacophores) and therefore the dimensionality of the distance-vector being
            enumerated. Behavior: if nPts < 2 the function returns 0 because fewer than two
            points have no inter-point distances to form a scaffold. Failure modes:
            requesting an nPts value not present in the global nPointDistDict will raise a
            KeyError when the function attempts to read nPointDistDict[nPts].
        bins (list): A list of discrete distance bins available to encode each pairwise
            distance. Each element of a returned scaffold tuple is an integer index into
            this list (valid indices are 0 .. len(bins)-1). Practical significance: these
            bins correspond to discretized distance ranges used by Pharm2D to group distances
            into fingerprint bins. Behavior: the function builds all combinations of bin
            indices of length nDists = len(nPointDistDict[nPts]) using GetAllCombinations.
            If len(bins) == 0 the underlying combination generator will produce no
            combinations and the function will return an empty list for nPts >= 2.
        useTriangleInequality (bool): If True (default), the function filters the generated
            bin-index combinations with ScaffoldPasses(combo, bins) to ensure the set of
            pairwise distances represented by the bin indices is realizable in Euclidean
            space (i.e., satisfies the triangle inequality for all relevant triples of
            points). Practical significance: enabling this prevents enumerating distance
            assignments that cannot correspond to actual 2D/3D placements of the specified
            points and thus reduces false-positive scaffolds for downstream pharmacophore
            matching or fingerprinting tasks. If False, the function returns all possible
            combinations of bin indices without triangle-inequality filtering. Default:
            True. Side effects: none specific beyond calling GetAllCombinations and
            ScaffoldPasses; those helper functions may themselves have side effects or raise
            exceptions.
    
    Returns:
        int or list of tuple of int: If nPts < 2 the function returns the integer 0 to
        indicate that no scaffold (no inter-point distances) can be constructed. For nPts >= 2
        the function returns a list of tuples; each tuple contains integer indices into the
        bins list. Each returned tuple has length nDists where nDists == len(nPointDistDict[nPts]),
        i.e., one entry per distinct pairwise distance for the given number of points. The
        tuples enumerate candidate scaffolds: if useTriangleInequality is True, the list
        contains only those tuples for which ScaffoldPasses(tuple, bins) returned True;
        otherwise the list contains every combination produced by GetAllCombinations([range(len(bins))] * nDists, noDups=0).
        Practical significance: the returned tuples can be used directly as discrete scaffold
        descriptors for Pharm2D fingerprinting and pharmacophore matching. Failure modes:
        a KeyError may be raised when accessing nPointDistDict[nPts] for unsupported nPts,
        and GetAllCombinations or ScaffoldPasses may raise exceptions propagated to the caller.
    
    Notes:
        - The function reads global helper structures/functions (nPointDistDict, GetAllCombinations,
          ScaffoldPasses) defined elsewhere in rdkit.Chem.Pharm2D.Utils; their semantics determine
          the ordering and interpretation of tuple elements.
        - The enumeration is combinatorial in len(bins) and nDists; for large inputs this can
          be computationally expensive. Use useTriangleInequality=True to reduce the result
          set to geometrically realizable scaffolds when appropriate.
    """
    from rdkit.Chem.Pharm2D.Utils import GetPossibleScaffolds
    return GetPossibleScaffolds(nPts, bins, useTriangleInequality)


################################################################################
# Source: rdkit.Chem.Pharm2D.Utils.GetTriangles
# File: rdkit/Chem/Pharm2D/Utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def rdkit_Chem_Pharm2D_Utils_GetTriangles(nPts: int):
    """rdkit.Chem.Pharm2D.Utils.GetTriangles returns the index triples that define the set of triangles used to represent an nPts-pharmacophore in RDKit's Pharm2D utilities. This function is used by the Pharm2D pharmacophore/fingerprint code to enumerate the combinations of three distance positions (points) that form the triangular features used in 3-point pharmacophore representations and downstream fingerprint calculations.
    
    The function computes a sliding-window set of triangles starting from the triple (0, 1, nPts-1) and incrementing each index in lock-step until the first index reaches nPts-2. The computed result is cached in the module-level dictionary _trianglesInPharmacophore to avoid recomputation for the same nPts on subsequent calls.
    
    Args:
        nPts (int): The number of points in the pharmacophore. This integer specifies how many ordered distance positions are present in the pharmacophore model for which triangle index triples are required. The function expects a non-negative integer; callers in the Pharm2D fingerprinting pipeline pass the number of distance bins or point positions used to construct triangular pharmacophore features.
    
    Returns:
        tuple or list: For nPts >= 3, returns a tuple of integer triples, where each triple is of the form (idx1, idx2, idx3). Each idxN is an integer index in the range 0..nPts-1 that refers to a position in the nPts-ordered pharmacophore; the set of triples enumerates the triangles (three-point combinations) used by Pharm2D. The number of returned triangles for nPts >= 3 is (nPts - 2). For nPts < 3, the function returns an empty list ([]), since fewer than three points cannot form a triangle.
    
    Behavior and side effects:
        - Caching: The function uses and mutates the global cache _trianglesInPharmacophore (a dict keyed by nPts). When triangles for a given nPts are not already cached, the function computes them, converts the computed list to a tuple, stores that tuple in the cache under the key nPts, and returns it. Subsequent calls with the same nPts will return the cached tuple object.
        - Deterministic ordering: The triangles are generated in a deterministic, sliding-window order starting from (0, 1, nPts-1) and advancing until the first index equals nPts-2; this deterministic order matters for reproducible fingerprint construction in Pharm2D workflows.
        - Input expectations and failure modes: The function expects an integer nPts. If nPts < 3 the function intentionally returns an empty list. Passing non-integer or otherwise invalid types is not checked by this function and may lead to unpredictable behavior; callers in RDKit should pass an integer number of points. Negative integers will be treated like any integer with the same numeric comparison semantics and will result in the empty-list return for nPts < 3.
        - No external resources are modified beyond the module-level cache described above.
    """
    from rdkit.Chem.Pharm2D.Utils import GetTriangles
    return GetTriangles(nPts)


################################################################################
# Source: rdkit.Chem.Pharm2D.Utils.GetUniqueCombinations
# File: rdkit/Chem/Pharm2D/Utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def rdkit_Chem_Pharm2D_Utils_GetUniqueCombinations(choices: list, classes: list):
    """rdkit.Chem.Pharm2D.Utils.GetUniqueCombinations returns the set of unique, deterministic mappings between provided class identifiers and one selected item from each corresponding choices list while avoiding selections that reuse the same item across multiple class positions. In the RDKit Pharm2D context this utility is used when generating combinatorial assignments of pharmacophore or feature labels (classes) to candidate feature instances (choices) for descriptor/fingerprint enumeration, ensuring each returned combination maps each class to a distinct feature and that equivalent assignments are deduplicated and returned in a stable order.
    
    Args:
        choices (list): A list with one element per class position; each element should be a sequence (for example a list) of candidate items that may be selected for that class. The function uses itertools.product(*choices) to iterate over every possible selection of one item from each element of choices. The practical significance in Pharm2D workflows is that choices enumerates alternative feature instances (e.g., different atoms or pharmacophore points) that could be assigned to each label when building 2D pharmacophore combinations.
        classes (list): A list of class identifiers corresponding position-by-position to entries of choices. len(classes) must equal len(choices); the function asserts this and will raise AssertionError if lengths differ. In the Pharm2D domain these are the feature labels (for example pharmacophore types or descriptor slots) that will be paired with chosen items from the matching entry of choices.
    
    Returns:
        list: A list of combinations. Each combination is a list of (class, choice) tuples representing an assignment of each class identifier to a selected item from the corresponding entry of choices. Combinations that would assign the same selected item to more than one class position are omitted (the function skips products where the selected items are not all distinct). Equivalent assignments that differ only by class ordering are deduplicated: for each valid product the function constructs a tuple of (class, choice) pairs, sorts those pairs to produce a canonical order, and uses a set to ensure uniqueness; the final return value is produced by sorting these canonical tuples and converting each back to a list to provide a stable, deterministic ordering across runs. Note that adding combinations to a set requires that the (class, choice) tuples be hashable; if class identifiers or choice items are not hashable, a TypeError may be raised. Also be aware that the number of products can grow combinatorially with the lengths of choices, so memory and runtime can become large for many or large choice lists. The function has no side effects and does not modify its inputs.
    """
    from rdkit.Chem.Pharm2D.Utils import GetUniqueCombinations
    return GetUniqueCombinations(choices, classes)


################################################################################
# Source: rdkit.Chem.Pharm2D.Utils.NumCombinations
# File: rdkit/Chem/Pharm2D/Utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def rdkit_Chem_Pharm2D_Utils_NumCombinations(nItems: int, nSlots: int):
    """rdkit.Chem.Pharm2D.Utils.NumCombinations: Compute the number of unordered combinations with repetition for placing a set of items into a fixed number of slots. This function implements the "stars and bars" combinatorial formula used in RDKit's Pharm2D utilities to count how many distinct multisets of size nSlots can be formed from nItems types (for example, counting ways to assign pharmacophore feature types to fingerprint slots when order does not matter and repeats are allowed).
    
    Args:
        nItems (int): The number of distinct item types (N in the combinatorial formula). In the Pharm2D context this corresponds to the number of distinct feature categories or bins that can be placed into slots. This must be an integer; passing values that are not integers or that lead to invalid binomial parameters will raise the underlying exception from the comb implementation (typically TypeError or ValueError).
        nSlots (int): The number of slots to fill (S in the combinatorial formula). In the Pharm2D context this is the number of positions in a descriptor or fingerprint that will be populated from the nItems types. This must be an integer; invalid values (negative or those that make the binomial arguments invalid) will cause the underlying comb call to raise an exception.
    
    Returns:
        int: The number of unordered combinations with repetition, computed by the formula res = (N + S - 1)! / ((N - 1)! * S!), which is implemented as comb(nItems + nSlots - 1, nSlots). This integer is the count of distinct multisets of size nSlots drawn from nItems types and is used in RDKit Pharm2D combinatoric calculations (e.g., enumerating feature assignments). Side effects: the function caches results in the module-global dictionary _numCombDict keyed by the tuple (nItems, nSlots); if a cached value exists it is returned immediately, otherwise the value is computed, stored in _numCombDict, and then returned. Failure modes: if inputs produce invalid arguments for the binomial coefficient (for example resulting in negative arguments) or are of incorrect types, the underlying comb implementation will raise TypeError or ValueError.
    """
    from rdkit.Chem.Pharm2D.Utils import NumCombinations
    return NumCombinations(nItems, nSlots)


################################################################################
# Source: rdkit.Chem.Pharm2D.Utils.OrderTriangle
# File: rdkit/Chem/Pharm2D/Utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def rdkit_Chem_Pharm2D_Utils_OrderTriangle(featIndices: list, dists: list):
    """rdkit.Chem.Pharm2D.Utils.OrderTriangle canonicalizes the ordering of a triangle of pharmacophore feature indices and their associated pairwise distances so that the same geometric/feature triangle is represented consistently for fingerprinting, indexing, and comparison tasks in RDKit's Pharm2D utilities.
    
    Args:
        featIndices (list): A length-3 list representing the feature class indices (typically integer identifiers of pharmacophore feature classes) at the three vertices of a triangle. The elements' order encodes which vertex is associated with which distance in the parallel dists list. In Pharm2D fingerprinting this list identifies the feature classes participating in the triangle; the function preserves the semantic association between positions in featIndices and entries in dists while returning a canonical ordering. The function requires exactly three entries and will raise an error otherwise.
        dists (list): A length-3 list of distances (numeric values, e.g. int or float) corresponding to the three edges of the triangle. Each element is associated position-wise with the entries of featIndices. These distances are used to determine a canonical permutation of the triangle so that identical triangles (up to vertex labeling) map to the same ordered representation used downstream in pharmacophore 2D fingerprint generation and comparison. The function requires exactly three entries and will raise an error otherwise.
    
    Behavior and algorithmic details:
    This function enforces a canonical ordering for the pair (featIndices, dists) for triangles used in RDKit Pharm2D code. It first validates that both featIndices and dists contain exactly three elements; if not, a ValueError is raised. If the three feature indices are all distinct, no reordering is performed and the input lists are returned unchanged (this preserves any existing correspondence and is the least disruptive option for already-unique vertices). If the feature indices are not all distinct (two equal or all three equal), the function computes three pairwise sums of distances (dSums[0] = dists[0] + dists[1], dSums[1] = dists[0] + dists[2], dSums[2] = dists[1] + dists[2]) and uses the maximum of these sums to select which vertex-edge arrangement should be placed first in the canonical representation. Ties and orientations are resolved deterministically by comparing the relevant individual distances; for the fully symmetric case (all three feature indices equal) the effect is to produce a permutation of dists that places the distances in a consistent order (examples in source show that [1,2,3] becomes [3,2,1]). For the two-equal case, the function identifies which two positions refer to the same feature and applies a deterministic permutation based on comparisons between the distances so that the repeated-feature vertices and their connecting edges are ordered consistently. The algorithm uses small, fixed index permutations (precomputed in code as ireorder and dreorder) to reorder featIndices and dists respectively.
    
    Side effects and mutability:
        The function does not modify the caller's lists in place; it constructs and returns new lists corresponding to the canonical ordering. No external state is modified.
    
    Failure modes and exceptions:
        ValueError is raised if featIndices or dists do not have exactly three elements, since the function is defined only for triangles. The function assumes the lists contain comparable numeric distance values and feature identifiers; passing elements that do not support comparison operations used in tie-breaking may raise TypeError or produce undefined ordering behavior.
    
    Returns:
        tuple(list, list): A pair (featIndices, dists) where both are lists of length 3. featIndices is the possibly-permuted list of feature indices corresponding to the canonical vertex ordering and dists is the correspondingly-permuted list of distances. This returned representation is intended for use in Pharm2D fingerprint generation, indexing, and equality checks so that equivalent triangles yield identical ordered tuples.
    """
    from rdkit.Chem.Pharm2D.Utils import OrderTriangle
    return OrderTriangle(featIndices, dists)


################################################################################
# Source: rdkit.Chem.Pharm2D.Utils.ScaffoldPasses
# File: rdkit/Chem/Pharm2D/Utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def rdkit_Chem_Pharm2D_Utils_ScaffoldPasses(combo: list, bins: list = None):
    """rdkit.Chem.Pharm2D.Utils.ScaffoldPasses checks whether a pharmacophore scaffold encoding (combo) is geometrically consistent by verifying that every contributing triangle of binned distances satisfies the triangle inequality. This function is used in the RDKit Pharm2D fingerprinting and pharmacophore-scaffold validation workflow to reject scaffold encodings that cannot correspond to a valid set of Euclidean distances for the pharmacophore points.
    
    Checks the scaffold encoded in combo (a list of binned distance indices) by retrieving all triangle index triples for the pharmacophore point count (via nDistPointDict and GetTriangles) and testing each triangle's three distances (looked up in bins) with BinsTriangleInequality. If any triangle fails the inequality, the scaffold is treated as invalid and the function returns False; if all triangles pass, it returns True.
    
    Args:
        combo (list): A list encoding the scaffold as binned distance indices. Each element is an integer index that selects a representative distance from the bins sequence. The length of combo determines the pharmacophore point count used to query nDistPointDict and GetTriangles to produce triangle index triples. In the Pharm2D domain, combo represents the contributing inter-point distance bins for a candidate scaffold; an invalid or inconsistent combo will cause the function to return False.
        bins (list or None): A sequence that maps bin indices (the integers stored in combo) to numeric distance values (typically floats) used to evaluate triangle inequalities. This parameter defaults to None; however, passing None is not useful in normal operation because the implementation indexes into bins using values from combo (bins[combo[x]]), which will raise a TypeError or IndexError if bins is None or if combo contains out-of-range indices. The caller must provide a bins list whose length and contents match the indices in combo so that numeric distances can be looked up for each triangle.
    
    Returns:
        bool: True if every triangle formed by the pharmacophore points (as determined from len(combo) via nDistPointDict and GetTriangles) has distances that satisfy the triangle inequality when mapped through bins; False if any triangle fails the inequality. There are no side effects beyond reading the provided structures. Possible failure modes include TypeError/IndexError if bins is None or does not contain entries for indices present in combo, and KeyError if nDistPointDict does not contain an entry for len(combo).
    """
    from rdkit.Chem.Pharm2D.Utils import ScaffoldPasses
    return ScaffoldPasses(combo, bins)


################################################################################
# Source: rdkit.Chem.Pharm2D.Utils.UniquifyCombinations
# File: rdkit/Chem/Pharm2D/Utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def rdkit_Chem_Pharm2D_Utils_UniquifyCombinations(combos: list):
    """UniquifyCombinations in rdkit.Chem.Pharm2D.Utils deduplicates a list of combinations by treating each inner combination as an unordered set of elements. This utility is useful in the RDKit Pharm2D context (cheminformatics/pharmacophore 2D utilities) when generating or comparing sets of pharmacophore point combinations or other element combinations where element order does not matter: it returns one representative tuple for each unique collection of elements regardless of their order in the input.
    
    Args:
        combos (list): A list whose elements are sequences representing combinations (for example, lists of atom or pharmacophore point identifiers). Each inner sequence is expected to support slicing (combo[:]) and an in-place sort method (k.sort()) or otherwise be convertible to a mutable, sortable sequence prior to calling this function. The function treats each inner sequence as an unordered collection of values and uses a sorted form of the inner sequence as the uniqueness key. In practice within RDKit Pharm2D code, combos will typically be a list of lists of integers or strings identifying features; if inner sequences are not sortable or contain incomparable elements, a TypeError or AttributeError will be raised.
    
    Returns:
        list: A list of tuples. Each tuple is a representative of a unique unordered combination found in the input list. For each group of input combinations that contain the same elements in any order, exactly one tuple appears in the returned list. The returned tuples are constructed from the original combinations (tuple(combo)), not from the sorted key, so the tuple returned for a given unique key is the last encountered input combination that mapped to that key. The order of tuples in the returned list corresponds to the order in which each unique sorted key was first encountered while iterating through the input list (i.e., insertion order of keys); subsequent duplicates overwrite the stored representative value but do not change its position in the output list.
    
    Behavior, side effects, defaults, and failure modes:
        - The function does not modify the outer list object passed as combos, but it relies on slicing and in-place sorting of each inner sequence (combo[:] followed by k.sort()). If inner sequences are immutable tuples, or lack a sort method, the code will raise an AttributeError when k.sort() is attempted; if elements of an inner sequence are not mutually comparable, k.sort() will raise a TypeError.
        - Uniqueness is determined by the sorted contents of each inner sequence. Combinations that differ only by element order are considered identical.
        - Among duplicates (combinations that sort to the same key), the stored representative tuple equals the last such input combination seen; however, the representative's position in the output list is determined by when that key was first inserted.
        - Time complexity is dominated by sorting each inner sequence: roughly O(N * M log M) where N is the number of combinations and M is the average length of a combination. Memory usage is O(U * M) where U is the number of unique sorted keys stored.
        - Typical use in RDKit Pharm2D workflows: remove redundant permutations of the same pharmacophore point sets before fingerprint generation or comparison to reduce duplicate work and ensure uniqueness of combination-based features.
    """
    from rdkit.Chem.Pharm2D.Utils import UniquifyCombinations
    return UniquifyCombinations(combos)


################################################################################
# Source: rdkit.Chem.Pharm3D.EmbedLib.AddExcludedVolumes
# File: rdkit/Chem/Pharm3D/EmbedLib.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def rdkit_Chem_Pharm3D_EmbedLib_AddExcludedVolumes(
    bm: numpy.ndarray,
    excludedVolumes: list,
    smoothIt: bool = True
):
    """Adds a set of excluded volumes to a bounds matrix used by RDKit's Pharm3D embedding routines.
    
    This function is used in pharmacophore modeling and 3D embedding workflows (RDKit Pharm3D) to extend an existing bounds matrix that encodes pairwise distance bounds between features/atoms so that one or more "excluded volume" objects (regions where a ligand atom must not occupy) are represented as additional rows/columns. The returned matrix is a new numpy.ndarray with dtype numpy.float64 and size increased by the number of excluded volumes; the original bm matrix is not modified. This function also assigns the computed matrix index to each ExcludedVolume object by setting its index attribute, and optionally applies triangle smoothing to the resulting bounds matrix to improve consistency for embedding algorithms.
    
    Args:
        bm (numpy.ndarray): The original bounds matrix used by Pharm3D embedding. This is a square 2D numpy array (shape (N, N)) of type numpy.float64 representing distance bounds between existing features/atoms. The function treats bm as read-only and copies its values into the top-left corner of the returned matrix; bm itself is not altered.
        excludedVolumes (list): A list of ExcludedVolume objects to append to the bounds matrix. Each ExcludedVolume must provide an exclusionDist attribute (a numeric value used to initialize distances from the excluded volume to existing atoms) and a featInfo iterable describing the excluded volume's defining features. Each featInfo item is expected to be a tuple (indices, minV, maxV) where indices is an iterable of integer indices referring to rows/columns in the original bm, minV is the numeric value to set in the new row at positions given by indices, and maxV is the numeric value to set in the new column at those positions. For each excluded volume in the list, this function sets the excluded volume object's index attribute to the new matrix index assigned to that volume.
        smoothIt (bool): If True (the default), apply DG.DoTriangleSmoothing to the resulting bounds matrix before returning. Triangle smoothing is used to enforce or improve consistency (for example, triangle inequality-style constraints) among the matrix entries to make the matrix better suited for downstream embedding algorithms. If False, the smoothing step is skipped and the raw assembled matrix is returned.
    
    Returns:
        numpy.ndarray: A new square bounds matrix of type numpy.float64 with shape (N + M, N + M) where N is the original bm.shape[0] and M is len(excludedVolumes). The original bm is left unchanged. The returned matrix contains:
        - The original bm copied into the top-left N x N block.
        - For each appended excluded volume at new index k = N + i:
          - The row k, columns [0:k) are initialized to vol.exclusionDist to represent the exclusion-based lower bounds from the excluded volume to existing features.
          - The column k, rows [0:k) are initialized to 1000.0 to represent a large upper bound for those entries.
          - For each (indices, minV, maxV) entry in vol.featInfo, asymmetric entries are set so that res[k, index] = minV and res[index, k] = maxV for each index in indices.
          - For interactions between excluded volumes, entries for later-added volumes are set so that res[k, j] = 0.0 and res[j, k] = 1000.0 for j > k to reflect initialization policy used by the embedding routines.
        If smoothIt is True, the returned matrix may be modified by DG.DoTriangleSmoothing to enforce consistency.
    
    Side effects and failure modes:
        - The function sets the index attribute on each ExcludedVolume object in excludedVolumes to the corresponding new matrix index (bm.shape[0] + position in excludedVolumes). Callers should be aware that excludedVolumes objects are mutated.
        - The function logs an error using the module logger and raises IndexError if any index referenced in an excluded volume's featInfo is outside the valid range for the assembled matrix. The logged message will include the bad indices and the shape of the matrix being written to.
        - The function always returns a new numpy.ndarray and does not modify the input bm; any downstream code must use the returned matrix for further embedding steps.
    """
    from rdkit.Chem.Pharm3D.EmbedLib import AddExcludedVolumes
    return AddExcludedVolumes(bm, excludedVolumes, smoothIt)


################################################################################
# Source: rdkit.Chem.Pharm3D.EmbedLib.CombiEnum
# File: rdkit/Chem/Pharm3D/EmbedLib.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def rdkit_Chem_Pharm3D_EmbedLib_CombiEnum(sequence: tuple):
    """rdkit.Chem.Pharm3D.EmbedLib.CombiEnum generates all combinations that pick one element from each subsequence in a tuple of subsequences. This generator is a small utility used in RDKit's Pharm3D embedding and pharmacophore-related code to enumerate alternative assignments (for example, alternative placements or feature choices) without materializing the full Cartesian product in memory; it yields each combination as a list in the same order as the input subsequences.
    
    Args:
        sequence (tuple): A tuple of subsequences (each subsequence is expected to be an iterable such as a tuple or list) that provide candidate values for each position in the combination. The role of this parameter in the Pharm3D/embedding domain is to represent, for each pharmacophore position or decision point, the set of alternatives to be enumerated. The function uses len(), indexing and slicing on this tuple. If the top-level tuple is empty (len(sequence) == 0) the generator yields a single empty list (one valid combination of zero choices). If any subsequence is empty, the generator will produce no combinations (i.e., it will be empty), because there is no valid choice for that position.
    
    Returns:
        generator: A generator that yields lists. Each yielded list represents one combination formed by selecting exactly one element from each subsequence in the input tuple, in the same order as the subsequences. For a non-empty input tuple of length N, each yielded list has length N. The generator evaluates lazily (combinations are produced on demand), does not modify the input tuple or its subsequences, and does not precompute all combinations (so it is memory-efficient for large enumerations). Typical failure modes include TypeError or iteration errors if the provided tuple or its elements are not indexable/iterable as expected; excessive recursion depth is possible for very large tuples because the implementation is recursive.
    """
    from rdkit.Chem.Pharm3D.EmbedLib import CombiEnum
    return CombiEnum(sequence)


################################################################################
# Source: rdkit.Chem.Pharm3D.EmbedLib.DownsampleBoundsMatrix
# File: rdkit/Chem/Pharm3D/EmbedLib.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def rdkit_Chem_Pharm3D_EmbedLib_DownsampleBoundsMatrix(
    bm: numpy.ndarray,
    indices: tuple,
    maxThresh: float = 4.0
):
    """DownsampleBoundsMatrix removes rows and corresponding columns from a bounds matrix that represent points (typically atom indices or pharmacophore feature indices) that are all farther than a given threshold from a supplied set of reference indices. This function is used in the RDKit Pharm3D embedding pipeline to reduce the size of the pairwise bounds matrix before an O(N^3) bounds-smoothing step, improving performance by excluding atoms/features unlikely to be part of the pharmacophore of interest.
    
    This function expects bm to be a 2D square NumPy array containing pairwise distance bounds (for example, upper bounds on inter-atomic distances used in pharmacophore embedding). The indices argument supplies the one or more "core" atomic/feature indices to which distances are compared: any point with at least one bound strictly less than maxThresh to any of these indices is retained; all specified indices are always retained. The threshold comparison uses a strict less-than (<) comparison. The input matrix is not modified in-place: when no rows/columns are removed a copy of bm is returned, and when rows/columns are removed a new reduced NumPy array is returned. The function preserves the dtype of the input bm (it will not implicitly upcast to float64).
    
    Args:
        bm (numpy.ndarray): A 2-D square NumPy array of pairwise bounds between points (shape (N, N)). In the Pharm3D/EmbedLib context this is typically a bounds matrix of inter-atomic or feature distances used for pharmacophore embedding and smoothing. The function assumes bm supports numeric comparisons with maxThresh; supplying a non-numeric or mismatched-shaped array may raise numpy TypeError or IndexError.
        indices (tuple): A tuple of integer indices identifying core points (atomic or pharmacophore feature indices) that should be kept and against which other points are tested for proximity. Duplicates in this tuple are ignored. Each index must be a valid row/column index for bm (0 <= index < N); otherwise numpy will raise IndexError.
        maxThresh (float): A numeric threshold (default 4.0) in the same units as the values in bm. Any point whose bound to any index in indices is strictly less than maxThresh (bm[idx, j] < maxThresh for some idx in indices) will be kept. The comparison is strict; equal-to values are not considered "close". Use a larger maxThresh to keep more points, and a smaller value to be more selective.
    
    Returns:
        numpy.ndarray: A NumPy array representing the downsampled bounds matrix. If indices is empty, returns an empty 2-D array with shape (0, 0) and the same dtype as bm. If every point is kept (no rows/columns removed), returns a copy of bm (same shape and dtype). Otherwise returns a new square NumPy array containing only the rows and columns corresponding to the kept indices; the returned matrix preserves the numeric dtype of the input bm and the original ordering of the kept indices (sorted by index value). Possible failure modes include IndexError for out-of-range indices, TypeError for non-comparable dtypes when performing bm < maxThresh, and unexpected behavior if bm is not square.
    """
    from rdkit.Chem.Pharm3D.EmbedLib import DownsampleBoundsMatrix
    return DownsampleBoundsMatrix(bm, indices, maxThresh)


################################################################################
# Source: rdkit.Chem.Pharm3D.EmbedLib.ReplaceGroup
# File: rdkit/Chem/Pharm3D/EmbedLib.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def rdkit_Chem_Pharm3D_EmbedLib_ReplaceGroup(
    match: list,
    bounds: numpy.ndarray,
    slop: float = 0.01,
    useDirs: bool = False,
    dirLength: float = 2.0
):
    """ReplaceGroup adds a new point representing the geometric center of a multi-point pharmacophore feature to a copy of an existing bounds matrix and returns the augmented bounds matrix and the index of the inserted center point. This function is used in the Pharm3D embedding utilities of RDKit to convert a feature defined by multiple equivalent points (for example, a ring or symmetric pharmacophore feature) into a single representative point for distance-based embedding and matching. The routine assumes the feature points form a regular polygon and are listed in neighbor order (point i adjacent to i+1 and the last adjacent to the first); it computes an approximate center-to-point and point-to-center distance range from the supplied pairwise bounds, expands those bounds by a fractional slop, applies a geometric scale factor for the polygon-to-center conversion, and writes the resulting distances into a newly allocated bounds matrix. If useDirs is True, an additional “direction” point is appended after the center point and its pairwise bounds with the center and original points are set using dirLength and slop.
    
    Args:
        match (list): A list of integer indices identifying the points (rows/columns) in the input bounds matrix that belong to the multi-point feature. These indices must reference valid rows/columns of bounds. The order of indices is significant: the function assumes they are listed in circular neighbor order (0 adjacent to 1, ..., last adjacent to 0). In the RDKit Pharm3D context, match selects the member feature points whose collective center is to replace them for embedding and matching tasks.
        bounds (numpy.ndarray): A square 2-D NumPy array containing pairwise bounds/distances used by the Pharm3D embedding routines. The function reads bounds[idx1, idx0] and bounds[idx0, idx1] for index pairs drawn from match to determine minimum and maximum edge distances for the polygon formed by the feature points. The array is not modified in place; a new, larger array is allocated and returned. The dtype is treated as numeric (float-compatible) and the shape is expected to be (N, N) where N is the number of existing points.
        slop (float, optional): Fractional tolerance applied to the computed minimum and maximum distances for the replacement center point. Default is 0.01 (1%). The function first multiplies the found maxVal by (1 + slop) and minVal by (1 - slop) before converting edge distances to center distances via a geometric scale factor. Note: slop is used fractionally for expanding/shrinking the min/max bounds, but is added/subtracted as an absolute amount when setting the feature-to-direction distances if useDirs is True.
        useDirs (bool, optional): If True, an additional direction point is appended after the new center point in the returned bounds matrix. This is used to represent a vectorial or directional constraint for the feature in Pharm3D embeddings. Default is False. When True, the returned bounds matrix is expanded by two rows/columns (one for the center and one for the direction point) relative to the original bounds; when False, the matrix is expanded by one row/column (for the center only).
        dirLength (float, optional): The nominal distance (float) used to set the bound between the center point and the appended direction point when useDirs is True. Default is 2.0. The code sets the center-to-direction bound to (dirLength + slop) and the direction-to-center bound to (dirLength - slop), and uses those values combined with the computed min/max center-to-point distances to set diagonal distances between the original points and the direction point via Pythagorean combinations.
    
    Returns:
        tuple: A 2-tuple (bm, replaceIdx) where:
            bm (numpy.ndarray): A newly allocated square NumPy array of dtype numpy.float64 and shape (N + 1 + enhanceSize, N + 1 + enhanceSize) containing the original bounds in the upper-left block and the newly written bounds entries for the inserted center point (and optional direction point). enhanceSize is 1 if useDirs is True (an extra direction point) and 0 otherwise. The function initializes certain placeholder entries to large values (1000.0) to indicate unconstrained/unset distances consistent with the Pharm3D embedding code.
            replaceIdx (int): The integer index assigned to the inserted center point in bm. This equals the original bounds.shape[0] (the first new row/column index) so callers can reference the inserted point for subsequent operations in the Pharm3D pipeline.
    
    Behavior, side effects, defaults, and failure modes:
        - The function does not modify the input bounds array; it constructs and returns a new array bm that contains the original bounds and the appended rows/columns for the replacement point and optional direction point.
        - The routine identifies minVal and maxVal from pairs bounds[idx1, idx0] and bounds[idx0, idx1] for consecutive index pairs around the polygon defined by match. It then expands those values by the fractional slop (maxVal *= (1 + slop); minVal *= (1 - slop)) and converts edge distances to center distances by multiplying by scaleFact = 1.0 / (2.0 * sin(pi / nPts)), which is valid under the regular-polygon geometric assumption. This scale factor is the geometric mapping used by Pharm3D to approximate center-to-vertex distances from edge lengths.
        - If useDirs is True, the function appends a direction point immediately after the center point and sets the center-direction bounds to dirLength +/- slop (absolute addition/subtraction). It then sets distances between the original feature points and the direction point using sqrt((center-direction)^2 + (center-point)^2) combinations to maintain compatible triangle inequalities for embedding.
        - Several placeholder entries are set to a large value (1000.0) to represent effectively unconstrained distances for downstream embedding logic; these are assigned to bm[:replaceIdx, replaceIdx] and, when useDirs is True, to bm[:replaceIdx + 1, replaceIdx + 1].
        - Default values: slop defaults to 0.01 (1% fractional tolerance), useDirs defaults to False (no direction point), and dirLength defaults to 2.0.
        - Failure modes and errors:
            - The function assumes match lists a multi-point feature; the geometric derivation assumes at least three points (nPts >= 3) to represent a regular polygon. Using an empty match or nPts == 1 will cause a division or domain error when computing sin(pi / nPts); callers should provide a valid list of feature indices. Although the code can mathematically run for nPts == 2, the regular-polygon assumption documented for pharmacophore features typically requires nPts >= 3.
            - All indices in match must be valid integer indices within the bounds matrix dimensions; an out-of-range index will raise an IndexError from NumPy.
            - The bounds argument must be a numeric, square NumPy ndarray; passing an object of incompatible shape or type may raise IndexError, TypeError, or value errors when the function attempts numeric operations.
            - The function performs numeric operations that could propagate NaNs or infinities if present in bounds; callers should ensure bounds contains finite numeric values for meaningful results.
    
    Examples (behavioral summary):
        - Given a 3x3 bounds matrix and match = [0, 1, 2], ReplaceGroup returns a 4x4 matrix and replaceIdx == 3. The original matrix remains unchanged; the returned matrix contains new center-to-point and point-to-center distance bounds computed as described above.
    """
    from rdkit.Chem.Pharm3D.EmbedLib import ReplaceGroup
    return ReplaceGroup(match, bounds, slop, useDirs, dirLength)


################################################################################
# Source: rdkit.Chem.Pharm3D.EmbedLib.isNaN
# File: rdkit/Chem/Pharm3D/EmbedLib.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def rdkit_Chem_Pharm3D_EmbedLib_isNaN(v: float):
    """Detect whether a floating-point value is a NaN (not-a-number) produced by the RDKit C++ layer, using platform-specific heuristics.
    
    This function exists to provide an OS-independent interface for detecting NaNs that originate on the C++ side of RDKit (for example, from numerical routines in Chem.Pharm3D embedding code). In RDKit workflows NaNs can indicate failed or invalid numerical results (such as failed 3D coordinate or energy computations) and must be detected so downstream Python code can handle or discard those values. Because Python itself raises a ZeroDivisionError for 1/0 and does not produce the same raw IEEE NaN objects the C++ layer can return, this function applies simple, historically observed checks that differ on Windows and non-Windows platforms to recognize C++-sourced NaNs.
    
    Behavior:
    - On Windows (sys.platform == 'win32') the function returns True when v != v, exploiting the IEEE property that NaN is not equal to itself; otherwise it falls through to return False.
    - On non-Windows platforms the function returns True when both comparisons v == 0 and v == 1 evaluate to True; this is a platform-specific heuristic present in the original implementation to detect certain C++->Python converted NaN values. If neither platform-specific test matches, the function returns False.
    - The function performs no mutation and has no side effects other than reading sys.platform and evaluating the given value using Python comparison operators. It does not attempt to construct NaNs in Python; it is intended to inspect values produced by the C++ side of RDKit.
    - If v is not a Python float, Python's normal comparison semantics apply; comparisons may succeed, return False, or raise exceptions (e.g., TypeError) which will propagate to the caller.
    
    Args:
        v (float): A floating-point numeric value to test for NaN. In the RDKit/Pharm3D context this is intended to be a value returned from C++ numerical routines (for example, embedding or force-field calculations) where an IEEE NaN may indicate an invalid or failed computation. The value is inspected with simple comparison-based heuristics that differ by operating system to determine whether it should be treated as NaN.
    
    Returns:
        bool: True if the value v matches the platform-specific NaN detection heuristics (treated as NaN), False otherwise. On Windows this corresponds to v != v being True; on non-Windows this corresponds to both v == 0 and v == 1 being True. If comparisons against v raise an exception (for example because v is a non-comparable type), that exception is propagated and no boolean is returned.
    """
    from rdkit.Chem.Pharm3D.EmbedLib import isNaN
    return isNaN(v)


################################################################################
# Source: rdkit.Chem.Scaffolds.MurckoScaffold.MurckoScaffoldSmilesFromSmiles
# File: rdkit/Chem/Scaffolds/MurckoScaffold.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def rdkit_Chem_Scaffolds_MurckoScaffold_MurckoScaffoldSmilesFromSmiles(
    smiles: str,
    includeChirality: bool = False
):
    """rdkit.Chem.Scaffolds.MurckoScaffold.MurckoScaffoldSmilesFromSmiles: Return the Bemis-Murcko scaffold as a SMILES string derived from an input SMILES.
    
    This function is part of the RDKit cheminformatics toolkit and is used to extract the Murcko (Bemis-Murcko) scaffold — the core ring systems and linker framework of a molecule without side-chain substituents — expressed as a SMILES string. It is commonly used in scaffold-based analysis workflows such as scaffold clustering, scaffold-based train/test splitting for machine learning on molecular datasets, and medicinal chemistry analyses where the core topology of molecules is compared. The implementation delegates to MurckoScaffoldSmiles(smiles=smiles, includeChirality=includeChirality), so behavior, performance characteristics, and error propagation follow that underlying routine.
    
    Args:
        smiles (str): The input molecule encoded as a SMILES string. This string is parsed by RDKit to build a molecule object from which the Murcko scaffold is extracted. The caller is responsible for providing a valid SMILES representation; invalid or unparsable SMILES will cause the underlying RDKit parsing/extraction routine to fail and that failure will propagate to the caller.
        includeChirality (bool): If True, stereochemical information (chirality) present in the input SMILES will be preserved in the output scaffold SMILES when possible. If False (the default), stereochemical specifications will be omitted from the returned scaffold SMILES. This flag controls only whether stereochemical annotations are retained in the output; it does not alter the scaffold topology extraction logic.
    
    Returns:
        str: A SMILES string representing the Murcko scaffold (the core rings and linkers) of the input molecule. The returned string is produced by the underlying MurckoScaffoldSmiles function and contains the molecular topology of the scaffold; if includeChirality was set to True and the underlying routine supports it, stereochemical descriptors will be included in the returned SMILES. Any errors encountered while parsing the input SMILES or extracting the scaffold (for example, invalid SMILES) are propagated from the underlying RDKit/MurckoScaffoldSmiles implementation rather than handled inside this function. No other side effects occur (the function does not mutate global state or input data).
    """
    from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmilesFromSmiles
    return MurckoScaffoldSmilesFromSmiles(smiles, includeChirality)


################################################################################
# Source: rdkit.Chem.TorsionFingerprints.CalculateTFD
# File: rdkit/Chem/TorsionFingerprints.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def rdkit_Chem_TorsionFingerprints_CalculateTFD(
    torsions1: list,
    torsions2: list,
    weights: list = None
):
    """Calculate the torsion deviation fingerprint (TFD) between two sets of torsion angles for conformer comparison.
    
    This function is used in the RDKit cheminformatics context to quantify how different two molecular conformations are in terms of their torsion (dihedral) angles. It computes, for each corresponding torsion, the minimal circular angular difference (in degrees) between any representative angles for that torsion in the two conformers, normalizes that difference by a per-torsion normalization factor, and returns an average (or weighted average) of these normalized deviations. This metric can be used as a descriptor/fingerprint for comparing 3D molecular conformations in similarity searches, clustering, or as an input feature for machine-learning models.
    
    Args:
        torsions1 (list): A list describing torsions for conformation 1. Each element must be an indexable 2-element sequence where the first element is an iterable of torsion angles (numeric, in degrees) representing equivalent angle values for that torsion (e.g., due to symmetry or multiple representations), and the second element is a numeric normalization value (divisor) used to scale the raw angular deviation for that torsion. The function iterates over tors1[0] and uses tors1[1] as the divisor; if tors1[1] is zero a ZeroDivisionError will be raised. This structure and role are required because the algorithm computes minimal circular differences among the angle lists and divides by the provided normalization factor to produce a normalized per-torsion deviation.
        torsions2 (list): A list describing torsions for conformation 2 with the same structure and semantics as torsions1. torsions1 and torsions2 must have identical lengths and corresponding elements represent the same torsional degrees of freedom in the two conformations. If the two lists have different lengths a ValueError is raised. Elements must contain numeric angle values in degrees; non-numeric or improperly structured elements may raise TypeError or IndexError.
        weights (list = None): Optional list of numeric weights, one per torsion, used to compute a weighted average of the normalized deviations. If provided, weights must have the same length as torsions1 and torsions2, otherwise a ValueError is raised. If weights is None (the default), all torsions are equally weighted (the average is taken over the number of torsions). If the sum of weights is zero, the function will avoid division by zero and return the un-divided sum of weighted deviations (i.e., sum(deviations) without normalization by sum_weights), which reflects an implementation detail to be aware of.
    
    Behavior and failure modes:
        The function treats angles as circular values in degrees: for any pair of angles t1 and t2 it computes the absolute difference abs(t1 - t2) and uses min(diff, 360.0 - diff) so that directional sign is ignored and wrap-around at 360 degrees is handled. For each torsion pair it finds the minimal such circular difference over all combinations of representative angles from torsions1 and torsions2. That minimal difference is then divided by the torsion-specific normalization value (torsion[1]) to produce a normalized deviation for that torsion. Deviations are multiplied by corresponding weights if weights are provided. The final TFD value is the sum of (possibly weighted) deviations divided by the sum of weights if that sum is non-zero; otherwise the raw sum of deviations is returned.
        The function raises ValueError when torsions1 and torsions2 have different lengths or when weights is provided with a length different from the torsion lists. A ZeroDivisionError will occur if any torsion element uses a normalization value of zero. Improper element structure (e.g., missing two elements per torsion or non-iterable angle lists) may raise IndexError or TypeError. The function has no external side effects and does not modify its inputs.
    
    Returns:
        float: The torsion deviation fingerprint (TFD) value as a floating-point number. This value represents the (weighted) average of per-torsion normalized minimal circular angular deviations between the two input conformations and is intended for use as a quantitative descriptor in RDKit workflows (e.g., 3D molecular operations, fingerprint/descriptor computation, similarity comparison, or machine-learning features).
    """
    from rdkit.Chem.TorsionFingerprints import CalculateTFD
    return CalculateTFD(torsions1, torsions2, weights)


################################################################################
# Source: rdkit.Chem.UnitTestPandasTools.getStreamIO
# File: rdkit/Chem/UnitTestPandasTools.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def rdkit_Chem_UnitTestPandasTools_getStreamIO(sdfString: str):
    """rdkit.Chem.UnitTestPandasTools.getStreamIO returns an in-memory binary stream (BytesIO) containing the UTF-8 encoded bytes of an SDF (Structure-Data File) string. This helper is used in RDKit unit tests and utility code (for example in UnitTestPandasTools) to provide a file-like binary object that RDKit readers and other code that expect a binary stream can consume without writing to disk.
    
    Args:
        sdfString (str): A text representation of an SDF file or SDF fragment. In the cheminformatics domain used by RDKit, an SDF contains one or more molecular records and associated data fields; this parameter supplies that content as a Python string. If sdfString is None, the function produces an empty binary stream. The function encodes this string to UTF-8 bytes for binary I/O; therefore the caller should provide text that is valid UTF-8 or can be losslessly represented in UTF-8.
    
    Returns:
        BytesIO: An instance of io.BytesIO containing the UTF-8 encoded bytes of sdfString (or empty if sdfString is None). The returned stream is positioned at the start (ready for reading) and can be passed directly to RDKit readers or other APIs that accept a binary file-like object. No file is created on disk; the side effect is only allocating the in-memory buffer. If a non-string, non-None object is passed as sdfString, calling encode may raise an AttributeError or TypeError—callers should ensure they pass a str or None to avoid such errors.
    """
    from rdkit.Chem.UnitTestPandasTools import getStreamIO
    return getStreamIO(sdfString)


################################################################################
# Source: rdkit.Chem.UnitTestSurf.readRegressionData
# File: rdkit/Chem/UnitTestSurf.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def rdkit_Chem_UnitTestSurf_readRegressionData(filename: str, col: int):
    """Return entries from a regression dataset file bundled with RDKit and used by RDKit unit tests.
    
    This generator function opens the file named by filename located in the repository source tree under RDConfig.RDCodeDir/Chem/test_data, reads it line by line, and for each non-comment line yields a _TestData tuple containing the 1-based line number, the SMILES string, the parsed RDKit molecule, and the expected numeric value parsed from column col. It is intended for use in RDKit's unit/regression testing workflows to feed expected values (floats) and molecules (Chem.Mol) into test code that compares computed properties against baseline values stored in the test_data files.
    
    Args:
        filename (str): The base filename of the CSV-style regression data file located in RDKit's test data directory. The function resolves this relative to RDConfig.RDCodeDir by joining RDConfig.RDCodeDir, 'Chem', 'test_data', and filename, then opens that file for reading. This parameter must be the exact filename present in that directory (for example 'my_regression.csv'); providing an incorrect filename will raise a FileNotFoundError when the generator is first iterated.
        col (int): Zero-based column index into the comma-separated fields on each non-comment line from which the expected numeric value will be read. The function uses split(',') on each line and accesses splitL[col] to obtain the value; therefore col must be an integer indexing an existing column in every data line. If col is out of range for a given line an IndexError will be raised during iteration.
    
    Behavior, side effects, and failure modes:
        The file is opened using os.path.join(RDConfig.RDCodeDir, 'Chem', 'test_data', filename) and read in text mode. Lines beginning with the character '#' are treated as comments and skipped. Each non-comment line is split on the comma character; the first field splitL[0] is treated as a SMILES string and passed to Chem.MolFromSmiles to produce an RDKit molecule (rdkit.Chem.rdchem.Mol). If Chem.MolFromSmiles returns None (indicating a malformed or unparseable SMILES), the function raises AssertionError with a message containing the line number and the offending SMILES. The value taken from splitL[col] is converted to float; if conversion fails a ValueError will be raised during iteration. Because the function yields results, it performs lazy I/O: errors related to file access, parsing, indexing, or conversion occur when the generator is iterated, not at the time the function is called. Line numbering is 1-based (the first line returned has lineNum == 1 after skipping comment lines is not considered for numbering). The function does not modify the input file.
    
    Returns:
        generator: A generator that yields _TestData objects for each non-comment line in the specified file. Each yielded _TestData contains four fields in order: line number (int) representing the 1-based line index in the file, smiles (str) the SMILES string parsed from the first comma-separated field, mol (rdkit.Chem.rdchem.Mol) the RDKit molecule produced by Chem.MolFromSmiles(smi), and expected (float) the numeric value parsed from the column specified by col. The generator is intended to be iterated by test code that compares computed molecular descriptors or regression outputs against the expected float values stored in the test data files.
    """
    from rdkit.Chem.UnitTestSurf import readRegressionData
    return readRegressionData(filename, col)


################################################################################
# Source: rdkit.Chem.inchi.InchiToInchiKey
# File: rdkit/Chem/inchi.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def rdkit_Chem_inchi_InchiToInchiKey(inchi: str):
    """rdkit.Chem.inchi.InchiToInchiKey: Return the InChIKey for a given InChI string.
    
    This function is a thin RDKit Python wrapper around the underlying rdinchi binding that converts a text InChI (IUPAC International Chemical Identifier) into its corresponding InChIKey, a compact hashed identifier commonly used in cheminformatics for indexing, deduplication, database keys, and fast exact matching of chemical structures. The function performs no modification of the input string; it delegates conversion to the rdinchi.InchiToInchiKey implementation and returns its result if the conversion is successful.
    
    Args:
        inchi (str): The input InChI string to be converted. This should be a textual IUPAC InChI representing a chemical structure (typically starts with the prefix "InChI=" for standard InChI values). The parameter is passed verbatim to the underlying rdinchi binding; no normalization or validation is performed by this wrapper beyond what rdinchi implements.
    
    Returns:
        str or None: The InChIKey string produced for the provided InChI when conversion succeeds. If the underlying rdinchi.InchiToInchiKey call returns a falsy value (indicating conversion failure, an invalid or unsupported InChI, or other internal error), this function returns None. Note that this wrapper does not catch exceptions raised by the underlying binding: if rdinchi.InchiToInchiKey raises an exception, that exception will propagate to the caller.
    
    Behavior and failure modes:
        This function is deterministic for a given valid InChI and has no side effects beyond calling the rdinchi binding. It returns a canonical hashed identifier on success and None when the conversion cannot be performed. Common reasons for returning None include malformed InChI strings, unsupported InChI versions/features, or internal conversion errors in the rdinchi library. Callers that require strict failure diagnostics should handle exceptions from the underlying binding and validate input InChI strings before calling this function.
    """
    from rdkit.Chem.inchi import InchiToInchiKey
    return InchiToInchiKey(inchi)


################################################################################
# Source: rdkit.Chem.inchi.MolBlockToInchiAndAuxInfo
# File: rdkit/Chem/inchi.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def rdkit_Chem_inchi_MolBlockToInchiAndAuxInfo(
    molblock: str,
    options: str = "",
    logLevel: int = None,
    treatWarningAsError: bool = False
):
    """Returns the standard InChI string and the InChI auxInfo for a molecule supplied as an MDL molfile (mol block). This function wraps the rdinchi.MolBlockToInchi call from the RDKit InChI interface to produce a machine-readable chemical identifier (InChI) and the InChI auxInfo (auxiliary layered information used for atom ordering, stereochemistry layers, and other mapping details), which are commonly used in cheminformatics workflows for canonical representation, searching, and interoperability.
    
    Args:
        molblock (str): The input molecule encoded as an MDL molfile / mol block text. This string provides the atomic coordinates, element types, bonding connectivity and any stereochemical annotations that the InChI API uses to compute the standardized InChI identifier and auxInfo. In the RDKit context this is the typical serialized molecule representation produced by MolToMolBlock and similar routines.
        options (str): A string of options passed verbatim to the underlying InChI API (rdinchi.MolBlockToInchi). These options control how the InChI is generated (for example, layers or treatements supported by the InChI library). The default is an empty string, which requests standard InChI generation behavior from the InChI API.
        logLevel (int): An integer key selecting a logging function from the module-level mapping logLevelToLogFunctionLookup. If set to None (the default) logging to that mapped function is disabled. If a non-None value is provided and is not present in logLevelToLogFunctionLookup, the function raises ValueError. When provided and valid, the mapped logging function will be called with the InChI API message on successful (retcode == 0) conversions; regardless of this mapping, warnings (retcode == 1) and errors (retcode != 0 and != 1) are emitted via the module logger as described below.
        treatWarningAsError (bool): If True (default False), any non-zero return code from the InChI API (retcode != 0), including warnings (retcode == 1), causes the function to raise InchiReadWriteError. The raised exception encodes the produced InChI string, the auxInfo string, and the textual message returned by the InChI API so callers can inspect partial results and the error text. If False, the function does not raise on InChI warnings and instead returns the InChI and auxInfo while emitting log messages.
    
    Behavior and side effects:
        This function calls rdinchi.MolBlockToInchi(molblock, options) and unpacks its five-tuple response as (inchi, retcode, message, logs, aux).
        If logLevel is not None, the function first verifies that logLevel exists in the module mapping logLevelToLogFunctionLookup; if it does not, a ValueError is raised.
        If logLevel is provided and the InChI API returned retcode == 0 (success), the corresponding mapped logging function is invoked with the API message. Independently of the logLevel mapping, if retcode != 0 the module-level logger emits a warning when retcode == 1 and an error for all other non-zero retcodes.
        If treatWarningAsError is True and retcode != 0, the function raises InchiReadWriteError(inchi, aux, message). The exception contains the InChI and auxInfo that the InChI API produced along with the API message, allowing callers to examine partial outputs produced in error cases.
        The function may also propagate exceptions raised by the underlying rdinchi implementation or other runtime errors; callers should handle such exceptions as appropriate.
    
    Failure modes:
        ValueError is raised if logLevel is not None but is not a key in logLevelToLogFunctionLookup.
        InchiReadWriteError is raised when treatWarningAsError is True and the InChI API returns any non-zero retcode; the exception encodes the inchi, aux, and message.
        Other exceptions from the underlying rdinchi.MolBlockToInchi call or from logging infrastructure may be propagated to the caller.
    
    Returns:
        tuple(str, str): A two-tuple (inchi, aux) where inchi is the standard InChI string produced by the InChI API for the input molblock and aux is the corresponding InChI auxInfo string. Both values are returned even if the InChI API emitted warnings; if treatWarningAsError is True and a non-zero retcode was returned, an InchiReadWriteError is raised instead of returning the tuple.
    """
    from rdkit.Chem.inchi import MolBlockToInchiAndAuxInfo
    return MolBlockToInchiAndAuxInfo(molblock, options, logLevel, treatWarningAsError)


################################################################################
# Source: rdkit.Chem.inchi.MolFromInchi
# File: rdkit/Chem/inchi.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def rdkit_Chem_inchi_MolFromInchi(
    inchi: str,
    sanitize: bool = True,
    removeHs: bool = True,
    logLevel: int = None,
    treatWarningAsError: bool = False
):
    """Construct an RDKit molecule (rdkit.Chem.rdchem.Mol) from an InChI string using the RDKit InChI API.
    
    Args:
        inchi (str): The input InChI string (IUPAC International Chemical Identifier) that encodes the chemical structure to be parsed. This string is passed to the underlying RDKit InChI API (rdinchi.InchiToMol) and should be a valid InChI representation produced by an InChI generator or obtained from cheminformatics data sources. The function converts this textual identifier into an RDKit molecule that can be used for downstream cheminformatics tasks (e.g., descriptor calculation, fingerprinting, substructure searches).
        sanitize (bool): If True (default), enable RDKit sanitization of the constructed molecule after parsing. Sanitization performs valence checks, aromaticity/perception fixes, kekulization where appropriate, and other integrity checks used across RDKit to make the Mol suitable for typical operations (2D/3D operations, fingerprinting, descriptor calculation). If False, the returned Mol will skip these sanitization steps; this may be useful for performance or for inspecting partially-invalid structures, but many RDKit operations assume a sanitized molecule and may fail or produce misleading results.
        removeHs (bool): If True (default), request removal of explicit hydrogen atoms during conversion. This parameter is meaningful when sanitization is enabled because hydrogen handling and implicit/explicit hydrogen normalization are part of sanitization. When True, the resulting rdkit.Chem.rdchem.Mol will typically have hydrogens removed (represented implicitly) which is the common representation used for fingerprinting and many descriptor calculations. When False, explicit hydrogens present in the InChI may be kept in the returned Mol.
        logLevel (int): An optional log level index selecting which RDKit logging function to use for informational messages produced by the InChI API. If None (default), logging from the InChI API is disabled and no informational InChI messages are emitted via the selected log function. If provided, the integer must be a key present in the internal logLevelToLogFunctionLookup mapping; otherwise the function will raise ValueError("Unsupported log level: %d" % logLevel). When a valid logLevel is provided and the InChI API returns a success code (retcode == 0), the corresponding log function will be invoked with the InChI message.
        treatWarningAsError (bool): If True (default False), treat any non-success return code from the InChI API as an error condition and raise an InchiReadWriteError exception. The exception contains the resulting molecule (if any) and the textual message from the InChI API, allowing callers to inspect partial results and the underlying error message. If False (default), non-success return codes are reported via RDKit logging: retcode == 1 triggers logger.warning(message); any other non-zero retcode triggers logger.error(message), and the function returns the molecule object when available.
    
    Returns:
        rdkit.Chem.rdchem.Mol or None: On successful parsing the function returns an rdkit.Chem.rdchem.Mol instance representing the chemical structure encoded by the input InChI. The Mol will have been sanitized and had hydrogens removed or retained according to the sanitize and removeHs parameters. If the underlying rdinchi.InchiToMol call raises a ValueError (indicating a low-level parse/IO failure), the function logs the error and returns None. Note that when treatWarningAsError is True and the InChI API returns a non-zero return code, the function raises InchiReadWriteError instead of returning; in that case the exception carries the partially-constructed Mol (if produced) and the InChI API message.
    
    Behavior and failure modes:
        The function calls rdinchi.InchiToMol(inchi, sanitize, removeHs) to perform conversion; that call returns a tuple (mol, retcode, message, log). If rdinchi.InchiToMol raises ValueError, MolFromInchi logs the error and returns None. If logLevel is not None, the integer must exist in the internal logLevelToLogFunctionLookup mapping; otherwise the function raises ValueError describing the unsupported level. When a valid logLevel is supplied and the InChI API returns retcode == 0 (success), the associated log function is called with the API message. For retcode != 0, the function emits a warning (retcode == 1) or error (other non-zero codes) via RDKit logger. If treatWarningAsError is True and retcode != 0, the function raises InchiReadWriteError(mol, message) containing the produced Mol and the API message. Users should ensure input InChI validity and choose sanitization and hydrogen handling consistent with downstream RDKit operations (many descriptors and fingerprints expect sanitized molecules with implicit hydrogens).
    """
    from rdkit.Chem.inchi import MolFromInchi
    return MolFromInchi(inchi, sanitize, removeHs, logLevel, treatWarningAsError)


################################################################################
# Source: rdkit.Dbase.DbUtils.GetTypeStrings
# File: rdkit/Dbase/DbUtils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def rdkit_Dbase_DbUtils_GetTypeStrings(
    colHeadings: list,
    colTypes: list,
    keyCol: str = None
):
    """rdkit.Dbase.DbUtils.GetTypeStrings returns a list of SQL column type declaration strings constructed from parallel lists of column headings and type descriptors. This function is used by RDKit database utilities (for example, when generating CREATE TABLE column specifications for the RDKit PostgreSQL cartridge and other code that needs textual SQL column definitions). It maps Python type markers in colTypes to SQL type keywords: float -> "double precision", int -> "integer", and any other marker -> "varchar(n)" where n is supplied in the type descriptor.
    
    Args:
        colHeadings (list): A list of column name strings. Each entry is used verbatim as the column identifier prefixing the SQL type token for the corresponding column. The i-th element of colHeadings is paired with the i-th element of colTypes to produce a single SQL type declaration string (for example, "mol double precision" or "name varchar(50)").
        colTypes (list): A list of type descriptor tuples. Each element is expected to be an indexable sequence whose first element (typ[0]) is a Python type object used as a marker: the code tests for typ[0] == float and typ[0] == int. If typ[0] == float the resulting SQL type is "double precision"; if typ[0] == int the resulting SQL type is "integer"; otherwise the function formats a varchar with the length taken from typ[1] (i.e., "varchar(%d)" % typ[1]). The function iterates over range(len(colTypes)) and indexes colHeadings with the same index, so colTypes controls the number and order of output strings.
        keyCol (str): Optional. A column name (one of the strings in colHeadings) to mark as the table primary key. If keyCol is equal to colHeadings[i] for some i, the corresponding SQL declaration is postfixed with " not null primary key" (for example, "id integer not null primary key"). The default is None, which leaves all columns unmarked as primary key.
    
    Returns:
        list: A list of SQL type declaration strings, one per element in colTypes. Each string is formed by concatenating the column name from colHeadings[i] with the SQL type keyword determined from colTypes[i] and, if applicable, the " not null primary key" suffix when the column matches keyCol. The returned list preserves the order of colTypes and therefore the association between headings and types.
    
    Behavior, defaults, and failure modes:
        The function performs no I/O and has no side effects other than returning a newly created list of strings; it does not modify colHeadings or colTypes. The function uses len(colTypes) to determine the number of output columns; if colHeadings is shorter than colTypes, indexing colHeadings[i] will raise an IndexError. Each element of colTypes is expected to be indexable with at least one element (typ[0]) and, for non-float/non-int markers, a second element typ[1] that is an integer length for varchar formatting; missing elements will raise an IndexError and incompatible types passed to the "%d" formatter may raise a TypeError or ValueError. The comparisons typ[0] == float and typ[0] == int require the descriptor to use those exact Python type objects as markers; other numeric-like markers will be handled by the varchar branch. The SQL keywords produced ("double precision", "integer", "varchar(n)") are chosen to be compatible with PostgreSQL (RDKit's PostgreSQL cartridge); other database backends may require different type names and may need additional translation before use.
    """
    from rdkit.Dbase.DbUtils import GetTypeStrings
    return GetTypeStrings(colHeadings, colTypes, keyCol)


################################################################################
# Source: rdkit.Dbase.StorageUtils.IndexToRDId
# File: rdkit/Dbase/StorageUtils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def rdkit_Dbase_StorageUtils_IndexToRDId(idx: int, leadText: str = "RDCmpd"):
    """rdkit.Dbase.StorageUtils.IndexToRDId converts an integer index into a canonical RDId string used by RDKit storage utilities and database cartridges to label compounds or records. The function produces a human-readable, fixed-format identifier string with zero-padded 3-digit blocks and a trailing checksum digit; this identifier is suitable for use in cheminformatics workflows (for example, when generating displayable compound IDs or storing/retrieving records in RDKit-backed molecular databases).
    
    The format produced is:
      leadText-xxx-xxx-xxx-y
    where each "xxx" is a zero-padded three-digit block derived from the integer index and "y" is a single checksum digit computed as the sum of the decimal digits of idx modulo 10. If the index has a nonzero millions component (idx >= 1_000_000), an additional leading millions block (also zero-padded to three digits) is included so the overall identifier can represent values in the millions while preserving the same block semantics.
    
    Args:
        idx (int): The non-negative integer index to convert into an RDId. In the RDKit domain this integer typically represents an internal numeric primary key for a compound or record; the function maps that numeric key to a stable, displayable identifier string. The index must be greater than or equal to zero; negative values are rejected. Large integers are supported; when idx >= 1_000_000 an extra millions block is emitted. The decimal digits of idx are summed to compute the final checksum digit.
        leadText (str): Prefix text to place at the start of the RDId (default: 'RDCmpd'). This string identifies the ID namespace or record type (for example, using 'RDCmpd' for standard RDKit compounds or a different prefix for alternative datasets). It is inserted verbatim followed by a hyphen before the numeric blocks.
    
    Returns:
        str: A formatted RDId string composed of the leadText, three-digit zero-padded numeric blocks, and a final checksum digit. Example outputs from the same algorithm: IndexToRDId(9) -> 'RDCmpd-000-009-9'; IndexToRDId(9009) -> 'RDCmpd-009-009-8'; IndexToRDId(9000009) -> 'RDCmpd-009-000-009-8'. The returned string is the only side effect; no persistent state is modified.
    
    Raises:
        ValueError: If idx is negative. The function enforces that indices must be >= 0 because negative indices have no meaningful representation in the RDId scheme used by RDKit storage utilities.
        TypeError: May be raised by Python operations if idx is not an integer-like value (for example, passing a non-numeric type); callers should pass an int to match the function signature.
    """
    from rdkit.Dbase.StorageUtils import IndexToRDId
    return IndexToRDId(idx, leadText)


################################################################################
# Source: rdkit.Dbase.StorageUtils.RDIdToInt
# File: rdkit/Dbase/StorageUtils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def rdkit_Dbase_StorageUtils_RDIdToInt(ID: str, validate: bool = 1):
    """rdkit.Dbase.StorageUtils.RDIdToInt converts a canonical RDId string used in RDKit storage utilities into the integer index that RDKit uses for internal indexing, database keys, and compact numeric representations in molecular storage and retrieval workflows.
    
    This function is used in RDKit's database/storage utilities (for example, the molecular database cartridge and other persistence layers) to map human-readable RDId identifiers like "RDCmpd-009-000-009-8" or "RDData_000_009_9" to a single integer index. The conversion treats the components between the first and last dash (or underscore, which is normalized to a dash) as fixed-width 3-digit blocks that form a base-1000 little-endian number: the rightmost interior block is the least significant block (multiplied by 1000^0), the next block to the left is multiplied by 1000^1, and so on. Algorithmic steps: if validation is enabled the function calls ValidateRDId(ID) and raises ValueError("Bad RD Id") if validation fails; underscores are replaced with hyphens; the ID is split on '-' and the tokens between the first and last are parsed in reverse order as decimal integers; the final integer is the sum of term_int * (1000**position) for each term. Typical practical significance: the returned integer is suitable for compact storage, numeric comparisons, indexing, and as a deterministic numeric key derived from the RDId string.
    
    Args:
        ID (str): The RDId string to convert. This is the canonical identifier used in RDKit storage contexts (examples from source: "RDCmpd-000-009-9", "RDCmpd-009-000-009-8", "RDData_000_009_9"). Underscores in this string are treated as hyphens; the function extracts the tokens between the first and last hyphen after normalization and interprets each token as a zero-padded 3-digit decimal block. If the provided ID does not conform to expected RDId formatting and validation is enabled, a ValueError is raised.
        validate (bool): Whether to validate the ID format before conversion. The default value in the function signature is 1 (truthy), so validation is performed by default. When True (or any truthy value), the function calls ValidateRDId(ID) and raises ValueError("Bad RD Id") if validation fails. When False (or falsy), the function skips the ValidateRDId check and proceeds to parse and convert the ID; in that case, malformed interior blocks may still cause a ValueError during integer conversion (for example, if a block is not numeric).
    
    Returns:
        int: A non-negative integer representing the RDId converted into a base-1000 positional integer index. The integer is computed by interpreting the interior 3-digit blocks of the normalized ID (between the first and last hyphen) as little-endian base-1000 digits and summing term_int * (1000**position). For example, "RDCmpd-000-009-9" yields 9 and "RDCmpd-009-000-009-8" yields 9000009. If validation fails or an interior token cannot be converted to an integer, the function raises ValueError("Bad RD Id") or the integer conversion will raise ValueError; there are no other side effects.
    """
    from rdkit.Dbase.StorageUtils import RDIdToInt
    return RDIdToInt(ID, validate)


################################################################################
# Source: rdkit.Dbase.StorageUtils.ValidateRDId
# File: rdkit/Dbase/StorageUtils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def rdkit_Dbase_StorageUtils_ValidateRDId(ID: str):
    """rdkit.Dbase.StorageUtils.ValidateRDId validates an RDId string used by RDKit storage/database utilities by checking format and a single-digit checksum (CRC). This function is intended for use in RDKit's cheminformatics and molecular database workflows (for example, the RDKit molecular database cartridge and compound identifiers such as "RDCmpd-000-009-9") to determine whether an identifier conforms to the expected segment structure and checksum rule.
    
    Args:
        ID (str): The identifier string to validate. This should be the textual RDId as used in RDKit storage and database contexts (for example "RDCmpd-000-009-9"). Underscores in the supplied ID are treated as equivalent to hyphens: the function replaces all '_' characters with '-' before validation. The function does not otherwise normalize letter case or validate the textual prefix; it only enforces the segment-count and digit/checksum rules described below.
    
    Behavior:
        The function performs the following checks and computation in order:
        1) Replaces underscores with hyphens in the provided ID (ID = ID.replace('_', '-')).
        2) Splits the modified string on hyphens. A valid RDId must produce at least four segments when split: a leading prefix (e.g., "RDCmpd"), one or more numeric groups, and a final segment that encodes the checksum digit. If the split yields fewer than four segments, the function deems the ID invalid and returns 0.
        3) For every character in every middle segment (all segments except the first and the last), the function attempts to convert the character to an integer digit. If any character in those middle segments is not a decimal digit (0–9), the function treats the ID as invalid and returns 0.
        4) The function sums all decimal digits found in the middle segments into an accumulator.
        5) It converts the final segment to an integer and interprets this integer as the checksum (CRC). If the final segment cannot be parsed as an integer, a ValueError will be raised by int(), and that exception will propagate to the caller (this is a documented failure mode).
        6) The checksum rule is: the accumulated sum of digits modulo 10 must equal the integer value of the final segment. If this equality holds, the ID is valid.
    
    Side effects:
        The function has no side effects on external state or on the original string passed by the caller. It operates on a local, modified copy of the input string. It does not read or write files, databases, or global variables.
    
    Defaults and complexity:
        There are no optional arguments or defaults beyond the single required ID parameter. The runtime is linear in the length of the input string (it inspects each character of the middle segments once).
    
    Failure modes:
        - Returns 0 when the ID has fewer than four hyphen-separated segments.
        - Returns 0 when any character in the middle segments is not a decimal digit.
        - Raises ValueError when the final segment cannot be converted to an integer (the function does not catch this exception).
        - The function does not validate the textual prefix (first segment) beyond its presence; incorrect prefix text will not by itself cause a validation failure unless other rules are violated.
    
    Returns:
        int: 1 if the ID passes the structural and checksum validation (note: the function may return the boolean True which behaves as 1 in Python), 0 if the ID is determined to be invalid according to the checks above (or the function returns 0 on encountering non-digit characters in the middle segments). A ValueError is raised if the final segment cannot be parsed as an integer.
    """
    from rdkit.Dbase.StorageUtils import ValidateRDId
    return ValidateRDId(ID)


################################################################################
# Source: rdkit.ML.Cluster.Murtagh.ClusterData
# File: rdkit/ML/Cluster/Murtagh.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def rdkit_ML_Cluster_Murtagh_ClusterData(
    data: list,
    nPts: int,
    method: str,
    isDistData: bool = 0
):
    """rdkit.ML.Cluster.Murtagh.ClusterData clusters the input data and returns a hierarchical cluster tree used by RDKit for cheminformatics and machine-learning workflows (for example clustering molecular descriptor or fingerprint vectors or clustering by pairwise distances).
    
    Args:
        data (list): The input data to cluster. In practice this is a list of lists (rows = observations, columns = features) or an array-like object containing feature vectors for each point when clustering feature data. When clustering distance data (see isDistData) this should instead contain the condensed distance vector arranged so that for i<j: d_ij = dists[j*(j-1)//2 + i]. This argument is immediately converted to a numpy.array inside the function and must therefore be compatible with numpy.array(data). The contents represent either feature vectors for nPts observations (when isDistData is False) or the condensed distance matrix elements (when isDistData is True).
        nPts (int): The number of points to be used in the clustering. For feature-vector input this should equal the number of rows (observations) present in data. For condensed distance input this should equal the number of points whose pairwise distances are stored in data. The function uses nPts together with the data to drive the underlying Murtagh clustering routines.
        method (str): A string that determines which Murtagh clustering algorithm to use. Valid, predefined values expected by the implementation are 'WARDS', 'SLINK', 'CLINK', and 'UPGMA'. The chosen method controls the linkage/merge rules used to build the hierarchical cluster tree; supplying an unsupported string will cause the underlying clustering routine to raise an error.
        isDistData (bool): Toggle indicating whether the provided data is a condensed distance matrix (True) or raw feature vectors (False). Default is False (the original default 0 is interpreted as False). When False the function reads feature vectors and infers the number of features as data.shape[1] and calls MurtaghCluster; when True the function treats data as a condensed symmetric distance representation and calls MurtaghDistCluster. For condensed distance input, the required storage order is: for i<j: d_ij = dists[j*(j-1)//2 + i].
    
    Returns:
        list: A single-entry list containing the cluster tree produced by the Murtagh clustering routines and post-processed by _ToClusters. The returned list is the hierarchical clustering representation used by RDKit downstream code (for example to examine merge operations, extract clusters at a given cutoff, or visualize a dendrogram). If clustering cannot proceed due to malformed input (for example mismatched nPts versus data shape, incorrect condensed distance length or format, or an unsupported method string) the function will raise an exception propagated from numpy.array conversion or from the underlying MurtaghCluster/MurtaghDistCluster/_ToClusters calls.
    
    Behavior and side effects:
        The function converts the supplied data to a numpy.array at the start. If isDistData is False it computes the number of features as data.shape[1] and calls MurtaghCluster(data, nPts, sz, method). If isDistData is True it calls MurtaghDistCluster(data, nPts, method). In both cases the integer index arrays and criteria returned by those routines are passed to _ToClusters(data, nPts, ia, ib, crit, isDistData=isDistData) which constructs the returned cluster tree. Side effects include potential memory allocation for the numpy array conversion and any exceptions raised by numpy or the underlying clustering routines. Default behavior assumes feature-vector input unless isDistData is explicitly set True. Failure modes include incompatible data shapes (for example fewer rows than nPts, or incorrect condensed distance vector length for nPts), non-numeric data that cannot be converted to a numpy array, and invalid method names; such conditions will raise errors from the called routines.
    """
    from rdkit.ML.Cluster.Murtagh import ClusterData
    return ClusterData(data, nPts, method, isDistData)


################################################################################
# Source: rdkit.ML.Cluster.Resemblance.FindMinValInList
# File: rdkit/ML/Cluster/Resemblance.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def rdkit_ML_Cluster_Resemblance_FindMinValInList(
    mat: numpy.ndarray,
    nObjs: int,
    minIdx: int = None
):
    """Finds the minimum value in a condensed metric matrix and returns the zero-based pair indices and the minimum value.
    
    This function is used in the RDKit ML clustering/ressemblance code to locate the smallest pairwise distance (or resemblance score) when the pairwise values for nObjs objects are stored in a condensed 1-D numpy array (upper-triangle excluding the diagonal). It is typically used during agglomerative clustering or similarity search steps to pick the closest pair of objects (for example, molecules or descriptors) as described by the RDKit ML Cluster Resemblance utilities. The implementation decodes a linear index in the condensed representation into a (row, column) pair with row < column.
    
    Args:
        mat (numpy.ndarray): A 1-D numpy array containing the condensed metric matrix values in row-major order for the upper triangle of an nObjs-by-nObjs symmetric matrix excluding the diagonal. The array length must equal nObjs * (nObjs - 1) / 2; an AssertionError is raised if this condition is not met. Elements are numeric values (distances or resemblance scores) and the function returns one of these elements as the minimum value.
        nObjs (int): The number of original objects (nodes) whose pairwise values are represented in mat. This value is used to validate the length of mat and to decode the condensed index into a pair of zero-based indices (row, column). Practically, nObjs corresponds to the number of molecules or feature vectors in a clustering or similarity computation.
        minIdx (int or None): Optional precomputed index in the condensed array mat that points to the minimum value. If provided, it must be a valid index into mat (0 <= minIdx < len(mat)); if it is out of bounds, an IndexError may be raised when accessing mat[minIdx]. If None (the default), numpy.argmin(mat) is used to compute the index of the first occurrence of the minimum value; when there are ties, numpy.argmin selects the first minimal element encountered, which this function then decodes.
    
    Returns:
        tuple: A 3-tuple (row, col, value) where:
            row (int): Zero-based row index of the pair in the original nObjs-by-nObjs matrix. This index is strictly less than col.
            col (int): Zero-based column index of the pair in the original matrix. This index is strictly greater than row.
            value: The minimum value itself as taken from mat[minIdx]; its type matches the dtype of elements stored in mat (typically a numpy scalar or Python numeric type).
        The returned indices identify the pair of objects (row, col) whose condensed pairwise value is minimal and can be used directly in clustering steps that require the indices of the closest pair.
    
    Behavior, side effects, and failure modes:
        - The function asserts that len(mat) == nObjs * (nObjs - 1) / 2. If this is not true, an AssertionError is raised to signal an inconsistent condensed matrix length.
        - If minIdx is None, numpy.argmin(mat) is invoked to locate the minimal element; this may be O(len(mat)) in time. If you have already computed the index elsewhere, supplying minIdx avoids this recomputation.
        - Ties for the minimal value are resolved by returning the first occurrence as determined by numpy.argmin.
        - If minIdx is supplied but outside the valid range, accessing mat[minIdx] will raise an IndexError.
        - The function decodes the condensed index assuming the standard packing of the upper triangle (row < column) into a 1-D array; it returns zero-based indices accordingly.
        - The implementation is straightforward and not optimized for extreme performance; for very large nObjs and frequent queries, consider maintaining a priority structure externally.
    """
    from rdkit.ML.Cluster.Resemblance import FindMinValInList
    return FindMinValInList(mat, nObjs, minIdx)


################################################################################
# Source: rdkit.ML.Cluster.Resemblance.ShowMetricMat
# File: rdkit/ML/Cluster/Resemblance.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def rdkit_ML_Cluster_Resemblance_ShowMetricMat(metricMat: list, nObjs: int):
    """rdkit.ML.Cluster.Resemblance.ShowMetricMat: display a condensed upper-triangular metric matrix as a formatted nObjs-by-nObjs table on standard output. This function is used in the RDKit ML/Clustering Resemblance code to visualize pairwise distances or similarity values computed for nObjs objects during clustering or analysis workflows; it reconstructs the full symmetric display from the condensed 1D storage used by the clustering routines and prints it for human inspection.
    
    Args:
        metricMat (list): A one-dimensional list containing the upper-triangular (excluding the diagonal) elements of a symmetric metric/distance matrix. The list must contain exactly nObjs * (nObjs - 1) / 2 elements and is expected to store entries in column-major order for the upper triangle such that the value for the pair (row, col) with row < col is accessed by the index (col * (col - 1)) / 2 + row. Each element should be a numeric value (e.g., float) representing the pairwise metric between two objects computed elsewhere in the RDKit clustering pipeline.
        nObjs (int): The number of objects (rows/columns) to display. The function will print nObjs lines, each containing nObjs columns. This integer determines how the one-dimensional metricMat is interpreted and how many entries are printed on each line.
    
    Returns:
        None: This function does not return a value. Its effect is a side effect: printing to standard output a formatted table with nObjs rows and nObjs columns. For positions where col <= row (the diagonal and the lower triangle) the function prints the placeholder string "   ---    " because those entries are not stored in metricMat; for positions where col > row it prints the numeric metric value formatted with a field width of 10 and six decimal places ("%10.6f"). Each row ends with a newline.
    
    Behavior and failure modes:
        The function asserts that len(metricMat) == nObjs * (nObjs - 1) / 2 and will raise an AssertionError with the message 'bad matrix length in FindMinValInList' if the provided list length does not match the expected condensed upper-triangular size. If metricMat contains non-numeric elements, or if its contents/ordering do not conform to the expected column-major upper-triangular layout, the printed table will be incorrect and printing may raise TypeError/ValueError or IndexError depending on the incorrect contents. The function writes directly to standard output and is intended for human-readable inspection rather than programmatic consumption; callers that need machine-readable output should reconstruct the full matrix themselves from metricMat and nObjs.
    """
    from rdkit.ML.Cluster.Resemblance import ShowMetricMat
    return ShowMetricMat(metricMat, nObjs)


################################################################################
# Source: rdkit.ML.Cluster.Standardize.StdDev
# File: rdkit/ML/Cluster/Standardize.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def rdkit_ML_Cluster_Standardize_StdDev(mat: numpy.ndarray):
    """rdkit.ML.Cluster.Standardize.StdDev computes a standardized version of an input data array by delegating to the internal statistics routine ML.Data.Stats.StandardizeMatrix. This function is intended for use in RDKit cheminformatics machine-learning and clustering workflows (for example, normalizing descriptor or fingerprint feature matrices prior to clustering or model fitting) and acts as a thin wrapper that forwards the provided numpy.ndarray to the Stats.StandardizeMatrix implementation.
    
    Args:
        mat (numpy.ndarray): A NumPy array containing the data to standardize. In the RDKit ML/cluster context this typically holds feature values such as molecular descriptors or fingerprint-derived features for a set of samples. The array must be suitable for consumption by ML.Data.Stats.StandardizeMatrix; if it contains non-numeric entries or an unsupported layout, the underlying routine may raise an error. This parameter is required and is passed unchanged to Stats.StandardizeMatrix.
    
    Returns:
        numpy.ndarray: The array returned by ML.Data.Stats.StandardizeMatrix(mat). This value is the standardized version of the input as produced by the Stats implementation and is intended to be used downstream in clustering and machine-learning pipelines within RDKit (for example, to ensure features are on comparable scales). If the underlying Stats.StandardizeMatrix raises an exception (for example due to invalid input types or numerical issues such as zero variance for a feature), that exception will propagate to the caller; no additional error handling is performed by this wrapper.
    """
    from rdkit.ML.Cluster.Standardize import StdDev
    return StdDev(mat)


################################################################################
# Source: rdkit.ML.Data.DataUtils.BuildDataSet
# File: rdkit/ML/Data/DataUtils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def rdkit_ML_Data_DataUtils_BuildDataSet(fileName: str):
    """rdkit.ML.Data.DataUtils.BuildDataSet builds an MLData.MLDataSet instance by reading a RDKit-style ".dat" file and parsing the variable metadata and example records needed for cheminformatics machine-learning workflows. The function is intended for use within RDKit's ML utilities to prepare datasets (examples and associated metadata such as variable names, quantile bounds, and point names) for downstream tasks such as descriptor-based modeling, training classifiers/regressors, or similarity analyses.
    
    Args:
        fileName (str): The filesystem path to a ".dat" file to read. This string must point to a readable file in the RDKit ML ".dat" format expected by ReadVars and ReadGeneralExamples. The file should contain variable declarations and example records; ReadVars(inFile) is called to extract varNames and qBounds (variable names and quantile/bound metadata) and ReadGeneralExamples(inFile) is called to extract ptNames and examples (point/example identifiers and example data). In practical terms, fileName is the primary input used to create a structured ML dataset from on-disk example data and metadata for cheminformatics machine-learning experiments.
    
    Returns:
        MLData.MLDataSet: An MLData.MLDataSet object constructed from the parsed file contents. The returned object encapsulates the examples (the per-example data vectors), varNames (the list of variable/feature names corresponding to example columns), qBounds (quantile or bound information parsed from the file and used by RDKit ML utilities for normalization/discretization or metadata purposes), and ptNames (optional point/example identifiers). This MLData.MLDataSet is ready for use with other RDKit ML functions (for example, model training, evaluation, or descriptor pipelines).
    
    Behavior and side effects:
        The function opens the file at fileName for reading using a context manager (with open(fileName, 'r') as inFile), which ensures the file is closed on return or error. It delegates parsing to ReadVars and ReadGeneralExamples, then constructs the MLData.MLDataSet via MLData.MLDataSet(examples, qBounds=qBounds, varNames=varNames, ptNames=ptNames). There are no other persistent side effects (no global state is modified).
    
    Failure modes and errors:
        If fileName does not refer to an existing or readable file, the call will raise FileNotFoundError or an I/O-related exception from the underlying open() call. If the file contents do not conform to the expected RDKit ".dat" format, parsing functions (ReadVars or ReadGeneralExamples) may raise ValueError or other parsing exceptions defined in the surrounding module. The MLData.MLDataSet constructor may raise exceptions if the parsed examples and metadata are inconsistent (for example, mismatched dimensions between examples and varNames). Encoding-related errors (e.g., UnicodeDecodeError) can occur if the file contains characters incompatible with the default text encoding used by open(). The caller should handle these exceptions or ensure the input file is valid and correctly encoded.
    """
    from rdkit.ML.Data.DataUtils import BuildDataSet
    return BuildDataSet(fileName)


################################################################################
# Source: rdkit.ML.Data.DataUtils.BuildQuantDataSet
# File: rdkit/ML/Data/DataUtils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def rdkit_ML_Data_DataUtils_BuildQuantDataSet(fileName: str):
    """BuildQuantDataSet constructs an RDKit quantitative machine-learning dataset (MLQuantDataSet) from a .qdat file.
    
    Args:
        fileName (str): Path to a .qdat file containing quantitative examples and metadata used by RDKit's ML utilities. In the RDKit cheminformatics and machine-learning workflow, this file typically encodes variable names (column headers), quantitative bounds for variables, sample/point identifiers, and the numeric example rows. The function opens fileName for reading as text, parses the file using the internal ReadVars and ReadQuantExamples helpers to extract varNames, qBounds, ptNames, and examples, and uses these to build the dataset. fileName should be a file-system path string accessible from the runtime environment; the file is opened with Python's default text encoding, so non-default encodings may require pre-conversion. No default values are used.
    
    Returns:
        MLData.MLQuantDataSet: An instance of MLData.MLQuantDataSet constructed from the parsed contents of the .qdat file. The returned object contains the example matrix (examples) and associated metadata: qBounds (quantitative bounds for variables), varNames (variable/feature names), and ptNames (sample identifiers). This MLQuantDataSet is intended for downstream quantitative (regression) machine-learning tasks within RDKit, such as training or evaluating models that consume descriptor/fingerprint-derived numeric features.
    
    Behavior and side effects:
        The function reads the entire .qdat file specified by fileName, calling ReadVars(inFile) to obtain varNames and qBounds and ReadQuantExamples(inFile) to obtain ptNames and examples. It then constructs and returns an MLData.MLQuantDataSet using these values. The function does not modify the input file; it only reads from it. File reading occurs immediately and the file handle is closed before the function returns.
    
    Failure modes and errors:
        If fileName does not exist or is not readable, a FileNotFoundError or OSError/IOError will be raised by the open call. If the .qdat file contents are malformed or do not conform to the expected format consumed by ReadVars or ReadQuantExamples, those helper functions or the MLQuantDataSet constructor may raise parsing-related exceptions (for example, ValueError) or other exceptions propagated to the caller. Consumers should ensure the .qdat file follows the format expected by RDKit's DataUtils utilities before calling this function.
    """
    from rdkit.ML.Data.DataUtils import BuildQuantDataSet
    return BuildQuantDataSet(fileName)


################################################################################
# Source: rdkit.ML.Data.DataUtils.CalcNPossibleUsingMap
# File: rdkit/ML/Data/DataUtils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def rdkit_ML_Data_DataUtils_CalcNPossibleUsingMap(
    data: list,
    order: list,
    qBounds: list,
    nQBounds: list = None,
    silent: bool = True
):
    """rdkit.ML.Data.DataUtils.CalcNPossibleUsingMap: calculate the number of possible discrete values for each variable in a dataset, using an ordering map and optional quantization bounds. This function is used in RDKit machine-learning and descriptor-processing workflows to determine how many discrete categories each variable can take, either by reading provided quantization bounds or by scanning integer-valued data. The result is commonly used when preparing molecular descriptors or other features for algorithms that require knowledge of categorical cardinality (for example, encoder sizing, histogram binning, or discrete-feature models).
    
    This function examines each variable index defined by the ordering map and returns a list of counts corresponding to the number of possible values for that variable. For variables that have an entry in qBounds (a list of quantization buckets), the function uses the length of that entry as the count. For variables without qBounds, the function inspects the dataset values (accessed according to order) and, if all observed values for that variable are integer-valued numbers of one of the recognized numeric types, computes the count as max_integer_value_seen + 1 (interpreting integer values as zero-based categories). Variables that are declared non-quantized via nQBounds (non-zero entry) are excluded from computation and yield the sentinel count produced by the function logic. The function can print diagnostic information when silent is False.
    
    Args:
        data (list): A list of examples (rows). Each example is an indexable sequence (for example, a tuple or list) of variable values. In RDKit ML workflows this typically contains molecular descriptor vectors or feature rows; values are accessed as data[row_index][order[col_index]] according to the mapping in order. The function iterates over these examples to infer maximum integer values for variables that lack explicit quantization bounds.
        order (list): A list mapping the function's variable indices (0..nVars-1) to the column indices in each example of data. For variable index i used inside this function, the corresponding value in a given example is data[row][order[i]]. The length of order determines the number of variables processed and must match the length of qBounds or nQBounds as required by the assertion below.
        qBounds (list): A list of quantization bounds for variables. Each entry qBounds[i] is expected to be a sequence (possibly empty) that lists quantization buckets for variable i; if qBounds[i] is non-empty, the function reads len(qBounds[i]) and uses that as the number of possible values for variable i without scanning data. qBounds must be provided (truthy) and have the same length as order unless nQBounds is provided and satisfies the assertion described below.
        nQBounds (list): Optional list used to mark variables as non-quantized. If provided and nQBounds[i] != 0 for a variable i, that variable is excluded from counting and assigned the internal sentinel value (which results in zero in the final returned count). If nQBounds is provided, it must have the same length as order. If both qBounds and nQBounds are provided, nQBounds is checked first for each variable index.
        silent (bool): If False, the function prints diagnostic information to standard output (for example, the order, qBounds and messages when a column is excluded during scanning). Default True suppresses these prints. Use False when debugging mapping/quantization issues in descriptor preprocessing.
    
    Returns:
        list: A list of integers, one per variable (same length as order), where each element is the computed number of possible discrete values for that variable. For variables with qBounds[i] non-empty, the returned value is len(qBounds[i]). For variables with only integer-valued data (recognized numeric types), the returned value is max_integer_value_seen + 1, where max_integer_value_seen is the maximum int(d) observed across examples for that variable (this treats integer values as zero-based categories). For variables excluded by nQBounds or for variables whose values are non-integer or of an unrecognized type according to the function's internal numeric type check, the function yields 0 (these are represented internally as -1 and converted to 0 by the final int(x) + 1 computation).
    
    Behavior and failure modes:
        - The function requires either qBounds (truthy) with len(qBounds) == len(order) or nQBounds with len(nQBounds) == len(order). If neither condition holds, the function raises an AssertionError with message 'order/qBounds mismatch'.
        - Quantization precedence: if nQBounds is provided and nQBounds[i] != 0, that variable is skipped and not scanned; otherwise, if qBounds[i] has length > 0, that length is used directly.
        - The function uses an exact type check (type(value) in (int, float, numpy.int64, numpy.int32, numpy.int16)) to detect numeric values. This means some numeric types commonly used in scientific Python (for example numpy.float64 or other numpy floating types) are not recognized by this exact check; such values are treated as non-discrete, causing the variable to be marked as excluded and the returned count to be 0. Integer-valued floats (Python float exactly equal to an integer) are accepted because float is in the recognized types and are considered integer-valued only when int(d) == d.
        - The function computes counts assuming integer category values are zero-based. If dataset values are not zero-based or contain gaps, the returned count equals maximum observed integer value plus one and may overestimate the number of distinct categories.
        - Side effects: printing to standard output when silent is False. The function performs no other persistent side effects.
        - The returned list elements are plain Python integers derived from internal integer arithmetic (int(x) + 1).
    """
    from rdkit.ML.Data.DataUtils import CalcNPossibleUsingMap
    return CalcNPossibleUsingMap(data, order, qBounds, nQBounds, silent)


################################################################################
# Source: rdkit.ML.Data.DataUtils.CountResults
# File: rdkit/ML/Data/DataUtils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def rdkit_ML_Data_DataUtils_CountResults(inData: list, col: int = -1, bounds: list = None):
    """rdkit.ML.Data.DataUtils.CountResults counts occurrences of either raw values or binned values taken from a specified column of a tabular dataset. This utility is part of RDKit's ML data utilities and is typically used when preparing descriptor or fingerprint data for machine-learning workflows (for example, computing class/histogram counts of a label column or discretizing a continuous property into bins for model input or analysis).
    
    Args:
        inData (list): A sequence (typically a list) of records/rows representing dataset entries used in RDKit ML tasks. Each element is expected to be an indexable container (for example, a tuple or list) from which a column value can be retrieved using inData[i][col]. inData provides the raw examples whose column values will be counted or placed into bins.
        col (int): The integer column index to extract from each record in inData. Defaults to -1, which selects the last element of each record (a common convention for label/target columns in RDKit ML data files). If col is out of range for a record, an IndexError will be raised.
        bounds (list): Optional list of numeric threshold values used to bin continuous column values into discrete bins. When bounds is None (the default), the function returns counts keyed by the raw extracted column values (useful for counting categorical labels). When bounds is provided, each extracted value is compared with bounds in order using the '<' operator and placed in the first bin index whose threshold is greater than the value; if the value is not less than any threshold, it is placed in the final bin whose index equals len(bounds). bounds must contain values comparable with the column values (and should be provided in ascending order for meaningful binning); if elements are not comparable, a TypeError or other comparison error may occur.
    
    Returns:
        dict: A dictionary mapping keys to integer counts summarizing occurrences in the specified column across inData. If bounds is None, keys are the raw values extracted from each record at index col and values are their integer counts (useful for categorical frequency counts). If bounds is provided, keys are integer bin indices (0 through len(bounds)) and values are the counts of records placed into each bin. No other side effects occur; the function does not modify inData or bounds. Note that the algorithm scans bounds linearly for each record, so runtime is O(N * B) where N is len(inData) and B is len(bounds).
    """
    from rdkit.ML.Data.DataUtils import CountResults
    return CountResults(inData, col, bounds)


################################################################################
# Source: rdkit.ML.Data.DataUtils.FilterData
# File: rdkit/ML/Data/DataUtils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def rdkit_ML_Data_DataUtils_FilterData(
    inData: list,
    val: int,
    frac: float,
    col: int = -1,
    indicesToUse: list = None,
    indicesOnly: bool = 0
):
    """FilterData filters a dataset to produce a subset in which the fraction of rows
    whose value in a specified column equals a given target value is approximately
    equal to a requested fraction. This function is part of RDKit's ML DataUtils
    and is typically used in cheminformatics machine-learning workflows to balance
    training or evaluation datasets (for example, balancing active/inactive labels
    from descriptor or fingerprint tables) by randomly selecting which rows to keep
    so that a specified class fraction is achieved as closely as possible.
    
    Args:
        inData (list): The input dataset as a sequence (e.g., list) of rows. Each
            row must be indexable (row[col] must be valid) and comparable to the
            integer target value val. Practical use: inData is typically a list of
            RDKit-derived records (tuples, lists, or other indexable objects)
            containing features, labels, or descriptors produced during RDKit
            preprocessing; the function does not modify inData and works on a
            shallow copy of its entries.
        val (int): The target integer value to filter on in the specified column.
            The function identifies rows where row[col] == val and treats them as
            the "target" class to preserve/toss as needed to reach the requested
            fraction. In cheminformatics use, val often represents a class label
            (e.g., 1 for actives, 0 for inactives) stored in a particular column.
        frac (float): The desired fraction (0 <= frac <= 1) of rows in the output
            that should have row[col] == val. The function raises ValueError if
            frac is outside [0, 1]. The function attempts to return a subset whose
            target-class fraction matches frac as closely as possible using integer
            counts and random selection; when an exact match is impossible because
            of integer constraints, it adjusts counts to the closest achievable
            values.
        col (int, optional): The column index in each row used to compare against
            val. Accepts negative indices using normal Python indexing semantics;
            default -1 selects the last element of each row. The function will
            raise ValueError('target column index out of range') if inData[0][col]
            is not a valid access, so every row is expected to have at least the
            indexed position.
        indicesToUse (list, optional): If provided, this is a list of integer
            indices into inData that selects a subset of rows to consider for
            filtering. The function will operate only on rows in this subset and
            compute the target fraction relative to the subset. Practical use:
            restrict balancing to a particular fold, preselected subset, or
            partition of a dataset. If indicesOnly is True the returned indices
            refer to indices in the original inData and are derived from this list.
            Default None means the entire inData list is used.
        indicesOnly (bool, optional): If False (default 0), the function returns
            two lists of row objects (kept and rejected) drawn from the considered
            subset of inData. If True, the function returns two lists of integer
            indices into the original inData array corresponding to kept and
            rejected rows. Note that the function internally sorts and then applies
            a random permutation when selecting and returning items, so results are
            returned in a randomized order and the selection is non-deterministic
            unless the underlying permutation/randomness that permutation() uses is
            controlled externally.
    
    Behavior and algorithmic details:
        The function first validates frac and the column index. It then builds a
        working list tmp consisting of either the entire inData (shallow-copied)
        or only the rows indexed by indicesToUse. It sorts tmp by the values in
        column col (by sorting indices according to tmp[index][col] and then
        reordering), and locates the contiguous block of rows whose column value
        equals val. Let nWithVal be the number of rows matching val and nOthers be
        the number of rows that do not. The current fraction currFrac = nWithVal / nOrig
        is compared to the requested frac. If currFrac < frac, the procedure keeps
        most or all target-class rows and randomly discards some of the other rows;
        if currFrac >= frac it keeps most or all other rows and randomly discards
        some of the target-class rows. The function computes integer targets for
        how many target-class and other-class rows to keep (nTgtFinal, nOthersFinal)
        using rounding and a loop that adjusts counts to ensure the achievable
        final fraction is as close as possible to frac. Selection uses a random
        permutation (via permutation()) to choose which rows to keep from the
        sorted lists. The function does not modify inData; it operates on copies
        (references) of rows and returns new lists.
    
    Side effects and randomness:
        The function is non-deterministic because it uses a random permutation to
        select which specific rows to keep when more rows exist than needed. It
        does not modify the original inData list or its elements (only returns
        references to them). There is no internal parameter to set the random seed;
        control of reproducibility depends on the implementation/state of
        permutation() used by RDKit.
    
    Failure modes and exceptions:
        Raises ValueError if frac < 0 or frac > 1 with message 'filter fraction out of bounds'.
        Raises ValueError('target column index out of range') if inData[0][col] is invalid.
        Raises ValueError('target value (%d) not found in data' % (val)) if no row in
        the considered subset has row[col] == val. If the requested frac cannot be
        achieved exactly due to integer-count constraints, the function returns the
        closest achievable selection as described above.
    
    Returns:
        tuple: A pair (res, rej). If indicesOnly is False, res is a list of rows
        (elements taken from inData or from the indicesToUse subset) that were kept,
        and rej is a list of rows that were rejected; both lists are returned in a
        random order determined by the internal permutation. If indicesOnly is True,
        res and rej are lists of integer indices into the original inData corresponding
        to kept and rejected rows, respectively (when indicesToUse was provided, the
        returned indices map back to the original inData using that list).
    """
    from rdkit.ML.Data.DataUtils import FilterData
    return FilterData(inData, val, frac, col, indicesToUse, indicesOnly)


################################################################################
# Source: rdkit.ML.Data.DataUtils.InitRandomNumbers
# File: rdkit/ML/Data/DataUtils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def rdkit_ML_Data_DataUtils_InitRandomNumbers(seed: tuple):
    """rdkit.ML.Data.DataUtils.InitRandomNumbers seeds RDKit's and Python's pseudo-random number generators to produce reproducible stochastic behavior in RDKit's cheminformatics and machine-learning workflows (for example, descriptor and fingerprint generation, conformer generation, 2D/3D molecular operations, and other routines that rely on randomness).
    
    This function applies a numeric seed to two global RNG implementations used within RDKit: the RDRandom generator provided by rdkit.RDRandom and the standard library random module. Seeding these global RNGs changes their internal state and therefore affects all subsequent calls that rely on either generator in the current Python process.
    
    Args:
        seed (tuple): A 2-tuple containing integers intended as random seeds. The implementation extracts seed[0] and uses that single integer value to seed both rdkit.RDRandom (via RDRandom.seed(seed[0])) and the standard Python random module (via random.seed(seed[0])). Although the docstring and callers may supply two integers, only the first element of the tuple is applied by this function; the second element is not used by the current implementation.
    
    Behavior and side effects:
        - Mutates global RNG state for rdkit.RDRandom and the Python random module. This affects reproducibility across RDKit functions and any other code that uses these RNGs in the same process.
        - Only the first element of the provided tuple is used; providing a 2-tuple is the documented convention but both RNGs receive the same first-element seed.
        - Designed to support reproducible machine-learning experiments and deterministic behavior for RDKit operations that rely on randomness (see README: descriptor and fingerprint generation, 2D/3D molecular operations).
    
    Failure modes and exceptions:
        - If seed is not a subscriptable tuple-like object or has no element at index 0, the code will raise a TypeError or IndexError when attempting to read seed[0].
        - If the extracted seed[0] value is not an integer, behavior depends on the underlying implementations of RDRandom.seed and random.seed; those calls may raise TypeError or ValueError or perform implicit conversion. Callers should supply an integer at seed[0] to avoid errors.
        - Because this function changes global state, using it in multi-threaded or concurrent scenarios may lead to race conditions or surprising interactions between components that expect independent RNG states.
    
    Returns:
        None: This function does not return a value. Its purpose is to produce side effects by seeding the global RNGs used by RDKit (rdkit.RDRandom) and Python's random module to control reproducibility of stochastic operations.
    """
    from rdkit.ML.Data.DataUtils import InitRandomNumbers
    return InitRandomNumbers(seed)


################################################################################
# Source: rdkit.ML.Data.DataUtils.TakeEnsemble
# File: rdkit/ML/Data/DataUtils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def rdkit_ML_Data_DataUtils_TakeEnsemble(
    vect: list,
    ensembleIds: tuple,
    isDataVect: bool = False
):
    """rdkit.ML.Data.DataUtils.TakeEnsemble extracts a subset of elements from a sequence (vect) according to ensemble member indices (ensembleIds). This utility is used in RDKit's machine-learning data handling to select predictions, feature values, or ensemble member outputs for further processing, evaluation, or storage. When vect is a "data vector" (isDataVect=True) that contains metadata at the first and last positions (for example a molecule identifier or label), those sentinel elements are preserved and the ensemble indices are adjusted to account for the metadata layout.
    
    Args:
        vect (list): The input sequence from which elements will be selected. In RDKit ML workflows this is typically a list containing ensemble outputs or a data vector that mixes metadata and ensemble values. The function does not modify the original list object; it builds and returns a new list with the selected elements. Elements are accessed by index using Python's zero-based indexing.
        ensembleIds (tuple): A tuple of integer indices specifying which ensemble member positions to extract from vect. Each element must be an int and is interpreted as a zero-based index into vect. If isDataVect is True, these indices are interpreted as referring to the ensemble portion of a data vector and are incremented by 1 internally to skip the first metadata element.
        isDataVect (bool = False): When False (default), vect is treated as a plain list of ensemble values and the returned list contains exactly the elements vect[x] for each x in ensembleIds, in the same order as ensembleIds. When True, vect is treated as a data vector that stores metadata at vect[0] and vect[-1]; in this mode the function returns a new list composed of vect[0], followed by vect[x+1] for each x in ensembleIds, followed by vect[-1]. Use this mode when vect includes leading/trailing metadata that must be preserved around the selected ensemble members.
    
    Returns:
        list: A new list containing the selected elements. If isDataVect is False, the returned list is [vect[x] for x in ensembleIds]. If isDataVect is True, the returned list is [vect[0]] + [vect[x+1] for x in ensembleIds] + [vect[-1]]. The original vect argument is not mutated; a fresh list object is returned.
    
    Behavior, side effects, defaults, and failure modes:
        - Default behavior: isDataVect defaults to False, meaning direct selection from vect by the indices in ensembleIds.
        - Indexing: ensembleIds must contain valid integer indices for the interpretation chosen; indices are treated as zero-based. When isDataVect is True, indices are internally incremented by 1 before accessing vect to account for the preserved leading metadata element.
        - Errors: If any index (after the optional +1 adjustment) is out of range for vect, Python will raise IndexError. If ensembleIds is not a tuple of integers or vect is not indexable as a list, a TypeError (or other standard Python exception) may be raised. The function does not perform additional type coercion or validation beyond standard list indexing behavior.
        - Use in RDKit: This function is intended for simple, fast extraction of ensemble member outputs or for preparing data vectors for downstream ML tasks in RDKit (for example, selecting a subset of model predictions from an ensemble while keeping record identifiers and labels intact). It performs no copying beyond constructing the returned list and has no external side effects.
    """
    from rdkit.ML.Data.DataUtils import TakeEnsemble
    return TakeEnsemble(vect, ensembleIds, isDataVect)


################################################################################
# Source: rdkit.ML.Data.DataUtils.TextToData
# File: rdkit/ML/Data/DataUtils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def rdkit_ML_Data_DataUtils_TextToData(
    reader: list,
    ignoreCols: list = [],
    onlyCols: list = None
):
    """rdkit.ML.Data.DataUtils.TextToData constructs an RDKit MLData.MLDataSet from rows of text, typically produced by a CSV reader or similar line-oriented iterator. This function is used in RDKit's cheminformatics and machine-learning workflows to convert tabular text data (for example, molecular identifiers plus descriptor or target columns produced during descriptor/fingerprint generation) into an MLData.MLDataSet that downstream RDKit ML utilities expect.
    
    Args:
        reader (list): An iterable object (signature types lists) that yields rows as sequences of string elements, for example a csv.reader. The first row returned by this iterator is treated as the header containing variable names. The reader is consumed: next(reader) is called to obtain the header and the remaining rows are iterated to build the dataset. If the iterator is empty, a StopIteration will propagate.
        ignoreCols (list): A list of header names to exclude from the output dataset. Defaults to an empty list, meaning no columns are ignored. This parameter is the literal list object provided by the caller; the function does not modify it. When onlyCols is not provided (None), ignoreCols determines which header names are skipped when selecting columns to keep.
        onlyCols (list): An optional list of header names that specifies the exact columns to include and their order in the output dataset. Defaults to None, which disables this behavior and uses ignoreCols logic instead. When provided, the function builds the output columns in the order of onlyCols by locating each name in the header; if a name in onlyCols does not match any header entry, the internal index for that requested name remains -1 and that index will be used when extracting data (which will select the last column in Python indexing semantics). onlyCols takes precedence over ignoreCols: if onlyCols is provided, ignoreCols is not used.
    
    Behavior and practical details:
        - The first row produced by reader is treated as the header and defines variable names for the dataset. The header values are used to match names in ignoreCols or onlyCols as described above.
        - For the selected columns, the first selected column is treated as the point (row) identifier (ptNames). Remaining selected columns are parsed as data values.
        - For each data cell (all columns except the point identifier), the function attempts to coerce the textual value to an integer (int). If that fails it attempts to coerce to a floating-point number (float). If both numeric conversions fail, the original string is preserved. This conversion policy produces Python int, float, or str values in the resulting dataset and is intended to produce typed numeric values for ML tasks when possible.
        - The function validates that every non-empty row returned by reader has the same number of columns as the header (nCols). If any row has a different number of elements than the header, a ValueError with message 'unequal line lengths' is raised.
        - Empty rows (rows for which len(splitLine) == 0) are skipped and do not contribute to the dataset.
        - The function constructs and returns an instance of MLData.MLDataSet (rdkit.ML.Data.MLDataSet) with the list of parsed value rows (vals), the tuple of selected variable names (varNames), and the list of point names (ptNames).
        - Side effects: the reader iterator is advanced (the header is consumed and subsequent rows are consumed while building the dataset). If the reader yields fewer rows than expected or is empty, a StopIteration or ValueError may be raised as described.
        - Defaults: ignoreCols defaults to an empty list and onlyCols defaults to None. Because ignoreCols is a mutable default, callers should be aware that providing their own list reference is typical practice, but this function does not mutate the ignoreCols list itself.
    
    Returns:
        MLData.MLDataSet: An instance of rdkit.ML.Data.MLDataSet constructed from the selected columns of the provided text iterator. The MLDataSet contains:
            - vals: a list of rows where each row is a list of parsed values (int, float, or str) corresponding to the selected data columns (excluding the point identifier column).
            - varNames: a tuple of variable names corresponding to the columns present in vals, in the order determined by ignoreCols or onlyCols.
            - ptNames: a list of point identifiers taken from the first selected column of each non-empty input row.
        If construction fails due to inconsistent row lengths, a ValueError is raised instead of returning.
    """
    from rdkit.ML.Data.DataUtils import TextToData
    return TextToData(reader, ignoreCols, onlyCols)


################################################################################
# Source: rdkit.ML.Data.Quantize.FindVarMultQuantBounds
# File: rdkit/ML/Data/Quantize.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def rdkit_ML_Data_Quantize_FindVarMultQuantBounds(
    vals: list,
    nBounds: int,
    results: list,
    nPossibleRes: int
):
    """FindVarMultQuantBounds finds multiple quantization bounds (cut points) for a single continuous variable so as to maximize information gain for a discrete target.
    This function is used in RDKit's ML/Data routines when discretizing continuous descriptor or feature values (for example descriptor values computed from molecules) into bins that are informative for a discrete result variable. The routine sorts the input variable together with its associated result codes, identifies valid cut start points, adjusts the requested number of bounds if there are too few candidate boundaries, and searches (via internal helpers) for the set of cuts that yields the highest information gain for the given number of result categories.
    
    Args:
        vals (list): Sequence of variable values to quantize. These are assumed to be numeric (floats) representing a continuous descriptor or feature (e.g., a molecular descriptor value). Each element in vals corresponds positionally to an element in results. The function asserts that len(vals) == len(results) and will raise AssertionError if they differ.
        nBounds (int): The requested number of quantization bounds (cut points) to find. This is an integer specifying how many boundaries to attempt to place between sorted values to partition the continuous variable into multiple bins. If the number of valid candidate start points found in the data is smaller than nBounds, the function reduces nBounds (nBounds is set to len(startPoints)-1) and will ensure at least one bound is used when possible.
        results (list): A list of result codes (should be integers) that give the discrete target/class for each corresponding entry in vals. These are used to compute information gain for candidate quantizations. The values in results should be consistent with nPossibleRes (see below); the function does not coercively validate that each code lies within a specific range but uses them as categorical labels when computing gains.
        nPossibleRes (int): An integer specifying the number of possible distinct values for the result variable (the number of classes). This parameter informs the information-gain calculations performed by the internal routines and should match the number of distinct categories represented in results.
    
    Returns:
        tuple(list, float): A 2-tuple where the first element is a list of quantization bounds (floats) and the second element is the information gain (float) associated with that quantization.
        The list of quantization bounds contains floating-point thresholds derived from midpoints between sorted adjacent vals at indices determined by the internal start-point calculation. If an index falls at the end of the sorted values the corresponding bound is the last sorted value; if an index is zero the bound is the first sorted value; otherwise the bound is the average of the two adjacent sorted values. The information gain is a numeric score computed by the internal search routine and represents the improvement in target-class purity achieved by the returned cuts.
    
    Behavior, defaults, and failure modes:
        - The function first verifies that vals and results have equal length and raises AssertionError if they do not.
        - If len(vals) == 0 (no data), the function returns an empty list of bounds and a very negative gain: ([], -1e8). This signals that no quantization is possible on empty data.
        - The routine sorts vals together with results (maintaining association) before any cut identification; the sorting determines candidate boundaries and is therefore deterministic given equal input ordering and Python's sort stability.
        - If the internal start-point detection finds no candidate start points, the function returns ([0], 0.0). This is a special-case result indicating that no informative split could be found and a single default bound of 0.0 is returned with zero gain.
        - If the number of candidate start points is less than the requested nBounds, nBounds is reduced to len(startPoints) - 1 to avoid invalid cut indices. If that reduction yields nBounds == 0 the function forces nBounds = 1 to attempt at least one cut when possible.
        - The function relies on internal helper routines (_FindStartPoints and _RecurseOnBounds) to identify start indices and to perform the combinatorial search for the best set of cuts. Exceptions or errors raised by those helpers (if any) will propagate to the caller.
        - The function does not modify its input lists; it creates sorted copies internally.
        - The returned bounds are chosen from midpoints between sorted variable values (or endpoints as described above); they are suitable as thresholds for discretizing vals into bins for downstream ML algorithms (e.g., building decision trees or preparing descriptor bins for classifiers).
        - No additional type coercion is performed; callers should pass numeric floats in vals and integer result codes in results consistent with nPossibleRes.
    
    Practical significance in RDKit:
        This routine is intended for use in RDKit's machine-learning preprocessing, e.g., discretizing continuous molecular descriptors or derived features so that information-theoretic splits (information gain) can be evaluated for classification tasks. The produced bounds and gain help determine informative binning of descriptor values for downstream models or for feature engineering in cheminformatics workflows.
    """
    from rdkit.ML.Data.Quantize import FindVarMultQuantBounds
    return FindVarMultQuantBounds(vals, nBounds, results, nPossibleRes)


################################################################################
# Source: rdkit.ML.Data.Quantize.FindVarQuantBound
# File: rdkit/ML/Data/Quantize.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def rdkit_ML_Data_Quantize_FindVarQuantBound(vals: list, results: list, nPossibleRes: int):
    """FindVarQuantBound finds a single-variable quantization boundary and its associated gain by delegating to FindVarMultQuantBounds with a multiplicity of 1. This function is a thin, historical wrapper kept for backwards compatibility in RDKit's ML data quantization utilities; it is used when converting continuous descriptor or feature values (vals) into a single discrete cut (one quantization boundary) for downstream machine-learning tasks such as descriptor binning, simple decision splits, or preprocessing of molecular descriptors.
    
    Args:
        vals (list): A list of continuous or ordered feature values to be quantized. In the RDKit ML/Quantize context, these are typically descriptor values computed for a set of molecules; the function treats this list as the variable to be partitioned and preserves the element type when returning the boundary (the returned boundary will be the same type as elements of vals where possible).
        results (list): A list of target outcomes or class labels corresponding to the entries in vals. In supervised quantization for machine learning (as used in RDKit descriptor preprocessing), results provides the observed outcomes used to evaluate candidate boundaries (for example class membership or binned response values). The length and ordering of results are expected to match vals; mismatches may lead to incorrect behavior or exceptions propagated from underlying routines.
        nPossibleRes (int): The number of distinct possible result values (classes) present in results. This integer guides the internal evaluation of information gain or impurity when FindVarMultQuantBounds computes candidate boundaries. It must reflect the actual number of categories represented in results; providing an incorrect count may produce incorrect gain estimates.
    
    Behavior and side effects:
        This function calls FindVarMultQuantBounds(vals, 1, results, nPossibleRes) and returns the first boundary from the bounds sequence produced by that call together with the gain value returned by FindVarMultQuantBounds. There are no other side effects: it does not modify its input lists in-place beyond any modifications performed by the underlying routine. The function is intentionally minimal and exists for historical API compatibility; use FindVarMultQuantBounds directly when multiple boundaries are required.
        Failure modes: if the underlying call to FindVarMultQuantBounds raises an exception (for example due to invalid inputs, mismatched lengths of vals and results, or internal computation errors), that exception propagates unchanged. If FindVarMultQuantBounds returns an empty bounds sequence, attempting to access bounds[0] will raise an IndexError. Callers should validate inputs (matching lengths, non-empty vals) or handle exceptions accordingly.
    
    Returns:
        tuple: A two-tuple (bound, gain) where bound is the single quantization boundary selected (the first element of the bounds list returned by FindVarMultQuantBounds, matching the element type of vals where applicable) and gain is the numeric score returned by FindVarMultQuantBounds that quantifies the quality of that boundary (for example information gain or reduction in impurity).
    """
    from rdkit.ML.Data.Quantize import FindVarQuantBound
    return FindVarQuantBound(vals, results, nPossibleRes)


################################################################################
# Source: rdkit.ML.Data.Quantize.feq
# File: rdkit/ML/Data/Quantize.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def rdkit_ML_Data_Quantize_feq(v1: float, v2: float, tol: float = 1e-08):
    """rdkit.ML.Data.Quantize.feq tests whether two floating-point values are equal within a specified absolute tolerance. This routine is used in RDKit's machine-learning and data-quantization code paths to guard against floating-point round-off when comparing descriptor, fingerprint, or other computed scalar feature values.
    
    Args:
        v1 (float): The first floating-point value to compare. In the RDKit context this is typically a computed descriptor or feature value extracted from a molecule or dataset. Its role is as one side of the equality test; passing non-finite values (NaN or infinity) will affect the comparison as described below.
        v2 (float): The second floating-point value to compare. As with v1, v2 is usually a computed scalar from cheminformatics calculations or ML feature pipelines. The function computes the absolute difference between v1 and v2 to determine equality.
        tol (float): The absolute tolerance used for the comparison. The function returns equality when abs(v1 - v2) < tol. The default value is 1e-08, chosen to tolerate typical floating-point rounding noise encountered when computing molecular descriptors or features in RDKit. This is an absolute (not relative) tolerance. Because the comparison uses a strict less-than test, a difference exactly equal to tol is considered not equal.
    
    Returns:
        int: 1 if the absolute difference between v1 and v2 is strictly less than tol (i.e., abs(v1 - v2) < tol), indicating they are considered equal within the specified tolerance; 0 otherwise. In practical RDKit usage, this lets calling code branch on a simple equal/not-equal result when quantizing or comparing floating-point features.
    
    Behavior and failure modes:
        This is a pure function with no side effects. It performs an absolute-difference comparison; it does not perform any relative-scaling or ULP-based checks. If either v1 or v2 is NaN, the comparison will evaluate as not equal (the function yields 0). If v1 and v2 are infinities of opposite sign or if their difference produces NaN, the result is not equal (0). Very large magnitudes may require a larger tol if callers intend to treat relative rather than absolute closeness as equality.
    """
    from rdkit.ML.Data.Quantize import feq
    return feq(v1, v2, tol)


################################################################################
# Source: rdkit.ML.Data.SplitData.SplitDataSet
# File: rdkit/ML/Data/SplitData.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def rdkit_ML_Data_SplitData_SplitDataSet(data: list, frac: float, silent: int = 0):
    """SplitDataSet splits a dataset into two subsets (training and hold-out) for machine-learning workflows in RDKit, typically used when preparing molecular descriptor or fingerprint example lists for model training and evaluation.
    
    This function takes a sequence of examples (for example, RDKit-calculated descriptors, fingerprints, or any per-molecule feature objects used in cheminformatics machine learning) and delegates the computation of integer indices to SplitIndices to partition the full index range [0, len(data)). The first returned list contains the examples selected for the "first" partition (commonly used as the training set) and the second returned list contains the remaining examples (commonly used as the hold-out/test set). SplitDataSet preserves the element selection order defined by the index lists returned from SplitIndices. SplitDataSet also emits a short summary message to standard output unless printing is suppressed via the silent parameter.
    
    Args:
        data (list): A Python list of examples to be split. In RDKit ML usage this is typically a list of per-molecule feature vectors, descriptor results, fingerprint objects, or any example objects that are indexable by integer positions. The function calls len(data) and accesses elements by integer indices (data[i]), so a TypeError will occur if the object does not support these operations.
        frac (float): The fraction in the interval [0.0, 1.0] of the original data to place into the first returned list (the "first" partition, e.g., training set). Values are interpreted as a proportion of the total number of examples; non-integer counts are handled by the underlying SplitIndices logic, so the actual number of items in the first partition will be approximately frac * len(data), subject to integer rounding performed by SplitIndices. If frac < 0.0 or frac > 1.0 a ValueError is raised.
        silent (int): Controls printing of brief informational messages to standard output. The default value 0 causes the function to print the number of points placed in the first partition and the hold-out set, which is useful for interactive or script-level feedback during dataset preparation. Any non-zero integer value suppresses these messages. Internally, SplitIndices is invoked with silent=1 so that only SplitDataSet's summary output is affected by this parameter.
    
    Returns:
        tuple: A 2-tuple (first_part, second_part) where each element is a list containing the examples from the input data selected for that partition. The first element corresponds to the portion of the data determined by frac (commonly used as the training set) and the second element corresponds to the remaining examples (commonly used as the hold-out/test set). Both lists contain references to the original elements from data in the order specified by the index lists produced by SplitIndices.
    
    Raises:
        ValueError: If frac is less than 0.0 or greater than 1.0.
        TypeError: If data does not support len(data) or integer indexing (data[i]), which are required to construct the returned partitions.
    """
    from rdkit.ML.Data.SplitData import SplitDataSet
    return SplitDataSet(data, frac, silent)


################################################################################
# Source: rdkit.ML.Data.SplitData.SplitIndices
# File: rdkit/ML/Data/SplitData.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def rdkit_ML_Data_SplitData_SplitIndices(
    nPts: int,
    frac: float,
    silent: bool = 1,
    legacy: bool = 0,
    replacement: bool = 0
):
    """rdkit.ML.Data.SplitData.SplitIndices: Split a set of integer indices (0..nPts-1) representing a dataset into two subsets according to a specified fraction. In RDKit machine-learning workflows this function is used to produce a primary set (commonly used as the training set) and a secondary hold-out set (commonly used as the test/validation set). The function supports three selection modes: (1) default: random permutation then split, (2) legacy: per-index Bernoulli selection using a random float, and (3) replacement: sampling the primary set with replacement.
    
    Args:
        nPts (int): The total number of points (elements) in the dataset to be split. Each index is assumed to be an integer from 0 to nPts-1. In cheminformatics and ML contexts this corresponds to the number of compounds, molecules, or feature vectors available to split. nPts should be an integer representing the dataset size; if nPts is zero or negative the resulting ranges and lists will be empty according to Python semantics (no explicit error is raised by this function for nPts).
        frac (float): The fraction of the data to be placed into the first returned set (the primary set, typically used as the training set). This value must lie between 0.0 and 1.0 inclusive; values outside this range raise ValueError('frac must be between 0.0 and 1.0 (frac=%f)'). The function uses int(nPts * frac) to compute the number of entries for the primary set in the non-legacy, non-replacement mode, so fractional results are floored to an integer count for that mode.
        silent (bool = 1): Toggle for printing summary statistics. When silent is truthy (the default value 1), no status is printed. When silent is falsy, the function prints two lines summarizing the number of points placed in the primary set and the hold-out set (for example: "Training with X (of Y) points." and "\tZ points are in the hold-out set."). This is a side effect useful for interactive experimentation but should be disabled (silent truthy) in batch pipelines to avoid extraneous output.
        legacy (bool = 0): If truthy, use the legacy per-index selection approach for backwards compatibility. In legacy mode the function iterates index i from 0 to nPts-1 and draws a uniform random float via RDKit's RDRandom.random(); if the float is less than frac the index is appended to the primary set, otherwise to the hold-out set. The number of indices in each set is therefore a random variable (it can vary between calls even for the same seed), and both returned lists are produced in ascending order of indices (the order in which indices were iterated). Use legacy=True only when the older selection semantics (variable-size primary set, ordered indices) are required.
        replacement (bool = 0): If truthy, select the primary set by sampling nTrain = int(nPts * frac) indices with replacement using RDKit's RDRandom.random() scaled by nPts. The sampled values are converted to int and any occasion where the raw value equals nPts is clamped to nPts-1 (ensuring indices are in 0..nPts-1). The primary set returned may contain duplicate indices. The hold-out set is then constructed as the ordered list of indices from 0..nPts-1 that do not appear in the sampled primary list; because of sampling with replacement, the hold-out set size can vary and will generally not equal nPts - nTrain. Use replacement=True when bootstrap-style sampling is required.
    
    Behavior and side effects:
        - Default mode (legacy=False, replacement=False): A list perm of 0..nPts-1 is created, RDKit's RDRandom.shuffle is used to permute the indices (the code passes Python's random.random as the random source), nTrain = int(nPts * frac) is computed, and the primary set is perm[:nTrain] (length fixed) while the hold-out set is perm[nTrain:] (length complementary). This mode produces a fixed-size primary set with no duplicates and randomized ordering.
        - Legacy mode: selection is index-wise and ordered; the size of the primary set is stochastic and indices are returned in ascending order.
        - Replacement mode: primary set has fixed length nTrain but may contain duplicates; hold-out set is the ascending ordered set of indices not sampled into the primary set.
        - Randomness source: all random decisions use RDKit's RDRandom module; reproducible behavior across runs requires initializing RDKit's random number generator (for example via RDKit.ML.Data.DataUtils.InitRandomNumbers). The function does not itself seed the RNG.
        - Printing: when silent is falsy the function prints two informational lines as described above.
    
    Failure modes:
        - A ValueError is raised if frac < 0.0 or frac > 1.0.
        - If replacement=True duplicates in the primary set are expected; downstream code that assumes unique indices must handle duplicates or avoid replacement mode.
        - The function does not explicitly validate that nPts is a non-negative integer; providing a non-integer or negative value will lead to Python runtime type/behavior (e.g., range(nPts) failing for non-integers) and is therefore unsupported.
    
    Returns:
        tuple of (list, list): A 2-tuple (primary_indices, holdout_indices). Both entries are Python lists of integers corresponding to indices in the original dataset (values in 0..nPts-1). The first list is the primary set determined by frac and the selected mode (commonly used as the training set in RDKit ML examples); the second list is the complementary hold-out set (commonly used as the test/validation set). The exact sizes and ordering of these lists depend on the mode: default mode yields fixed-size primary list with randomized order, legacy mode yields variable-size ordered lists, and replacement mode yields a fixed-length primary list that may contain duplicates and an ordered hold-out list of the indices not sampled.
    """
    from rdkit.ML.Data.SplitData import SplitIndices
    return SplitIndices(nPts, frac, silent, legacy, replacement)


################################################################################
# Source: rdkit.ML.Data.Stats.FormCorrelationMatrix
# File: rdkit/ML/Data/Stats.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def rdkit_ML_Data_Stats_FormCorrelationMatrix(mat: numpy.ndarray):
    """rdkit.ML.Data.Stats.FormCorrelationMatrix forms and returns the Pearson correlation matrix for variables encoded as columns in a 2-D data array.
    
    This function is intended for use in RDKit machine-learning and cheminformatics workflows (for example, descriptor or fingerprint analysis and feature-selection preprocessing). Given a 2-D numpy.ndarray where each row is an observation and each column is a variable (descriptor), the function computes the pairwise Pearson correlation coefficient for every variable pair using a direct-sum formula equivalent to the covariance-based Pearson correlation. The implementation computes nVars = len(mat[0]) (number of variables/columns) and N = len(mat) (number of observations/rows), forms an N-by-nVars view of each column and computes sums, sums of squares, and cross-products to produce a symmetric nVars-by-nVars correlation matrix. Diagonal entries will be 1.0 in the absence of zero variance; if a variable has zero sample variance the corresponding correlations (including the diagonal) are set to 0 by this implementation.
    
    Args:
        mat (numpy.ndarray): A 2-D array of numeric data with shape (N, nVars) where N is the number of observations (rows) and nVars is the number of variables (columns). Each column is treated as a separate variable whose pairwise Pearson correlation with every other column is computed. The function uses indexing mat[:, i] and mat[:, j], so mat must support 2-D numpy-style slicing; passing an array with zero rows or a non-2-D array will raise an IndexError or other exceptions. The function does not modify mat in place.
    
    Returns:
        numpy.ndarray: A new square, symmetric numpy.ndarray of shape (nVars, nVars) and floating dtype (the code uses 'd' for a double-precision float buffer) containing the pairwise Pearson correlation coefficients. Each element res[i, j] is computed as
        (N * sum(x * y) - sum(x) * sum(y)) / sqrt((N * sum(x^2) - sum(x)^2) * (N * sum(y^2) - sum(y)^2))
        where x and y are the column vectors for variables i and j. If the denominator for a pair is exactly zero (zero variance for one or both variables), the implementation sets the correlation for that pair to 0. Note that due to floating-point round-off the expression under the square root can become slightly negative, which may lead to NaNs in the result; such numerical issues are not specially handled by the function.
    
    Side effects and failure modes:
        The function allocates and returns a new numpy.ndarray and does not alter the input array. It expects mat to be a numpy.ndarray with at least one row and at least one column; calling it with an empty array (zero rows) or a 1-D array will generally raise an IndexError or TypeError. If mat contains non-numeric entries or is not a proper 2-D numeric numpy.ndarray, the behavior is undefined and Python or NumPy exceptions may be raised. Numerical instability (round-off) can produce small negative values inside the square root leading to NaNs; the function does not guard against such pathological floating-point cases.
    """
    from rdkit.ML.Data.Stats import FormCorrelationMatrix
    return FormCorrelationMatrix(mat)


################################################################################
# Source: rdkit.ML.Data.Stats.FormCovarianceMatrix
# File: rdkit/ML/Data/Stats.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def rdkit_ML_Data_Stats_FormCovarianceMatrix(mat: numpy.ndarray):
    """FormCovarianceMatrix forms and returns the sample covariance matrix for a set of observations represented as a NumPy array. This function is used in RDKit's machine-learning and descriptor workflows to compute covariances between features (for example prior to PCA, feature whitening, or other statistical analyses of molecular descriptors and fingerprints). The implementation centers the input data by subtracting the per-column means (computed over rows/observations) and then computes the unbiased covariance estimator by multiplying the transposed centered data with the centered data and dividing by (nPts - 1).
    
    Args:
        mat (numpy.ndarray): A 2-D numeric array containing the data to analyze. The first axis (mat.shape[0]) is interpreted as the number of observations (nPts, e.g., molecules or samples) and the second axis as features/variables (nFeatures, e.g., descriptor values). The function computes the column-wise mean vector, subtracts that mean from each row in-place (centering the data), and then uses the centered data to compute covariances. The input must be a numpy.ndarray; its values should be numeric (floating or integer types convertible to floating point) so that mean subtraction and dot-product produce meaningful numeric results. Because the data are modified in-place, callers that need to retain the original uncentered data should pass a copy (for example, mat.copy()).
    
    Returns:
        numpy.ndarray: The sample covariance matrix of the input features, shaped (nFeatures, nFeatures). This is computed as (mat^T dot mat) / (nPts - 1) after centering, i.e., the unbiased estimator commonly used in statistical analysis and machine-learning preprocessing.
    
    Behavior and failure modes:
        The function determines nPts from mat.shape[0]. If nPts <= 1, division by (nPts - 1) will raise a ZeroDivisionError or produce invalid values; callers must ensure at least two observations are provided. The function mutates mat by subtracting the per-column mean from each row; this side effect is intentional for in-place centering and improves memory efficiency for large descriptor matrices common in cheminformatics. If mat contains NaNs or infinities, the resulting covariance matrix will reflect those values (NaNs/infinities propagate). If mat is not two-dimensional or does not behave like a numeric numpy.ndarray, operations such as summation, in-place subtraction, or numpy.dot may raise exceptions (TypeError, ValueError) consistent with numpy's behavior.
    """
    from rdkit.ML.Data.Stats import FormCovarianceMatrix
    return FormCovarianceMatrix(mat)


################################################################################
# Source: rdkit.ML.Data.Stats.MeanAndDev
# File: rdkit/ML/Data/Stats.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def rdkit_ML_Data_Stats_MeanAndDev(vect: list, sampleSD: bool = 1):
    """rdkit.ML.Data.Stats.MeanAndDev computes the arithmetic mean and standard deviation of a numeric vector. This function is intended for use in RDKit workflows (for example when processing descriptor or fingerprint vectors for machine learning and normalization tasks) and implements the standard formulas used when summarizing a list of numeric descriptor values.
    
    Args:
        vect (list): A list of numeric values representing a 1-D data vector (for example, a descriptor vector produced by RDKit). The function converts this list to a NumPy array with dtype 'd' (double precision) using numpy.array(vect, 'd') before computation. If vect is empty (length 0) the function returns (0., 0.). If vect contains values that cannot be converted to double precision floats, numpy will raise a TypeError or ValueError during conversion; those exceptions are not handled inside this function.
        sampleSD (bool): Controls whether the standard deviation is the sample standard deviation (True) or the population standard deviation (False). When True (the default value in the signature is 1, which is truthy and selects the sample estimator), the deviation is computed as sqrt(sum((x - mean)^2) / (n - 1)) for n > 1. When False, the deviation is computed as sqrt(sum((x - mean)^2) / n). For n <= 0 the function returns (0., 0.). For n == 1 the mean is the single value and the deviation is returned as 0.0. This choice matters in statistical preprocessing for machine-learning tasks: sampleSD (True) provides the unbiased estimator for population variance when working with a sample.
    
    Returns:
        tuple: A pair (mean, dev) where mean is a float giving the arithmetic mean of the input vector and dev is a float giving the standard deviation as selected by sampleSD. Both values are returned as Python floats (derived from NumPy double precision). No in-place modification of the input list is performed; the function creates an internal NumPy array for computation. In edge cases the function returns (0.0, 0.0) for empty input and (x, 0.0) for a single-element input, and it will propagate NumPy conversion errors for non-numeric inputs.
    """
    from rdkit.ML.Data.Stats import MeanAndDev
    return MeanAndDev(vect, sampleSD)


################################################################################
# Source: rdkit.ML.Data.Stats.PrincipalComponents
# File: rdkit/ML/Data/Stats.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def rdkit_ML_Data_Stats_PrincipalComponents(mat: numpy.ndarray, reverseOrder: bool = 1):
    """rdkit.ML.Data.Stats.PrincipalComponents performs a principal components analysis (PCA) on a numeric data matrix by forming a correlation matrix from the input data, computing its eigenvalue decomposition, and returning the eigenvalues and corresponding eigenvectors ordered by magnitude. In the RDKit cheminformatics and machine-learning context described in the project README, this function is intended for use with molecular descriptor or fingerprint matrices to identify dominant variance directions for dimensionality reduction, feature analysis, or as a preprocessing step for downstream ML models.
    
    Args:
        mat (numpy.ndarray): A numeric data matrix provided as a NumPy array. In typical RDKit workflows this will be an observations-by-features matrix of molecular descriptors or similar feature vectors. The function passes this array to FormCorrelationMatrix(mat) to produce a square correlation matrix. mat must contain numeric values suitable for correlation computation; passing non-numeric or incorrectly shaped arrays may raise TypeError or cause FormCorrelationMatrix to fail. The practical significance is that mat encodes the variables whose principal components (directions of maximal variance) are being sought.
        reverseOrder (bool = 1): A boolean flag indicating the ordering of the returned eigenvalues and eigenvectors. If truthy (the default value of 1 is treated as True), the eigenvalues and eigenvectors are sorted from largest to smallest eigenvalue (descending order), which is the common order for PCA when selecting leading components. If False, they are returned in ascending order. This parameter has no side effects beyond affecting output order.
    
    Returns:
        tuple: A tuple (eigenVals, eigenVects) where:
            eigenVals (numpy.ndarray): A one-dimensional NumPy array of real eigenvalues of the correlation matrix. Its length equals the dimension of the square correlation matrix produced from mat. The eigenvalues are ordered according to reverseOrder: descending when reverseOrder is True (default), ascending otherwise. The eigenvalues represent the variance explained along each principal component direction and are used to rank components by importance in dimensionality-reduction workflows.
            eigenVects (numpy.ndarray): A two-dimensional NumPy array whose rows correspond to the eigenvectors associated with the entries of eigenVals, in the same order. Each row is the eigenvector for the corresponding eigenvalue in eigenVals, and the number of rows (and columns) matches the dimension of the correlation matrix. These eigenvectors define the principal component directions in the original feature space and are used to project original data into the principal component coordinate system.
    
    Behavior, defaults, and failure modes:
        The function first constructs a correlation matrix by calling FormCorrelationMatrix(mat). It then computes eigenvalues and eigenvectors using numpy.linalg.eig on that correlation matrix. If the computed eigenvalues or eigenvectors have complex dtypes, the implementation attempts to extract their real components via getattr(..., "real", ...); this will silently discard imaginary parts if present, which may result in data loss or incorrect results when the correlation matrix yields genuinely complex eigenpairs (rare for real symmetric correlation matrices but possible if FormCorrelationMatrix returns a non-symmetric or numerically unstable matrix). The function sorts eigenpairs by eigenvalue magnitude using numpy.argsort and reverses the order when reverseOrder is truthy. No modification is made to the input mat. Errors raised by FormCorrelationMatrix or numpy.linalg.eig (for example numpy.linalg.LinAlgError for a non-square or ill-conditioned matrix) propagate to the caller. Numerical results may vary slightly across NumPy versions and platforms due to floating-point differences.
    
    Side effects:
        None on inputs. The function returns new NumPy arrays and does not modify mat in-place.
    """
    from rdkit.ML.Data.Stats import PrincipalComponents
    return PrincipalComponents(mat, reverseOrder)


################################################################################
# Source: rdkit.ML.Data.Stats.R2
# File: rdkit/ML/Data/Stats.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def rdkit_ML_Data_Stats_R2(orig: numpy.ndarray, residSum: float):
    """rdkit.ML.Data.Stats.R2 computes the coefficient of determination (R²) for a set of predictions relative to observed target values. In the RDKit machine-learning context (for example when evaluating regression models built from molecular descriptors or fingerprints), this function quantifies the fraction of variance in the observed values that is explained by the model: it implements the common formula R² = 1 - RSS / TSS where RSS is the provided residual sum and TSS is the total sum of squares of the observed values around their mean.
    
    Args:
        orig (numpy.ndarray): An array of observed (target) values from the dataset used to evaluate predictions. This function converts orig to a NumPy array internally and uses the length of the first axis (orig.shape[0]) as the sample count n. The observed values are used to compute the mean and the total sum of squared deviations from the mean (TSS). In RDKit ML workflows, orig typically contains experimental or reference property values (e.g., measured activity, property descriptors) against which model predictions are compared. The array must therefore represent the set of actual target values corresponding to the predictions whose residuals were accumulated into residSum.
        residSum (float): The residual sum of squares (RSS) for the same set of predictions and observed values, i.e., typically the sum over samples of (prediction - observed)^2. This scalar is treated as a float and is used as the numerator in the R² calculation via R² = 1 - residSum / oVar where oVar is the total sum of squared deviations of orig from its mean. In practical RDKit model-evaluation code, residSum is produced by summing squared residuals across the dataset.
    
    Returns:
        float: The computed R² value when orig contains one or more observations and the total sum of squares (TSS) computed from orig is non-zero. The value is computed as 1.0 - residSum / oVar where oVar = sum((orig - mean(orig))**2). This return value expresses the proportion of variance in orig explained by the model predictions used to produce residSum.
        tuple: (0.0, 0.0) is returned when orig has no observations (n <= 0). This legacy behavior originates from the implementation's early exit and yields a two-element tuple of floats rather than the usual single float; callers should handle this special-case return explicitly.
    
    Behavior and failure modes:
        - The function converts orig to a NumPy array and uses the first dimension length as the number of samples. If orig is not shaped as expected, the behavior will follow NumPy's array conversion and indexing semantics.
        - If orig has zero variance (all values equal), the computed denominator oVar will be zero. In that case the expression residSum / oVar is undefined and the function will either raise a division-by-zero error or produce an undefined/Infinity/NaN result depending on the underlying NumPy/Python scalar behavior. Callers should check for zero variance in orig before calling this function or handle exceptions/NaN/inf results.
        - residSum is expected to be the sum of squared residuals (non-negative for ordinary squared residuals). If a negative residSum is supplied (which is atypical), the returned R² can exceed 1.0; if residSum > oVar the returned R² will be negative. These are mathematically consistent outcomes of the formula but may indicate incorrect inputs.
        - No in-place modification of orig or residSum occurs; the only side effect is allocation of an internal NumPy array copy of orig.
        - Note that the implementation contains a legacy early-return of (0.0, 0.0) for empty input (n <= 0) which is inconsistent with the normal single-float return; users integrating this function into RDKit ML pipelines should handle that special case.
    """
    from rdkit.ML.Data.Stats import R2
    return R2(orig, residSum)


################################################################################
# Source: rdkit.ML.Data.Stats.StandardizeMatrix
# File: rdkit/ML/Data/Stats.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def rdkit_ML_Data_Stats_StandardizeMatrix(mat: numpy.ndarray):
    """StandardizeMatrix standardizes a numeric observation-by-feature matrix by subtracting the column-wise means and dividing by the column-wise sample standard deviations. This function is used in RDKit's machine-learning and descriptor-processing workflows (for example, preparing descriptor or fingerprint matrices for downstream modeling) to center each feature (column) to zero mean and scale it to unit variance so that features contribute comparably to distance- or variance-based algorithms.
    
    Args:
        mat (numpy.ndarray): A numeric 2D array representing a matrix of observations (rows, n objects) by features (columns, variables). The function treats len(mat) as the number of objects (nObjs) and computes column-wise means with sum(mat, 0). The array is expected to contain numeric values suitable for arithmetic operations (float or castable to float). Practical significance: mat is the feature matrix produced by descriptor/fingerprint calculators or other preprocessing steps in cheminformatics and ML pipelines; standardizing this matrix is commonly required before clustering, PCA, regression, or other statistical learning methods.
    
    Returns:
        numpy.ndarray: A new numpy.ndarray with the same shape as mat containing the standardized values (for each column j, (mat[:, j] - mean_j) / std_j). The returned array is the standardized matrix used for downstream machine-learning tasks.
    
    Behavior and side effects:
        The function computes avgs = sum(mat, 0) / float(nObjs) and subtracts these column means from mat in place (mat -= avgs). This means the input array mat is modified: after the call its columns have had their means removed. The function then computes sample standard deviations devs as sqrt(sum(mat * mat, 0) / float(nObjs - 1)) using the mean-subtracted mat and attempts a vectorized division newMat = mat / devs to produce the returned standardized matrix. If a Python OverflowError is raised during the vectorized division, the function falls back to a safe per-column loop: it allocates a numpy.zeros(mat.shape, 'd') array and fills each column i with mat[:, i] / devs[i] only when devs[i] != 0.0; columns with devs[i] == 0.0 remain zeros in the returned matrix. Note that because mat was mean-centered in place prior to this division, columns that were constant in the original mat become all zeros in mat after centering and produce zeros in the returned newMat when devs is zero.
    
    Failure modes and important constraints:
        If nObjs = len(mat) is less than 2, the computation of devs uses a denominator of (nObjs - 1) and will raise a ZeroDivisionError; therefore the function requires at least two rows/observations. Division by zero in the vectorized division (if devs contains zeros) may produce infinite values or runtime warnings depending on the numpy configuration; an OverflowError is specifically caught and triggers the per-column safe fallback, but other floating-point issues (such as producing inf or NaN) may not be caught. The function does not validate input dimensionality beyond using sum(mat, 0) and indexing by columns; callers should provide a numeric numpy.ndarray representing observations-by-features.
    """
    from rdkit.ML.Data.Stats import StandardizeMatrix
    return StandardizeMatrix(mat)


################################################################################
# Source: rdkit.ML.Data.Stats.TransformPoints
# File: rdkit/ML/Data/Stats.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def rdkit_ML_Data_Stats_TransformPoints(tFormMat: numpy.ndarray, pts: list):
    """rdkit.ML.Data.Stats.TransformPoints transforms a set of numeric point vectors by centering them on their centroid and applying a linear transformation matrix. This function is intended for machine-learning and cheminformatics workflows in RDKit (for example, operating on 2D/3D atomic coordinate sets used when computing descriptors or aligning molecular point sets) where a set of points must be zero-centered and multiplied by a transformation matrix.
    
    Args:
        tFormMat (numpy.ndarray): A numeric transformation matrix provided as a NumPy array. This matrix is used with numpy.dot to transform each centered point. It must be a 2-D numeric numpy.ndarray whose dimensions are compatible with the dimensionality of the point vectors in pts (i.e., shapes must allow numpy.dot(tFormMat, point) to succeed). The function does not modify tFormMat.
        pts (list): A sequence of point vectors provided as a Python list. Each element is expected to be a numeric vector (for example a 1-D numpy.ndarray) or pts may be a 2-D numeric array (a list of vectors or an array of shape (n_points, point_dim)). The function converts pts to a numpy.array internally, computes the centroid (mean of the points), subtracts that centroid from each point (zero-centers them), and then applies tFormMat to each centered point.
    
    Behavior and side effects:
        The function first converts pts to a NumPy array and computes the number of points nPts. It computes the centroid avgP as sum(pts) / nPts and subtracts avgP from each point so the point set is centered at the origin. It then computes the transformed points by applying numpy.dot(tFormMat, centered_point) for each point and returns those transformed vectors as a Python list. The original pts list and the original tFormMat are not modified by this function; all arithmetic is performed on internal NumPy arrays and the returned result is a new list of numpy.ndarray objects.
        If pts is provided as a 2-D numeric array, that is supported; if provided as a list of numpy arrays, those are accepted as well.
    
    Failure modes and exceptions:
        If pts is empty (nPts == 0), a division by zero occurs when computing the centroid and a ZeroDivisionError (or a related exception) will be raised. If pts cannot be converted to a numeric numpy.array (for example, contain non-numeric items) a numpy exception will be raised. If the shapes of tFormMat and the point vectors are incompatible for matrix multiplication, numpy.dot will raise an exception (commonly ValueError); callers must ensure dimensional compatibility before calling. Numeric errors from numpy (TypeError, ValueError, etc.) may propagate to the caller.
    
    Returns:
        list: A Python list of numpy.ndarray objects. The list length equals the number of input points. Each element is the transformed numeric vector resulting from numpy.dot(tFormMat, (point - centroid)). These returned arrays represent the input points first centered about their mean and then linearly transformed by tFormMat.
    """
    from rdkit.ML.Data.Stats import TransformPoints
    return TransformPoints(tFormMat, pts)


################################################################################
# Source: rdkit.ML.Descriptors.CompoundDescriptors.GetAllDescriptorNames
# File: rdkit/ML/Descriptors/CompoundDescriptors.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def rdkit_ML_Descriptors_CompoundDescriptors_GetAllDescriptorNames(
    db: str,
    tbl1: str,
    tbl2: str,
    user: str = "sysdba",
    password: str = "masterkey"
):
    """Gets possible descriptor names and their human-readable descriptions from a relational database for use with RDKit machine-learning descriptor workflows.
    
    This function connects to a database using rdkit.Dbase.DbConnection.DbConnect and queries two tables: tbl1, which is treated as the source of columns that represent descriptor values (the feature columns used in ML models), and tbl2, which is treated as a lookup table that contains descriptor metadata (columns named 'property' and 'notes' are assumed to exist in tbl2 and are used to form descriptor names and descriptions respectively). The returned values can be used by downstream RDKit descriptor- and fingerprint-based machine-learning pipelines to build feature lists and accompanying documentation for each descriptor.
    
    Args:
        db (str): The name or connection identifier of the database to use. This is passed verbatim to rdkit.Dbase.DbConnection.DbConnect to open a connection and identifies the database instance or file that stores the descriptor tables. If this value is incorrect or the database is unreachable, DbConnect will raise an exception and no descriptors will be returned.
        tbl1 (str): The name of the table from which descriptor value columns are read. The function calls conn.GetColumnNames(table=tbl1) to obtain a list of column names; these column names are treated as descriptor (feature) names for ML workflows. If tbl1 does not exist or the user lacks permission, the underlying database call will raise an error.
        tbl2 (str): The name of the table that stores metadata about descriptors. This function calls conn.GetColumns('property,notes', table=tbl2) and expects tbl2 to expose a 'property' column (descriptor identifier) and a 'notes' column (human-readable description). The entries read from tbl2 are transformed so that the property name is uppercased and paired with its notes to form descriptor metadata entries. If tbl2 lacks the expected columns or the query fails, the function will propagate the database error.
        user (str): The database user name used for authentication when opening the connection. Defaults to "sysdba", which is the conventional default in the RDKit example code; provide a different user name if your database requires it. Authentication failures will raise an exception from the DB connection layer.
        password (str): The password for database authentication. Defaults to "masterkey" in the RDKit example code; in production use, supply the appropriate password. Authentication failures will raise an exception from the DB connection layer.
    
    Returns:
        tuple: A 2-tuple (col_names, col_descriptors) where:
            col_names (list): A list of column names gathered from tbl1 (in the same order returned by conn.GetColumnNames) with additional descriptor names appended from the module-level iterable countOptions (see notes). These names represent the feature columns that downstream RDKit ML code will treat as descriptors.
            col_descriptors (list): A list of descriptor metadata entries. For rows obtained from tbl2, each entry is a tuple (PROPERTY_NAME, notes) where PROPERTY_NAME is the uppercased value of the 'property' column and notes is the corresponding 'notes' column value; additional entries appended from countOptions are included as provided (typically tuples of (name, description)). This list aligns with the combined set of names returned in col_names and is intended to supply human-readable descriptions alongside each descriptor.
    
    Behavior, side effects, defaults, and failure modes:
        - The function opens a database connection by calling rdkit.Dbase.DbConnection.DbConnect(db, user=user, password=password). This has the side effect of establishing a network/file handle and may raise exceptions on connection or authentication failure.
        - It calls conn.GetColumnNames(table=tbl1) to read descriptor column names from tbl1 and conn.GetColumns('property,notes', table=tbl2) to read metadata from tbl2. These calls rely on the RDKit database abstraction layer and will raise exceptions if tables or columns are missing or permission is denied.
        - Rows read from tbl2 are transformed so that the property name is converted to upper case (x[0].upper()) while preserving the notes value (x[1]) to form metadata tuples.
        - After reading from the database, the function iterates over a module-level iterable named countOptions and appends its entries to both the col_names and col_descriptors lists. The module must define countOptions as an iterable of (name, description) pairs prior to calling this function; if countOptions is missing or not iterable, a NameError or TypeError will be raised.
        - Default authentication values are user="sysdba" and password="masterkey", matching the original RDKit example usage, but these should be overridden in production environments.
        - The function does not close the database connection explicitly in this snippet; connection lifecycle (closing) is managed by the DbConnect/connection object or surrounding code. If the connection object requires explicit closing, callers should ensure proper resource cleanup to avoid leaks.
        - No input validation beyond what the underlying DB layer provides is performed; malformed table names or unexpected schema shapes will result in the underlying calls raising exceptions.
    
    Notes:
        - This function is used within RDKit workflows to enumerate available molecular descriptor features and to collect human-readable descriptions for documentation, model training, and feature selection.
        - It assumes tbl2 contains 'property' and 'notes' columns as stated above; these are mapped to (PROPERTY, notes) in the descriptor metadata.
        - The module-level countOptions iterable is expected to supply additional (name, description) descriptor entries to be appended to the returned lists. If you depend on countOptions, ensure it is defined and populated before calling this function.
    """
    from rdkit.ML.Descriptors.CompoundDescriptors import GetAllDescriptorNames
    return GetAllDescriptorNames(db, tbl1, tbl2, user, password)


################################################################################
# Source: rdkit.ML.Descriptors.Parser.AVG
# File: rdkit/ML/Descriptors/Parser.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def rdkit_ML_Descriptors_Parser_AVG(strArg: str, composList: list, atomDict: dict):
    """rdkit.ML.Descriptors.Parser.AVG calculates the average (mean) value of a molecular descriptor expression across a supplied composition vector. In the RDKit machine-learning and descriptor generation context, this function is used to convert per-atom descriptor expressions into a single scalar feature for a molecule by weighting per-atom descriptor values by atom counts from a composition list and returning the count-weighted mean. The function expects the descriptor expression to contain the placeholder token "DEADBEEF" which will be replaced by each atom symbol from the composition during evaluation.
    
    Args:
        strArg (str): A Python expression in string form that computes a descriptor value for a single atom when the placeholder token "DEADBEEF" is replaced by that atom's symbol. Practical usage in RDKit workflows is to provide expressions that reference atomic properties (for example, when generating per-atom descriptor values for machine-learning features). This string is evaluated via Python's eval for each atom occurrence; therefore the expression must be syntactically valid and use only names available in the evaluation context. The required placeholder "DEADBEEF" must appear in the string; otherwise the same expression will be evaluated repeatedly without per-atom substitution.
        composList (list): A composition vector provided as an iterable of 2-tuples (atom, count) or similar pairs, where the first element is an atom identifier (typically a string symbol) and the second element is the multiplicity/count for that atom in the composition. The function iterates over composList, substitutes the atom identifier into strArg, evaluates the resulting expression, multiplies the result by count, and accumulates the weighted sum. composList must therefore be structured so that each element can be unpacked into two values; otherwise a TypeError will be raised.
        atomDict (dict): An atomic dictionary giving additional per-atom data; included for API compatibility with descriptor-parsing workflows in RDKit where atom-related lookup tables are commonly passed. In this implementation atomDict is not referenced by the function (it is accepted but unused), so passing it has no effect on the computation here. Callers should still provide the dictionary when integrating this function into pipelines that expect the three-argument signature.
    
    Returns:
        float: The count-weighted mean descriptor value computed as the accumulated weighted descriptor sum divided by the total count of atoms (sum of the second elements in composList). This scalar is suitable as a single feature value in RDKit descriptor and fingerprint pipelines for machine learning.
    
    Behavior, side effects, defaults, and failure modes:
        The function performs no in-place modification of its inputs; it only reads composList and strArg. It repeatedly replaces the literal substring "DEADBEEF" in strArg with each atom identifier and evaluates the resulting Python expression with eval. Because eval executes arbitrary Python code, this poses a security risk if strArg comes from an untrusted source; callers must ensure the expression is safe. If the total atom count (the sum of all counts in composList) is zero, a ZeroDivisionError will be raised when computing the mean. Evaluation of the expression can raise standard Python exceptions such as SyntaxError, NameError, TypeError, or other runtime errors if the expression is invalid, references undefined names, or returns non-numeric results; such exceptions propagate to the caller. The atomDict parameter is ignored by this function implementation but retained for compatibility with other descriptor parser functions in RDKit.
    """
    from rdkit.ML.Descriptors.Parser import AVG
    return AVG(strArg, composList, atomDict)


################################################################################
# Source: rdkit.ML.Descriptors.Parser.CalcMultipleCompoundsDescriptor
# File: rdkit/ML/Descriptors/Parser.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def rdkit_ML_Descriptors_Parser_CalcMultipleCompoundsDescriptor(
    composVect: list,
    argVect: tuple,
    atomDict: dict,
    propDictList: list
):
    """Calculates the numeric value of a user-specified descriptor expression for a list of compounds using per-atom and per-compound descriptor data.
    
    Args:
        composVect (list): A list (vector) where each element is itself a sequence (list/tuple) that encodes the composition information for one compound. The elements and their meaning follow the composition format used by the related helper _CalcSingleCompoundDescriptor; composVect provides one composition entry per compound for which a descriptor value will be produced. composVect determines the number of output entries (the returned list length) and is not modified by this function.
        argVect (tuple): A tuple with three elements that together define how the final descriptor value is computed from atomic and compound-level data: (1) AtomicDescriptorNames: a sequence (list/tuple) of names for the atomic descriptors that will be referenced in the expression as $1, $2, etc.; (2) CompoundDsscriptorNames: a sequence (list/tuple) of names for the compound-level descriptors that will be referenced in the expression as $a, $b, etc.; (3) Expr: a string containing the expression to evaluate to obtain the final numeric descriptor for a compound. The function performs textual substitutions on Expr using the provided descriptor name lists and then builds a Python expression to be evaluated for each compound. If argVect does not contain exactly three elements or contains invalid types, preparation of the evaluation expression will fail and the function will return a result vector filled with the error sentinel (see Returns).
        atomDict (dict): A dictionary mapping atom identifiers (or other keys used by the composition representation) to per-atom descriptor dictionaries. Each atomic entry is itself a dictionary whose keys are atomic descriptor names (matching names in AtomicDescriptorNames) and whose values are numeric descriptor values for that atom. atomDict supplies the atomic-level numeric values used when substituting atomic variables ($1, $2, ...) into the expression. This argument is read-only; the function does not alter atomDict.
        propDictList (list): A list (vector) of per-compound property/descriptor dictionaries (one dictionary per compound). Each element is a dictionary (propDict) containing compound-level descriptors referenced by names in CompoundDsscriptorNames. propDictList must align with composVect by index: propDictList[i] provides the compound-level descriptors for the composition composVect[i]. If propDictList is shorter than composVect, an IndexError can occur when the function accesses propDictList[i] (this error is not caught by the per-compound evaluation try/except and will propagate).
    
    Returns:
        list: A list of numeric descriptor values with the same length as composVect. For each index i the returned value is the numeric result of evaluating the prepared expression for the compound described by composVect[i] using atomDict and propDictList[i]. If an exception occurs during the global preparation phase (textual substitutions, method resolution using helper functions such as _SubForCompoundDescriptors, _SubForAtomicVars, or _SubMethodArgs, or if required globals such as knownMethods are missing), the function returns immediately a list of length len(composVect) filled with the sentinel value -666. If evaluation of the expression for an individual compound raises an exception, that compound's entry in the returned list will be -666 while other entries may still contain valid computed values.
    
    Behavior and failure modes:
        This function first constructs an evaluable Python expression by substituting atomic and compound variable tokens in the provided Expr string using internal helper functions (_SubForCompoundDescriptors, _SubForAtomicVars, _SubMethodArgs) and by resolving any known methods via the global knownMethods mapping. If that preparation step fails for any reason (invalid argVect contents, missing helper functions, or substitution errors), the function returns a list populated entirely with -666 sentinels. After successful preparation it iterates over composVect, sets up per-compound local variables (propDict and compos) from propDictList and composVect, and uses Python's eval() to compute the numeric result for each compound. Any exception raised by eval() for a specific compound is caught and results in -666 for that compound. Note that access to propDictList[i] is performed before the eval try/except; therefore mismatched lengths between composVect and propDictList or indexing errors will raise exceptions that are not handled by the per-compound eval except and will propagate out of the function. The function uses Python eval() and thus will execute code present in the prepared expression; callers must ensure expressions are safe and trusted. The function has no other side effects beyond reading the provided arguments and any referenced module-level helpers.
    """
    from rdkit.ML.Descriptors.Parser import CalcMultipleCompoundsDescriptor
    return CalcMultipleCompoundsDescriptor(composVect, argVect, atomDict, propDictList)


################################################################################
# Source: rdkit.ML.Descriptors.Parser.CalcSingleCompoundDescriptor
# File: rdkit/ML/Descriptors/Parser.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def rdkit_ML_Descriptors_Parser_CalcSingleCompoundDescriptor(
    compos: tuple,
    argVect: tuple,
    atomDict: dict,
    propDict: list
):
    """CalcSingleCompoundDescriptor calculates the descriptor value for a single compound by substituting atomic and compound variables into a user-provided expression and evaluating that expression. This function is part of RDKit's ML.Descriptors.Parser pipeline used to generate numeric descriptors for machine-learning workflows (see RDKit descriptors documentation). It expects a composition description, a descriptor specification tuple, a dictionary of per-atom descriptor dictionaries, and a list of per-compound properties, and it returns the evaluated numeric descriptor value for that composition.
    
    Args:
        compos (tuple): A composition vector/tuple that encodes the compound composition. Each entry should be a pair (element_identifier, amount) as in the original usage example '[("Fe",1.),("Pt",2.),("Rh",0.02)]'. In the RDKit ML descriptor domain this provides the stoichiometric or fractional amounts of constituent species used to combine atomic descriptors into a compound-level descriptor.
        argVect (tuple): A three-element tuple that specifies how to build and evaluate the descriptor expression. Element 0 is AtomicDescriptorNames: a list/tuple of atomic descriptor names that define the meaning of placeholders like $1, $2 etc. Element 1 is CompoundDescriptorNames: a list/tuple of compound-level descriptor names that define the meaning of placeholders like $a, $b etc. Element 2 is Expr: a string containing the expression to evaluate after substitutions. The function uses these three components to perform ordered substitutions (_SubForCompoundDescriptors, _SubForAtomicVars, _SubMethodArgs in the source) to produce a Python expression that is then evaluated to compute the descriptor.
        atomDict (dict): A dictionary mapping atomic identifiers (e.g., element symbols or atom labels used in compos) to dictionaries of atomic descriptor values. Each value is itself a dict whose keys are atomic descriptor names (as listed in argVect[0]) and whose values are the numeric atomic descriptor values used when assembling the compound descriptor. In practice this supplies the per-atom numeric inputs required by the expression produced from argVect.
        propDict (list): A list of descriptors/properties for the composition (compound-level properties). These are the values referenced by the compound descriptor names given in argVect[1] and are substituted into the expression under the name 'propDict' during processing. In RDKit usage this contains precomputed composition properties that the expression may combine with atomic descriptor aggregates to produce the final value.
    
    Returns:
        The numeric value of the computed descriptor for the given composition when evaluation succeeds. If an error occurs during the substitution or evaluation phases, the function returns the sentinel value -666 to indicate failure. When the RDKit debugging flag (__DEBUG) is enabled, substitution failures raise a RuntimeError('Failure 1') and evaluation failures raise RuntimeError('Failure 2') instead of returning -666; in debug mode additional diagnostic output is emitted (tracebacks printed, and evaluation failures append diagnostic information to the file RDConfig.RDCodeDir + '/ml/descriptors/log.txt'). Note that the function constructs and evaluates a Python expression (via eval) after performing textual substitutions, so malformed expressions, missing descriptor names in atomDict/propDict, or runtime errors in the evaluated expression will trigger the described failure behavior.
    """
    from rdkit.ML.Descriptors.Parser import CalcSingleCompoundDescriptor
    return CalcSingleCompoundDescriptor(compos, argVect, atomDict, propDict)


################################################################################
# Source: rdkit.ML.Descriptors.Parser.DEV
# File: rdkit/ML/Descriptors/Parser.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def rdkit_ML_Descriptors_Parser_DEV(strArg: str, composList: list, atomDict: dict):
    """rdkit.ML.Descriptors.Parser.DEV: calculate the composition-weighted mean absolute deviation of a descriptor expression across a molecular composition.
    
    This function computes the average deviation (mean absolute deviation) of a descriptor value defined by a Python expression string across a composition vector. It is intended for use in cheminformatics descriptor generation (RDKit descriptors and fingerprint features for machine learning). The routine first computes the composition-weighted mean by calling MEAN(strArg, composList, atomDict) and then computes the sum over composition entries of the absolute difference between each per-atom descriptor value and that mean, weighted by the atom counts, divided by the total number of atoms in the composition. The descriptor expression is evaluated with Python's eval after substituting the placeholder token 'DEADBEEF' with an atomic symbol from the composition; therefore the expression must evaluate to a numeric value for each atom symbol.
    
    Args:
        strArg (str): A Python expression in string form that evaluates to a numeric descriptor value for a single atomic symbol. The expression must include the literal placeholder 'DEADBEEF' where the atomic symbol should be inserted (for example, "atomic_weight('DEADBEEF') * 1.0" or "someDescriptor(DEADBEEF)"), so that the code replaces 'DEADBEEF' with an atom symbol from composList before evaluation. The expression will be evaluated using eval() in the current Python execution environment; any names or functions referenced by the expression must be available in that environment. This function calls MEAN(strArg, composList, atomDict) internally, so strArg must also be acceptable to MEAN.
        composList (list): A composition vector describing the composition to average over. Each element is expected to be a pair (atom, num) where atom is an atomic identifier (commonly a string atomic symbol such as 'C', 'H', 'O') and num is the count/weight of that atom in the composition (an int or numeric value). The function iterates over composList, replacing 'DEADBEEF' with atom and weighting contributions by num. The total weight is the sum of the num values.
        atomDict (dict): An atomic dictionary passed through the descriptor calculation pipeline. This dictionary typically maps atomic identifiers (keys) to atomic properties or descriptor lookup data used by descriptor expressions or helper functions. Although DEV does not directly index atomDict, it is provided to satisfy the common interface and is forwarded to MEAN(strArg, composList, atomDict) which may require it. The contents and structure of atomDict must be appropriate for the descriptor expressions and for MEAN.
    
    Returns:
        float: The composition-weighted mean absolute deviation of the descriptor expression across composList. Computed as (sum_over_atoms |value(atom) - mean| * count(atom)) / total_count. The return is a Python float.
    
    Behavior, side effects, defaults, and failure modes:
        - The function calls MEAN(strArg, composList, atomDict) to obtain the composition-weighted mean. MEAN must be available in the same execution context and accept the same arguments.
        - For each (atom, num) in composList, the function constructs tStr by replacing every occurrence of the literal substring 'DEADBEEF' in strArg with the atom value and then evaluates tStr with eval(tStr). The evaluation must produce a numeric result; non-numeric results will lead to TypeError or other runtime errors during the subtraction and absolute-value operations.
        - The function multiplies the absolute deviation by num and accumulates these weighted deviations. It then divides by the total of num values (nSoFar) to produce the mean absolute deviation.
        - If the total count nSoFar is zero (for example, composList is empty or all counts are zero), a ZeroDivisionError will be raised when dividing by nSoFar.
        - Because the function uses eval(), it can execute arbitrary Python code present in strArg after replacement. This can be a security risk if strArg comes from untrusted sources. Ensure that strArg is validated or that eval is used in a trusted context.
        - The function does not modify composList or atomDict; its observable effect is the numeric return value only.
        - Any exceptions raised by MEAN, eval, or arithmetic operations (NameError, SyntaxError, ZeroDivisionError, TypeError, ValueError, etc.) propagate to the caller. Ensure that the environment provides any names or functions referenced in strArg and that composList contains valid (atom, num) pairs.
    """
    from rdkit.ML.Descriptors.Parser import DEV
    return DEV(strArg, composList, atomDict)


################################################################################
# Source: rdkit.ML.Descriptors.Parser.HAS
# File: rdkit/ML/Descriptors/Parser.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def rdkit_ML_Descriptors_Parser_HAS(strArg: str, composList: list, atomDict: dict):
    """rdkit.ML.Descriptors.Parser.HAS determines whether a parsed, atom-substituted expression is present in a composition vector used by RDKit descriptor parsing and machine-learning descriptor generation.
    
    Args:
        strArg (str): A comma-separated string containing two Python expressions used by the parser. The first expression is a Python expression that must include the literal placeholder "DEADBEEF" and, after replacing that placeholder with an atom symbol from composList, should evaluate (via eval) to a container (for example a list, set, string, or other object that supports membership testing). The second expression is a Python expression that is evaluated (via eval) and whose value is tested for membership in the container produced by the first expression. This string form is the form emitted/consumed by the RDKit descriptor parser when evaluating "HAS"-style conditions in descriptor specification strings.
        composList (list): The composition vector for a molecule or fragment, supplied as a list of 2-tuples (atom, count) or similar iterable of (atom, multiplicity) pairs. The function iterates over this list and substitutes each atom (the first element of each tuple) into the first expression in strArg by replacing the "DEADBEEF" placeholder. This argument represents the parsed atomic composition used during descriptor calculation and machine-learning feature generation.
        atomDict (dict): An atomic dictionary provided by the caller (for example mapping atom symbols to properties) that is part of the descriptor parser API. The current implementation does not reference this dictionary; it is accepted for API compatibility with other parser functions and may be used by callers or future implementations.
    
    Returns:
        int: Function returns one of three integers indicating the outcome of the membership test performed across the composition vector:
            1: At least one atom substitution produced a container in which the evaluated second expression was found (successful hit). In the RDKit descriptor parsing domain this indicates the "HAS" condition is satisfied for the molecule/composition and can influence inclusion of a descriptor or a boolean feature used in machine-learning pipelines.
            0: No atom substitution produced a container containing the evaluated second expression (no hit). This indicates the "HAS" condition is not satisfied.
            -666: The input strArg did not contain a comma-separated pair of expressions (split produced a single part) and therefore cannot be evaluated; this is a sentinel error code used by the parser to indicate malformed input.
    
    Behavior and side effects:
        The function splits strArg on a comma. If the split yields at least two parts, the first part is treated as a container-expression template and the second part as a target-expression. For each (atom, _) entry in composList the literal substring "DEADBEEF" in the first expression is replaced by the atom string and both expressions are evaluated using Python's eval(). The result of the second expression is checked for membership in the result of the first expression (using the "in" operator). If any substitution yields a positive membership test the function returns 1 immediately; if none do it returns 0. If strArg cannot be split into two expressions the function returns -666 and performs no evaluations.
        The function has no persistent side effects on global state or its input arguments. However, it executes arbitrary Python code via eval(), so evaluating untrusted or user-supplied strArg may execute arbitrary code and create side effects external to the function. Callers in cheminformatics or ML pipelines (as in RDKit) must ensure that strArg is trusted or sanitized before calling this function.
    
    Failure modes and exceptions:
        If the expressions produced after splitting and substitution are syntactically invalid, or evaluation raises exceptions (NameError, SyntaxError, TypeError, etc.), those exceptions propagate to the caller. The function does not catch eval-related exceptions internally. If composList does not contain 2-tuples but another iterable of atoms, the function only uses the first element of each item as the atom string; mismatched shapes may cause exceptions during iteration or substitution. The special return value -666 signals malformed strArg (no comma-separated pair) rather than raising an exception.
    """
    from rdkit.ML.Descriptors.Parser import HAS
    return HAS(strArg, composList, atomDict)


################################################################################
# Source: rdkit.ML.Descriptors.Parser.MAX
# File: rdkit/ML/Descriptors/Parser.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def rdkit_ML_Descriptors_Parser_MAX(strArg: str, composList: list, atomDict: dict):
    """rdkit.ML.Descriptors.Parser.MAX computes the maximum value of a per-atom descriptor expression across a molecular composition. This function is used by RDKit descriptor parsing machinery to evaluate a textual descriptor expression for each atom in a supplied composition and return the largest numeric result, which is useful when generating descriptors or features for machine-learning workflows that require an aggregate (maximum) per-molecule value.
    
    Args:
        strArg (str): A Python expression given as a string that defines the per-atom descriptor to evaluate. The expression must include the literal placeholder string "DEADBEEF" which will be replaced, for each element of composList, by the atom symbol (for example "C", "O", "N"). After substitution the resulting string is evaluated with Python's eval() and is expected to produce a numeric value (float-compatible). This argument is central to how different atomic properties or descriptor formulas are specified in the RDKit descriptor parser.
        composList (list): A list representing the composition to iterate over. Based on the implementation this should be an iterable of 2-tuples where each element unpacks as (atom_symbol, multiplicity) — e.g. ("C", 3). Only the first element of each tuple (the atom symbol) is used by this function. The list defines which atom symbols will be substituted into strArg and evaluated; it therefore determines the set of values from which the maximum is taken.
        atomDict (dict): A dictionary mapping atom symbols to atom-specific data (for example atomic properties or cached values). Although accepted to match the parser API used in RDKit descriptor generation, this particular implementation does not read from atomDict; it is present for API compatibility with other parser functions and higher-level descriptor infrastructure that supply an atomic dictionary.
    
    Returns:
        float: The maximum numeric value obtained by evaluating the expression (strArg with "DEADBEEF" replaced by each atom symbol) for every entry in composList. The return value is intended to be a scalar descriptor used in downstream descriptor vectors or ML feature arrays.
    
    Behavior, side effects, defaults, and failure modes:
        The function constructs a string for each atom by replacing the exact substring "DEADBEEF" in strArg with the atom symbol from composList and then evaluates that string using Python's eval(). There are no persistent side effects (no modification of inputs), but use of eval() means arbitrary code in strArg will be executed in the current Python environment; therefore strArg must be treated as untrusted input only if the caller controls it. Expected behavior requires that composList is non-empty; if composList is empty, calling max() on the accumulated results will raise a ValueError. If any element of composList cannot be unpacked into two values (atom, multiplicity), a TypeError will be raised by the unpacking in the loop. If the substituted expression is syntactically invalid or references undefined names, eval() may raise SyntaxError, NameError, or other exceptions. If eval() yields non-numeric values, the semantics of max() apply but the result may not be a float; callers should ensure the expression produces numeric (float-compatible) outputs to match the documented return type. This implementation also ignores atomDict, so any intent to use atomic data from atomDict must be incorporated into strArg via a scope accessible to eval().
    """
    from rdkit.ML.Descriptors.Parser import MAX
    return MAX(strArg, composList, atomDict)


################################################################################
# Source: rdkit.ML.Descriptors.Parser.MIN
# File: rdkit/ML/Descriptors/Parser.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def rdkit_ML_Descriptors_Parser_MIN(strArg: str, composList: list, atomDict: dict):
    """rdkit.ML.Descriptors.Parser.MIN calculates the minimum numeric value of a descriptor expression across the atoms present in a molecular composition. It is used in the RDKit descriptors/fingerprint generation context to evaluate a per-atom expression for each entry in a composition vector and return the smallest resulting numeric value; this can serve as a feature or descriptor value for machine-learning workflows described in the RDKit README.
    
    This function evaluates a Python expression provided as a string for each atom in composList. The expression must include the literal placeholder DEADBEEF which will be replaced by the atom symbol (string) from composList before evaluation. The evaluated results are collected and the minimum of those results is returned. The function relies on Python's eval() to compute the expression, so the expression may reference names available in the current evaluation environment (for example, atomDict if the expression uses that name), and the expression may perform arbitrary computation permitted by eval.
    
    Args:
        strArg (str): A Python expression in string form that uses the placeholder DEADBEEF to indicate where an atom symbol should be inserted. Practical usage: for per-atom property lookup or computation in descriptor generation you might write expressions that reference atomic properties (for example, "atomDict['DEADBEEF']['someProp'] * 2") and pass that as strArg. The function will replace every occurrence of the exact substring DEADBEEF with the atom symbol from composList before calling eval on the resulting string. The expression must be valid Python and must evaluate to a numeric value for each atom to produce meaningful descriptor output. Using untrusted input here is unsafe because eval will execute arbitrary code.
        composList (list): A composition vector representing the atoms in the molecule; in code this is iterated as for atom, _ in composList, so each element is expected to be a two-item sequence where the first item is the atom symbol (string) and the second item is typically a count or multiplicity (ignored by this function). Practical significance: the atom symbols provided here are substituted into strArg to compute per-atom descriptor values which are then compared to produce the minimum.
        atomDict (dict): An atomic dictionary mapping atom symbols (strings) to atomic data (for example, property dictionaries). Although the current implementation of MIN does not directly index atomDict itself, it is passed to this function and is available to any expression evaluated via strArg if the expression refers to the name atomDict. In RDKit workflows this dictionary commonly contains per-element properties used by descriptor expressions.
    
    Behavior and side effects:
        For each element of composList the function extracts the atom symbol (the first component of the element), substitutes that symbol for every occurrence of DEADBEEF in strArg, evaluates the resulting expression with Python's eval(), and appends the result to an internal accumulator. After all atoms are processed, the function returns the minimum value from the accumulator. There are no other side effects performed by the function itself, but the evaluated expression may have side effects (it can call functions or mutate accessible objects) because eval executes arbitrary code. No conversion of the evaluated values to a specific numeric type is performed by the function; callers should ensure expressions yield numeric results compatible with downstream usage in RDKit descriptor and machine-learning pipelines.
    
    Failure modes and important notes:
        If composList is empty, min(accum) will raise a ValueError because there are no values to compare. If any substituted expression is not valid Python or raises an exception during eval (NameError, KeyError, TypeError, ZeroDivisionError, etc.), that exception will propagate to the caller. If the expressions do not evaluate to comparable numeric types, the result of min() may be unexpected or may raise a TypeError. Because eval() is used, passing untrusted or user-supplied strings for strArg is a security risk and should be avoided or sandboxed in production contexts.
    
    Returns:
        float: The minimum numeric value produced by evaluating the provided expression for each atom in composList. This value represents the smallest per-atom descriptor computed across the composition and is intended for use as a descriptor feature in RDKit's machine-learning and descriptor-generation workflows.
    """
    from rdkit.ML.Descriptors.Parser import MIN
    return MIN(strArg, composList, atomDict)


################################################################################
# Source: rdkit.ML.Descriptors.Parser.SUM
# File: rdkit/ML/Descriptors/Parser.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def rdkit_ML_Descriptors_Parser_SUM(strArg: str, composList: list, atomDict: dict):
    """rdkit.ML.Descriptors.Parser.SUM: Compute the weighted sum of an evaluated per-atom descriptor expression over a composition vector used in RDKit descriptor generation and machine-learning feature preparation.
    
    Args:
        strArg (str): A string expression representing an atomic descriptor formula in which the literal token 'DEADBEEF' is used as a placeholder for an atom symbol. In practice, callers construct strArg from descriptor definitions so that replacing 'DEADBEEF' with an atom symbol (e.g., "C", "O", "N") yields a valid Python expression that evaluates to a numeric value for that atom. This string is evaluated with Python's eval() for each atom in composList; therefore strArg must be a syntactically correct expression that returns a numeric value when evaluated for each substituted atom. Because eval() is used, strArg must be trusted or sanitized to avoid security risks.
        composList (list): A list of two-element tuples (atom, num) representing the composition vector of a molecule or fragment. atom is expected to be a string atomic identifier that will replace the 'DEADBEEF' token in strArg. num is a numeric weight (integer or float) indicating how many times that atom contributes (for example, atom counts in a molecular formula). The function iterates over composList and multiplies the evaluated per-atom descriptor value by num to produce a contribution to the total. If composList contains entries that are not two-element (atom, num) pairs or if num is not numeric, a runtime exception (TypeError, ValueError) or incorrect result may occur.
        atomDict (dict): A dictionary mapping atom identifiers to atom-specific data (for example, atomic properties or precomputed values). Although this parameter is part of the function signature for API compatibility with other parser functions in the RDKit descriptors framework, the current implementation does not read from atomDict. Callers may supply the atomic dictionary for consistency with higher-level code; it will not be modified by this function. If future versions of the parser change behavior, atomDict is available for lookups.
    
    Behavior and side effects:
        The function initializes an accumulator to 0.0, then for each (atom, num) pair in composList replaces the substring 'DEADBEEF' in strArg with the atom identifier and evaluates the resulting string using eval(). The evaluated numeric result is multiplied by num and added to the accumulator. The function returns the final accumulator as a float. There are no other side effects: inputs are not mutated. However, using eval() means the evaluated expression can access names in the calling environment and execute arbitrary code; therefore strArg must come from a trusted source. Common failure modes include SyntaxError or NameError from eval() if the substituted expression is invalid, TypeError/ValueError if numerical operations fail, or unexpected results if composList contents are malformed.
    
    Returns:
        float: The numeric sum of the per-atom evaluated descriptor values multiplied by their corresponding counts/weights from composList. This value is suitable for use as a scalar molecular descriptor in downstream RDKit descriptor lists or machine-learning feature vectors.
    """
    from rdkit.ML.Descriptors.Parser import SUM
    return SUM(strArg, composList, atomDict)


################################################################################
# Source: rdkit.ML.InfoTheory.BitRank.AnalyzeSparseVects
# File: rdkit/ML/InfoTheory/BitRank.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def rdkit_ML_InfoTheory_BitRank_AnalyzeSparseVects(bitVects: list, actVals: list):
    """AnalyzeSparseVects computes information gains for sparse bit-vector features (SBVs) relative to binary activity labels, returning a per-bit ranking and the list of gain values. This function is used in the RDKit cheminformatics/ML context to score fingerprint bits (sparse bit vectors commonly produced for molecular descriptors) by how informative each bit is for predicting a binary activity, making it useful for feature selection and bit ranking in classification tasks.
    
    Args:
        bitVects (list): A sequence (list) of sparse bit-vector objects (SBVs) representing molecular fingerprint features. Each element is expected to implement GetSize() to report the total number of possible bit positions and GetOnBits() to return an iterable of integer bit indices that are set in that vector. The order of elements in bitVects must correspond exactly to the order of labels in actVals. These SBVs are the inputs whose individual bit positions are evaluated for information content with respect to the binary activities.
        actVals (list): A sequence (list) of binary activity values (e.g., booleans or integers interpreted by truthiness) with the same length and sample ordering as bitVects. Each entry indicates whether the corresponding sample (fingerprint) is active/positive (truthy) or inactive/negative (falsy). These labels are used to count occurrences of each bit among active and inactive samples and build 2x2 contingency tables for information-gain calculation.
    
    Returns:
        tuple: A pair (res, gains) describing per-bit statistics and scores.
            res (list): A list of tuples for bits that appear in at least one input vector. Each tuple has the form (bit_index, gain, nAct, nInact) where bit_index (int) is the bit position, gain (float) is the information gain computed for that bit using a 2x2 contingency table, nAct (int) is the number of active samples that have that bit set, and nInact (int) is the number of inactive samples that have that bit set. The order of entries in res corresponds to the order in which bits are iterated (increasing bit index).
            gains (list): A list of float information-gain values corresponding to the same bits reported in res, in the same order. This list can be used directly for ranking or statistical analysis of feature importance.
    
    Behavior, side effects, and failure modes:
        - The function counts, for each bit position across all provided SBVs, how many active and inactive samples have that bit set. It constructs a 2x2 contingency table for each bit and computes information gain via entropy.InfoGain(resTbl), returning both the detailed tuple list and the raw gains list.
        - Inputs are not modified; internal temporary arrays (numpy integer arrays) are used for counting.
        - The function requires that len(bitVects) == len(actVals); if they differ a ValueError is raised with message 'var and activity lists should be the same length'.
        - If bitVects is empty, or if elements of bitVects do not implement GetSize() and GetOnBits(), the function will raise AttributeError or IndexError as appropriate; callers must supply valid SBV-like objects.
        - actVals entries are interpreted by Python truthiness (truthy values count as active); non-binary values are not explicitly validated and may lead to misleading counts if they are neither clearly truthy nor falsy.
        - The number of bits nBits is taken from bitVects[0].GetSize(); all SBVs are expected to be compatible with that bit-space size. If a provided SBV contains bit indices outside [0, nBits-1], behavior is undefined.
        - The information-gain computation relies on the entropy.InfoGain function; any exceptions raised by that call (e.g., due to invalid contingency tables) will propagate to the caller.
    
    Practical significance in RDKit/cheminformatics:
        - Use this function when you have a set of RDKit fingerprint-like sparse bit vectors for a dataset of molecules and binary activity labels, and you want to identify or rank fingerprint bits that are most informative for distinguishing active vs inactive molecules. The output provides both per-bit counts and information-gain scores commonly used in feature selection and interpretable ML on chemical datasets.
    """
    from rdkit.ML.InfoTheory.BitRank import AnalyzeSparseVects
    return AnalyzeSparseVects(bitVects, actVals)


################################################################################
# Source: rdkit.ML.InfoTheory.BitRank.CalcInfoGains
# File: rdkit/ML/InfoTheory/BitRank.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def rdkit_ML_InfoTheory_BitRank_CalcInfoGains(
    bitVects: list,
    actVals: list,
    nPossibleActs: int,
    nPossibleBitVals: int = 2
):
    """Calculates the information gain for each bit (feature) across a set of samples with associated activity values.
    
    This function is part of RDKit's ML.InfoTheory.BitRank utilities and is used in cheminformatics and machine-learning workflows (for example, QSAR and classification on molecular fingerprints) to quantify how much information each fingerprint bit provides about the activity labels. For each bit position shared by the IntVector-like entries in bitVects, the function computes a single numeric information-gain score that can be used for feature ranking or selection.
    
    Args:
        bitVects (list): A list (sequence) of IntVector-like objects representing per-sample bit vectors (for example binary or small-integer fingerprint features for molecules). Each element corresponds to one sample; all elements are expected to have the same length (number of bits). The function reads bit values by index to form conditional counts used to compute information gain.
        actVals (list): A list (sequence) of activity values corresponding to the samples in bitVects. There must be exactly one activity value per element in bitVects and the lists must be the same length. actVals holds discrete activity labels (e.g., class indices) used to compute class-conditional counts for entropy-based information gain.
        nPossibleActs (int): The integer number of distinct possible activity values (the number of classes). This tells the internal counting routines how many activity categories to allocate and is required for correct computation of per-class counts used by the entropy/info-gain calculation.
        nPossibleBitVals (int): Optional; integer maximum number of distinct values any bit may take. Default is 2, which is the common case for binary fingerprint bits. If bits can take more than two values (e.g., small integer counts), set this to that maximum to ensure correct counting and entropy evaluation.
    
    Returns:
        list of floats: A sequence of floating-point information-gain scores, one per bit position. The returned sequence has length equal to the number of bits in each IntVector in bitVects (nBits). Each value is a non-negative float quantifying the reduction in uncertainty about actVals obtained by knowing the bit value; higher values indicate greater discriminatory power of that bit for the provided activity labels.
    
    Behavior, side effects, defaults, and failure modes:
        - The function first checks that len(bitVects) == len(actVals); if not, it raises ValueError('var and activity lists should be the same length').
        - The number of bits (nBits) is inferred from len(bitVects[0]); therefore bitVects must be non-empty and its first element must expose a length equal to the expected number of bits. If bitVects is empty or elements are not indexable in the expected way, an IndexError or TypeError may be raised by the implementation.
        - For each bit index, per-bit class/value counts are computed by calling FormCounts(...) with the supplied nPossibleActs and nPossibleBitVals; these counts are then converted to an information-gain score using entropy.InfoGain(...). Errors propagated from FormCounts or entropy.InfoGain (for example due to invalid types or inconsistent values) are not caught and will propagate to the caller.
        - nPossibleBitVals defaults to 2 to support binary fingerprint bits; callers must set it explicitly when bits can take a larger range of integer values.
        - There are no external side effects: the function does not modify bitVects or actVals; it returns computed scores. The implementation uses NumPy internally for accumulation, but the documented return is a sequence of floats suitable for feature ranking in downstream ML code.
    """
    from rdkit.ML.InfoTheory.BitRank import CalcInfoGains
    return CalcInfoGains(bitVects, actVals, nPossibleActs, nPossibleBitVals)


################################################################################
# Source: rdkit.ML.InfoTheory.BitRank.FormCounts
# File: rdkit/ML/InfoTheory/BitRank.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def rdkit_ML_InfoTheory_BitRank_FormCounts(
    bitVects: list,
    actVals: list,
    whichBit: int,
    nPossibleActs: int,
    nPossibleBitVals: int = 2
):
    """rdkit.ML.InfoTheory.BitRank.FormCounts generates a counts (contingency) matrix for a single fingerprint bit across activity classes; it is used by RDKit's machine-learning/InfoTheory/BitRank utilities to summarize how many occurrences of each bit value co-occur with each activity value when analyzing fingerprint-like integer vectors (IntVectors) for descriptor and fingerprint-based models.
    
    This function constructs a 2-D counts matrix with shape (nPossibleBitVals, nPossibleActs) where rows correspond to possible values of the selected bit and columns correspond to possible activity classes. The resulting matrix is typically used for information-theoretic scoring or ranking of fingerprint bits in cheminformatics and machine-learning workflows within RDKit (for example, to compute mutual information or other statistics between bit presence/values and activity labels).
    
    Args:
        bitVects (list): a sequence containing IntVectors (RDKit-style integer-indexable vectors representing fingerprint bit values). Each element provides integer values for bits; the function reads the element at index whichBit from each vector to determine the bit value for that sample. The length of this sequence must equal the length of actVals; otherwise a ValueError is raised. Elements must support integer indexing and return integers suitable as indices into the output row dimension.
        actVals (list): a sequence of integer activity labels (class indices) aligned with bitVects such that actVals[i] is the activity value for the sample represented by bitVects[i]. Values are used as indices into the output column dimension and must lie in the range [0, nPossibleActs-1]; values outside that range will raise an IndexError at runtime.
        whichBit (int): the integer index of the bit to examine within each IntVector in bitVects. This selects which position to read from each vector; if this index is out of range for an element in bitVects an IndexError will be raised.
        nPossibleActs (int): the integer number of possible activity values (the number of columns in the returned counts matrix). This defines the second dimension of the output and must be consistent with the range of integers found in actVals.
        nPossibleBitVals (int = 2): optional integer specifying the maximum number of distinct bit values to account for (the number of rows in the returned counts matrix). By default this is 2, which is appropriate for binary fingerprint bits (0/1). If bitVects contain values outside the range [0, nPossibleBitVals-1], an IndexError will occur when incrementing the corresponding count.
    
    Returns:
        Numeric array: a Numeric (numpy) 2-D array of integers with shape (nPossibleBitVals, nPossibleActs) where entry [b,a] is the count of samples whose bit at whichBit has value b and whose activity value is a. The array is newly allocated and returned; the function has no side effects on its inputs.
    
    Raises and failure modes: The function raises ValueError if bitVects and actVals have different lengths. It may raise IndexError if whichBit is out of range for an element of bitVects or if values read from bitVects or actVals are outside the ranges implied by nPossibleBitVals or nPossibleActs. Type-related errors may arise if elements of bitVects are not indexable integer containers or if actVals elements are not integers. This function is intended primarily for internal use within RDKit's InfoTheory/BitRank tooling for fingerprint analysis.
    """
    from rdkit.ML.InfoTheory.BitRank import FormCounts
    return FormCounts(bitVects, actVals, whichBit, nPossibleActs, nPossibleBitVals)


################################################################################
# Source: rdkit.ML.InfoTheory.entropy.PyInfoEntropy
# File: rdkit/ML/InfoTheory/entropy.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def rdkit_ML_InfoTheory_entropy_PyInfoEntropy(results: numpy.ndarray):
    """rdkit.ML.InfoTheory.entropy.PyInfoEntropy computes the information-theoretic (Shannon) entropy in bits for a discrete empirical distribution defined by observed counts. This function is part of RDKit's machine-learning/info-theory utilities and is used to quantify the uncertainty or information content of a categorical outcome distribution derived from observed frequencies (for example, when evaluating the distribution of fingerprint hits, descriptor value categories, or class label counts in cheminformatics ML tasks).
    
    Args:
        results (numpy.ndarray): A 1D numeric array containing the observed counts for each possible outcome of a discrete variable. Each element is the number of times that outcome was observed (for example, for three possible outcomes observed 5, 6 and 1 times, results would be numpy.array([5, 6, 1])). The array must be numeric (integer or float); elements are expected to be non-negative counts. The function treats this array as raw counts (not probabilities) and normalizes by the sum of its elements to form the empirical probability distribution. If the array contains zeros for some outcomes, those zero probabilities are handled safely (the implementation avoids taking log(0) by substituting a safe value when computing the logarithm while preserving the final contribution of zero-probability outcomes as zero). If the sum of results is zero (no observations), the function returns 0. Using a non-1D array, a non-numeric dtype, or negative counts may raise an exception or produce incorrect results; callers should ensure results is a one-dimensional numeric counts array appropriate for an empirical distribution.
    
    Returns:
        float: The Shannon entropy of the empirical distribution computed from results, expressed in bits (log base 2). This is computed by normalizing results to probabilities and summing -p * log2(p) across outcomes; zero-count outcomes contribute zero to the sum. If the total count is zero, the function returns 0. The function has no side effects and does not modify the input array.
    """
    from rdkit.ML.InfoTheory.entropy import PyInfoEntropy
    return PyInfoEntropy(results)


################################################################################
# Source: rdkit.ML.InfoTheory.entropy.PyInfoGain
# File: rdkit/ML/InfoTheory/entropy.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def rdkit_ML_InfoTheory_entropy_PyInfoGain(varMat: numpy.ndarray):
    """rdkit.ML.InfoTheory.entropy.PyInfoGain computes the information gain (expected reduction in Shannon entropy) for a single discrete variable given a contingency table of observed counts. In the RDKit machine-learning and information-theory utilities this function is used to evaluate how much knowing the value of a chemical descriptor or discrete feature (the variable) reduces uncertainty about an outcome or class label (the result), e.g., during feature selection or decision-tree split evaluation on molecular descriptor/fingerprint data.
    
    Args:
        varMat (numpy.ndarray): A 2-D numeric array (contingency table) of observed counts where rows correspond to the possible values of the variable and columns correspond to the possible result/outcome values. Each entry varMat[i, j] is the number of occurrences where the variable takes its i-th value and the result is the j-th outcome. For example, for a variable with 4 possible values and a result with 3 possible values, varMat would be shaped (4, 3). The array is interpreted as counts (non-negative numerics). The function sums rows to obtain Sv (the counts for each variable value) and sums columns to obtain S (the overall counts per result), following Mitchell's notation used in the implementation.
    
    Returns:
        float: The expected information gain, computed as H(S) - (1/|S|) * sum_i |Sv_i| * H(Sv_i), where H(...) is the Shannon entropy computed by the module's InfoEntropy function, |S| is the total number of observations (sum of all entries in varMat), and Sv_i is the i-th row of varMat (counts for the i-th variable value). If the total count |S| is zero (no observations in varMat), the function returns 0.0. The returned value represents the reduction in uncertainty about the result when the variable value is known; higher values indicate a more informative variable for predicting the result.
    
    Behavior and failure modes:
        The function computes per-row entropies using InfoEntropy and weights them by the row totals to obtain the expected conditional entropy. There are no side effects (no modification of varMat). The function expects a 2-D numpy.ndarray; if a different shape or a non-array object is supplied, numpy indexing or arithmetic operations will raise the corresponding runtime exception. Entries should represent non-negative counts; negative or non-finite values (NaN or inf) will produce incorrect entropies or propagate NaNs through the calculation. The function does not explicitly validate input types or values beyond relying on numpy operations, so callers should ensure varMat is a well-formed contingency table of counts for meaningful results.
    """
    from rdkit.ML.InfoTheory.entropy import PyInfoGain
    return PyInfoGain(varMat)


################################################################################
# Source: rdkit.ML.MLUtils.VoteImg.BuildVoteImage
# File: rdkit/ML/MLUtils/VoteImg.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def rdkit_ML_MLUtils_VoteImg_BuildVoteImage(
    nModels: int,
    data: list,
    values: list,
    trueValues: list = [],
    sortTrueVals: int = 0,
    xScale: int = 10,
    yScale: int = 2,
    addLine: int = 1
):
    """rdkit.ML.MLUtils.VoteImg.BuildVoteImage constructs a visualization image that encodes the per-model “votes” (outputs) for a set of examples produced by an ensemble of models in RDKit's machine-learning utilities. This function is intended for use in RDKit ML workflows to inspect and present how each model in a composite votes across examples: rows correspond to examples, columns correspond to models, pixel intensities encode vote strengths, and an optional purple vertical separator can mark the model/example boundary. The resulting image is produced using PIL (Pillow) and numpy and is suitable for visual debugging, reporting, or inclusion in ML result summaries.
    
    Args:
        nModels (int): the number of models in the composite. This determines the width (number of columns before scaling) of the vote image; the function will create nModels columns representing each model's vote for every example. It must match the second dimension of the per-example vote data provided in data.
        data (list): the results of voting. A sequence (typically a list) of per-example vote vectors that is convertible to a numpy array of integer type (the code uses numpy.integer). Each element corresponds to one example and should contain nModels numeric vote values. The function converts this input into a numpy integer array internally, scales it to 0–255, and uses it as the pixel intensity matrix. If data lengths or inner lengths do not match nModels or the lengths of values/trueValues, numpy or indexing operations will raise errors.
        values (list): predicted values for each example. A sequence with one entry per example (same length as data). When sortTrueVals is zero or when no trueValues are provided, this sequence is used to determine the ordering of examples (numpy.argsort(values)) before the image is built. This ordering is useful to visualize predictions sorted by predicted score so users can inspect model agreement patterns across the prediction ranking.
        trueValues (list = []): true values for each example. Optional sequence with one entry per example (same length as data). When provided (non-empty) and when sortTrueVals is nonzero, the examples are sorted by these trueValues (numpy.argsort(trueValues)) before constructing the image; this is useful for visualizing votes sorted by the ground-truth label or measurement. The default is an empty list, which signals that no ground-truth sorting should be applied.
        sortTrueVals (int = 0): if nonzero, instructs the function to sort examples by trueValues; otherwise examples are sorted by values. This integer is used as a boolean flag (0 means false, nonzero means true). When set and trueValues is non-empty, the function computes an order array with numpy.argsort(trueValues) and reorders data accordingly; when unset (0) the order is computed with numpy.argsort(values). If the chosen sorting array does not have the same length as data, numpy.argsort will raise an exception.
        xScale (int = 10): number of pixels per vote in the x direction (horizontal scaling factor). After an initial image of width nModels and height equal to number of examples is built, the image is resized by multiplying the width by xScale. This integer controls horizontal magnification for presentation and does not change the computed ordering or pixel intensities except for nearest-neighbor/antialiasing effects introduced by PIL's resize.
        yScale (int = 2): number of pixels per example in the y direction (vertical scaling factor). After building the initial image, the image height (number of examples) is multiplied by yScale during resize. This integer controls vertical magnification for display.
        addLine (int = 1): if nonzero, a purple vertical separator line is drawn between the model-vote area and the example boundary. Implementationally, when addLine is nonzero the PIL image is converted to 'RGB' mode and ImageDraw.Draw.line is used with color (128, 0, 128). The drawn line position differs by one pixel depending on whether trueValues were provided (the code uses nModels - 3 when trueValues is non-empty, otherwise nModels - 2). If addLine is zero, no separator line is drawn and the image remains in grayscale ('L') mode.
    
    Behavior, side effects, defaults, and failure modes:
        The function computes nData = len(data) and converts data into a numpy array of integer type (numpy.integer). It determines the ordering of examples using numpy.argsort on trueValues if sortTrueVals is nonzero and trueValues is non-empty; otherwise it uses numpy.argsort on values. The data rows are reordered by this order. Pixel intensities are normalized by scaling the integer data so that the maximum observed vote value maps to 255 (data = data * 255 / maxVal) and then cast to unsigned byte ('B') before creating a PIL Image via Image.frombytes with mode 'L' and initial size (nModels, nData). If addLine is true, the image is converted to 'RGB' and a purple vertical line is drawn at a model-edge x coordinate as described above. Finally, the image is resized to (nModels * xScale, nData * yScale) and returned.
    
        Default parameter behavior: trueValues defaults to an empty list meaning no ground-truth sorting; sortTrueVals defaults to 0 (use values for sorting); xScale and yScale default to 10 and 2 respectively to provide a readable magnified image; addLine defaults to 1 to include the separator by default.
    
        Potential failure modes: if data cannot be converted to a numpy integer array (e.g., incompatible element types), numpy will raise a TypeError or ValueError. If the per-example length (inner dimension) of data does not equal nModels or if values/trueValues do not have length equal to len(data), numpy.argsort or the reindexing operations will raise IndexError or ValueError. If all vote values are zero so that maxVal == 0, the division data * 255 / maxVal will produce a division-by-zero condition (runtime warning or error depending on numpy configuration) and may yield invalid pixel values. The function relies on PIL (Pillow) and numpy being available; missing imports will raise ImportError. The function performs no in-place modification of the caller's lists (it rebinds the local name data), but it does allocate numpy arrays and temporary PIL images.
    
    Returns:
        PIL.Image.Image: a Pillow Image representing the vote matrix. If addLine is zero the image mode is 'L' (grayscale) with size (nModels * xScale, nData * yScale). If addLine is nonzero the image mode is 'RGB' and contains a purple separator line at the model/example boundary; the returned image size is (nModels * xScale, nData * yScale). The image encodes models as columns and examples as rows after any sorting specified by sortTrueVals and trueValues/values.
    """
    from rdkit.ML.MLUtils.VoteImg import BuildVoteImage
    return BuildVoteImage(
        nModels,
        data,
        values,
        trueValues,
        sortTrueVals,
        xScale,
        yScale,
        addLine
    )


################################################################################
# Source: rdkit.ML.MLUtils.VoteImg.VoteAndBuildImage
# File: rdkit/ML/MLUtils/VoteImg.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def rdkit_ML_MLUtils_VoteImg_VoteAndBuildImage(
    composite: list,
    data: list,
    badOnly: int = 0,
    sortTrueVals: int = 0,
    xScale: int = 10,
    yScale: int = 2,
    addLine: int = 1
):
    """VoteAndBuildImage constructs a visual representation (PIL image) of ensemble voting results produced by a composite model over a set of examples. This function is intended for use in RDKit-based cheminformatics and machine-learning workflows where inspecting how multiple submodels (an ensemble) vote on molecular examples (for example, classification based on descriptors or fingerprints) helps diagnose model behavior, visualize consensus, and identify misclassified examples.
    
    The function collects votes from the provided composite model using CollectVotes, prints summary information about the number of models and misclassifications to standard output, and then constructs an image with BuildVoteImage that lays out votes and example labels according to pixel scaling parameters and optional sorting/separation behavior.
    
    Args:
        composite (list): A composite model represented as a list of submodels or voting components. In the RDKit ML context this is an ensemble (e.g., multiple classifiers built from molecular descriptors/fingerprints) whose individual votes are combined. The length of this list is used to compute nModels = len(composite) + 3 inside the function; the function prints the number of submodels (len(composite)) to stdout. The composite list is passed to CollectVotes to obtain per-model votes, numeric values, true labels, and a misclassification count.
        data (list): The examples to be voted upon, provided as a list. In cheminformatics use this will typically be a list of example records (molecules, feature vectors, or dataset rows) compatible with the composite model and with CollectVotes. These examples are the items for which votes are collected and visualized.
        badOnly (int): If nonzero, only incorrect votes (examples misclassified by the ensemble) will be included in the image; if zero (the default) all examples are shown. Default: 0. This flag controls filtering applied before the image is constructed and thereby focuses visualization on failure cases when set.
        sortTrueVals (int): If nonzero, votes and examples are sorted so that the trueValues (true labels) are presented in order; if zero (the default) sorting is by the numeric vote values returned by CollectVotes. Default: 0. This affects the ordering of rows/columns in the returned image and therefore how consensus and class-wise patterns appear.
        xScale (int): Number of pixels per vote in the horizontal (x) direction. Default: 10. This scaling parameter controls horizontal resolution allocated per model vote in the image produced by BuildVoteImage; increasing it makes the vote columns wider, which can improve readability for many models.
        yScale (int): Number of pixels per example in the vertical (y) direction. Default: 2. This scaling parameter controls vertical resolution per example row in the image; increasing it increases the height for each example, useful when labels or separators need more visual space.
        addLine (int): If nonzero (default 1), draw a separating purple line between the block that shows votes and the block that shows example labels in the generated image; if zero, omit the line. Default: 1. This is a purely visual aid to distinguish the voting matrix from example annotations.
    
    Returns:
        PIL image: A PIL image object constructed by BuildVoteImage that visualizes the voting results for the provided examples and composite model. The image encodes per-model votes, ordering determined by sortTrueVals, filtering determined by badOnly, and pixel dimensions controlled by xScale and yScale; when addLine is nonzero a purple separator line is drawn. The function also has the side effects of printing the number of submodels (len(composite)) and the count of misclassified examples (misCount) to standard output. Exceptions raised by CollectVotes or BuildVoteImage (for example due to incompatible composite/data structures or missing PIL support) are propagated to the caller.
    """
    from rdkit.ML.MLUtils.VoteImg import VoteAndBuildImage
    return VoteAndBuildImage(
        composite,
        data,
        badOnly,
        sortTrueVals,
        xScale,
        yScale,
        addLine
    )


################################################################################
# Source: rdkit.ML.SLT.Risk.BurgesRiskBound
# File: rdkit/ML/SLT/Risk.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def rdkit_ML_SLT_Risk_BurgesRiskBound(VCDim: int, nData: int, nWrong: int, conf: float):
    """rdkit.ML.SLT.Risk.BurgesRiskBound calculates Burges's formulation of the risk bound (Eqn. 3 in Burges, "A Tutorial on Support Vector Machines for Pattern Recognition", Data Mining and Knowledge Discovery, 1998) and returns an upper bound on the true misclassification risk used in machine-learning analyses (for example SVMs) within the RDKit ML/SLT workflow.
    
    This function combines an empirical error term (fraction of misclassified training examples) with a structural risk term derived from the VC-dimension to produce a single scalar risk bound. It is intended for use in statistical learning evaluations (binary classification scenarios, as noted in the original implementation) when one wants a theoretically motivated upper bound on generalization error for a hypothesis class characterized by its VC dimension.
    
    Args:
        VCDim (int): The Vapnik–Chervonenkis (VC) dimension of the hypothesis class being evaluated. In the context of RDKit's ML utilities, VCDim represents the capacity/complexity of the classifier family (h in Burges's notation). It must be provided as an integer; if supplied incorrectly (non-numeric or not an integer), a TypeError or later numeric-errors may occur.
        nData (int): The number of data points (training examples) used to compute the empirical error (l in Burges's notation). This integer is used as a denominator in the empirical error and to scale the structural term. Passing nData == 0 will cause a division-by-zero error in the implementation; non-integer numeric types may lead to unexpected results or exceptions.
        nWrong (int): The number of data points from the nData that were misclassified (used to compute empirical risk). This integer is converted to a float fraction rEmp = float(nWrong) / nData. Supplying values inconsistent with nData (e.g., nWrong > nData) will still compute a numeric result but may not be meaningful in a probability interpretation.
        conf (float): The confidence parameter (denoted eta in the code and in Burges's formulation) used to scale the logarithmic confidence term in the structural risk. This float is used inside a natural logarithm (math.log(eta / 4.0)) and therefore must be such that the argument to the log is positive; improper values can raise a ValueError from math.log. The value represents the confidence level used when deriving the probabilistic bound.
    
    Behavior and implementation details:
        - The implementation maps input names to Burges's notation as follows: h = VCDim, l = nData, eta = conf. It then computes numerator = h * (log(2*l/h) + 1) - log(eta/4) and structRisk = sqrt(numerator / l). The empirical risk rEmp is computed as float(nWrong) / l. The function returns rEmp + structRisk.
        - There are no side effects (the function does not modify global state or external resources); it is purely computational.
        - The function assumes the user is evaluating a classifier where VC-dimension based bounds are meaningful (the original implementation notes this is technically valid for binary classification).
        - The implementation has been validated against the Burges paper and follows the formula from Eqn. 3 of that review article.
    
    Failure modes and exceptions:
        - ZeroDivisionError will occur if nData is zero because the code divides by l (nData).
        - ValueError may be raised by math.log if the arguments to the log (for example eta/4.0 or 2*l/h) are non-positive, or by math.sqrt if the expression inside the square root is negative. These indicate the provided numeric inputs are incompatible with the mathematical operations required by Burges's bound.
        - TypeError or other numeric exceptions may arise if arguments are of incorrect types (e.g., non-numeric objects).
        - The bound returned is a numeric upper bound computed from the inputs; for some combinations of inputs (e.g., inconsistent counts or extreme confidence values) the numeric result may not correspond to a valid probability in [0,1] even though the quantity represents an upper bound on misclassification risk in the theoretical derivation.
    
    Returns:
        float: A scalar value equal to the empirical misclassification rate (nWrong / nData) plus the structural risk term derived from VCDim and conf. This float is intended to be an upper bound on the true misclassification probability (theoretical generalization risk) as formulated by Burges. If an exception occurs (see Failure modes), no return value is produced.
    """
    from rdkit.ML.SLT.Risk import BurgesRiskBound
    return BurgesRiskBound(VCDim, nData, nWrong, conf)


################################################################################
# Source: rdkit.ML.SLT.Risk.CherkasskyRiskBound
# File: rdkit/ML/SLT/Risk.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def rdkit_ML_SLT_Risk_CherkasskyRiskBound(
    VCDim: int,
    nData: int,
    nWrong: int,
    conf: float,
    a1: float = 1.0,
    a2: float = 2.0
):
    """rdkit.ML.SLT.Risk.CherkasskyRiskBound computes a provable upper bound on the true (generalization) risk of a classifier using the Cherkassky & Mulier formulation (Equations 4.22 and 4.23, Learning From Data, Wiley 1998). This implementation is intended for use in the RDKit ML/SLT risk-bounding utilities to support model evaluation and selection where the VC dimension of a hypothesis class is known or estimated.
    
    Args:
        VCDim (int): The VC dimension of the classifier/hypothesis class (denoted h in the code and Cherkassky's notation). This integer quantifies model capacity and appears as h in the bound formula; it must be positive (h >= 1) for the bound to be meaningful and to avoid division by zero in intermediate logarithms. In cheminformatics and ML workflows within RDKit, VCDim expresses the capacity of the learned model (for example, a classifier built on molecular descriptors or fingerprints).
        nData (int): The number of labeled data points used to compute the empirical error (denoted n in the code). This must be a positive integer (n >= 1). nData is used to compute the empirical risk rEmp = nWrong / nData and to scale the structural term in the bound; passing nData == 0 will raise a ZeroDivisionError.
        nWrong (int): The number of misclassified data points in the sample (an integer in the range 0..nData). This is used to compute the empirical risk rEmp = float(nWrong) / nData. Supplying values outside 0..nData can produce an empirical risk outside [0,1] and lead to misleading or invalid bounds.
        conf (float): The confidence parameter eta used in the probabilistic bound (denoted eta in the code). This is the probability with which the bound should hold and must be strictly positive (conf > 0). Typical usage sets conf in (0,1], e.g., 0.05 for 95% confidence. The implementation uses math.log(conf / 4.0) so conf <= 0 will cause a math domain error.
        a1 (float = 1.0): A scalar constant that scales the numerator of the structural term (default 1.0). The code multiplies the numerator by a1 and divides by nData to form eps; therefore, a1 == 0.0 will set eps to zero and usually lead to division-by-zero behavior when used inside the square-root expression. The original formulation restricts this constant to 0 <= a1 <= 4. The default a1=1.0 is chosen by analogy to Burges's paper and is a commonly used setting in practice.
        a2 (float = 2.0): A scalar constant that appears inside the logarithmic term log(a2 * nData / VCDim) (default 2.0). The original formulation restricts this constant to 0 <= a2 <= 2. The value of a2 influences the model-complexity penalty in the structural risk term; a2 <= 0 or VCDim <= 0 or nData <= 0 will cause math domain or division errors in the logarithm.
    
    Behavior and formula:
        The function implements the Cherkassky/Mulier bound by mapping inputs to local variables used in the book: h = VCDim, n = nData, eta = conf, and rEmp = nWrong / nData. It computes
        numerator = h * (log(a2 * n / h) + 1) - log(eta / 4)
        eps = a1 * numerator / n
        structRisk = eps / 2 * (1 + sqrt(1 + (4 * rEmp / eps)))
        and returns rEmp + structRisk.
        The returned float therefore represents an upper bound on the true risk (probability of error) with confidence at least conf, combining the empirical risk rEmp and a data- and capacity-dependent structural term derived from the cited equations.
    
    Side effects and defaults:
        This function has no side effects: it does not modify inputs, global state, or filesystem; it returns a float computed deterministically from its arguments. The defaults a1=1.0 and a2=2.0 follow the original code and literature notes; users may adjust them within the documented ranges to explore different constant choices in the bound.
    
    Failure modes and exceptions:
        The function will raise a ZeroDivisionError if nData == 0 (division by nData) or if VCDim == 0 leads to division by zero inside the logarithm. It will raise a ValueError or math domain error if any argument causes invalid inputs to math.log or math.sqrt (for example, conf <= 0, a2 * nData / VCDim <= 0, or eps < 0). Supplying a1 == 0 will typically cause a division-by-zero or invalid sqrt when eps == 0. The caller must ensure the constraints implied by the Cherkassky formulation are respected: VCDim >= 1, nData >= 1, 0 <= nWrong <= nData, conf > 0 (typically conf in (0,1]), and the recommended ranges 0 <= a1 <= 4, 0 <= a2 <= 2 to obtain meaningful, numerically stable bounds.
    
    Returns:
        float: A floating-point upper bound on the classifier's true risk (generalization error) computed as empirical risk plus a structural risk term from Cherkassky and Mulier (Equations 4.22/4.23). The returned value is intended to be interpreted as a probability-like quantity (typically in [0,1] when inputs are valid and constraints are respected), and it holds with probability at least conf under the assumptions of the bound.
    """
    from rdkit.ML.SLT.Risk import CherkasskyRiskBound
    return CherkasskyRiskBound(VCDim, nData, nWrong, conf, a1, a2)


################################################################################
# Source: rdkit.ML.SLT.Risk.CristianiRiskBound
# File: rdkit/ML/SLT/Risk.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def rdkit_ML_SLT_Risk_CristianiRiskBound(VCDim: int, nData: int, nWrong: int, conf: float):
    """Compute the Cristiani & Shawe-Taylor VC-theory-based risk bound (Theorem 4.6 in
    "An Introduction to Support Vector Machines" by Cristiani and Shawe-Taylor,
    Cambridge University Press, 2000). This function implements the same notation
    used in the book and returns an upper bound on the true classification risk
    (probability of misclassification on new examples) as the sum of an empirical
    risk term and a structural (capacity) term derived from the VC dimension.
    
    Args:
        VCDim (int): The VC dimension of the hypothesis class. This integer
            quantifies the capacity/complexity of the classifier family being
            analyzed and appears in the structural term of the bound. In the
            implementation this value is assigned to the internal variable d and
            is used inside a logarithm; therefore it must be a positive integer
            (>= 1) for the formula to be mathematically well-defined.
        nData (int): The number of data points (training examples) used to
            compute the empirical risk. This integer is assigned to the internal
            variable l. It must be positive (> 0) because it appears in
            denominators and inside log expressions; if nData == 0 the function
            will raise a ZeroDivisionError or a math domain error.
        nWrong (int): The number of data points misclassified in the training
            set (empirical errors). This integer is assigned to the internal
            variable k and is used to compute the empirical risk term rEmp = 2*k/l.
            Practically, it should lie in the range [0, nData]. Values outside
            this range will produce values that are not meaningful as an empirical
            error count and can lead to misleading or nonsensical bounds.
        conf (float): The confidence parameter (denoted delta in the bound),
            representing the probability that the bound fails. This float is
            assigned to the internal variable delta and appears inside a logarithm
            as log2(4.0 / delta). It must be strictly positive (> 0). In the
            typical statistical interpretation the returned bound then holds with
            probability at least 1 - conf (so typical choices are small positive
            numbers, e.g., 0.05), but the function only enforces positivity, not
            an upper bound of 1.
    
    Returns:
        float: A numerical upper bound on the true misclassification risk computed
        as rEmp + structRisk, where rEmp = 2 * nWrong / nData is the empirical risk
        term and structRisk = sqrt((4.0 / nData) * (VCDim * log2((2.0 * e * nData) /
        VCDim) + log2(4.0 / conf))) is the structural (capacity) term derived from
        the VC dimension. The returned float is the sum of these two terms and is
        intended as an upper bound on the expected generalization error given the
        inputs. Note that the bound may be loose and in some parameter regimes may
        exceed 1 or otherwise appear counterintuitive; this behavior is a known
        property of VC-style bounds and is indicated in the original implementation.
    
    Behavior and failure modes:
        This function is pure (no side effects) and performs only arithmetic.
        It relies on math.log2 and math.sqrt and will raise exceptions for invalid
        numeric inputs: if nData <= 0 a ZeroDivisionError or math domain error will
        occur; if VCDim <= 0 a division-by-zero or math domain error will occur;
        if conf <= 0 a math domain error will occur because log2(4.0 / conf) is
        undefined. Because nWrong is used directly in the empirical term, negative
        nWrong or nWrong > nData produce values that are not meaningful in the
        context of counting misclassifications and can yield misleading bounds.
        No input coercion is performed; callers should validate integer-type
        semantics and positivity before calling if necessary.
    
    Practical significance within RDKit/ML context:
        This implementation is part of the RDKit ML/SLT risk utilities and is useful
        when analyzing classifier generalization in cheminformatics workflows that
        use VC-theory-based capacity control (for example, when evaluating the
        expected error of simple classifiers on molecular descriptor/fingerprint
        feature sets). Users should treat the result as a theoretical upper bound
        informed by VC dimension and sample size, not as a precise empirical
        probability; empirical validation on held-out data is recommended.
    """
    from rdkit.ML.SLT.Risk import CristianiRiskBound
    return CristianiRiskBound(VCDim, nData, nWrong, conf)


################################################################################
# Source: rdkit.ML.Scoring.Scoring.CalcAUC
# File: rdkit/ML/Scoring/Scoring.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def rdkit_ML_Scoring_Scoring_CalcAUC(scores: list, col: int):
    """Calculates the area under the receiver operating characteristic (ROC) curve (AUC) for a set of scored molecules.
    
    This function is part of RDKit's ML/scoring utilities and is used to quantify the performance of a ranking or binary scoring produced for molecules (for example, scores coming from descriptor- or fingerprint-based models). It delegates to CalcROC(scores, col) to compute the ROC curve (an object expected to expose FPR and TPR sequences), then numerically integrates the ROC curve using the trapezoidal rule over false positive rate (FPR) to produce the AUC value. The AUC summarizes classifier performance: higher values indicate better separation between positive and negative classes in the scored list.
    
    Args:
        scores (list): A list of per-molecule score records as produced by RDKit scoring utilities. Each element in this list is expected to be indexable (for example, a tuple or list) so that the integer column index given by col can be used to extract the numeric score or label required by CalcROC. The function passes this list directly to CalcROC; any structural or type requirements on the elements are therefore those required by CalcROC.
        col (int): The zero-based integer column index used to select the numeric value from each record in scores that will be used to compute the ROC curve. In RDKit scoring workflows, col selects which field in each score record represents the classifier score (or label) to be used for ranking and ROC computation.
    
    Returns:
        float: The computed area under the ROC curve. The value is produced by integrating (using the trapezoidal rule) the TPR vs FPR arrays returned by CalcROC: AUC = 0.5 * sum((FPR[i+1]-FPR[i])*(TPR[i+1]+TPR[i])) over the ordered points. For typical, valid ROC inputs this value summarizes discrimination performance (commonly interpreted on a 0.0–1.0 scale). If the provided scores list has fewer than two entries, the loop does not accumulate area and the function will return 0.0. Any exceptions or errors raised by CalcROC (for example, due to malformed scores entries or incompatible column indices) are propagated to the caller; the function itself has no side effects beyond calling CalcROC.
    """
    from rdkit.ML.Scoring.Scoring import CalcAUC
    return CalcAUC(scores, col)


################################################################################
# Source: rdkit.ML.Scoring.Scoring.CalcBEDROC
# File: rdkit/ML/Scoring/Scoring.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def rdkit_ML_Scoring_Scoring_CalcBEDROC(scores: list, col: int, alpha: float):
    """rdkit.ML.Scoring.Scoring.CalcBEDROC computes the BEDROC (Boltzmann-enhanced discrimination of ROC) score used to quantify "early recognition" performance in virtual screening campaigns, following Truchon & Bayly, J. Chem. Inf. Model. 47, 488-508 (2007).
    
    This function expects a ranked list of samples (e.g., molecules) with per-sample data and a column indicating which samples are considered "active" (true positives). It uses an internal RIE (Robust Initial Enhancement) helper to compute the unnormalized enrichment and then normalizes that value to produce the BEDROC score. BEDROC emphasizes retrieval of active compounds at the top of a ranked list and is commonly used in cheminformatics and machine-learning workflows within RDKit to evaluate virtual screening and ranking algorithms.
    
    Args:
        scores (list or numpy.ndarray): A two-dimensional sequence (samples × features) containing per-sample data. The 0th index corresponds to the sample (row) and each element scores[sample_id] is an indexable sequence representing that sample's data. The rows in scores must be sorted in ranked order with lower indices representing "better" (higher-priority) predictions; the metric assumes the input is pre-ranked. The column specified by col is used to determine true actives: scores[sample_id][col] == True iff that sample is considered active. Providing data that is not 2D, not indexable as described, or not pre-sorted will make the metric meaningless or cause indexing/type errors.
        col (int): Integer column index in each sample vector that indicates the true label for that sample. For a given sample_id, scores[sample_id][col] must compare equal to True exactly when the sample is active. If col is out of range for the per-sample vectors, an IndexError (or equivalent) will be raised by the underlying operations.
        alpha (float): Hyperparameter from the original BEDROC formulation that controls how strongly to weight early enrichment (the "top" of the ranked list). Larger values of alpha place more emphasis on top-ranked actives. This float is passed directly into the RIE calculation; extreme values may increase numerical sensitivity because exponential functions are used internally.
    
    Returns:
        float: The BEDROC score (a normalized value derived from RIE) quantifying early recognition for the provided ranked list. If there are no active samples in scores (no entries with scores[sample_id][col] == True), the function returns 0.0. If every sample is active, the function returns 1.0. For intermediate cases the return value is the normalized RIE mapped to the 0.0–1.0 range according to the original BEDROC formula. No other side effects occur; errors such as invalid indexing or wrong input shapes will propagate as standard Python exceptions (IndexError, TypeError, etc.).
    """
    from rdkit.ML.Scoring.Scoring import CalcBEDROC
    return CalcBEDROC(scores, col, alpha)


################################################################################
# Source: rdkit.ML.Scoring.Scoring.CalcEnrichment
# File: rdkit/ML/Scoring/Scoring.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def rdkit_ML_Scoring_Scoring_CalcEnrichment(scores: list, col: int, fractions: list):
    """rdkit.ML.Scoring.Scoring.CalcEnrichment determines enrichment factors for ranked scoring results produced in cheminformatics / virtual screening workflows (as used in RDKit machine-learning and scoring contexts). The function computes, for each requested top fraction of the ranked list, the fold-enrichment of actives recovered in that top fraction relative to the expectation from random selection. This is typically used to quantify early-recovery performance of scoring functions or virtual screening models when a boolean active/inactive indicator is stored in a specific column of each score record.
    
    Args:
        scores (list): A list of score records (rows) for molecules/items produced by a scoring or ranking procedure. Each element in this list must be an indexable sequence (for example, a tuple or list) such that scores[i][col] yields a truthy value when the i-th record is an active (hit) and a falsy value otherwise. The function uses the order of this list as the ranking (it expects the list to be ordered from best-ranked to worst-ranked so that "top fractions" refer to leading entries). The length of this list determines the total number of molecules/items evaluated; an empty list causes a ValueError.
        col (int): The integer column index into each score record that identifies activity (the active/inactive flag). This index must be valid for every element of scores; if an element does not support indexing at this position an IndexError or TypeError may be raised. The value at this index is interpreted as boolean-like: truthy means active, falsy means inactive.
        fractions (list): A list of fractional thresholds (each a numeric value) specifying the proportions of the ranked list at which to report enrichment (for example, 0.01 for the top 1%). Each fraction must satisfy 0 <= fraction <= 1. The function converts each fraction into a count with math.ceil(len(scores) * fraction) to determine how many top-ranked entries to consider for that fraction. An empty fractions list causes a ValueError. Fractions outside [0, 1] cause a ValueError.
    
    Behavior, algorithm, and failure modes:
        The function first validates inputs: it raises ValueError if scores or fractions are empty, and raises ValueError if any fraction is outside the closed interval [0, 1]. It then computes an integer threshold count for each fraction using ceiling(len(scores) * fraction) and appends the total number of records as a sentinel to ease threshold processing. It iterates over the ranked scores in list order, counting actives encountered; whenever the iteration index crosses a threshold for the next fraction the function computes an intermediate enrichment value based on the number of actives seen so far and the number of items examined, appends it to an internal list, and proceeds. After finishing the scan, it normalizes the intermediate values by the total number of actives to produce fold-enrichment values. If there are no actives in the entire scores list the function returns a list of zeros with length equal to len(fractions). The function has no external side effects (it does not modify inputs) but may raise IndexError or TypeError if col is invalid for some score record or if records are not indexable. Small or zero fractions are supported (fraction == 0 yields a threshold count of zero via ceil and will be handled by the iteration logic), but users should verify that the resulting thresholds match their intended selection behavior.
    
    Returns:
        list: A list of floating-point enrichment factors, one per entry in fractions in the same order. Each returned value represents the fold-enrichment of actives recovered in the corresponding top fraction of the ranked scores relative to random expectation. If no actives are present in scores the returned list contains zeros.
    """
    from rdkit.ML.Scoring.Scoring import CalcEnrichment
    return CalcEnrichment(scores, col, fractions)


################################################################################
# Source: rdkit.ML.Scoring.Scoring.CalcRIE
# File: rdkit/ML/Scoring/Scoring.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def rdkit_ML_Scoring_Scoring_CalcRIE(scores: list, col: int, alpha: float):
    """rdkit.ML.Scoring.Scoring.CalcRIE computes the Robust Initial Enhancement (RIE) enrichment metric used to quantify early-recognition performance of ranking/scoring methods in virtual screening (original definition: Sheridan et al., J. Chem. Inf. Comput. Sci. 2001). The function is part of RDKit's ML/Scoring utilities and is used to evaluate how well a scoring list prioritizes “active” entries near the top of a ranked list.
    
    Args:
        scores (list): A sequence (Python list) containing per-item records/rows produced by scoring or screening pipelines. Each element is expected to be an indexable record (for example, a tuple, list, or other sequence) from which a column can be selected using the integer index given by col. In cheminformatics/virtual-screening usage, scores typically contains both a predicted score/rank and a label column identifying which entries are actives/decoys; this function reads the column indicated by col to determine activity labels or values used by the RIE calculation. Providing a plain list of numeric scores without a selectable column is not compatible with the col argument.
        col (int): Integer column index into each element of scores that identifies the activity indicator or numeric value used by the RIE calculation. In typical use within RDKit screening benchmarks, this selects the column that marks actives (for example 1/0 or True/False) or provides the reference value required by the implementation. The index is zero-based and must be valid for every element in scores.
        alpha (float): Positive floating point parameter that controls how strongly early-ranking positions are weighted in the RIE calculation. Larger alpha values increase emphasis on the top-ranked portion of the list, making the metric more sensitive to very early enrichment. The value is passed through directly to the underlying _RIEHelper routine; no implicit clipping or defaulting is performed by CalcRIE.
    
    This function calls an internal helper (_RIEHelper) to compute RIE and returns only the primary RIE scalar produced by that helper. There are no side effects (no global state modification, no file or I/O operations). No default values are assumed for col or alpha; they must be supplied by the caller. Typical domain usage: compare alternative scoring functions or virtual-screening protocols by computing RIE on ranked results to quantify early enrichment (as per Sheridan et al., 2001).
    
    Failure modes and error propagation: If scores is not a list-like container of indexable records, or if col is out of range for any record, the underlying operations will raise Python exceptions (for example IndexError or TypeError). If alpha is not a numeric float or is non-positive in contexts where the helper requires positivity, a ValueError or TypeError may be raised by _RIEHelper or by numeric operations; such exceptions are propagated to the caller. The function does not validate or coerce element types beyond relying on the internal helper, so callers should ensure inputs match expected formats to avoid runtime errors.
    
    Returns:
        float: The Robust Initial Enhancement (RIE) scalar computed for the provided scores, column selection, and alpha parameter. This single floating-point value quantifies early enrichment (higher values indicate stronger early recognition of actives in typical virtual-screening contexts).
    """
    from rdkit.ML.Scoring.Scoring import CalcRIE
    return CalcRIE(scores, col, alpha)


################################################################################
# Source: rdkit.ML.Scoring.Scoring.CalcROC
# File: rdkit/ML/Scoring/Scoring.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def rdkit_ML_Scoring_Scoring_CalcROC(scores: list, col: int):
    """rdkit.ML.Scoring.Scoring.CalcROC: Compute a Receiver Operating Characteristic (ROC) curve from a list of scored records, producing cumulative true positive and false positive rates at each position in the provided list. This function is intended for use in cheminformatics and ML workflows with RDKit (for example virtual screening or classifier evaluation) where a list of scored samples and a column index indicating active/inactive labels are available; it returns the cumulative TPR and FPR vectors that can be plotted or used to compute AUC.
    
    Args:
        scores (list): A list of records (for example tuples or lists) containing scoring/classification results for individual molecules or samples. Each record is indexed with col to determine whether that sample is considered "active" (truthy value) or "inactive" (falsy value). The function processes this list in the supplied order and computes cumulative counts; for a conventional ROC curve the caller should provide scores sorted by predicted score (e.g., descending predicted activity) so that rates correspond to varying classification thresholds.
        col (int): Integer column index into each element of scores that holds the active/inactive indicator used as the ground-truth label. The value at scores[i][col] is interpreted as a boolean: truthy means active (positive class), falsy means inactive (negative class). This parameter selects which position in each record encodes the binary label.
    
    Returns:
        RocCurve: A namedtuple('RocCurve', ['FPR', 'TPR']) with two attributes:
            FPR (list): A list of floats of length len(scores). Each entry is the cumulative false positive rate FP/(TN+FP) after processing the corresponding prefix of the input list. Values are normalized to the range [0, 1] when there is at least one inactive sample; if there are zero inactives the returned values remain as cumulative counts and no normalization for FPR is performed.
            TPR (list): A list of floats of length len(scores). Each entry is the cumulative true positive rate TP/(TP+FN) after processing the corresponding prefix of the input list. Values are normalized to the range [0, 1] when there is at least one active sample; if there are zero actives the returned values remain as cumulative counts and no normalization for TPR is performed.
    
    Raises:
        ValueError: If scores is an empty list. The function requires at least one record to compute cumulative rates.
    
    Behavior and notes:
        The function iterates the scores list once (O(n) time, O(n) additional memory for TPR/FPR lists) and counts active/inactive samples cumulatively. It first computes cumulative counts (TP and FP) at each index and then divides by the total number of actives and inactives respectively when those totals are greater than zero. There are no side effects: the input list is not modified and the function returns a new RocCurve object. If all samples are active or all are inactive, only one of the rate vectors will be normalized to [0,1]; the other vector will remain as raw cumulative counts because the code avoids division by zero. To obtain a standard ROC curve that captures classifier threshold behavior, provide scores ordered by predicted score (e.g., highest predicted probability of activity first).
    """
    from rdkit.ML.Scoring.Scoring import CalcROC
    return CalcROC(scores, col)


################################################################################
# Source: rdkit.sping.PDF.pdfdoc.MakeFontDictionary
# File: rdkit/sping/PDF/pdfdoc.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def rdkit_sping_PDF_pdfdoc_MakeFontDictionary(startpos: int, count: int):
    """rdkit.sping.PDF.pdfdoc.MakeFontDictionary: Construct and return a PDF font dictionary fragment that maps standard PDF font resource names (/F1, /F2, ...) to PDF indirect object references. This function is used by RDKit's PDF assembly code (sping.PDF.pdfdoc) when generating PDF output for molecular images or other RDKit-generated graphics; it assumes that the font objects already exist in the PDF file as consecutive indirect objects beginning at the given start object number.
    
    Args:
        startpos (int): The PDF indirect object number at which the first font object resides. In the PDF object reference format used by the code, each font entry is rendered as "<object-number> 0 R". startpos therefore defines the base object number for the sequence of font object references inserted into the dictionary. This value must be an integer representing a valid PDF object number in the assembled file; using a negative or non-integer value will produce an incorrect or malformed PDF fragment.
        count (int): The number of font entries to include in the dictionary. The function will create entries named /F1 through /F<count>, each referencing sequential object numbers startpos through startpos + count - 1. This must be an integer; a non-integer will likely cause a TypeError during string formatting, and a negative value will produce a syntactically valid string but one that does not represent a correct set of PDF object references.
    
    Returns:
        str: A string containing the PDF dictionary fragment for font resources. The returned string begins with "  <<", contains one line per font of the form "\t\t/Fi <n> 0 R " where i is 1..count and n is startpos + (i-1), and ends with "\t\t>>", with LINEEND appended after each line. There are no side effects (the function does not modify files or global state); the caller is expected to insert the returned string into the larger PDF document stream. If count is zero the function returns a minimal dictionary with no /F entries. Note that the function does not perform validation of PDF object numbers beyond simple string construction, so invalid startpos/count combinations will result in an invalid PDF resource dictionary rather than an exception.
    """
    from rdkit.sping.PDF.pdfdoc import MakeFontDictionary
    return MakeFontDictionary(startpos, count)


################################################################################
# Source: rdkit.sping.PDF.pdfgeom.bezierArc
# File: rdkit/sping/PDF/pdfgeom.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def rdkit_sping_PDF_pdfgeom_bezierArc(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    startAng: float = 0,
    extent: float = 90
):
    """rdkit.sping.PDF.pdfgeom.bezierArc: Compute cubic Bezier control points that approximate an elliptical arc inscribed in the rectangle defined by two corner points. This function is used by RDKit's PDF/graphics geometry routines to convert an arc (used when rendering 2D molecular diagrams and other vector graphics) into one or more cubic Bezier segments suitable for PDF or other vector backends that represent curves with Bezier control points.
    
    The function interprets (x1, y1) and (x2, y2) as opposite corners of the enclosing rectangle and normalizes them so the returned ellipse fits that rectangle. The coordinate system assumed by the algorithm has X increasing to the right and Y increasing downwards (typical raster/PDF coordinate orientation used by RDKit PDF geometry code). Angles are expressed in degrees, with 0 degrees pointing to the right (positive X axis) and angles increasing counter-clockwise. The arc starts at startAng and spans extent degrees; a negative extent produces a clockwise sweep. For extents whose absolute value exceeds 90 degrees the arc is subdivided into multiple cubic Bezier segments (each covering at most 90 degrees) to maintain the standard approximation accuracy used in RDKit's PDF geometry code. The returned list contains one tuple per cubic Bezier segment; each tuple has eight floats (x1, y1, x2, y2, x3, y3, x4, y4) representing the start point, the two control points, and the end point of that segment in that order. These coordinates are in the same coordinate system used by the inputs and are ready to be consumed by RDKit's PDF drawing routines or any other renderer that accepts cubic Bezier segments.
    
    Args:
        x1 (float): X coordinate of the first corner of the rectangle that encloses the elliptical arc. This input is used to determine the horizontal extent (rx) of the ellipse; the function normalizes x1 and x2 so the left edge is min(x1, x2) and the right edge is max(x1, x2). In RDKit this coordinate is in the same units used for page/canvas coordinates when producing PDF/vector output.
        y1 (float): Y coordinate of the first corner of the rectangle that encloses the elliptical arc. Because the coordinate system used increases downward, the function normalizes the pair (y1, y2) so the top/bottom orientation matches the internal representation (it sets the internal y1 to the larger of the two input Y values). This value, together with y2, determines the vertical radius (ry) of the ellipse for PDF rendering of molecular graphics.
        x2 (float): X coordinate of the second corner of the rectangle that encloses the elliptical arc. Combined with x1 it sets the center and horizontal radius of the ellipse. The function swaps/normalizes x1 and x2 as needed so downstream geometry calculations assume x1 <= x2.
        y2 (float): Y coordinate of the second corner of the rectangle that encloses the elliptical arc. Combined with y1 it sets the center and vertical radius of the ellipse. The function swaps/normalizes y1 and y2 as needed so downstream geometry calculations assume the internal representation where y1 is the larger value (since Y increases downward).
        startAng (float): Start angle of the arc in degrees. 0 degrees points to the right (positive X axis) and angles increase counter-clockwise. The arc begins at this angle measured from the center of the ellipse defined by the rectangle. The default is 0. In RDKit usage, setting startAng selects the angular position on the ellipse where the arc begins when converting vector shapes for PDF output.
        extent (float): Angular extent of the arc in degrees (how far the arc spans from startAng). Positive values produce a counter-clockwise sweep, negative values produce a clockwise sweep. If |extent| > 90 the function automatically subdivides the arc into multiple cubic Bezier segments (each with at most 90 degrees) to preserve approximation accuracy, matching the approach used by RDKit's PDF geometry utilities. The default is 90.
    
    Returns:
        list: A list of tuples, one tuple per cubic Bezier segment approximating the requested elliptical arc. Each tuple contains eight floats in the order (x1, y1, x2, y2, x3, y3, x4, y4). The first pair (x1, y1) is the start point of that segment, (x2, y2) and (x3, y3) are the two Bezier control points, and (x4, y4) is the end point. These coordinates are expressed in the same page/canvas coordinate system as the inputs and can be passed directly to RDKit's PDF/vector drawing routines or any renderer that accepts cubic Bezier segments.
    
    Behavior and side effects:
        The function is pure (no external side effects) and returns newly constructed coordinate data only. It normalizes the input rectangle so that the ellipse is computed consistently: the internal x1 is set to min(original x1, original x2), internal x2 to max(...), internal y1 to max(original y1, original y2) and internal y2 to min(...), reflecting the downward-increasing Y coordinate system used by RDKit PDF geometry. If the absolute value of extent is greater than 90 degrees the arc is divided into Nfrag = ceil(|extent| / 90) fragments and each fragment is converted to one cubic Bezier tuple; fragAngle = extent / Nfrag is the angle covered by each fragment. The function uses the standard "kappa" factor (4/3 * (1 - cos(halfAngle)) / sin(halfAngle), with halfAngle = fragAngle*pi/360) to compute control points that approximate an elliptical arc with cubic Beziers, matching the common technique used in vector graphics and RDKit's PDF geometry code.
    
    Failure modes and exceptions:
        The function does not perform explicit type checking beyond using the numeric values provided. If fragAngle is zero (for example extent == 0) or if sin(halfAng) evaluates to zero, the internal computation of the kappa factor will perform a division by zero and raise a ZeroDivisionError. Callers should avoid passing an extent that leads to fragAngle == 0 or otherwise ensure extent is a nonzero value that results in a valid halfAngle. Invalid numeric inputs (NaN, infinities) will propagate through the arithmetic and may produce exceptions or invalid output.
    
    Examples of practical usage in RDKit:
        - Converting an elliptical arc used in a 2D molecular depiction into one or more cubic Bezier segments for inclusion in a PDF drawing command sequence.
        - Generating control points for arcs when exporting molecular diagrams to vector formats that require cubic Beziers rather than native arc primitives.
    """
    from rdkit.sping.PDF.pdfgeom import bezierArc
    return bezierArc(x1, y1, x2, y2, startAng, extent)


################################################################################
# Source: rdkit.sping.PDF.pdfmetrics.parseAFMfile
# File: rdkit/sping/PDF/pdfmetrics.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def rdkit_sping_PDF_pdfmetrics_parseAFMfile(filename: str):
    """rdkit.sping.PDF.pdfmetrics.parseAFMfile parses an Adobe Font Metrics (AFM) file and returns a fixed-size array of character widths used by RDKit's PDF generation code to measure and lay out text. This is an ultra-crude, permissive parser intended to extract numeric widths from the "StartCharMetrics" / "EndCharMetrics" section of an AFM file so higher-level PDF routines can compute text extents when creating PDF output.
    
    Args:
        filename (str): Path to a text AFM file on disk. The function opens the file with open(filename, 'r') and reads all lines into memory (readlines()), so the entire file is loaded at once. The caller is responsible for passing a readable path; FileNotFoundError or other I/O errors will propagate if the file cannot be opened. The function does not perform explicit file closing (it relies on the temporary file object returned by open(...).readlines()).
    
    Returns:
        list: A list of integers of length 255 where each index is treated as a character code (CID) and the value is the width extracted from the AFM file for that CID. The list is initialized to zeros and populated by parsing lines in the AFM "StartCharMetrics" / "EndCharMetrics" block. Each metric line is expected to be semicolon-separated; the first field must contain a token pair where the second token is the CID (an integer), and the second field must contain a token pair where the second token is the width (an integer). The function assigns widths[int(cid)] = int(width) for each parsed metric line.
    
    Behavior, side effects, and failure modes:
        - The parser is case-insensitive when locating the StartCharMetrics and EndCharMetrics markers: it searches for the lowercase substrings 'startcharmetrics' and 'endcharmetrics' in each line. Lines between the first 'startcharmetrics' and the subsequent 'endcharmetrics' are treated as metric lines and parsed.
        - The widths array is preallocated to a fixed size of 255 entries. If a parsed CID is outside the range 0..254, an IndexError will be raised when assigning into widths.
        - The parser expects metric lines to contain semicolon-separated chunks and specific token patterns; malformed lines (wrong number of chunks, non-integer CID or width, unexpected tokenization) will raise ValueError or IndexError during parsing.
        - The function attempts to default any unset widths to the width of the ASCII space character (CID 32), but due to an implementation bug it uses the equality operator (==) instead of assignment (=) when performing that fallback. As written, that fallback is a no-op and unset widths remain zero. Callers relying on space-width fallback must handle this externally or patch the implementation.
        - If the AFM file does not contain the expected StartCharMetrics/EndCharMetrics block, the function will return the initial all-zero widths list (subject to the bug above).
        - The function reads the entire file into memory; very large AFM files will increase memory usage accordingly.
        - Typical exceptions that can propagate to the caller include FileNotFoundError, IOError/OSError for file access problems, ValueError for unexpected numeric conversions, and IndexError for out-of-range CID assignments.
    
    Practical significance:
        - In the RDKit project, this function is used by the PDF metrics subsystem to obtain character advance widths from AFM font descriptions. Those widths are necessary for measuring text extents, implementing simple text layout, and producing correctly spaced text in PDF outputs generated by RDKit tools. Because the parser is intentionally simple and brittle, it is suitable only for well-formed AFM files matching the expected layout and should be replaced or hardened if used in production workflows that must tolerate diverse AFM formats.
    """
    from rdkit.sping.PDF.pdfmetrics import parseAFMfile
    return parseAFMfile(filename)


################################################################################
# Source: rdkit.sping.PDF.pdfutils.cacheImageFile
# File: rdkit/sping/PDF/pdfutils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def rdkit_sping_PDF_pdfutils_cacheImageFile(filename: str):
    """Processes an image file for PDF embedding and writes a cached, Flate-compressed,
    Ascii85-encoded image file alongside the original image file.
    
    This function is part of RDKit's sping.PDF.pdfutils utilities and is used to
    prepare image data for inclusion in PDF content streams. It opens the image at
    the given filename using the Python Imaging Library (PIL), converts the image to
    RGB, extracts raw image bytes, compresses them with zlib (Flate), encodes the
    compressed bytes with an Ascii Base85 encoder, formats simple PDF inline image
    tokens (BI, ID, EI) and a minimal image dictionary (/W, /H, /BPC, /CS, /F) and
    writes the result to a sidecar file with the same base name and an ".a85"
    extension. The produced .a85 file is intended as a cached representation that
    can be embedded into PDF streams by other pdfutils code to avoid repeated
    reprocessing of the original image.
    
    Args:
        filename (str): Path to the source image file to be processed. This must be
            a path readable by PIL.Image.open and should refer to an image format
            supported by PIL (for example PNG or JPEG). The string is used to open
            the image, to derive image dimensions written into the cached file's
            inline-image dictionary (/W width and /H height), and to construct the
            output filename by replacing the original file extension with ".a85".
            The function does not validate that the path is absolute or relative;
            any IO/OS semantics follow the underlying Python filesystem calls.
    
    Returns:
        None: This function does not return a value. Side effects:
        - Creates a new file named os.path.splitext(filename)[0] + '.a85' in binary
          write mode and writes a small textual representation of the image suitable
          for embedding in a PDF content stream. The written content includes the
          tokens 'BI', an inline-image dictionary line of the form
          '/W <width> /H <height> /BPC 8 /CS /RGB /F [/A85 /Fl]', the 'ID' token,
          the Ascii85-encoded, zlib-compressed image bytes broken into 60-character
          lines, and the 'EI' token, with module-level LINEEND separators between
          lines. After successfully writing the file, the function prints
          "cached image as <cachedname>" to standard output.
    
    Behavioral details, defaults, and failure modes:
    - Image processing: The function opens the image with PIL.Image.open(filename)
      and converts it to RGB with .convert('RGB'). It then calls .tobytes() to get
      raw image data. The code contains an assertion assert len(raw) == imgwidth *
      imgheight which will raise AssertionError if the length of raw bytes does not
      match the product of the extracted width and height. Because .convert('RGB')
      often yields three bytes per pixel, this assertion can fail for images where
      the raw byte length is not equal to width*height; callers should be aware that
      this check is enforced by the current implementation and may raise for some
      images.
    - Compression and encoding: The raw bytes are compressed using zlib.compress
      (Flate) and then encoded via the module-level _AsciiBase85Encode function;
      both operations operate on byte strings. The function assumes _AsciiBase85Encode
      is available in the same module and returns a string containing the Ascii85
      representation.
    - Line wrapping: The encoded data is split into lines of up to 60 characters
      using a StringIO read loop and written with LINEEND (a module-level string)
      separating lines; the code assumes LINEEND and StringIO are defined in the
      module scope.
    - Output file: The cached output is opened with open(cachedname, 'wb') and the
      concatenation LINEEND.join(code) + LINEEND is written as bytes. The function
      therefore assumes that code items and LINEEND are of types that can be
      concatenated and encoded as bytes; if they are text, the write may raise a
      TypeError unless the module ensures proper encoding elsewhere.
    - Exceptions: This function can raise several exceptions originating from its
      operations:
      - IOError/FileNotFoundError if the input filename does not exist or is not
        readable by PIL.
      - PIL.UnidentifiedImageError or other PIL exceptions if the file is not a
        supported image.
      - AssertionError if the raw byte length does not match imgwidth*imgheight
        (see above).
      - NameError if required module-level symbols (_AsciiBase85Encode, StringIO,
        LINEEND, os) are not defined in the module scope.
      - zlib.error if compression fails (unlikely for valid byte input).
      - OSError or IOError during output file creation or writing.
    - Scope and intended usage: This function is intended as a simple caching
      helper within RDKit's PDF utilities to precompute a Flate-compressed,
      Ascii85-encoded representation of an image suitable for inline PDF image
      embedding. It is not a general-purpose image-to-PDF converter and does not
      perform validation beyond the minimal checks described above. Callers who
      integrate this function into workflows that embed images into PDF streams
      should ensure the required module-level helpers and constants exist and may
      need to handle the assertion and IO exceptions in their own code.
    """
    from rdkit.sping.PDF.pdfutils import cacheImageFile
    return cacheImageFile(filename)


################################################################################
# Source: rdkit.sping.PDF.pdfutils.cachedImageExists
# File: rdkit/sping/PDF/pdfutils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def rdkit_sping_PDF_pdfutils_cachedImageExists(filename: str):
    """rdkit.sping.PDF.pdfutils.cachedImageExists determines whether a cached ASCII85-encoded image file (.a85) exists for a given file and whether that cached file is at least as new as the original file. This function is used by RDKit's PDF utilities (sping.PDF.pdfutils) to avoid regenerating embedded images when a previously created, encoded cache is available and up-to-date for PDF composition of cheminformatics visualizations.
    
    The function derives the cache file path by replacing the original file's extension with ".a85" (using os.path.splitext) and checks the filesystem for that cached file. It compares modification timestamps (os.stat(...)[8], the st_mtime field in the os.stat_result sequence) of the original and cached files. The function performs only read/query operations on the filesystem (no writes) and therefore has no side effects beyond filesystem access. Typical failure modes are that os.stat on the original filename will raise an OSError (for example if the original file does not exist or permissions prevent access); such exceptions are not caught within this function and will propagate to the caller. If the cached file is absent, the function returns 0.
    
    Args:
        filename (str): Path to the original image file for which a cached ".a85" counterpart is being checked. In the RDKit PDF generation context, this is the image file produced during molecule rendering or other visualization steps; the function expects a filesystem-accessible path string. The function constructs the cache filename by taking the base name of this path (everything up to but excluding the final dot extension) and appending ".a85" in the same directory.
    
    Returns:
        int: 1 if a cached file with the same base name and a ".a85" extension exists and its modification time is equal to or newer than the modification time of the provided filename (i.e., the cached image can be reused). 0 if no cached ".a85" file exists or if the cached file is older than the original. Note that this function returns integer flags (0 or 1) rather than Python bools. Exceptions from os.stat (e.g., file not found or permission errors on the original filename) will propagate to the caller.
    """
    from rdkit.sping.PDF.pdfutils import cachedImageExists
    return cachedImageExists(filename)


################################################################################
# Source: rdkit.utils.chemutils.ConfigToNumElectrons
# File: rdkit/utils/chemutils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def rdkit_utils_chemutils_ConfigToNumElectrons(
    config: str,
    ignoreFullD: bool = 0,
    ignoreFullF: bool = 0
):
    """Counts the number of electrons appearing in an electronic configuration string used in RDKit cheminformatics utilities.
    
    This function is used in RDKit code paths that need a simple integer count of valence electrons extracted from an electronic configuration string (for example in heuristics for valence checking, bonding, or charge handling). The implementation parses a space-separated configuration string, sums integer superscript counts from each orbital token after the first token, and optionally ignores full d or f shells when those shells appear and meet the exact fullness criteria used in the code. There are no external side effects; the function returns an integer electron count computed from the input string.
    
    Args:
        config (str): The electronic configuration string to parse. The string is split on space characters and tokens at indices 1..N are interpreted as orbital entries of the form "<shell><orbital>^<count>" (for example "2s^2" or "3d^10"). Note that the implementation intentionally starts processing at the second token (index 1) and therefore the first token (index 0) is ignored; callers should supply a leading token (commonly an element symbol or a placeholder) if they intend all orbital tokens to be considered. A configuration with fewer than two space-separated tokens yields a count of 0. Malformed tokens that do not contain a "^" or whose superscript part cannot be converted to int will raise a ValueError or IndexError as produced by the underlying Python operations.
        ignoreFullD (bool): If true (nonzero), full d shells are not counted toward the returned electron total. Concretely, when this flag is true and an orbital token contains the letter "d" and its superscript integer equals 10, that token's electrons are treated as 0 instead of 10, but only when the configuration has more than two tokens (len(config.split(' ')) > 2). Default is 0 (treated as False in Python); use True/1 to enable ignoring full d shells. This behavior is intended to support domain-specific conventions in counting valence electrons for transition metals within RDKit utilities.
        ignoreFullF (bool): If true (nonzero), full f shells are not counted toward the returned electron total. Concretely, when this flag is true and an orbital token contains the letter "f" and its superscript integer equals 14, that token's electrons are treated as 0 instead of 14, but only when the configuration has more than two tokens (len(config.split(' ')) > 2). Default is 0 (treated as False in Python); use True/1 to enable ignoring full f shells. This behavior is intended to support domain-specific conventions in counting valence electrons for lanthanides/actinides within RDKit utilities.
    
    Returns:
        int: The number of valence electrons computed from the configuration string according to the parsing rules above. The returned integer is the sum of the integer superscripts extracted from processed orbital tokens, after applying the optional ignoreFullD/ignoreFullF filters. No other side effects occur.
    
    Raises:
        IndexError: If an orbital token does not contain a "^" and the code attempts to index the split result.
        ValueError: If the superscript part of an orbital token cannot be converted to int.
        TypeError: If config is not a string (the function calls str.split on the input).
    """
    from rdkit.utils.chemutils import ConfigToNumElectrons
    return ConfigToNumElectrons(config, ignoreFullD, ignoreFullF)


################################################################################
# Source: rdkit.utils.chemutils.GetAtomicData
# File: rdkit/utils/chemutils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def rdkit_utils_chemutils_GetAtomicData(
    atomDict: dict,
    descriptorsDesired: list,
    dBase: str = "/opt/conda/share/RDKit/Data/atomdb.gdb",
    table: str = "atomic_data",
    where: str = "",
    user: str = "sysdba",
    password: str = "masterkey",
    includeElCounts: int = 0
):
    """GetAtomicData pulls atomic property descriptors from an RDKit atomic database and populates a provided dictionary with per-atom descriptor dictionaries keyed by atomic name. This function is used within the RDKit cheminformatics library to load atomic data (for example, electronic configuration strings and numeric descriptors) from a local database file into an in-memory mapping that downstream code (descriptor calculators, fingerprint generators, machine-learning featurizers) can query. The function connects to a database using rdkit.Dbase.DbModule.connect, issues a SQL SELECT for the requested descriptor columns, and for each row creates a dictionary of descriptor-name: value entries which it stores into the supplied atomDict under the atom NAME. If includeElCounts is nonzero, additional valence electron count fields (NVAL, NVAL_NO_FULL_F, NVAL_NO_FULL_D, NVAL_NO_FULL) are computed from the atomic CONFIG string using ConfigToNumElectrons and added to each atom entry.
    
    Args:
        atomDict (dict): A mutable mapping that will be populated in-place. After successful execution, atomDict[name] will be a dict of descriptor names to values for each atom name produced by the database. In the RDKit domain this mapping is typically used to provide per-atom lookup data (atomic configuration, formal charges, custom numeric descriptors) required by descriptor calculators and other cheminformatics algorithms. The function mutates this dict; no new dict is returned.
        descriptorsDesired (list): A list of descriptor column names (strings) to request from the database. Values in this list are converted to uppercase before use. The function ensures that 'NAME' is included (it will be appended if missing) because NAME is used as the key for atomDict. If includeElCounts is nonzero, 'CONFIG' will also be appended if not present because electron-count fields are derived from the CONFIG column. Any entries in the code-local extraFields set ('NVAL', 'NVAL_NO_FULL_F', 'NVAL_NO_FULL_D', 'NVAL_NO_FULL') are removed from descriptorsDesired prior to the SELECT because those fields are computed by this function rather than pulled directly from the database.
        dBase (str = "/opt/conda/share/RDKit/Data/atomdb.gdb"): Path or identifier of the database to connect to. This string is passed to rdkit.Dbase.DbModule.connect along with user and password to open the atomic data database file used by RDKit. The default value is the RDKit distribution atom database path; supplying a different path allows loading custom atomic data files.
        table (str = "atomic_data"): Nominal database table name for atomic data. In the current implementation this parameter is accepted but not used: the SQL command is built using the literal table name 'atomic_data' (see implementation). Do not rely on this parameter to change the table unless the implementation is updated.
        where (str = ""): An optional SQL clause fragment appended verbatim to the SELECT statement after the table name. Because the implementation concatenates this fragment directly into the SQL command, include any required leading keyword (for example, "WHERE element = 'C'") and ensure it is valid for the target database. Passing an empty string results in no WHERE filter and selects all rows.
        user (str = "sysdba"): Username to pass to rdkit.Dbase.DbModule.connect when opening the database. This parameter is significant when the DB backend requires authentication; for the default RDKit atom DB the bundled credentials are typically sufficient.
        password (str = "masterkey"): Password to pass to rdkit.Dbase.DbModule.connect when opening the database. See user for significance and default usage.
        includeElCounts (int = 0): Integer flag (0 or nonzero) indicating whether valence electron count fields should be computed and added to each atom's descriptor dict. If nonzero, the function ensures 'CONFIG' is requested from the DB (appending it if necessary), and after loading each row it calls ConfigToNumElectrons to compute and add the fields 'NVAL', 'NVAL_NO_FULL_F', 'NVAL_NO_FULL_D', and 'NVAL_NO_FULL'. These computed fields are commonly used in electronic-structure informed descriptors and ML features.
    
    Behavior and side effects:
        The function converts descriptorsDesired entries to uppercase, appends mandatory descriptors as described above, removes any pre-existing computed electron-count field names, and constructs a comma-separated SELECT column list. It connects to the database via rdkit.Dbase.DbModule.connect(dBase, user, password) and acquires a cursor. The SQL command executed is of the form "select <COLUMNS> from atomic_data <WHERE_FRAGMENT>" where <COLUMNS> is the comma-joined descriptors and <WHERE_FRAGMENT> is the provided where string. If executing the SQL raises an exception, the function prints a diagnostic "Problems executing command:" followed by the SQL command and returns early (None). A failure to connect to the database (DbModule.connect raising an exception) is not caught by this function and will propagate to the caller. On success, the function iterates the fetched rows; for each row it builds a dictionary mapping each requested descriptor name to the corresponding column value, obtains the atom name from the 'NAME' entry, and stores the descriptor dictionary in atomDict under that name. If includeElCounts is nonzero, the function reads the 'CONFIG' value and uses ConfigToNumElectrons to compute and insert the additional NVAL* fields for that atom. The function does not close the connection or cursor explicitly in the current implementation, and the table parameter is ignored.
    
    Failure modes and caveats:
        If the database connection fails, an exception from DbModule.connect will propagate to the caller. If the SQL execution fails (for example, due to invalid column names or a malformed where clause) the function prints an error message and returns None without modifying atomDict further. If the number of columns returned by the database does not match the number of requested descriptors, indexing into the row by position may raise an IndexError or result in incorrect mappings. If includeElCounts is nonzero but the CONFIG column is missing or contains unexpected values, ConfigToNumElectrons may raise an exception; the function does not handle such exceptions. Because the SQL is assembled by string concatenation, ensure descriptorsDesired entries and the where fragment are safe and valid for the database backend in use.
    
    Returns:
        None: This function does not return a value. Instead it mutates the provided atomDict in-place by adding an entry for each atom name fetched from the database; each atom entry is a dict of descriptor-name: value pairs. On SQL execution failure the function returns early (None) after printing a diagnostic. Exceptions from the database connection or descriptor conversion functions may propagate to the caller.
    """
    from rdkit.utils.chemutils import GetAtomicData
    return GetAtomicData(
        atomDict,
        descriptorsDesired,
        dBase,
        table,
        where,
        user,
        password,
        includeElCounts
    )


################################################################################
# Source: rdkit.utils.chemutils.SplitComposition
# File: rdkit/utils/chemutils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def rdkit_utils_chemutils_SplitComposition(compStr: str):
    """SplitComposition parses a simple chemical composition string into an ordered list of element/count pairs suitable for lightweight stoichiometry and descriptor preprocessing in cheminformatics workflows (RDKit context). The function is intended for very simple, flat composition strings (for example 'Fe3Al') and returns a sequence of element symbols with their associated counts that can be consumed by code that computes elemental contributions, simple composition-based descriptors, or stoichiometric checks.
    
    This parser uses a fixed regular expression that recognizes an uppercase letter followed by an optional lowercase letter as the element symbol and an optional numeric portion (digits and optional decimal point) as the count. The numeric portion, when present, is converted to a floating-point value; when absent the count is reported as the integer 1. The method does not modify external state and returns a new list; it is not intended to parse complex chemical formula syntax such as parentheses, hydration/dot notation, charges, isotopic labels, nested groups, or other annotations.
    
    Args:
        compStr (str): A simple chemical composition string to be parsed. This should be a flat formula where element symbols are one uppercase letter optionally followed by one lowercase letter, and an optional numeric coefficient (integer or decimal) immediately follows the symbol. Practical examples in the RDKit domain include input used to compute element-based descriptors or to check simple stoichiometries, for example 'Fe3Al' or 'C6H6'. If compStr is not a string a TypeError may be raised by the underlying regex operations. Leading/trailing whitespace is not specially normalized by this function; such whitespace will simply be ignored by the regex if it does not match the pattern.
    
    Returns:
        list: An ordered list of 2-tuples representing the parsed composition. Each tuple is (element_symbol, count) where element_symbol is a str (the matched one- or two-character element symbol) and count is numeric: it is a float when a numeric part is present in the input (converted via float()), and it is the integer 1 when no numeric part follows the element symbol. For example, input 'Fe3Al' yields [('Fe', 3.0), ('Al', 1)]. If the input contains no matches the function returns an empty list. The returned list is a new object and there are no side effects. Failure modes include inability to handle complex formula constructs (parentheses, multipliers on groups, dot hydration, charges, isotopes) and possible exceptions if a non-string is provided.
    """
    from rdkit.utils.chemutils import SplitComposition
    return SplitComposition(compStr)


################################################################################
# Source: rdkit.utils.listutils.CompactListRepr
# File: rdkit/utils/listutils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def rdkit_utils_listutils_CompactListRepr(lst: list):
    """CompactListRepr provides a compact, human-readable string representation of a sequence by collapsing consecutive identical elements into run-length notation. This function is part of rdkit.utils.listutils and is intended as a small utility within the RDKit cheminformatics toolkit to make lists that frequently occur in cheminformatics workflows (for example, atom index lists, fingerprint bit arrays, or descriptor sequences) easier to inspect in logs, debugging output, or textual summaries.
    
    Args:
        lst (list): The input sequence to summarize. By signature this is documented as a Python list; the implementation operates on any object that supports len() and indexing and compares elements with != (the original examples in the source show that tuples and strings are handled as sequences of elements/characters). The function inspects the sequence element by element, groups consecutive equal elements, and uses repr() on each representative element when building the output. The caller should pass the sequence to be summarized; passing non-sequence objects that do not implement len() and indexing will raise a TypeError or other Python exceptions coming from the attempted operations.
    
    Returns:
        str: A compact string that describes the input sequence as a concatenation of run components. Each run of identical consecutive elements is rendered as "[<repr(element)>]" when the run length is 1, or as "[<repr(element)>]*<count>" when the run length is greater than 1. Components are joined with '+' signs. For an empty input sequence the function returns the literal string '[]'. This return value is a new string (the function has no side effects on the input sequence) and is intended for display/logging purposes (for example, to succinctly show repetitive patterns in fingerprint bit vectors or lists of atom labels). The function runs in linear time with respect to the length of the input sequence and uses additional memory proportional to the number of runs; unexpected behavior can occur if elements mutate during iteration or if element equality semantics are non-standard.
    """
    from rdkit.utils.listutils import CompactListRepr
    return CompactListRepr(lst)


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
