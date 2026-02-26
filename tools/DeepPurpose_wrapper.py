"""
Regenerated Google-style docstrings for module 'DeepPurpose'.
README source: others/readme/DeepPurpose/README.md
Generated at: 2025-12-02T01:16:01.105705Z

Total functions: 31
"""


################################################################################
# Source: DeepPurpose.pybiomed_helper.CalculateAAComposition
# File: DeepPurpose/pybiomed_helper.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def DeepPurpose_pybiomed_helper_CalculateAAComposition(ProteinSequence: str):
    """DeepPurpose.pybiomed_helper.CalculateAAComposition computes the percent composition of the 20 standard amino acids for a given protein sequence. This function is a simple, per-sequence featurization used in DeepPurpose for protein encoding and preprocessing steps that support downstream tasks such as drug-target interaction (DTI) prediction, protein-protein interaction (PPI) prediction, and protein function prediction. The returned composition can be used as a lightweight input feature or baseline encoding in model training, evaluation, repurposing, and virtual screening workflows described in the DeepPurpose README.
    
    Args:
        ProteinSequence (str): A protein sequence provided as a Python string of single-letter amino acid codes. The function expects a "pure" protein sequence composed of the standard one-letter amino acid codes present in the module-level AALetter list (20 amino acids). ProteinSequence is treated literally (case-sensitive): occurrences are counted using str.count for each code in AALetter and the denominator is the total length len(ProteinSequence). Therefore, passing lowercase letters, whitespace, digits, symbols, or non-standard letters will alter counts and produce skewed percentage values because they contribute to the sequence length but are not matched to AALetter entries. Passing an empty string (zero-length ProteinSequence) will lead to a division-by-zero failure (ZeroDivisionError). There are no implicit conversions or side effects; if inputs might be lowercase or contain non-amino-acid characters, normalize or validate them before calling this function.
    
    Returns:
        dict: A dictionary mapping each amino acid one-letter code (each key corresponds to an element of the module-level AALetter sequence of 20 codes) to a float representing that residue's composition as a percentage of the input sequence. Each percentage is computed as (count of that letter in ProteinSequence / len(ProteinSequence)) * 100 and rounded to three decimal places. The values are suitable for use as numeric protein features in DeepPurpose pipelines; due to rounding the values may not sum to exactly 100.0. No other side effects occur.
    """
    from DeepPurpose.pybiomed_helper import CalculateAAComposition
    return CalculateAAComposition(ProteinSequence)


################################################################################
# Source: DeepPurpose.pybiomed_helper.CalculateAADipeptideComposition
# File: DeepPurpose/pybiomed_helper.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def DeepPurpose_pybiomed_helper_CalculateAADipeptideComposition(ProteinSequence: str):
    """CalculateAADipeptideComposition computes a fixed-length numerical feature vector that encodes the amino acid (AA) composition, dipeptide composition, and 3-mer (tripeptide/spectrum) composition for a given protein primary sequence. In the DeepPurpose toolkit (used for drug-target interaction prediction, protein property/function prediction, PPI, and related molecular modeling tasks), this function is used to transform a raw protein sequence (single-letter amino acid codes) into a deterministic numeric representation (feature vector) suitable for machine learning models, virtual screening, and repurposing workflows.
    
    Args:
        ProteinSequence (str): A protein primary sequence provided as a string of single-letter amino acid codes (e.g., "MSEQ..."). In the DeepPurpose context, this sequence represents the target protein whose composition-based encoding will be used as input features for models (DTI, PPI, ProteinPred). The function expects a "pure" protein sequence; sequences containing non-standard characters, whitespace, or gaps are not guaranteed to be handled and may cause errors or incorrect features. The caller is responsible for providing an appropriately preprocessed sequence (uppercase or lowercase single-letter codes as accepted by the helper composition functions in this module).
    
    Returns:
        numpy.ndarray: A one-dimensional numpy array containing the concatenated numeric composition values in the exact order produced by the function: first the amino-acid composition vector, then the dipeptide composition vector, and finally the 3-mer (spectrum) composition vector. The concatenated feature vector length is 8420 (the sum of the lengths of the individual composition vectors). Each element is a numeric value (count or normalized composition, as produced by the underlying helper functions CalculateAAComposition, CalculateDipeptideComposition, and GetSpectrumDict). This array is suitable for direct use as input features to downstream DeepPurpose models and machine learning pipelines.
    
    Behavior and side effects:
        The function builds an intermediate dictionary that aggregates named composition features from CalculateAAComposition(ProteinSequence), CalculateDipeptideComposition(ProteinSequence), and GetSpectrumDict(ProteinSequence), in that exact sequence. It then returns a numpy array of the dictionary's values in insertion order (i.e., AA features followed by dipeptide features followed by 3-mer features). There are no external side effects (no file I/O or network access). The function is deterministic for a given input sequence.
    
    Defaults and practical significance:
        The produced 8420-dimensional vector is a composition-based encoding commonly used in DeepPurpose workflows when simpler, alignment-free protein encodings are desired. It is useful for rapid featurization in large-scale virtual screening, repurposing, and as one of multiple protein encodings when training or applying DTI and protein prediction models.
    
    Failure modes and error handling:
        This function relies on the helper functions CalculateAAComposition, CalculateDipeptideComposition, and GetSpectrumDict. If ProteinSequence is empty, contains invalid or non-amino-acid characters, or is otherwise incompatible with those helpers, the helper functions may raise exceptions or produce undefined results; such exceptions will propagate to the caller. If the caller needs both feature names and values, they should invoke the helper functions directly, because this function returns only the numeric vector of values (the intermediate dictionary with feature names is not returned).
    """
    from DeepPurpose.pybiomed_helper import CalculateAADipeptideComposition
    return CalculateAADipeptideComposition(ProteinSequence)


################################################################################
# Source: DeepPurpose.pybiomed_helper.CalculateConjointTriad
# File: DeepPurpose/pybiomed_helper.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def DeepPurpose_pybiomed_helper_CalculateConjointTriad(proteinsequence: str):
    """CalculateConjointTriad computes the Conjoint Triad (CTriad) encoding for a protein sequence and returns the 343-dimensional count vector used as a fixed-length protein descriptor in DeepPurpose models (for tasks such as drug–target interaction prediction, protein–protein interaction prediction, and protein function prediction). This function maps the input single-letter amino acid sequence to a 1..7 grouped alphabet via the helper function _Str2Num, then counts occurrences of every length-3 contiguous group triad over the grouped alphabet (7^3 = 343 possible triads) and returns those counts in a numpy array in a deterministic order that DeepPurpose expects for downstream encoding/learning pipelines.
    
    Args:
        proteinsequence (str): A pure protein sequence given as a Python string of single-letter amino acid codes (e.g., "MSTNPKPQR"). This argument is the primary input to the Conjoint Triad encoding: it is first converted by the internal helper _Str2Num into a string of digits '1'..'7' representing amino-acid groups, and then all contiguous length-3 group triads are counted. In the DeepPurpose context, this parameter represents the target protein whose sequence-derived descriptor will be used as an input feature vector to DTI, PPI, or protein-function prediction models. The caller must provide a sequence consisting only of standard single-letter amino-acid characters (no whitespace or non-letter characters); providing an empty string is permitted and yields an all-zero feature vector.
    
    Returns:
        numpy.ndarray: A 1-D numpy array of length 343 containing integer counts for each Conjoint Triad feature. The ordering of elements corresponds to the lexicographic order produced by the nested loops over group indices i, j, k (i from 1 to 7, j from 1 to 7, k from 1 to 7), i.e., features for triads "111", "112", ..., "117", "121", ..., "777". Each element is the count of occurrences of that specific grouped triad in the grouped representation of the input sequence. This fixed-length numeric vector is suitable for concatenation with other encodings or direct input to machine learning models in the DeepPurpose framework.
    
    Behavior, side effects, and failure modes:
        - The function has no side effects: it does not modify global state or write files; it only returns the computed numpy array.
        - Internally it calls _Str2Num(proteinsequence) to map amino-acid letters to group digits; any exceptions raised by _Str2Num (for example due to invalid or non-standard characters) are propagated to the caller.
        - If proteinsequence is not a str, a TypeError may be raised by upstream operations; callers should pass a Python string.
        - An empty proteinsequence results in a numpy array of zeros (no triads found).
        - Runtime and memory cost scale linearly with sequence length for the mapping step and counting is performed via string.count for each of the 343 triads; for extremely long sequences this may incur increased CPU time but the output size remains constant (343).
        - The output format and ordering are deterministic and intended to align with other DeepPurpose components that expect the Conjoint Triad 343-feature layout.
    """
    from DeepPurpose.pybiomed_helper import CalculateConjointTriad
    return CalculateConjointTriad(proteinsequence)


################################################################################
# Source: DeepPurpose.pybiomed_helper.CalculateDipeptideComposition
# File: DeepPurpose/pybiomed_helper.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def DeepPurpose_pybiomed_helper_CalculateDipeptideComposition(ProteinSequence: str):
    """Calculate the composition of dipeptides for a given protein sequence, returning a fixed-size feature dictionary commonly used as a simple protein encoding in DeepPurpose workflows (for example as an input feature for DTI, PPI, or protein function prediction models). This function iterates over the module-level AALetter sequence to form every ordered pair of amino-acid one-letter codes and computes the percentage composition of each dipeptide in the provided ProteinSequence.
    
    Args:
        ProteinSequence (str): A protein primary sequence provided as a contiguous string of amino-acid one-letter codes (e.g., "MSTNPKPQR..."). This argument is treated as a raw sequence and is not validated for allowed characters inside this function; characters not present in the module-level AALetter will simply produce zero counts for dipeptides containing them. The function uses Python's str.count(substring) to count occurrences of each dipeptide substring in ProteinSequence; note that str.count does not count overlapping occurrences, so repeated-residue patterns (for example "AAA" with dipeptide "AA") will be counted according to Python's non-overlapping substring semantics, which can undercount overlapping dipeptides compared with a sliding-window count.
    
    Returns:
        dict: A dictionary mapping each dipeptide string (constructed from the ordered pairs in the module-level AALetter) to a float percentage value. Each value is computed as (count_of_dipeptide / (LengthSequence - 1)) * 100 and rounded to two decimal places, where LengthSequence is the length of ProteinSequence. The function populates entries for the full Cartesian product of AALetter (the implementation produces 400 dipeptide keys when AALetter contains 20 amino-acid letters), so the returned dict has a fixed set of keys regardless of which dipeptides actually occur in ProteinSequence. Practical significance: these percentage features are intended to be used as an input encoding for DeepPurpose models (e.g., for drug-target interaction, protein-protein interaction, or protein function prediction) where simple, fixed-length handcrafted descriptors are useful.
    
    Behavior and failure modes:
        The function computes LengthSequence = len(ProteinSequence) and divides counts by (LengthSequence - 1). If ProteinSequence has length less than 2, this division will raise a ZeroDivisionError. The function has no other side effects (it does not modify global state) but it depends on the module-level variable AALetter to determine the dipeptide keys; if AALetter is changed, the set and order of keys in the returned dict will change accordingly. Because counts are obtained via str.count, overlapping dipeptide occurrences are not counted, and rounding to two decimal places means the sum of returned percentages may not equal exactly 100. No input normalization (such as uppercasing) or validation is performed by this function; callers should pre-process sequences as needed for the DeepPurpose pipeline.
    """
    from DeepPurpose.pybiomed_helper import CalculateDipeptideComposition
    return CalculateDipeptideComposition(ProteinSequence)


################################################################################
# Source: DeepPurpose.pybiomed_helper.GetAAComposition
# File: DeepPurpose/pybiomed_helper.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def DeepPurpose_pybiomed_helper_GetAAComposition(ProteinSequence: str):
    """DeepPurpose.pybiomed_helper.GetAAComposition calculates the normalized composition of amino acids in a protein sequence and returns a dictionary of fractional frequencies for use as a simple protein encoding feature in DeepPurpose models (DTI, PPI, Protein Function Prediction, virtual screening, and related tasks).
    
    This function computes, for each amino acid code listed in the module-level AALetter sequence, the fraction of characters in the provided ProteinSequence that match that code. The output is intended to represent the composition of the 20 standard amino acids (one entry per code in AALetter) as a normalized feature vector suitable for downstream machine learning models in the DeepPurpose toolkit.
    
    Args:
        ProteinSequence (str): A protein primary sequence provided as a string of single-letter amino acid codes. This function treats the string as case-sensitive and counts occurrences using str.count. ProteinSequence should ideally use uppercase single-letter codes corresponding to the entries in the module-level AALetter sequence; otherwise counts for mismatched case or non-standard characters will be zero unless those characters appear in AALetter. An empty string will lead to a division-by-zero error (ZeroDivisionError). No mutation or normalization is performed on ProteinSequence by this function.
    
    Returns:
        dict: A dictionary mapping each amino acid single-letter code from the module-level AALetter to a float representing its fractional composition in ProteinSequence. Each value is computed as (count of that letter in ProteinSequence) / (length of ProteinSequence) and rounded to three decimal places using round(..., 3). The keys of the returned dict are exactly the elements iterated from AALetter; therefore, the function's practical output corresponds to the composition of the 20 standard amino acids only if AALetter contains those 20 standard uppercase single-letter codes. Due to rounding, the sum of values may be approximately but not exactly 1.0. No external side effects occur, but correct operation depends on AALetter being defined in the module scope; if AALetter is missing or contains unexpected entries, the returned dictionary keys and semantics will reflect that.
    """
    from DeepPurpose.pybiomed_helper import GetAAComposition
    return GetAAComposition(ProteinSequence)


################################################################################
# Source: DeepPurpose.pybiomed_helper.GetAPseudoAAC
# File: DeepPurpose/pybiomed_helper.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def DeepPurpose_pybiomed_helper_GetAPseudoAAC(
    ProteinSequence: str,
    lamda: int = 30,
    weight: float = 0.5
):
    """DeepPurpose.pybiomed_helper.GetAPseudoAAC computes type II pseudo-amino acid composition (PAAC) descriptors for a protein sequence and returns a fixed-length numerical encoding useful as a protein feature vector in DeepPurpose tasks such as drug–target interaction (DTI) prediction, protein function prediction, and other protein-based machine learning pipelines.
    
    Args:
        ProteinSequence (str): A pure protein sequence represented as a string of standard single-letter amino acid codes (for example, "MSTNPKPQR"). This sequence is the primary input from which PAAC descriptors are computed. The length of ProteinSequence determines the maximum allowed lamda: lamda should not be greater than len(ProteinSequence). Supplying non-string values or sequences containing invalid/non-standard characters may cause internal helper routines (GetAPseudoAAC1/GetAPseudoAAC2) to raise exceptions or produce incorrect descriptor values.
        lamda (int): Non-negative integer that specifies the number of sequence-order correlation (pseudo) components to compute. The function returns 20 + lamda descriptors: 20 conventional amino-acid composition components plus lamda pseudo-amino-acid composition components. Typical practical choices are small-to-moderate integers (for example, 0, 5, 10, 15). Requirements and failure modes: lamda must be an integer >= 0 and should not exceed the length of ProteinSequence; if lamda = 0 the function returns only the conventional 20-dimensional amino acid composition; if lamda is larger than the sequence length or negative, the computation is not valid and the called helper functions may raise an error or produce meaningless output. Default: 30.
        weight (float): Weighting factor (float) used to balance the conventional 20 amino-acid composition components against the additional lamda pseudo components. This parameter affects the relative contribution of sequence-order information versus simple composition in the resulting descriptors. Recommended practical range based on original PAAC conventions is approximately 0.05 to 0.7; using values outside this range may lead to descriptor sets that emphasize one component excessively or behave differently than typical PAAC encodings. Supplying non-float types may be accepted by Python (via implicit conversion) but is not recommended; negative weights or extremely large weights may produce atypical or unstable descriptor magnitudes. Default: 0.5.
    
    Returns:
        dict: A dictionary mapping descriptor names (strings) to numeric descriptor values (typically floats). The returned dictionary contains 20 + lamda PAAC descriptors: keys and values produced by two internal helper routines called within this function. Specifically, GetAPseudoAAC1(ProteinSequence, lamda, weight) contributes the conventional 20 amino-acid composition components and GetAPseudoAAC2(ProteinSequence, lamda, weight) contributes the lamda pseudo (sequence-order-correlation) components; this function merges their outputs and returns the combined mapping. If lamda = 0, the dictionary contains only the 20 conventional amino-acid composition entries. There are no external side effects (no file I/O or global state modification) beyond computing and returning this dictionary.
    
    Behavior notes, side effects, and failure modes:
        - This function is a pure computational helper intended to produce fixed-length numeric encodings of protein sequences for downstream machine learning within the DeepPurpose toolkit (e.g., DTI, protein property prediction, repurposing workflows). It calls GetAPseudoAAC1 and GetAPseudoAAC2 internally and merges their outputs.
        - The length of the returned feature vector depends directly on lamda (20 + lamda). Choosing lamda that is too large relative to sequence length is invalid; the function does not enforce domain-specific substitutions or sequence cleaning, so the caller should ensure ProteinSequence contains valid amino-acid single-letter codes and is sufficiently long.
        - Defaults (lamda=30, weight=0.5) are provided for convenience but may not be appropriate for all datasets or model architectures; users should tune lamda and weight according to dataset size, protein lengths, and modeling needs.
        - Invalid parameter types or out-of-range values (e.g., negative lamda, non-string ProteinSequence, non-finite weight) can cause exceptions in the helper functions or produce invalid descriptors; callers should validate inputs before invoking this function.
    """
    from DeepPurpose.pybiomed_helper import GetAPseudoAAC
    return GetAPseudoAAC(ProteinSequence, lamda, weight)


################################################################################
# Source: DeepPurpose.pybiomed_helper.GetAPseudoAAC1
# File: DeepPurpose/pybiomed_helper.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def DeepPurpose_pybiomed_helper_GetAPseudoAAC1(
    ProteinSequence: str,
    lamda: int = 30,
    weight: float = 0.5
):
    """GetAPseudoAAC1 computes the Type II pseudo-amino acid composition (APAAC) features used by DeepPurpose to encode proteins for tasks such as drug–target interaction (DTI) prediction, protein–protein interaction (PPI) prediction, protein function prediction, and other molecular modeling workflows. Concretely, this function returns the first 20 APAAC descriptors (one per standard amino acid in the library's AALetter ordering) that combine normalized amino acid composition with sequence-order correlation information derived from hydrophobicity and hydrophilicity. These features are commonly used as fixed-length protein descriptors that capture both composition and short-range sequence-order effects for downstream machine learning models in DeepPurpose.
    
    Args:
        ProteinSequence (str): The protein primary sequence provided as a string of single-letter amino acid codes (e.g., "MST..."). This sequence is the input whose amino acid composition and sequence-order correlation factors are used to compute APAAC descriptors. The function delegates parsing and validation to helper functions GetAAComposition and GetSequenceOrderCorrelationFactorForAPAAC; if the sequence contains non-standard letters, those helpers determine the resulting behavior (they may return zeros or raise an exception).
        lamda (int): The number of sequence-order correlation tiers to include when computing the aggregate sequence-order effect (the code sums correlation factors for k = 1..lamda). This integer controls how much sequence-order information beyond single-residue composition contributes to the denominator normalization. Default is 30. Larger lamda increases the contribution of the right-hand sequence-order term (rightpart) to the normalization; extremely large values may increase runtime and numerical values of the rightpart.
        weight (float): A scalar weight applied to the aggregated sequence-order term before combining it with the composition-based term. The denominator used to normalize composition is (1 + weight * rightpart). Default is 0.5. Note that if weight * rightpart equals -1, a ZeroDivisionError will occur; negative weights are permitted by the signature but may produce unexpected normalization or division errors.
    
    Returns:
        dict: A dictionary mapping feature names to floating-point values. Each key is a string of the form "APAACn" where n is an integer index starting at 1 corresponding to the position of the amino acid in the global AALetter ordering (typically the 20 standard amino acids). Each value is a float equal to the normalized amino acid composition for that residue after incorporation of sequence-order correlations, rounded to three decimal places. The returned dictionary therefore contains one APAAC descriptor per amino acid letter in AALetter (commonly 20 descriptors) and is ready to be used as a fixed-length protein feature vector in DeepPurpose models.
    
    Behavioral notes, side effects, and failure modes:
        - This function computes rightpart = sum_{k=1..lamda} GetSequenceOrderCorrelationFactorForAPAAC(ProteinSequence, k) and AAC = GetAAComposition(ProteinSequence), then sets result["APAACi"] = round(AAC[residue] / (1 + weight * rightpart), 3) for each residue in AALetter.
        - The function relies on the global AALetter ordering and the helper functions GetAAComposition and GetSequenceOrderCorrelationFactorForAPAAC; their implementations determine handling of edge cases (e.g., unknown letters, empty sequences).
        - There are no file or network side effects; the function performs only in-memory computation.
        - Time complexity grows with lamda and sequence length due to repeated calls to the correlation-factor helper; setting lamda very large increases runtime.
        - If weight * rightpart == -1 (for example, with certain negative weight choices), a ZeroDivisionError will occur. If the provided ProteinSequence is empty or contains unexpected characters, behavior depends on the helper functions and may raise exceptions or return feature values of zero.
    """
    from DeepPurpose.pybiomed_helper import GetAPseudoAAC1
    return GetAPseudoAAC1(ProteinSequence, lamda, weight)


################################################################################
# Source: DeepPurpose.pybiomed_helper.GetAPseudoAAC2
# File: DeepPurpose/pybiomed_helper.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def DeepPurpose_pybiomed_helper_GetAPseudoAAC2(
    ProteinSequence: str,
    lamda: int = 30,
    weight: float = 0.5
):
    """GetAPseudoAAC2 computes the type II pseudo-amino acid composition (PAAC) correlation features for a protein sequence as implemented in DeepPurpose.pybiomed_helper. It returns the "last lamda" block of PAAC descriptors (the sequence-order correlation terms based on hydrophobicity and hydrophilicity) that are commonly appended to the 20 conventional amino-acid composition features when encoding proteins for tasks in the DeepPurpose framework (for example, drug-target interaction, protein-protein interaction, and protein function prediction).
    
    This function iteratively calls GetSequenceOrderCorrelationFactorForAPAAC for k = 1..lamda to collect two sequence-order correlation factors per k (hydrophobicity- and hydrophilicity-based factors), normalizes them using the standard PAAC normalization term temp = 1 + weight * sum(rightpart), and returns these normalized correlation features scaled to percentages and rounded to three decimal places. The returned descriptors are named "PAAC21" through "PAAC{20+2*lamda}" (i.e., the PAAC indices immediately following the 20 conventional amino-acid composition descriptors).
    
    Args:
        ProteinSequence (str): Protein primary sequence as a single-letter amino-acid string. This is the input sequence to be encoded into PAAC correlation features. In DeepPurpose this sequence is used as the protein input encoding for models that accept APAAC/PAAC-style descriptors. The function delegates sequence-order correlation calculations to GetSequenceOrderCorrelationFactorForAPAAC, so ProteinSequence must be acceptable to that helper (for example, containing recognized amino-acid characters); invalid characters or an incompatible format may cause the helper to raise an exception.
        lamda (int): Number of correlation tiers (default 30). For each integer k from 1 to lamda the function obtains two correlation factors, producing 2 * lamda descriptors in total. Practically, lamda controls the maximal sequence separation over which order correlations are computed: larger lamda captures longer-range sequence-order information but increases computation and may be invalid or uninformative for very short sequences. If lamda <= 0, no correlation descriptors are produced and an empty result dictionary is returned. The function does not internally constrain lamda beyond using it as the loop bound; any required validation should be done by the caller or will surface as an error from the underlying helper.
        weight (float): Weighting factor (default 0.5) applied to the sequence-order correlation terms during normalization. The implementation uses temp = 1 + weight * sum(rightpart) as the normalization denominator and computes each returned value as (weight * correlation_value / temp) * 100, rounded to three decimal places. In practice, weight balances the relative contribution of the sequence-order correlations against the implicit 20 amino-acid composition terms (which are not computed by this function). Passing different weight values adjusts how strongly these correlation features influence downstream models.
    
    Returns:
        dict: Mapping of descriptor name strings to float values. Keys follow the form "PAAC21", "PAAC22", ..., up to "PAAC{20+2*lamda}", corresponding to the 2*lamda sequence-order correlation descriptors produced for k = 1..lamda (two per k). Values are the normalized contributions expressed as percentages (multiplied by 100) and rounded to three decimal places. If no descriptors are produced (for example, lamda <= 0), an empty dict is returned.
    
    Behavior and side effects:
        This function is pure (no I/O or global state modifications) and returns a new dictionary constructed from computed correlation factors. It calls GetSequenceOrderCorrelationFactorForAPAAC lamda times; therefore runtime and computational cost increase linearly with lamda and depend on the behavior of that helper. The function computes only the PAAC "last lamda" block (sequence-order correlation features) and does not compute or return the first 20 amino-acid composition features. Users who require a full PAAC vector (20 composition + 2*lamda correlation features) should combine the output of this function with conventional amino-acid composition counts or use a higher-level encoder in DeepPurpose that provides the complete vector.
    
    Failure modes and exceptions:
        The function does no explicit input validation beyond relying on Python types and the helper function. Errors can propagate from GetSequenceOrderCorrelationFactorForAPAAC (for example, due to invalid sequence characters, insufficient sequence length for requested k, or other domain-specific validation failures). If such errors occur, they will be raised to the caller. Large lamda values relative to sequence length can lead to meaningless correlations or exceptions from the underlying helper.
    """
    from DeepPurpose.pybiomed_helper import GetAPseudoAAC2
    return GetAPseudoAAC2(ProteinSequence, lamda, weight)


################################################################################
# Source: DeepPurpose.pybiomed_helper.GetCorrelationFunction
# File: DeepPurpose/pybiomed_helper.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def DeepPurpose_pybiomed_helper_GetCorrelationFunction(
    Ri: str = "S",
    Rj: str = "D",
    AAP: list = []
):
    """GetCorrelationFunction computes a numeric correlation measure between two amino acids based on a provided set of amino-acid property dictionaries.
    
    This function is used in the DeepPurpose protein-feature processing pipeline to quantify similarity/difference between two amino acids (given by their single-letter codes) across multiple biochemical or biophysical properties. In the DeepPurpose toolkit (used for drug–target interaction prediction, protein–protein interaction prediction, protein function prediction, and related tasks), this per-residue correlation can be used to build features, substitution metrics, or similarity matrices that feed downstream models for virtual screening, repurposing, or other predictive tasks.
    
    Args:
        Ri (str): Single-letter code of the first amino acid (default "S"). Ri is treated as the key used to look up property values in each property dictionary after normalization. In the DeepPurpose context this typically denotes an amino acid such as "S" for serine; the function expects Ri to match the keys present in each normalized property mapping produced by NormalizeEachAAP.
        Rj (str): Single-letter code of the second amino acid (default "D"). Rj plays the same role as Ri but for the comparison partner (for example "D" for aspartic acid). It must match keys in the normalized property dictionaries returned by NormalizeEachAAP.
        AAP (list): A list of amino-acid-property entries, where each entry is a dict mapping single-letter amino-acid codes to numeric property values (e.g., hydrophobicity, volume, charge-related scores). Each element in AAP is normalized by calling NormalizeEachAAP(AAP[i]) before use. AAP represents the collection of property descriptors over which the per-property squared differences are averaged to produce the correlation metric.
    
    Behavior, defaults, side effects, and failure modes:
        The function iterates over each property dictionary in AAP, calls NormalizeEachAAP on that dictionary to obtain a normalized mapping (expected to map amino-acid single-letter strings to numeric values), computes the squared difference between the normalized values for Ri and Rj for that property, sums those squared differences across all properties, divides the sum by the number of properties (len(AAP)), rounds the result to three decimal places, and returns it. The default Ri and Rj values ("S" and "D") are convenience defaults and correspond to typical single-letter amino-acid codes; they are meaningful only if those keys exist in the property mappings.
    
        The function does not perform implicit expansion of types: AAP must be a list and each element must be a dict-like mapping compatible with NormalizeEachAAP. If AAP is empty, a ZeroDivisionError will occur due to division by zero when computing the mean over properties. If Ri or Rj are not present as keys in the normalized mapping returned by NormalizeEachAAP for any property, a KeyError will be raised. If AAP or its elements are not of the expected types (for example, AAP is not a list or an element is not a dict), a TypeError or other exception may be raised by NormalizeEachAAP or by the indexing operations. Any side effects depend on the behavior of NormalizeEachAAP; this function itself does not intentionally persistently mutate the AAP list but it does call NormalizeEachAAP for each entry.
    
    Returns:
        float: The mean squared difference between the normalized property values of Ri and Rj across all provided properties, rounded to three decimal places. This value is non-negative and represents a simple averaged squared-distance-based correlation metric used by DeepPurpose to quantify per-residue dissimilarity for downstream modeling and analysis.
    """
    from DeepPurpose.pybiomed_helper import GetCorrelationFunction
    return GetCorrelationFunction(Ri, Rj, AAP)


################################################################################
# Source: DeepPurpose.pybiomed_helper.GetPseudoAAC
# File: DeepPurpose/pybiomed_helper.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def DeepPurpose_pybiomed_helper_GetPseudoAAC(
    ProteinSequence: str,
    lamda: int = 30,
    weight: float = 0.05,
    AAP: list = []
):
    """GetPseudoAAC computes type I pseudo-amino acid composition (PAAC) descriptors for a protein sequence and returns them as a merged dictionary. It implements the PAAC encoding used as a protein feature/encoding in DeepPurpose models for tasks such as drug–target interaction (DTI) prediction, virtual screening, and protein function prediction. The function internally calls GetPseudoAAC1 and GetPseudoAAC2 and combines their outputs into a single dict of descriptors.
    
    This function converts a single protein primary sequence into numerical descriptors that capture both conventional amino acid composition (20 standard amino acids) and additional sequence-order correlation factors up to a user-specified rank (lamda). These descriptors are commonly used as input features to machine learning and deep learning models in cheminformatics and bioinformatics workflows described in the DeepPurpose README (e.g., DTI, protein function prediction, repurposing and virtual screening). The magnitude of the additional pseudo components is scaled relative to conventional amino acid composition by the weight parameter, matching the PAAC formulation referenced in the original implementation.
    
    Args:
        ProteinSequence (str): The primary amino acid sequence of the protein to encode. This should be a plain string composed of single-letter amino acid codes (e.g., "MTEITAAMVKEL..."). The sequence length determines the maximum allowable lamda (see lamda entry). This argument is the biological input the descriptor will characterize and is used by GetPseudoAAC1 and GetPseudoAAC2 to compute composition and correlation terms. No side effects occur on this string; it is not modified in place.
        lamda (int): Non-negative integer specifying the maximum rank of sequence-order correlation factors to include in the pseudo-amino acid composition. Practical significance: the total number of output descriptors produced by the PAAC encoding equals 20 + lamda. Typical choices (from the original PAAC usage) are integers such as 0, 1, 2, ..., and the example documentation suggests values like 15 or 20. Constraints and failure modes: lamda must be a non-negative integer and should NOT be larger than the length of ProteinSequence; if lamda is larger than the sequence length or is negative, the resulting descriptors are invalid or the helper functions called by GetPseudoAAC may raise an error. Default: 30 (as in the function signature); callers should choose lamda appropriate for their sequence lengths and modeling needs.
        weight (float): Floating-point scaling factor that weights the contribution of the additional PseAA components (sequence-order correlation terms) relative to the conventional 20-D amino acid composition. Practical significance: higher weight increases the relative influence of correlation features in downstream models. The original guidance recommends selecting weight within the region from 0.05 to 0.7 (0.05 is a commonly used default). The function accepts any float but values outside the recommended range may produce nonstandard descriptors; extreme values can dominate or effectively suppress conventional composition. Default: 0.05.
        AAP (list): A list of physicochemical property specifications required to compute the sequence-order correlation factors. In practice, each element of AAP is expected to be a dict-like mapping that defines one amino-acid property profile (for example, hydrophobicity values for the 20 amino acids). Practical significance: these properties are used to compute the correlation (lamda-ranked) terms; without a non-empty AAP, the pseudo components cannot be computed meaningfully. The original implementation states the user "must specify some properties into AAP"; the default empty list (AAP=[]) is provided for API compatibility but will result in missing or invalid pseudo components unless populated with appropriate property dicts. The function does not modify the list passed in.
    
    Returns:
        dict: A dictionary containing the combined PAAC descriptors computed by this function. The returned mapping includes the conventional 20 amino-acid composition descriptors plus lamda additional pseudo-amino acid composition descriptors (total dimensionality = 20 + lamda). If lamda == 0, the output corresponds to the 20-D amino acid composition only. The returned dict is the merged result of GetPseudoAAC1(ProteinSequence, lamda, weight, AAP) and GetPseudoAAC2(ProteinSequence, lamda, weight, AAP). There are no other side effects such as file I/O. If inputs violate the documented constraints (e.g., lamda greater than sequence length, empty or incorrectly formatted AAP entries, non-string ProteinSequence), the function may produce invalid descriptor values or propagate errors raised by the internal helper functions; callers should validate inputs before invoking this function.
    """
    from DeepPurpose.pybiomed_helper import GetPseudoAAC
    return GetPseudoAAC(ProteinSequence, lamda, weight, AAP)


################################################################################
# Source: DeepPurpose.pybiomed_helper.GetPseudoAAC1
# File: DeepPurpose/pybiomed_helper.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def DeepPurpose_pybiomed_helper_GetPseudoAAC1(
    ProteinSequence: str,
    lamda: int = 30,
    weight: float = 0.05,
    AAP: list = []
):
    """Compute the first 20 type I pseudo-amino acid composition (PAAC) descriptors for a protein sequence used as a protein encoding in DeepPurpose models for drug-target interaction (DTI), protein property prediction, PPI, and related tasks. This function implements the Type I pseudo-amino acid composition formula: it obtains the standard amino acid composition (AAC) for the 20 canonical amino acids and scales/normalizes those values by a term that incorporates sequence-order correlation factors computed up to a user-specified lag (lamda). The result is a dictionary of 20 PAAC features labeled "PAAC1" .. "PAAC20" corresponding to the amino acids in the global AALetter ordering; these features are suitable as fixed-length protein descriptors for machine learning models in DeepPurpose (for example, as input encodings for CNN/Transformer/other protein encoders).
    
    Args:
        ProteinSequence (str): The protein primary sequence as a string of single-letter amino acid codes (e.g., "MKT..."). This sequence is the domain object being encoded into PAAC descriptors; it must use the amino acid letters expected by the global AALetter mapping used by DeepPurpose helper functions. The function computes AAC and sequence-order correlation factors for this sequence and thus requires the sequence to be a valid protein string for the helper functions GetAAComposition and GetSequenceOrderCorrelationFactor to process. No in-place modification of ProteinSequence occurs.
        lamda (int = 30): The maximum lag (non-negative integer) for sequence-order correlation factor calculations. The function sums correlation factors for lags 1..lamda (looping lamda times) via GetSequenceOrderCorrelationFactor and uses that sum to compute the PAAC normalization denominator. In DeepPurpose applications this parameter determines how many neighbor-distance correlations are used to inject sequence-order information into the otherwise composition-only AAC; larger lamda captures longer-range correlations but increases computed rightpart. Default value is 30 as in the original implementation.
        weight (float = 0.05): The weighting factor that balances original amino acid composition and the aggregated sequence-order correlation contribution in the Type I PAAC normalization term. The denominator used to normalize AAC is computed as temp = 1 + weight * rightpart, where rightpart is the sum over correlation factors for lags up to lamda. Typical default is 0.05 to give modest influence to correlation terms; changing this value alters the relative contribution of sequence-order information in the resulting PAAC features.
        AAP (list = []): A list of amino-acid property scales or descriptors that are forwarded to GetSequenceOrderCorrelationFactor to compute sequence-order correlation factors. In practice in DeepPurpose this argument contains numeric property(s) per amino acid (e.g., hydrophobicity, polarity scales) required by the helper function; the exact expected inner structure and length should match the contract of GetSequenceOrderCorrelationFactor. If left as the empty list (default), the behavior depends on that helper function (it may treat as "no properties" or raise an error); this function does not itself validate AAP beyond passing it through.
    
    Returns:
        dict: A mapping from keys "PAAC1" through "PAAC20" to float values (rounded to three decimal places) representing the first 20 type I pseudo-amino acid composition descriptors. Each key corresponds to the canonical amino acids in the global AALetter order (the same ordering used by GetAAComposition). Values are computed as AAC[amino_acid] / temp where temp = 1 + weight * rightpart and rightpart is the summed correlation factors for lags 1..lamda. These PAAC features are ready to be used as fixed-length protein encodings in downstream DeepPurpose models (for example as input features for DNNs, repurposing/virtual screening pipelines, or comparative descriptor tables).
    
    Behavior, side effects, defaults, and failure modes:
        This is a pure computational helper: it reads ProteinSequence and inputs, calls GetAAComposition and GetSequenceOrderCorrelationFactor, and returns a new dictionary; it does not mutate global state intentionally. Default parameters are lamda=30, weight=0.05, and AAP=[]. The function rounds each returned PAAC value to three decimal places before inserting into the result dict.
        Potential failure modes include:
        - If ProteinSequence contains characters (letters) not present in the global AALetter mapping, GetAAComposition or the dictionary access AAC[i] may raise a KeyError or produce incorrect counts; callers should ensure sequences use the expected single-letter codes.
        - If AAP does not match the expected structure for GetSequenceOrderCorrelationFactor (e.g., wrong length or types), that helper may raise errors; this function does not validate AAP.
        - If weight and rightpart combine to make temp equal to zero, a ZeroDivisionError will occur when normalizing AAC; with the default non-negative weight and typical non-negative correlation sums this is unlikely, but if custom negative weight values are supplied or the helper functions return unexpected negative rightpart, this error may arise.
        - Negative lamda results in zero correlation terms (the for-loop range will be empty) which yields PAAC equal to AAC / 1.0; callers should pass lamda consistent with the desired sequence-order depth.
        Use this function to produce standardized PAAC descriptors for model inputs, feature tables, or for downstream analysis in DeepPurpose workflows (DTI, drug property prediction, PPI, protein function prediction, repurposing and virtual screening).
    """
    from DeepPurpose.pybiomed_helper import GetPseudoAAC1
    return GetPseudoAAC1(ProteinSequence, lamda, weight, AAP)


################################################################################
# Source: DeepPurpose.pybiomed_helper.GetPseudoAAC2
# File: DeepPurpose/pybiomed_helper.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def DeepPurpose_pybiomed_helper_GetPseudoAAC2(
    ProteinSequence: str,
    lamda: int = 30,
    weight: float = 0.05,
    AAP: list = []
):
    """GetPseudoAAC2 computes the Type I pseudo-amino acid composition (PAAC) features for a protein sequence and returns the final lamda PAAC components as a dictionary keyed by standard PAAC feature names. In the DeepPurpose library these PAAC features are used as protein encodings for downstream tasks such as drug–target interaction (DTI) prediction, virtual screening, and protein property/function prediction. This implementation computes lamda sequence-order correlation factors by calling GetSequenceOrderCorrelationFactor and then applies the standard PAAC normalization and weighting to produce percentage-scaled descriptors rounded to three decimal places.
    
    Args:
        ProteinSequence (str): The primary amino-acid sequence of the protein to encode. This string is passed verbatim to GetSequenceOrderCorrelationFactor to compute sequence-order correlation factors; it should use the one-letter amino-acid code expected by that helper function. The sequence is the biological input whose physicochemical and order-dependent properties are summarized by the returned PAAC features for use in DeepPurpose model encodings and downstream ML workflows.
        lamda (int): The number of sequence-order (pseudo) components to compute and return. The function computes lamda correlation factors (calls GetSequenceOrderCorrelationFactor for orders 1..lamda) and produces lamda PAAC entries named "PAAC21" through "PAAC{20+lamda}" by default. The default is 30, producing keys PAAC21..PAAC50. lamda controls the depth of order information captured: larger values add higher-order sequence-order components. If lamda is zero, the function returns an empty dictionary (no PAAC entries).
        weight (float): The weighting factor applied to all sequence-order correlation factors during normalization. The function uses this value in the normalization denominator temp = 1 + weight * sum(correlation_factors) and in the numerator weight * correlation_factor, then multiplies by 100 and rounds to three decimals. The default is 0.05. If weight is zero, all returned PAAC entries will be 0.0 because sequence-order terms are multiplied by weight prior to scaling.
        AAP (list): A list passed unchanged to GetSequenceOrderCorrelationFactor that represents amino-acid physicochemical property scales or other per-residue property vectors required to compute sequence-order correlation factors. The exact expected contents and format of AAP are determined by GetSequenceOrderCorrelationFactor; if an empty list is provided (the default), the called helper function's internal behavior or defaults determine how correlation factors are computed.
    
    Returns:
        dict: A dictionary mapping PAAC feature names (str) to their computed values (float). Keys are of the form "PAAC{n}" where n ranges from 21 to 20+lamda (e.g., with the default lamda=30 the keys are "PAAC21".."PAAC50"). Each value is computed as round(weight * correlation_factor / (1 + weight * sum(correlation_factors)) * 100, 3) and represents the normalized, percentage-scaled contribution of that sequence-order component to the Type I pseudo-amino acid composition. The returned dictionary contains exactly lamda entries unless lamda is zero (empty dict).
    
    Behavior, side effects, defaults, and failure modes:
        This function has no side effects beyond calling GetSequenceOrderCorrelationFactor lamda times. It rounds every output value to three decimal places and scales by 100 to express results as percentages. Defaults are lamda=30, weight=0.05, and AAP=[]. The function relies on GetSequenceOrderCorrelationFactor to compute individual correlation factors; therefore invalid ProteinSequence formats, incompatible or malformed AAP contents, or errors inside GetSequenceOrderCorrelationFactor will propagate as exceptions from this function. Numeric stability depends on the values returned by GetSequenceOrderCorrelationFactor and the provided weight; extremely large correlation factors may affect normalization but the implementation prevents division by zero because the denominator is 1 + weight * sum(correlation_factors).
    """
    from DeepPurpose.pybiomed_helper import GetPseudoAAC2
    return GetPseudoAAC2(ProteinSequence, lamda, weight, AAP)


################################################################################
# Source: DeepPurpose.pybiomed_helper.GetQuasiSequenceOrder
# File: DeepPurpose/pybiomed_helper.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def DeepPurpose_pybiomed_helper_GetQuasiSequenceOrder(
    ProteinSequence: str,
    maxlag: int = 30,
    weight: float = 0.1
):
    """Compute quasi-sequence-order descriptors for a protein sequence following Chou (2000), producing a numeric feature vector used as an input encoding for downstream models in DeepPurpose (e.g., DTI, PPI, protein-function prediction, virtual screening and repurposing workflows).
    
    This function implements the quasi-sequence-order (QSO) descriptor family described in:
    Kuo-Chen Chou. Prediction of Protein Subcellar Locations by Incorporating Quasi-Sequence-Order Effect. Biochemical and Biophysical Research Communications 2000, 278, 477-483.
    It computes multiple QSO components using two internal distance matrices and four internal procedures (GetQuasiSequenceOrder1SW, GetQuasiSequenceOrder2SW, GetQuasiSequenceOrder1Grant, GetQuasiSequenceOrder2Grant). The resulting 1-D numeric vector encodes sequence-order correlation information (the quasi-sequence-order effect) that augments simple composition features and is intended for use as model input features in machine learning pipelines in DeepPurpose (e.g., for drug-target interaction prediction, protein-protein interaction prediction, or protein function prediction).
    
    Args:
        ProteinSequence (str): A pure protein primary sequence given as a string of standard single-letter amino acid codes (for example, 'MKT...'). This sequence is the biological input to be encoded into quasi-sequence-order descriptors. The sequence must consist of valid amino-acid letters; passing non-standard characters or whitespace may produce incorrect descriptors or cause the internal helper functions to raise exceptions. The method assumes the sequence length is larger than maxlag (see below); very short sequences that do not satisfy that assumption may lead to undefined behavior or errors in the internal computations.
        maxlag (int): The maximum lag (non-negative integer) used when computing sequence-order correlation terms. This parameter controls how far apart along the primary sequence pairwise residue correlations are computed: correlations for residue pairs separated by up to maxlag positions are included. The default is 30, matching common choices in the QSO literature and in the original implementation. Practically, choose maxlag relative to typical protein lengths in your dataset; the sequence length should be greater than maxlag for meaningful results.
        weight (float): A numeric weight factor that scales the relative contribution of sequence-order correlation terms versus composition-like terms in the final descriptor vector. Typical choices from the literature use small positive values (the default is 0.1) — this balances the influence of order information against residue composition. Adjusting this parameter changes feature scaling and can affect downstream model training.
    
    Behavior and side effects:
        - This function is pure with respect to user-visible state: it does not modify the input string or global state, but it calls internal helper functions and reads internal distance matrices (_Distance1 and _Distance2) to compute descriptors.
        - The function concatenates descriptors returned by four internal procedures in the following insertion order: results from GetQuasiSequenceOrder1SW, then GetQuasiSequenceOrder2SW, then GetQuasiSequenceOrder1Grant, then GetQuasiSequenceOrder2Grant. In Python 3.7+ the dict insertion order is preserved, so the returned array reflects this concatenation order.
        - The internal distance matrices correspond to two published residue-distance definitions used in QSO computations (the code labels these paths using the suffixes "SW" and "Grant"); these matrices quantify physicochemical distances between amino acids and are required to compute the sequence-order correlation terms.
        - Default parameter values (maxlag=30, weight=0.1) follow common practice from the QSO literature but can be adjusted to match dataset characteristics and modeling needs.
    
    Failure modes and input validation:
        - The function expects ProteinSequence to be a non-empty string of standard amino-acid single-letter codes. Passing sequences with invalid characters, numbers, or gaps may cause incorrect outputs or raise errors in the internal helpers.
        - If the sequence length is not larger than maxlag, the QSO terms that rely on lagged correlations cannot be computed meaningfully; in such cases the internal helper functions may raise exceptions or return incomplete descriptors. Ensure len(ProteinSequence) > maxlag for reliable results.
        - Numerical stability: extremely large maxlag values or unusual weight choices could lead to very small or large numeric descriptor values; downstream preprocessing (scaling/normalization) is typically required before using these descriptors in ML models.
    
    Returns:
        numpy.ndarray: A one-dimensional numpy array of floating-point quasi-sequence-order descriptors. The array is the concatenation (in insertion order) of the numeric outputs from the four internal QSO computation routines (1SW, 2SW, 1Grant, 2Grant). This vector is suitable as a fixed-length numeric feature representation of the input protein for machine learning pipelines (for example, as input to scikit-learn or PyTorch models used in DeepPurpose).
    """
    from DeepPurpose.pybiomed_helper import GetQuasiSequenceOrder
    return GetQuasiSequenceOrder(ProteinSequence, maxlag, weight)


################################################################################
# Source: DeepPurpose.pybiomed_helper.GetQuasiSequenceOrder1
# File: DeepPurpose/pybiomed_helper.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def DeepPurpose_pybiomed_helper_GetQuasiSequenceOrder1(
    ProteinSequence: str,
    maxlag: int = 30,
    weight: float = 0.1,
    distancematrix: dict = {}
):
    """Compute the first 20 quasi-sequence-order (QSO) descriptors for a protein sequence.
    
    This function implements the QSO-1 descriptor calculation used as a protein encoding in DeepPurpose for tasks such as drug-target interaction (DTI) prediction, virtual screening, repurposing, and protein-function prediction. It aggregates sequence-order coupling information across lags up to maxlag using GetSequenceOrderCouplingNumber, combines that aggregate with the amino-acid composition from GetAAComposition, and returns a normalized set of 20 descriptor values (one per standard amino-acid single-letter code in AALetter). The descriptors quantify how amino-acid composition is modulated by sequence-order (local coupling) information and are intended to be used as fixed-length numeric features for machine learning models in molecular and protein modeling workflows included in the DeepPurpose toolkit.
    
    Args:
        ProteinSequence (str): A protein primary sequence given as a string of amino-acid single-letter codes. This sequence is the input whose quasi-sequence-order descriptors are computed. Practical significance: this is the biological sequence (target) used by DeepPurpose encoders when preparing protein features for DTI, PPI, or protein-function prediction models. If the sequence contains non-standard letters, downstream helper functions (GetAAComposition or GetSequenceOrderCouplingNumber) may raise a KeyError or other exception.
        maxlag (int = 30): The maximum lag (positive integer) used when summing sequence-order coupling numbers. The function calls GetSequenceOrderCouplingNumber for lags 1 through maxlag and accumulates those coupling numbers into the rightpart normalization term. Practical significance: larger maxlag incorporates longer-range sequence-order relationships into the QSO descriptors; this parameter controls how much sequence-order (vs. composition) influences the final features. Default is 30. If maxlag is non-integer or negative, the behavior depends on GetSequenceOrderCouplingNumber and may raise an error.
        weight (float = 0.1): A scalar weight applied to the aggregated coupling term when computing the normalization denominator temp = 1 + weight * rightpart. Practical significance: weight balances the relative contribution of sequence-order coupling versus plain amino-acid composition in the final normalized QSO values. Default is 0.1. If weight is set so that temp equals zero (e.g., a negative weight paired with a particular rightpart), a ZeroDivisionError may occur when normalizing; the default value is chosen to avoid this in typical sequences.
        distancematrix (dict = {}): A dictionary passed through to GetSequenceOrderCouplingNumber to define pairwise physicochemical or distance measures used when computing coupling numbers. Practical significance: this allows use of different distance or property matrices (for example, matrices encoding physicochemical distances between amino acids) to tailor the coupling calculation to a specific property. The exact expected key/value structure is that consumed by GetSequenceOrderCouplingNumber; this function forwards distancematrix without modification. If the provided dictionary lacks required entries, the called helper may raise an exception.
    
    Returns:
        dict: A mapping of descriptor names to float values. Keys are strings "QSO1", "QSO2", ..., "QSO20" corresponding to the standard amino-acid single-letter codes in the module-level AALetter ordering, and values are the normalized quasi-sequence-order descriptor values rounded to six decimal places. These are computed as AAC[aa] / (1 + weight * sum_{lag=1..maxlag} coupling_number(lag)), where AAC is the amino-acid composition from GetAAComposition and coupling_number(lag) is obtained from GetSequenceOrderCouplingNumber. Practical significance: the returned dict is a fixed-length numeric feature vector suitable for input to machine-learning models in the DeepPurpose pipeline.
    
    Behavior and side effects:
        - The function is pure (no I/O, no global state mutation) and returns a new dictionary; it does not write files or modify global variables.
        - Internally it calls GetSequenceOrderCouplingNumber repeatedly for lags 1..maxlag and GetAAComposition once. Computational cost scales roughly linearly with maxlag and the length of ProteinSequence.
        - Values are rounded to six decimal places before being placed in the returned dictionary.
        - Default parameters (maxlag=30, weight=0.1, empty distancematrix) follow conventions used by DeepPurpose and related QSO implementations; these defaults aim to provide sensible sequence-order sensitivity while avoiding numerical instability.
    
    Failure modes and notes:
        - If ProteinSequence contains characters not handled by GetAAComposition or if AALetter does not cover the characters, a KeyError or similar exception may be raised.
        - If maxlag is invalid (non-integer, negative), the helper GetSequenceOrderCouplingNumber or the loop may raise an error.
        - If weight and the aggregated coupling term produce temp == 0, a ZeroDivisionError will occur during normalization.
        - Any errors originating from GetSequenceOrderCouplingNumber or GetAAComposition (for example, due to an incompatible distancematrix) will propagate to the caller.
        - The function assumes a 20-letter standard amino-acid alphabet (hence 20 QSO descriptors) as used elsewhere in DeepPurpose; if AALetter is changed, the number and naming of returned descriptors will change accordingly.
    """
    from DeepPurpose.pybiomed_helper import GetQuasiSequenceOrder1
    return GetQuasiSequenceOrder1(ProteinSequence, maxlag, weight, distancematrix)


################################################################################
# Source: DeepPurpose.pybiomed_helper.GetQuasiSequenceOrder1Grant
# File: DeepPurpose/pybiomed_helper.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def DeepPurpose_pybiomed_helper_GetQuasiSequenceOrder1Grant(
    ProteinSequence: str,
    maxlag: int = 30,
    weight: float = 0.1,
    distancematrix: dict = {'GW': 0.923, 'GV': 0.464, 'GT': 0.272, 'GS': 0.158, 'GR': 1.0, 'GQ': 0.467, 'GP': 0.323, 'GY': 0.728, 'GG': 0.0, 'GF': 0.727, 'GE': 0.807, 'GD': 0.776, 'GC': 0.312, 'GA': 0.206, 'GN': 0.381, 'GM': 0.557, 'GL': 0.591, 'GK': 0.894, 'GI': 0.592, 'GH': 0.769, 'ME': 0.879, 'MD': 0.932, 'MG': 0.569, 'MF': 0.182, 'MA': 0.383, 'MC': 0.276, 'MM': 0.0, 'ML': 0.062, 'MN': 0.447, 'MI': 0.058, 'MH': 0.648, 'MK': 0.884, 'MT': 0.358, 'MW': 0.391, 'MV': 0.12, 'MQ': 0.372, 'MP': 0.285, 'MS': 0.417, 'MR': 1.0, 'MY': 0.255, 'FP': 0.42, 'FQ': 0.459, 'FR': 1.0, 'FS': 0.548, 'FT': 0.499, 'FV': 0.252, 'FW': 0.207, 'FY': 0.179, 'FA': 0.508, 'FC': 0.405, 'FD': 0.977, 'FE': 0.918, 'FF': 0.0, 'FG': 0.69, 'FH': 0.663, 'FI': 0.128, 'FK': 0.903, 'FL': 0.131, 'FM': 0.169, 'FN': 0.541, 'SY': 0.615, 'SS': 0.0, 'SR': 1.0, 'SQ': 0.358, 'SP': 0.181, 'SW': 0.827, 'SV': 0.342, 'ST': 0.174, 'SK': 0.883, 'SI': 0.478, 'SH': 0.718, 'SN': 0.289, 'SM': 0.44, 'SL': 0.474, 'SC': 0.185, 'SA': 0.1, 'SG': 0.17, 'SF': 0.622, 'SE': 0.812, 'SD': 0.801, 'YI': 0.23, 'YH': 0.678, 'YK': 0.904, 'YM': 0.268, 'YL': 0.219, 'YN': 0.512, 'YA': 0.587, 'YC': 0.478, 'YE': 0.932, 'YD': 1.0, 'YG': 0.782, 'YF': 0.202, 'YY': 0.0, 'YQ': 0.404, 'YP': 0.444, 'YS': 0.612, 'YR': 0.995, 'YT': 0.557, 'YW': 0.244, 'YV': 0.328, 'LF': 0.139, 'LG': 0.596, 'LD': 0.944, 'LE': 0.892, 'LC': 0.296, 'LA': 0.405, 'LN': 0.452, 'LL': 0.0, 'LM': 0.062, 'LK': 0.893, 'LH': 0.653, 'LI': 0.013, 'LV': 0.133, 'LW': 0.341, 'LT': 0.397, 'LR': 1.0, 'LS': 0.443, 'LP': 0.309, 'LQ': 0.376, 'LY': 0.205, 'RT': 0.808, 'RV': 0.914, 'RW': 1.0, 'RP': 0.796, 'RQ': 0.668, 'RR': 0.0, 'RS': 0.86, 'RY': 0.859, 'RD': 0.305, 'RE': 0.225, 'RF': 0.977, 'RG': 0.928, 'RA': 0.919, 'RC': 0.905, 'RL': 0.92, 'RM': 0.908, 'RN': 0.69, 'RH': 0.498, 'RI': 0.929, 'RK': 0.141, 'VH': 0.649, 'VI': 0.135, 'EM': 0.83, 'EL': 0.854, 'EN': 0.599, 'EI': 0.86, 'EH': 0.406, 'EK': 0.143, 'EE': 0.0, 'ED': 0.133, 'EG': 0.779, 'EF': 0.932, 'EA': 0.79, 'EC': 0.788, 'VM': 0.12, 'EY': 0.837, 'VN': 0.38, 'ET': 0.682, 'EW': 1.0, 'EV': 0.824, 'EQ': 0.598, 'EP': 0.688, 'ES': 0.726, 'ER': 0.234, 'VP': 0.212, 'VQ': 0.339, 'VR': 1.0, 'VT': 0.305, 'VW': 0.472, 'KC': 0.871, 'KA': 0.889, 'KG': 0.9, 'KF': 0.957, 'KE': 0.149, 'KD': 0.279, 'KK': 0.0, 'KI': 0.899, 'KH': 0.438, 'KN': 0.667, 'KM': 0.871, 'KL': 0.892, 'KS': 0.825, 'KR': 0.154, 'KQ': 0.639, 'KP': 0.757, 'KW': 1.0, 'KV': 0.882, 'KT': 0.759, 'KY': 0.848, 'DN': 0.56, 'DL': 0.841, 'DM': 0.819, 'DK': 0.249, 'DH': 0.435, 'DI': 0.847, 'DF': 0.924, 'DG': 0.697, 'DD': 0.0, 'DE': 0.124, 'DC': 0.742, 'DA': 0.729, 'DY': 0.836, 'DV': 0.797, 'DW': 1.0, 'DT': 0.649, 'DR': 0.295, 'DS': 0.667, 'DP': 0.657, 'DQ': 0.584, 'QQ': 0.0, 'QP': 0.272, 'QS': 0.461, 'QR': 1.0, 'QT': 0.389, 'QW': 0.831, 'QV': 0.464, 'QY': 0.522, 'QA': 0.512, 'QC': 0.462, 'QE': 0.861, 'QD': 0.903, 'QG': 0.648, 'QF': 0.671, 'QI': 0.532, 'QH': 0.765, 'QK': 0.881, 'QM': 0.505, 'QL': 0.518, 'QN': 0.181, 'WG': 0.829, 'WF': 0.196, 'WE': 0.931, 'WD': 1.0, 'WC': 0.56, 'WA': 0.658, 'WN': 0.631, 'WM': 0.344, 'WL': 0.304, 'WK': 0.892, 'WI': 0.305, 'WH': 0.678, 'WW': 0.0, 'WV': 0.418, 'WT': 0.638, 'WS': 0.689, 'WR': 0.968, 'WQ': 0.538, 'WP': 0.555, 'WY': 0.204, 'PR': 1.0, 'PS': 0.196, 'PP': 0.0, 'PQ': 0.228, 'PV': 0.244, 'PW': 0.72, 'PT': 0.161, 'PY': 0.481, 'PC': 0.179, 'PA': 0.22, 'PF': 0.515, 'PG': 0.376, 'PD': 0.852, 'PE': 0.831, 'PK': 0.875, 'PH': 0.696, 'PI': 0.363, 'PN': 0.231, 'PL': 0.357, 'PM': 0.326, 'CK': 0.887, 'CI': 0.304, 'CH': 0.66, 'CN': 0.324, 'CM': 0.277, 'CL': 0.301, 'CC': 0.0, 'CA': 0.114, 'CG': 0.32, 'CF': 0.437, 'CE': 0.838, 'CD': 0.847, 'CY': 0.457, 'CS': 0.176, 'CR': 1.0, 'CQ': 0.341, 'CP': 0.157, 'CW': 0.639, 'CV': 0.167, 'CT': 0.233, 'IY': 0.213, 'VA': 0.275, 'VC': 0.165, 'VD': 0.9, 'VE': 0.867, 'VF': 0.269, 'VG': 0.471, 'IQ': 0.383, 'IP': 0.311, 'IS': 0.443, 'IR': 1.0, 'VL': 0.134, 'IT': 0.396, 'IW': 0.339, 'IV': 0.133, 'II': 0.0, 'IH': 0.652, 'IK': 0.892, 'VS': 0.322, 'IM': 0.057, 'IL': 0.013, 'VV': 0.0, 'IN': 0.457, 'IA': 0.403, 'VY': 0.31, 'IC': 0.296, 'IE': 0.891, 'ID': 0.942, 'IG': 0.592, 'IF': 0.134, 'HY': 0.821, 'HR': 0.697, 'HS': 0.865, 'HP': 0.777, 'HQ': 0.716, 'HV': 0.831, 'HW': 0.981, 'HT': 0.834, 'HK': 0.566, 'HH': 0.0, 'HI': 0.848, 'HN': 0.754, 'HL': 0.842, 'HM': 0.825, 'HC': 0.836, 'HA': 0.896, 'HF': 0.907, 'HG': 1.0, 'HD': 0.629, 'HE': 0.547, 'NH': 0.78, 'NI': 0.615, 'NK': 0.891, 'NL': 0.603, 'NM': 0.588, 'NN': 0.0, 'NA': 0.424, 'NC': 0.425, 'ND': 0.838, 'NE': 0.835, 'NF': 0.766, 'NG': 0.512, 'NY': 0.641, 'NP': 0.266, 'NQ': 0.175, 'NR': 1.0, 'NS': 0.361, 'NT': 0.368, 'NV': 0.503, 'NW': 0.945, 'TY': 0.596, 'TV': 0.345, 'TW': 0.816, 'TT': 0.0, 'TR': 1.0, 'TS': 0.185, 'TP': 0.159, 'TQ': 0.322, 'TN': 0.315, 'TL': 0.453, 'TM': 0.403, 'TK': 0.866, 'TH': 0.737, 'TI': 0.455, 'TF': 0.604, 'TG': 0.312, 'TD': 0.83, 'TE': 0.812, 'TC': 0.261, 'TA': 0.251, 'AA': 0.0, 'AC': 0.112, 'AE': 0.827, 'AD': 0.819, 'AG': 0.208, 'AF': 0.54, 'AI': 0.407, 'AH': 0.696, 'AK': 0.891, 'AM': 0.379, 'AL': 0.406, 'AN': 0.318, 'AQ': 0.372, 'AP': 0.191, 'AS': 0.094, 'AR': 1.0, 'AT': 0.22, 'AW': 0.739, 'AV': 0.273, 'AY': 0.552, 'VK': 0.889}
):
    """GetQuasiSequenceOrder1Grant computes the first 20 quasi-sequence-order (QSO) descriptors for a protein sequence and returns them as a dictionary suitable for use as protein features in DeepPurpose models (for example DTI, PPI, protein function prediction, or compound–protein interaction feature pipelines). The function combines amino-acid composition with sequence-order coupling information (summed over lags 1..maxlag using a provided amino-acid distance matrix) and normalizes the composition by a factor that depends on the coupling sum and a user-specified weight. The resulting 20 numeric descriptors capture global composition adjusted by sequence-order information and are returned with keys "QSOgrant1" .. "QSOgrant20" corresponding to the module-level AALetter ordering of the 20 standard amino acids.
    
    Args:
        ProteinSequence (str): Protein primary sequence represented as a string of one-letter amino-acid codes. This argument is the biological input whose composition and short-range sequence-order statistics are being encoded. The function relies on helper routines GetAAComposition and GetSequenceOrderCouplingNumber that expect standard one-letter amino-acid symbols; nonstandard letters or lowercase input may lead to incorrect counts or downstream lookup errors in the distance matrix.
        maxlag (int): Maximum lag (non-negative integer) used when summing sequence-order coupling numbers. The function sums coupling numbers for lags 1 through maxlag (inclusive in effect because the implementation iterates range(maxlag) and calls coupling with lag i+1). Larger maxlag increases the contribution of longer-range sequence-order effects to the normalization term (rightpart) and therefore changes the scaling of the returned QSO descriptors; very large values increase computation and may be biologically less meaningful. The default value is 30.
        weight (float): Scaling weight applied to the summed sequence-order coupling numbers when forming the normalization denominator temp = 1 + weight * rightpart. This parameter controls the relative influence of sequence-order coupling (rightpart) versus raw amino-acid composition. Typical default is 0.1 (as used in published QSO descriptor formulations); users can increase it to emphasize sequence-order effects or decrease it to reduce their influence.
        distancematrix (dict): Pairwise amino-acid distance matrix expressed as a dictionary mapping two-letter keys (concatenated one-letter codes, e.g., "AG" for A vs G) to numeric distances. This matrix is used by GetSequenceOrderCouplingNumber to compute coupling between residues separated by a specific lag. The default dictionary supplied to the function provides a precomputed set of distances (as in the module-level _Distance2) for the standard amino acids. The function expects that all residue-pair lookups required by the chosen maxlag and the sequence are present in this dictionary; missing entries will raise a KeyError in the coupling calculation.
    
    Behavior and side effects:
        The function is pure (no external side effects) and returns a new dictionary. Internally it:
        - calls GetSequenceOrderCouplingNumber(ProteinSequence, lag, distancematrix) for lag = 1..maxlag and sums the results into rightpart;
        - calls GetAAComposition(ProteinSequence) to obtain the frequency (composition) for each amino acid in the module-level AALetter ordering;
        - computes temp = 1 + weight * rightpart and divides each amino-acid composition value by temp;
        - rounds each resulting descriptor to six decimal places and stores it using keys "QSOgrant1" .. "QSOgrantN" where N equals len(AALetter) (20 for standard amino acids).
        The function relies on module-level symbols GetSequenceOrderCouplingNumber, GetAAComposition, and AALetter. Rounding to six decimals may truncate small numeric differences and is performed before returning.
    
    Failure modes and validation:
        - TypeError or ValueError may result if ProteinSequence is not a string, maxlag is not an integer, or weight is not numeric.
        - If distancematrix lacks required pair keys for residue pairs encountered at the requested lags, GetSequenceOrderCouplingNumber will raise a KeyError.
        - Negative maxlag will effectively result in no coupling terms being summed (rightpart = 0.0) due to Python range behavior; callers should pass a non-negative integer.
        - Extremely large maxlag increases runtime and may produce coupling contributions that are numerically unstable depending on the distance values.
        - The function does not perform extensive input sanitization; callers should ensure ProteinSequence uses expected one-letter codes and that distancematrix covers all needed residue pairs.
    
    Returns:
        dict: A dictionary of 20 (module-defined) quasi-sequence-order descriptors keyed as "QSOgrant1", "QSOgrant2", ..., corresponding to the module-level AALetter ordering of amino acids. Each value is a float equal to round((composition_of_residue / (1 + weight * rightpart)), 6), where composition_of_residue is the fraction/frequency returned by GetAAComposition and rightpart is the summed sequence-order coupling over lags 1..maxlag computed with distancematrix. These descriptors are intended to be used as normalized protein feature inputs in DeepPurpose machine-learning pipelines (e.g., for DTI, PPI, protein function prediction, virtual screening, or repurposing workflows).
    """
    from DeepPurpose.pybiomed_helper import GetQuasiSequenceOrder1Grant
    return GetQuasiSequenceOrder1Grant(ProteinSequence, maxlag, weight, distancematrix)


################################################################################
# Source: DeepPurpose.pybiomed_helper.GetQuasiSequenceOrder1SW
# File: DeepPurpose/pybiomed_helper.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def DeepPurpose_pybiomed_helper_GetQuasiSequenceOrder1SW(
    ProteinSequence: str,
    maxlag: int = 30,
    weight: float = 0.1,
    distancematrix: dict = {'GW': 0.923, 'GV': 0.464, 'GT': 0.272, 'GS': 0.158, 'GR': 1.0, 'GQ': 0.467, 'GP': 0.323, 'GY': 0.728, 'GG': 0.0, 'GF': 0.727, 'GE': 0.807, 'GD': 0.776, 'GC': 0.312, 'GA': 0.206, 'GN': 0.381, 'GM': 0.557, 'GL': 0.591, 'GK': 0.894, 'GI': 0.592, 'GH': 0.769, 'ME': 0.879, 'MD': 0.932, 'MG': 0.569, 'MF': 0.182, 'MA': 0.383, 'MC': 0.276, 'MM': 0.0, 'ML': 0.062, 'MN': 0.447, 'MI': 0.058, 'MH': 0.648, 'MK': 0.884, 'MT': 0.358, 'MW': 0.391, 'MV': 0.12, 'MQ': 0.372, 'MP': 0.285, 'MS': 0.417, 'MR': 1.0, 'MY': 0.255, 'FP': 0.42, 'FQ': 0.459, 'FR': 1.0, 'FS': 0.548, 'FT': 0.499, 'FV': 0.252, 'FW': 0.207, 'FY': 0.179, 'FA': 0.508, 'FC': 0.405, 'FD': 0.977, 'FE': 0.918, 'FF': 0.0, 'FG': 0.69, 'FH': 0.663, 'FI': 0.128, 'FK': 0.903, 'FL': 0.131, 'FM': 0.169, 'FN': 0.541, 'SY': 0.615, 'SS': 0.0, 'SR': 1.0, 'SQ': 0.358, 'SP': 0.181, 'SW': 0.827, 'SV': 0.342, 'ST': 0.174, 'SK': 0.883, 'SI': 0.478, 'SH': 0.718, 'SN': 0.289, 'SM': 0.44, 'SL': 0.474, 'SC': 0.185, 'SA': 0.1, 'SG': 0.17, 'SF': 0.622, 'SE': 0.812, 'SD': 0.801, 'YI': 0.23, 'YH': 0.678, 'YK': 0.904, 'YM': 0.268, 'YL': 0.219, 'YN': 0.512, 'YA': 0.587, 'YC': 0.478, 'YE': 0.932, 'YD': 1.0, 'YG': 0.782, 'YF': 0.202, 'YY': 0.0, 'YQ': 0.404, 'YP': 0.444, 'YS': 0.612, 'YR': 0.995, 'YT': 0.557, 'YW': 0.244, 'YV': 0.328, 'LF': 0.139, 'LG': 0.596, 'LD': 0.944, 'LE': 0.892, 'LC': 0.296, 'LA': 0.405, 'LN': 0.452, 'LL': 0.0, 'LM': 0.062, 'LK': 0.893, 'LH': 0.653, 'LI': 0.013, 'LV': 0.133, 'LW': 0.341, 'LT': 0.397, 'LR': 1.0, 'LS': 0.443, 'LP': 0.309, 'LQ': 0.376, 'LY': 0.205, 'RT': 0.808, 'RV': 0.914, 'RW': 1.0, 'RP': 0.796, 'RQ': 0.668, 'RR': 0.0, 'RS': 0.86, 'RY': 0.859, 'RD': 0.305, 'RE': 0.225, 'RF': 0.977, 'RG': 0.928, 'RA': 0.919, 'RC': 0.905, 'RL': 0.92, 'RM': 0.908, 'RN': 0.69, 'RH': 0.498, 'RI': 0.929, 'RK': 0.141, 'VH': 0.649, 'VI': 0.135, 'EM': 0.83, 'EL': 0.854, 'EN': 0.599, 'EI': 0.86, 'EH': 0.406, 'EK': 0.143, 'EE': 0.0, 'ED': 0.133, 'EG': 0.779, 'EF': 0.932, 'EA': 0.79, 'EC': 0.788, 'VM': 0.12, 'EY': 0.837, 'VN': 0.38, 'ET': 0.682, 'EW': 1.0, 'EV': 0.824, 'EQ': 0.598, 'EP': 0.688, 'ES': 0.726, 'ER': 0.234, 'VP': 0.212, 'VQ': 0.339, 'VR': 1.0, 'VT': 0.305, 'VW': 0.472, 'KC': 0.871, 'KA': 0.889, 'KG': 0.9, 'KF': 0.957, 'KE': 0.149, 'KD': 0.279, 'KK': 0.0, 'KI': 0.899, 'KH': 0.438, 'KN': 0.667, 'KM': 0.871, 'KL': 0.892, 'KS': 0.825, 'KR': 0.154, 'KQ': 0.639, 'KP': 0.757, 'KW': 1.0, 'KV': 0.882, 'KT': 0.759, 'KY': 0.848, 'DN': 0.56, 'DL': 0.841, 'DM': 0.819, 'DK': 0.249, 'DH': 0.435, 'DI': 0.847, 'DF': 0.924, 'DG': 0.697, 'DD': 0.0, 'DE': 0.124, 'DC': 0.742, 'DA': 0.729, 'DY': 0.836, 'DV': 0.797, 'DW': 1.0, 'DT': 0.649, 'DR': 0.295, 'DS': 0.667, 'DP': 0.657, 'DQ': 0.584, 'QQ': 0.0, 'QP': 0.272, 'QS': 0.461, 'QR': 1.0, 'QT': 0.389, 'QW': 0.831, 'QV': 0.464, 'QY': 0.522, 'QA': 0.512, 'QC': 0.462, 'QE': 0.861, 'QD': 0.903, 'QG': 0.648, 'QF': 0.671, 'QI': 0.532, 'QH': 0.765, 'QK': 0.881, 'QM': 0.505, 'QL': 0.518, 'QN': 0.181, 'WG': 0.829, 'WF': 0.196, 'WE': 0.931, 'WD': 1.0, 'WC': 0.56, 'WA': 0.658, 'WN': 0.631, 'WM': 0.344, 'WL': 0.304, 'WK': 0.892, 'WI': 0.305, 'WH': 0.678, 'WW': 0.0, 'WV': 0.418, 'WT': 0.638, 'WS': 0.689, 'WR': 0.968, 'WQ': 0.538, 'WP': 0.555, 'WY': 0.204, 'PR': 1.0, 'PS': 0.196, 'PP': 0.0, 'PQ': 0.228, 'PV': 0.244, 'PW': 0.72, 'PT': 0.161, 'PY': 0.481, 'PC': 0.179, 'PA': 0.22, 'PF': 0.515, 'PG': 0.376, 'PD': 0.852, 'PE': 0.831, 'PK': 0.875, 'PH': 0.696, 'PI': 0.363, 'PN': 0.231, 'PL': 0.357, 'PM': 0.326, 'CK': 0.887, 'CI': 0.304, 'CH': 0.66, 'CN': 0.324, 'CM': 0.277, 'CL': 0.301, 'CC': 0.0, 'CA': 0.114, 'CG': 0.32, 'CF': 0.437, 'CE': 0.838, 'CD': 0.847, 'CY': 0.457, 'CS': 0.176, 'CR': 1.0, 'CQ': 0.341, 'CP': 0.157, 'CW': 0.639, 'CV': 0.167, 'CT': 0.233, 'IY': 0.213, 'VA': 0.275, 'VC': 0.165, 'VD': 0.9, 'VE': 0.867, 'VF': 0.269, 'VG': 0.471, 'IQ': 0.383, 'IP': 0.311, 'IS': 0.443, 'IR': 1.0, 'VL': 0.134, 'IT': 0.396, 'IW': 0.339, 'IV': 0.133, 'II': 0.0, 'IH': 0.652, 'IK': 0.892, 'VS': 0.322, 'IM': 0.057, 'IL': 0.013, 'VV': 0.0, 'IN': 0.457, 'IA': 0.403, 'VY': 0.31, 'IC': 0.296, 'IE': 0.891, 'ID': 0.942, 'IG': 0.592, 'IF': 0.134, 'HY': 0.821, 'HR': 0.697, 'HS': 0.865, 'HP': 0.777, 'HQ': 0.716, 'HV': 0.831, 'HW': 0.981, 'HT': 0.834, 'HK': 0.566, 'HH': 0.0, 'HI': 0.848, 'HN': 0.754, 'HL': 0.842, 'HM': 0.825, 'HC': 0.836, 'HA': 0.896, 'HF': 0.907, 'HG': 1.0, 'HD': 0.629, 'HE': 0.547, 'NH': 0.78, 'NI': 0.615, 'NK': 0.891, 'NL': 0.603, 'NM': 0.588, 'NN': 0.0, 'NA': 0.424, 'NC': 0.425, 'ND': 0.838, 'NE': 0.835, 'NF': 0.766, 'NG': 0.512, 'NY': 0.641, 'NP': 0.266, 'NQ': 0.175, 'NR': 1.0, 'NS': 0.361, 'NT': 0.368, 'NV': 0.503, 'NW': 0.945, 'TY': 0.596, 'TV': 0.345, 'TW': 0.816, 'TT': 0.0, 'TR': 1.0, 'TS': 0.185, 'TP': 0.159, 'TQ': 0.322, 'TN': 0.315, 'TL': 0.453, 'TM': 0.403, 'TK': 0.866, 'TH': 0.737, 'TI': 0.455, 'TF': 0.604, 'TG': 0.312, 'TD': 0.83, 'TE': 0.812, 'TC': 0.261, 'TA': 0.251, 'AA': 0.0, 'AC': 0.112, 'AE': 0.827, 'AD': 0.819, 'AG': 0.208, 'AF': 0.54, 'AI': 0.407, 'AH': 0.696, 'AK': 0.891, 'AM': 0.379, 'AL': 0.406, 'AN': 0.318, 'AQ': 0.372, 'AP': 0.191, 'AS': 0.094, 'AR': 1.0, 'AT': 0.22, 'AW': 0.739, 'AV': 0.273, 'AY': 0.552, 'VK': 0.889}
):
    """GetQuasiSequenceOrder1SW computes the first 20 quasi-sequence-order (QSOSW) descriptors for a single protein sequence using a Schneider–Wrede style pairwise amino-acid distance matrix. These descriptors combine normalized amino-acid composition with sequence-order coupling information (summed coupling numbers across lags) and are intended for use as fixed-length protein encodings in DeepPurpose models for tasks such as drug–target interaction (DTI), protein property prediction, PPI, and protein function prediction described in the DeepPurpose README.
    
    Args:
        ProteinSequence (str): Protein primary sequence provided as a one-letter amino-acid string (e.g., "MTEYK..."). This function uses the sequence to compute amino-acid composition (via GetAAComposition) and to evaluate sequence-order coupling numbers (via GetSequenceOrderCouplingNumber). Practical significance: this argument is the raw biological input representing the target protein whose QSOSW descriptors will be used as features in downstream machine learning models (DTI, virtual screening, repurposing).
        maxlag (int): Maximum lag (positive integer) used when summing sequence-order coupling numbers across sequence separations from 1 up to maxlag (the code uses range(maxlag) and passes i+1 to the coupling routine). Default is 30. Role: larger maxlag includes longer-range sequence-order correlations; smaller maxlag focuses on local order. Failure modes: very large values may increase computation time and may produce KeyError or other errors if downstream coupling routines expect sequence length >= lag.
        weight (float): Weighting factor (default 0.1) that balances the contribution of sequence-order coupling information against raw amino-acid composition. The final amino-acid composition components are normalized by temp = 1 + weight * rightpart, where rightpart is the sum of coupling numbers for all lags. Practical significance: increasing weight emphasizes sequence-order (global/positional) information relative to composition; decreasing weight emphasizes composition. Failure modes: negative or extreme values will change normalization behavior and may produce unexpected encodings; the function does not validate numeric ranges beyond using the value in arithmetic.
        distancematrix (dict): Mapping of two-letter amino-acid pair keys (e.g., "AW", "GA", "YY") to float distance values that quantify physicochemical or evolutionary distances between amino-acid types. The function uses this matrix when computing sequence-order coupling numbers (GetSequenceOrderCouplingNumber). The signature default is a pre-defined SW-style distance dictionary (the repository default _Distance1) appropriate for QSOSW calculation. Practical significance: supplying an alternative distance matrix allows using different pairwise metrics; keys must cover all amino-acid pairs present in ProteinSequence. Failure modes: missing keys for any required pair will typically raise a KeyError or propagate errors from GetSequenceOrderCouplingNumber.
    
    Behavior and details:
        - Computes rightpart as the sum over lags 1..maxlag of GetSequenceOrderCouplingNumber(ProteinSequence, lag, distancematrix). This accumulates sequence-order coupling information used to modulate composition.
        - Computes amino-acid composition AAC for ProteinSequence using GetAAComposition; AAC is expected to return frequencies per amino-acid token from the global AALetter list (20 standard amino acids).
        - Normalizes each amino-acid composition entry by temp = 1 + weight * rightpart and rounds the result to six decimal places.
        - Produces exactly one descriptor per amino-acid letter in AALetter; in typical usage with 20 standard amino acids, this yields 20 descriptors named "QSOSW1" through "QSOSW20". These keys correspond, in order, to the amino-acid ordering in the module-level AALetter list used elsewhere in DeepPurpose.
        - No external side effects (does not modify global state) beyond calling helper functions. Computation time scales with sequence length and maxlag.
        - Dependence: relies on GetSequenceOrderCouplingNumber, GetAAComposition, and the module constant AALetter. If those helpers raise errors (e.g., due to invalid residues or missing distance entries) those errors propagate to the caller.
        - Typical usage in DeepPurpose: produce a compact, fixed-length protein feature vector suitable for concatenation with drug encodings for DTI model inputs or for standalone protein prediction models.
    
    Returns:
        dict: A dictionary mapping descriptor names to float values, e.g., {"QSOSW1": 0.012345, ..., "QSOSW20": 0.000000}. Each value is the normalized amino-acid composition entry for the corresponding amino-acid in AALetter after modulation by sequence-order information, rounded to six decimal places. If ProteinSequence contains non-standard characters or the provided distancematrix lacks required pair keys, the function may raise KeyError or propagate exceptions from the helper routines; it does not return None on error.
    """
    from DeepPurpose.pybiomed_helper import GetQuasiSequenceOrder1SW
    return GetQuasiSequenceOrder1SW(ProteinSequence, maxlag, weight, distancematrix)


################################################################################
# Source: DeepPurpose.pybiomed_helper.GetQuasiSequenceOrder2
# File: DeepPurpose/pybiomed_helper.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def DeepPurpose_pybiomed_helper_GetQuasiSequenceOrder2(
    ProteinSequence: str,
    maxlag: int = 30,
    weight: float = 0.1,
    distancematrix: dict = {}
):
    """GetQuasiSequenceOrder2 computes the last maxlag quasi-sequence-order (QSO) descriptors for a protein sequence. These QSO descriptors are numerical protein-encoding features used by DeepPurpose for downstream tasks such as drug-target interaction (DTI) prediction, protein-protein interaction (PPI) prediction, protein function prediction, and other molecular modeling workflows described in the DeepPurpose README. The implementation follows the common quasi-sequence-order descriptor formulation where coupling numbers for sequence-order effects are combined, weighted, and normalized to produce a fixed set of descriptors that capture local and long-range amino-acid relationship information.
    
    Args:
        ProteinSequence (str): Amino-acid sequence of the protein for which QSO descriptors are computed. The sequence is passed verbatim to internal helper functions GetSequenceOrderCouplingNumber and GetAAComposition; non-standard characters or unusual alphabets may cause those helpers to raise errors or produce unexpected results. In DeepPurpose workflows, this string typically comes from protein sequence datasets used for DTI/ProteinPred/PPI tasks (e.g., FASTA-style sequences).
        maxlag (int = 30): Number of sequence-order lags to include (k values from 1 to maxlag). The function computes coupling numbers for each lag and returns maxlag QSO descriptors named "QSO21" through "QSO{20+maxlag}". A typical default of 30 produces descriptors QSO21..QSO50. If maxlag is non-positive or not an integer, the behavior depends on Python iteration semantics but is not validated in this function (calling code should pass a positive integer).
        weight (float = 0.1): Weighting factor applied to each coupling-number term before normalization. The weight scales the contribution of sequence-order coupling numbers relative to the implicit zeroth-order term in the normalization denominator. The default 0.1 is commonly used in QSO formulations; changing weight will proportionally change the returned descriptor values according to the formula described below.
        distancematrix (dict = {}): Distance matrix supplied as a dictionary and forwarded to GetSequenceOrderCouplingNumber to compute pairwise amino-acid coupling numbers for each lag. The exact expected structure and keys of this dictionary are defined by GetSequenceOrderCouplingNumber (typically a mapping encoding amino-acid pair distances). If an empty dict is provided, the called helper may use internal defaults or raise an error if distances are required; missing entries in distancematrix can cause KeyError or computation errors in the helper functions.
    
    Detailed behavior:
        1. For each lag k in 1..maxlag the function calls GetSequenceOrderCouplingNumber(ProteinSequence, k, distancematrix) and collects the returned numeric coupling numbers into a list named rightpart. These coupling numbers quantify sequence-order-dependent relationships at distance k along the sequence and are used as the numerator components of QSO descriptors.
        2. The function also calls GetAAComposition(ProteinSequence) to compute amino-acid composition (AAC). In this implementation AAC is computed but not used in subsequent calculations; this call is retained for compatibility with related QSO algorithms or for side-effect compatibility with other implementations.
        3. A normalization denominator temp is computed as 1 + weight * sum(rightpart). This ensures the returned descriptors are scaled relative to a base term (1) plus the weighted total of coupling numbers, following the standard QSO normalization.
        4. For each lag k (1..maxlag) the function produces a descriptor with key "QSO{20+k}" (for example, lag 1 -> "QSO21") and value equal to round(weight * rightpart[k-1] / temp, 6). Values are rounded to six decimal places.
        5. The result is returned as a Python dict mapping descriptor names (strings) to float values.
    
    Returns:
        dict: Mapping of descriptor names to float values. The dict contains maxlag entries with keys "QSO21", "QSO22", ..., "QSO{20+maxlag}". Each value is a floating-point number equal to the normalized, weighted coupling number for the corresponding lag, rounded to six decimal places. Example: for maxlag=3 the returned keys will be "QSO21", "QSO22", and "QSO23".
    
    Side effects and performance:
        - The function is pure with respect to its own inputs (it does not modify ProteinSequence or distancematrix in-place) but it does call external helpers (GetSequenceOrderCouplingNumber and GetAAComposition) which may perform their own computations or validations.
        - Time complexity scales linearly with maxlag and with the cost of computing coupling numbers for each lag; larger maxlag increases runtime proportionally.
        - No files are read or written by this function.
    
    Failure modes and cautions:
        - If ProteinSequence is not a str or contains unexpected characters, the helper functions may raise exceptions (TypeError, KeyError) or return incorrect values.
        - If distancematrix is missing required entries for the helper function, a KeyError or similar error may be raised by GetSequenceOrderCouplingNumber.
        - No validation is performed on maxlag or weight; passing nonsensical values (negative integers, non-integer maxlag, nan/inf weight) may lead to incorrect outputs or runtime errors.
        - AAC is computed but unused in this implementation; if callers expect AAC-based descriptors to be present they should use a different helper or extend this function.
    
    Practical significance:
        - The returned QSO descriptors are a compact numeric representation of sequence-order information used as protein encodings in DeepPurpose models for DTI, PPI, ProteinPred, and related tasks. These descriptors can be concatenated with drug encodings (e.g., SMILES-based embeddings, molecular fingerprints, or GNN-derived features) to train predictive models for binding affinity, interaction classification, property prediction, repurposing, and virtual screening described in the DeepPurpose README.
    """
    from DeepPurpose.pybiomed_helper import GetQuasiSequenceOrder2
    return GetQuasiSequenceOrder2(ProteinSequence, maxlag, weight, distancematrix)


################################################################################
# Source: DeepPurpose.pybiomed_helper.GetQuasiSequenceOrder2Grant
# File: DeepPurpose/pybiomed_helper.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def DeepPurpose_pybiomed_helper_GetQuasiSequenceOrder2Grant(
    ProteinSequence: str,
    maxlag: int = 30,
    weight: float = 0.1,
    distancematrix: dict = {'GW': 0.923, 'GV': 0.464, 'GT': 0.272, 'GS': 0.158, 'GR': 1.0, 'GQ': 0.467, 'GP': 0.323, 'GY': 0.728, 'GG': 0.0, 'GF': 0.727, 'GE': 0.807, 'GD': 0.776, 'GC': 0.312, 'GA': 0.206, 'GN': 0.381, 'GM': 0.557, 'GL': 0.591, 'GK': 0.894, 'GI': 0.592, 'GH': 0.769, 'ME': 0.879, 'MD': 0.932, 'MG': 0.569, 'MF': 0.182, 'MA': 0.383, 'MC': 0.276, 'MM': 0.0, 'ML': 0.062, 'MN': 0.447, 'MI': 0.058, 'MH': 0.648, 'MK': 0.884, 'MT': 0.358, 'MW': 0.391, 'MV': 0.12, 'MQ': 0.372, 'MP': 0.285, 'MS': 0.417, 'MR': 1.0, 'MY': 0.255, 'FP': 0.42, 'FQ': 0.459, 'FR': 1.0, 'FS': 0.548, 'FT': 0.499, 'FV': 0.252, 'FW': 0.207, 'FY': 0.179, 'FA': 0.508, 'FC': 0.405, 'FD': 0.977, 'FE': 0.918, 'FF': 0.0, 'FG': 0.69, 'FH': 0.663, 'FI': 0.128, 'FK': 0.903, 'FL': 0.131, 'FM': 0.169, 'FN': 0.541, 'SY': 0.615, 'SS': 0.0, 'SR': 1.0, 'SQ': 0.358, 'SP': 0.181, 'SW': 0.827, 'SV': 0.342, 'ST': 0.174, 'SK': 0.883, 'SI': 0.478, 'SH': 0.718, 'SN': 0.289, 'SM': 0.44, 'SL': 0.474, 'SC': 0.185, 'SA': 0.1, 'SG': 0.17, 'SF': 0.622, 'SE': 0.812, 'SD': 0.801, 'YI': 0.23, 'YH': 0.678, 'YK': 0.904, 'YM': 0.268, 'YL': 0.219, 'YN': 0.512, 'YA': 0.587, 'YC': 0.478, 'YE': 0.932, 'YD': 1.0, 'YG': 0.782, 'YF': 0.202, 'YY': 0.0, 'YQ': 0.404, 'YP': 0.444, 'YS': 0.612, 'YR': 0.995, 'YT': 0.557, 'YW': 0.244, 'YV': 0.328, 'LF': 0.139, 'LG': 0.596, 'LD': 0.944, 'LE': 0.892, 'LC': 0.296, 'LA': 0.405, 'LN': 0.452, 'LL': 0.0, 'LM': 0.062, 'LK': 0.893, 'LH': 0.653, 'LI': 0.013, 'LV': 0.133, 'LW': 0.341, 'LT': 0.397, 'LR': 1.0, 'LS': 0.443, 'LP': 0.309, 'LQ': 0.376, 'LY': 0.205, 'RT': 0.808, 'RV': 0.914, 'RW': 1.0, 'RP': 0.796, 'RQ': 0.668, 'RR': 0.0, 'RS': 0.86, 'RY': 0.859, 'RD': 0.305, 'RE': 0.225, 'RF': 0.977, 'RG': 0.928, 'RA': 0.919, 'RC': 0.905, 'RL': 0.92, 'RM': 0.908, 'RN': 0.69, 'RH': 0.498, 'RI': 0.929, 'RK': 0.141, 'VH': 0.649, 'VI': 0.135, 'EM': 0.83, 'EL': 0.854, 'EN': 0.599, 'EI': 0.86, 'EH': 0.406, 'EK': 0.143, 'EE': 0.0, 'ED': 0.133, 'EG': 0.779, 'EF': 0.932, 'EA': 0.79, 'EC': 0.788, 'VM': 0.12, 'EY': 0.837, 'VN': 0.38, 'ET': 0.682, 'EW': 1.0, 'EV': 0.824, 'EQ': 0.598, 'EP': 0.688, 'ES': 0.726, 'ER': 0.234, 'VP': 0.212, 'VQ': 0.339, 'VR': 1.0, 'VT': 0.305, 'VW': 0.472, 'KC': 0.871, 'KA': 0.889, 'KG': 0.9, 'KF': 0.957, 'KE': 0.149, 'KD': 0.279, 'KK': 0.0, 'KI': 0.899, 'KH': 0.438, 'KN': 0.667, 'KM': 0.871, 'KL': 0.892, 'KS': 0.825, 'KR': 0.154, 'KQ': 0.639, 'KP': 0.757, 'KW': 1.0, 'KV': 0.882, 'KT': 0.759, 'KY': 0.848, 'DN': 0.56, 'DL': 0.841, 'DM': 0.819, 'DK': 0.249, 'DH': 0.435, 'DI': 0.847, 'DF': 0.924, 'DG': 0.697, 'DD': 0.0, 'DE': 0.124, 'DC': 0.742, 'DA': 0.729, 'DY': 0.836, 'DV': 0.797, 'DW': 1.0, 'DT': 0.649, 'DR': 0.295, 'DS': 0.667, 'DP': 0.657, 'DQ': 0.584, 'QQ': 0.0, 'QP': 0.272, 'QS': 0.461, 'QR': 1.0, 'QT': 0.389, 'QW': 0.831, 'QV': 0.464, 'QY': 0.522, 'QA': 0.512, 'QC': 0.462, 'QE': 0.861, 'QD': 0.903, 'QG': 0.648, 'QF': 0.671, 'QI': 0.532, 'QH': 0.765, 'QK': 0.881, 'QM': 0.505, 'QL': 0.518, 'QN': 0.181, 'WG': 0.829, 'WF': 0.196, 'WE': 0.931, 'WD': 1.0, 'WC': 0.56, 'WA': 0.658, 'WN': 0.631, 'WM': 0.344, 'WL': 0.304, 'WK': 0.892, 'WI': 0.305, 'WH': 0.678, 'WW': 0.0, 'WV': 0.418, 'WT': 0.638, 'WS': 0.689, 'WR': 0.968, 'WQ': 0.538, 'WP': 0.555, 'WY': 0.204, 'PR': 1.0, 'PS': 0.196, 'PP': 0.0, 'PQ': 0.228, 'PV': 0.244, 'PW': 0.72, 'PT': 0.161, 'PY': 0.481, 'PC': 0.179, 'PA': 0.22, 'PF': 0.515, 'PG': 0.376, 'PD': 0.852, 'PE': 0.831, 'PK': 0.875, 'PH': 0.696, 'PI': 0.363, 'PN': 0.231, 'PL': 0.357, 'PM': 0.326, 'CK': 0.887, 'CI': 0.304, 'CH': 0.66, 'CN': 0.324, 'CM': 0.277, 'CL': 0.301, 'CC': 0.0, 'CA': 0.114, 'CG': 0.32, 'CF': 0.437, 'CE': 0.838, 'CD': 0.847, 'CY': 0.457, 'CS': 0.176, 'CR': 1.0, 'CQ': 0.341, 'CP': 0.157, 'CW': 0.639, 'CV': 0.167, 'CT': 0.233, 'IY': 0.213, 'VA': 0.275, 'VC': 0.165, 'VD': 0.9, 'VE': 0.867, 'VF': 0.269, 'VG': 0.471, 'IQ': 0.383, 'IP': 0.311, 'IS': 0.443, 'IR': 1.0, 'VL': 0.134, 'IT': 0.396, 'IW': 0.339, 'IV': 0.133, 'II': 0.0, 'IH': 0.652, 'IK': 0.892, 'VS': 0.322, 'IM': 0.057, 'IL': 0.013, 'VV': 0.0, 'IN': 0.457, 'IA': 0.403, 'VY': 0.31, 'IC': 0.296, 'IE': 0.891, 'ID': 0.942, 'IG': 0.592, 'IF': 0.134, 'HY': 0.821, 'HR': 0.697, 'HS': 0.865, 'HP': 0.777, 'HQ': 0.716, 'HV': 0.831, 'HW': 0.981, 'HT': 0.834, 'HK': 0.566, 'HH': 0.0, 'HI': 0.848, 'HN': 0.754, 'HL': 0.842, 'HM': 0.825, 'HC': 0.836, 'HA': 0.896, 'HF': 0.907, 'HG': 1.0, 'HD': 0.629, 'HE': 0.547, 'NH': 0.78, 'NI': 0.615, 'NK': 0.891, 'NL': 0.603, 'NM': 0.588, 'NN': 0.0, 'NA': 0.424, 'NC': 0.425, 'ND': 0.838, 'NE': 0.835, 'NF': 0.766, 'NG': 0.512, 'NY': 0.641, 'NP': 0.266, 'NQ': 0.175, 'NR': 1.0, 'NS': 0.361, 'NT': 0.368, 'NV': 0.503, 'NW': 0.945, 'TY': 0.596, 'TV': 0.345, 'TW': 0.816, 'TT': 0.0, 'TR': 1.0, 'TS': 0.185, 'TP': 0.159, 'TQ': 0.322, 'TN': 0.315, 'TL': 0.453, 'TM': 0.403, 'TK': 0.866, 'TH': 0.737, 'TI': 0.455, 'TF': 0.604, 'TG': 0.312, 'TD': 0.83, 'TE': 0.812, 'TC': 0.261, 'TA': 0.251, 'AA': 0.0, 'AC': 0.112, 'AE': 0.827, 'AD': 0.819, 'AG': 0.208, 'AF': 0.54, 'AI': 0.407, 'AH': 0.696, 'AK': 0.891, 'AM': 0.379, 'AL': 0.406, 'AN': 0.318, 'AQ': 0.372, 'AP': 0.191, 'AS': 0.094, 'AR': 1.0, 'AT': 0.22, 'AW': 0.739, 'AV': 0.273, 'AY': 0.552, 'VK': 0.889}
):
    """Compute the last maxlag quasi-sequence-order descriptors (Grant variant) for a single protein sequence and return them as a dense dictionary of normalized numeric descriptors. This function implements the "quasi-sequence-order" feature construction used in protein encoding for downstream DeepPurpose tasks (drug-target interaction prediction, protein function prediction, PPI, etc.). Practically, it calls GetSequenceOrderCouplingNumber to compute sequence-order coupling numbers for successive lags and returns the final maxlag descriptors normalized by a factor that includes the provided weight. The computed descriptors capture positional/physicochemical relationships between amino acids using the supplied pairwise distance matrix and are intended to be used as fixed-length protein feature inputs to DeepPurpose models, repurposing and virtual screening pipelines.
    
    Args:
        ProteinSequence (str): A single protein amino-acid sequence represented as a string of one-letter amino-acid codes (for example, "MSTNPKPQR"). This is the sequence whose quasi-sequence-order descriptors are computed. The sequence is passed to internal helpers GetSequenceOrderCouplingNumber and GetAAComposition; characters in ProteinSequence must therefore be compatible with the keys expected by distancematrix and by those helpers. Supplying non-standard characters or letters not covered by distancematrix may raise a KeyError or cause the helper functions to fail.
        maxlag (int): Number of successive lag coupling terms to compute and include in the returned descriptor set. The function computes coupling numbers for lag = 1..maxlag (via repeated calls to GetSequenceOrderCouplingNumber) and returns exactly maxlag entries (with keys described below). Default is 30. maxlag should be an integer; non-positive or zero values result in an empty descriptor mapping (no lag descriptors computed).
        weight (float): Scaling weight applied to each coupling number when forming the returned quasi-sequence-order descriptors. Each returned value is computed as (weight * coupling_number_for_lag) / (1 + weight * sum_of_all_computed_coupling_numbers). The default is 0.1. This parameter controls the relative contribution of sequence-order terms versus the implicit normalization constant; changing weight rescales and rebalances the resulting descriptors used downstream by DeepPurpose models.
        distancematrix (dict): A mapping (dict) from two-letter amino-acid pair strings (str) to float distances (for example, 'AC': 0.112) that encodes the physicochemical/structural distance between amino acids for computing coupling numbers. This matrix is used by GetSequenceOrderCouplingNumber to compute sequence-order coupling for each lag. The default provided is a comprehensive dictionary of pairwise values used in the original implementation. The caller must ensure that all amino-acid pair keys required by the sequence and lag computations exist in this dict; missing keys will raise an exception in the helper routines.
    
    Behavior and side effects:
        The function computes a list of coupling numbers by calling GetSequenceOrderCouplingNumber(ProteinSequence, lag, distancematrix) for lag = 1..maxlag. It also calls GetAAComposition(ProteinSequence) (the resulting amino-acid composition is computed but not returned; this call is retained for compatibility with related descriptor routines and may have a small CPU cost). The final returned descriptor values are normalized by temp = 1 + weight * sum_of_all_coupling_numbers and rounded to six decimal places. There are no writes to disk or changes to global state performed by this function; its only side effects are the calls to the referenced helper functions and their associated CPU/memory usage.
    
    Failure modes and validation notes:
        If ProteinSequence contains characters not recognized by the helper functions or not present in distancematrix keys, the internal calls (GetSequenceOrderCouplingNumber or GetAAComposition) may raise a KeyError or other exception. If maxlag is not an integer, a TypeError may be raised when used in iteration; if it is negative or zero, the function will produce an empty descriptor dictionary. If distancematrix values are not numeric, arithmetic errors may occur. The function does not perform extensive input validation beyond relying on the helper functions; callers should ensure inputs conform to expected types and content.
    
    Returns:
        dict: A dictionary mapping descriptor names to numeric values. The keys are strings of the form "QSOgrant{N}" where N ranges from 21 to 20 + maxlag (for example, if maxlag == 30 the returned keys are "QSOgrant21" through "QSOgrant50"). Each value is a float equal to round(weight * coupling_number_for_lag / (1 + weight * sum_of_all_coupling_numbers), 6). These numeric descriptors are intended to be used directly as fixed-length protein feature inputs in DeepPurpose model training, prediction, repurposing, and virtual screening pipelines.
    """
    from DeepPurpose.pybiomed_helper import GetQuasiSequenceOrder2Grant
    return GetQuasiSequenceOrder2Grant(ProteinSequence, maxlag, weight, distancematrix)


################################################################################
# Source: DeepPurpose.pybiomed_helper.GetQuasiSequenceOrder2SW
# File: DeepPurpose/pybiomed_helper.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def DeepPurpose_pybiomed_helper_GetQuasiSequenceOrder2SW(
    ProteinSequence: str,
    maxlag: int = 30,
    weight: float = 0.1,
    distancematrix: dict = {'GW': 0.923, 'GV': 0.464, 'GT': 0.272, 'GS': 0.158, 'GR': 1.0, 'GQ': 0.467, 'GP': 0.323, 'GY': 0.728, 'GG': 0.0, 'GF': 0.727, 'GE': 0.807, 'GD': 0.776, 'GC': 0.312, 'GA': 0.206, 'GN': 0.381, 'GM': 0.557, 'GL': 0.591, 'GK': 0.894, 'GI': 0.592, 'GH': 0.769, 'ME': 0.879, 'MD': 0.932, 'MG': 0.569, 'MF': 0.182, 'MA': 0.383, 'MC': 0.276, 'MM': 0.0, 'ML': 0.062, 'MN': 0.447, 'MI': 0.058, 'MH': 0.648, 'MK': 0.884, 'MT': 0.358, 'MW': 0.391, 'MV': 0.12, 'MQ': 0.372, 'MP': 0.285, 'MS': 0.417, 'MR': 1.0, 'MY': 0.255, 'FP': 0.42, 'FQ': 0.459, 'FR': 1.0, 'FS': 0.548, 'FT': 0.499, 'FV': 0.252, 'FW': 0.207, 'FY': 0.179, 'FA': 0.508, 'FC': 0.405, 'FD': 0.977, 'FE': 0.918, 'FF': 0.0, 'FG': 0.69, 'FH': 0.663, 'FI': 0.128, 'FK': 0.903, 'FL': 0.131, 'FM': 0.169, 'FN': 0.541, 'SY': 0.615, 'SS': 0.0, 'SR': 1.0, 'SQ': 0.358, 'SP': 0.181, 'SW': 0.827, 'SV': 0.342, 'ST': 0.174, 'SK': 0.883, 'SI': 0.478, 'SH': 0.718, 'SN': 0.289, 'SM': 0.44, 'SL': 0.474, 'SC': 0.185, 'SA': 0.1, 'SG': 0.17, 'SF': 0.622, 'SE': 0.812, 'SD': 0.801, 'YI': 0.23, 'YH': 0.678, 'YK': 0.904, 'YM': 0.268, 'YL': 0.219, 'YN': 0.512, 'YA': 0.587, 'YC': 0.478, 'YE': 0.932, 'YD': 1.0, 'YG': 0.782, 'YF': 0.202, 'YY': 0.0, 'YQ': 0.404, 'YP': 0.444, 'YS': 0.612, 'YR': 0.995, 'YT': 0.557, 'YW': 0.244, 'YV': 0.328, 'LF': 0.139, 'LG': 0.596, 'LD': 0.944, 'LE': 0.892, 'LC': 0.296, 'LA': 0.405, 'LN': 0.452, 'LL': 0.0, 'LM': 0.062, 'LK': 0.893, 'LH': 0.653, 'LI': 0.013, 'LV': 0.133, 'LW': 0.341, 'LT': 0.397, 'LR': 1.0, 'LS': 0.443, 'LP': 0.309, 'LQ': 0.376, 'LY': 0.205, 'RT': 0.808, 'RV': 0.914, 'RW': 1.0, 'RP': 0.796, 'RQ': 0.668, 'RR': 0.0, 'RS': 0.86, 'RY': 0.859, 'RD': 0.305, 'RE': 0.225, 'RF': 0.977, 'RG': 0.928, 'RA': 0.919, 'RC': 0.905, 'RL': 0.92, 'RM': 0.908, 'RN': 0.69, 'RH': 0.498, 'RI': 0.929, 'RK': 0.141, 'VH': 0.649, 'VI': 0.135, 'EM': 0.83, 'EL': 0.854, 'EN': 0.599, 'EI': 0.86, 'EH': 0.406, 'EK': 0.143, 'EE': 0.0, 'ED': 0.133, 'EG': 0.779, 'EF': 0.932, 'EA': 0.79, 'EC': 0.788, 'VM': 0.12, 'EY': 0.837, 'VN': 0.38, 'ET': 0.682, 'EW': 1.0, 'EV': 0.824, 'EQ': 0.598, 'EP': 0.688, 'ES': 0.726, 'ER': 0.234, 'VP': 0.212, 'VQ': 0.339, 'VR': 1.0, 'VT': 0.305, 'VW': 0.472, 'KC': 0.871, 'KA': 0.889, 'KG': 0.9, 'KF': 0.957, 'KE': 0.149, 'KD': 0.279, 'KK': 0.0, 'KI': 0.899, 'KH': 0.438, 'KN': 0.667, 'KM': 0.871, 'KL': 0.892, 'KS': 0.825, 'KR': 0.154, 'KQ': 0.639, 'KP': 0.757, 'KW': 1.0, 'KV': 0.882, 'KT': 0.759, 'KY': 0.848, 'DN': 0.56, 'DL': 0.841, 'DM': 0.819, 'DK': 0.249, 'DH': 0.435, 'DI': 0.847, 'DF': 0.924, 'DG': 0.697, 'DD': 0.0, 'DE': 0.124, 'DC': 0.742, 'DA': 0.729, 'DY': 0.836, 'DV': 0.797, 'DW': 1.0, 'DT': 0.649, 'DR': 0.295, 'DS': 0.667, 'DP': 0.657, 'DQ': 0.584, 'QQ': 0.0, 'QP': 0.272, 'QS': 0.461, 'QR': 1.0, 'QT': 0.389, 'QW': 0.831, 'QV': 0.464, 'QY': 0.522, 'QA': 0.512, 'QC': 0.462, 'QE': 0.861, 'QD': 0.903, 'QG': 0.648, 'QF': 0.671, 'QI': 0.532, 'QH': 0.765, 'QK': 0.881, 'QM': 0.505, 'QL': 0.518, 'QN': 0.181, 'WG': 0.829, 'WF': 0.196, 'WE': 0.931, 'WD': 1.0, 'WC': 0.56, 'WA': 0.658, 'WN': 0.631, 'WM': 0.344, 'WL': 0.304, 'WK': 0.892, 'WI': 0.305, 'WH': 0.678, 'WW': 0.0, 'WV': 0.418, 'WT': 0.638, 'WS': 0.689, 'WR': 0.968, 'WQ': 0.538, 'WP': 0.555, 'WY': 0.204, 'PR': 1.0, 'PS': 0.196, 'PP': 0.0, 'PQ': 0.228, 'PV': 0.244, 'PW': 0.72, 'PT': 0.161, 'PY': 0.481, 'PC': 0.179, 'PA': 0.22, 'PF': 0.515, 'PG': 0.376, 'PD': 0.852, 'PE': 0.831, 'PK': 0.875, 'PH': 0.696, 'PI': 0.363, 'PN': 0.231, 'PL': 0.357, 'PM': 0.326, 'CK': 0.887, 'CI': 0.304, 'CH': 0.66, 'CN': 0.324, 'CM': 0.277, 'CL': 0.301, 'CC': 0.0, 'CA': 0.114, 'CG': 0.32, 'CF': 0.437, 'CE': 0.838, 'CD': 0.847, 'CY': 0.457, 'CS': 0.176, 'CR': 1.0, 'CQ': 0.341, 'CP': 0.157, 'CW': 0.639, 'CV': 0.167, 'CT': 0.233, 'IY': 0.213, 'VA': 0.275, 'VC': 0.165, 'VD': 0.9, 'VE': 0.867, 'VF': 0.269, 'VG': 0.471, 'IQ': 0.383, 'IP': 0.311, 'IS': 0.443, 'IR': 1.0, 'VL': 0.134, 'IT': 0.396, 'IW': 0.339, 'IV': 0.133, 'II': 0.0, 'IH': 0.652, 'IK': 0.892, 'VS': 0.322, 'IM': 0.057, 'IL': 0.013, 'VV': 0.0, 'IN': 0.457, 'IA': 0.403, 'VY': 0.31, 'IC': 0.296, 'IE': 0.891, 'ID': 0.942, 'IG': 0.592, 'IF': 0.134, 'HY': 0.821, 'HR': 0.697, 'HS': 0.865, 'HP': 0.777, 'HQ': 0.716, 'HV': 0.831, 'HW': 0.981, 'HT': 0.834, 'HK': 0.566, 'HH': 0.0, 'HI': 0.848, 'HN': 0.754, 'HL': 0.842, 'HM': 0.825, 'HC': 0.836, 'HA': 0.896, 'HF': 0.907, 'HG': 1.0, 'HD': 0.629, 'HE': 0.547, 'NH': 0.78, 'NI': 0.615, 'NK': 0.891, 'NL': 0.603, 'NM': 0.588, 'NN': 0.0, 'NA': 0.424, 'NC': 0.425, 'ND': 0.838, 'NE': 0.835, 'NF': 0.766, 'NG': 0.512, 'NY': 0.641, 'NP': 0.266, 'NQ': 0.175, 'NR': 1.0, 'NS': 0.361, 'NT': 0.368, 'NV': 0.503, 'NW': 0.945, 'TY': 0.596, 'TV': 0.345, 'TW': 0.816, 'TT': 0.0, 'TR': 1.0, 'TS': 0.185, 'TP': 0.159, 'TQ': 0.322, 'TN': 0.315, 'TL': 0.453, 'TM': 0.403, 'TK': 0.866, 'TH': 0.737, 'TI': 0.455, 'TF': 0.604, 'TG': 0.312, 'TD': 0.83, 'TE': 0.812, 'TC': 0.261, 'TA': 0.251, 'AA': 0.0, 'AC': 0.112, 'AE': 0.827, 'AD': 0.819, 'AG': 0.208, 'AF': 0.54, 'AI': 0.407, 'AH': 0.696, 'AK': 0.891, 'AM': 0.379, 'AL': 0.406, 'AN': 0.318, 'AQ': 0.372, 'AP': 0.191, 'AS': 0.094, 'AR': 1.0, 'AT': 0.22, 'AW': 0.739, 'AV': 0.273, 'AY': 0.552, 'VK': 0.889}
):
    """DeepPurpose.pybiomed_helper.GetQuasiSequenceOrder2SW computes the last maxlag quasi-sequence-order (QSOSW) descriptors for a single protein sequence and returns them as a dictionary of normalized coupling-number features. These descriptors are numerical features used by DeepPurpose as part of the protein encoding pipeline for downstream tasks such as drug-target interaction (DTI) prediction, protein-protein interaction (PPI) prediction, protein function prediction and other molecular modeling use cases described in the DeepPurpose README. The function implements the same per-lag coupling-number normalization strategy used by related QS descriptor routines in this module and relies on pairwise amino-acid distance values to capture sequence-order effects.
    
    Args:
        ProteinSequence (str): A protein primary sequence represented as a string of single-letter amino-acid codes (e.g., 'MQDRVKRPMNAFIVWSRDQRRKMALEN'). This is the input sequence for which the quasi-sequence-order descriptors are computed. The sequence must be provided as a Python str. If the sequence contains non-standard letters not handled by the helper routines (GetSequenceOrderCouplingNumber / GetAAComposition), those helpers may raise errors (KeyError or custom validation errors).
        maxlag (int): The number of successive sequence-order coupling lags to compute (default 30). The function computes coupling numbers for lags 1..maxlag via GetSequenceOrderCouplingNumber and returns that many QSOSW descriptors. If maxlag <= 0 the function will produce no QSOSW descriptors and return an empty dict. maxlag must be an integer; passing a non-integer will raise a TypeError when used in range().
        weight (float): A non-negative float weight (default 0.1) used to scale and normalize the coupling-number contributions. The returned QSOSW descriptors are computed as (weight * coupling_number) / (1 + weight * sum(all_coupling_numbers)). Changing weight adjusts the relative contribution of the sequence-order (coupling) part versus the implicit amino-acid composition baseline used in normalization. Negative weights are not prevented by this function but will change the normalization denominator and may lead to unexpected descriptor signs or magnitudes.
        distancematrix (dict): A mapping from two-letter amino-acid pair strings (e.g., 'AK', 'GY') to numeric pairwise distance values (default provided _Distance1 dict). This dictionary supplies the per-residue-pair distances used by GetSequenceOrderCouplingNumber to compute each lag's coupling number. The dict must contain all keys required by the internal coupling-number routine; missing keys will typically cause KeyError in the helper. The default distancematrix is a precomputed pairwise amino-acid distance table used by the module (exposed as _Distance1 in the source).
    
    Behavior and details:
        The function computes a list rightpart of length maxlag where each element rightpart[i] is GetSequenceOrderCouplingNumber(ProteinSequence, i+1, distancematrix). These coupling numbers quantify sequence-order correlations at increasing residue separation (lag). The function also calls GetAAComposition(ProteinSequence) and assigns it to AAC; this call is retained for consistency with related quasi-sequence-order computations in the codebase though AAC is not used in the returned values of this specific function.
        The returned descriptors are normalized by the factor temp = 1 + weight * sum(rightpart). For lag index k (0-based in rightpart), the descriptor value is weight * rightpart[k] / temp. Each returned numeric value is rounded to 6 decimal places before insertion into the result dictionary.
        Keys in the returned dictionary follow the exact naming convention used in the source: for lags 1..maxlag the function returns entries keyed "QSOSW21", "QSOSW22", ..., "QSOSW{20+maxlag}". In other words, the function starts numbering returned keys at 21 and continues sequentially for maxlag descriptors (this naming matches the module's descriptor index scheme where the first 20 indices are reserved for amino-acid composition features in related descriptor sets).
        The routine has no side effects (it does not modify global state or files); it is a pure computation that returns a dict.
    
    Performance and complexity:
        Time complexity is proportional to maxlag multiplied by the cost of computing each coupling number (which in turn depends on protein length); expect roughly O(maxlag * L) behavior where L is the sequence length. Memory overhead is modest: an O(maxlag) list plus the returned dict.
    
    Failure modes and validation notes:
        If ProteinSequence is not a str, a TypeError or unexpected behavior may occur.
        If maxlag is not an integer, a TypeError occurs when used in range(); if maxlag is negative or zero the function returns an empty dict.
        If distancematrix lacks required pair keys needed by GetSequenceOrderCouplingNumber, a KeyError (or error raised by that helper) will be raised.
        If weight is such that the normalization denominator temp equals zero (highly unlikely for typical non-negative weight and finite coupling numbers given temp = 1 + weight * sum(...)), numerical instability may occur; the current implementation does not explicitly guard against division-by-zero beyond the formula.
        The function delegates sequence validation (allowed amino-acid letters) and per-pair lookup behavior to the helper routines GetSequenceOrderCouplingNumber and GetAAComposition; consult those functions for additional validation rules and parameter choices.
    
    Returns:
        dict: A dictionary mapping descriptor names (str) to float values. Each key is a string of the form "QSOSW{n}" (n runs from 21 to 20+maxlag). Each value is the normalized, weighted sequence-order coupling descriptor for the corresponding lag, rounded to 6 decimal places. These numeric descriptors are intended to be used as protein feature inputs in DeepPurpose models (for example, as part of the full quasi-sequence-order feature vector used in DTI/compound-protein modeling).
    """
    from DeepPurpose.pybiomed_helper import GetQuasiSequenceOrder2SW
    return GetQuasiSequenceOrder2SW(ProteinSequence, maxlag, weight, distancematrix)


################################################################################
# Source: DeepPurpose.pybiomed_helper.GetSequenceOrderCorrelationFactor
# File: DeepPurpose/pybiomed_helper.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def DeepPurpose_pybiomed_helper_GetSequenceOrderCorrelationFactor(
    ProteinSequence: str,
    k: int = 1,
    AAP: list = []
):
    """DeepPurpose.pybiomed_helper.GetSequenceOrderCorrelationFactor computes the sequence order correlation factor for a single protein sequence using a gap (offset) of k and a set of amino-acid property mappings. This scalar descriptor quantifies average pairwise correlation between amino acids separated by k positions in the sequence and is intended for use in DeepPurpose protein encoding and feature-extraction pipelines (for example as part of generating numerical descriptors used in drug–target interaction (DTI), protein–protein interaction (PPI), or protein property prediction models).
    
    Args:
        ProteinSequence (str): A pure protein sequence represented as a string of amino-acid single-letter codes (e.g., "MSTNPKPQR"). This argument is the sequence whose sequence order correlation factor will be computed. In the DeepPurpose workflow this sequence acts as the raw input for per-sequence descriptor calculation; the function does not validate characters, so passing non-standard letters or lowercase may cause downstream errors in the correlation computation.
        k (int): The gap (integer offset) between two amino acids whose correlation is measured. The function iterates over all index pairs (i, i+k) for i in [0, len(ProteinSequence)-k-1], computes a correlation value for each pair, and returns the average. Default is 1. Practically, larger k captures longer-range sequence order effects; k must be a non-negative integer less than the sequence length (len(ProteinSequence)) to avoid a division-by-zero condition.
        AAP (list): A list of amino-acid property mappings; each element is expected to be a dict that maps amino-acid single-letter codes to numeric property values (for example hydrophobicity, polarity, or other physicochemical scales). These property dicts are passed to the helper GetCorrelationFunction to compute the correlation between two amino acids given the chosen properties. In DeepPurpose this list provides the physicochemical context used to translate residue identity pairs into numeric correlation scores. The default empty list is allowed by the signature but will typically cause GetCorrelationFunction to raise an error (TypeError/KeyError) because required property values will be missing.
    
    Behavior and details:
        The function computes the correlation factor by:
        1) Determining the sequence length L = len(ProteinSequence).
        2) For each index i from 0 to L - k - 1, extracting AA1 = ProteinSequence[i] and AA2 = ProteinSequence[i + k].
        3) Calling GetCorrelationFunction(AA1, AA2, AAP) for each pair and collecting the returned numeric correlation values.
        4) Averaging the collected correlation values by dividing their sum by (L - k).
        5) Rounding the average to three decimal places and returning it as a Python float.
        The function itself has no persistent side effects (it does not modify global state or its inputs), but it depends on the availability and correct behavior of GetCorrelationFunction and on the correctness of the AAP entries.
    
    Defaults and failure modes:
        - Default k is 1 (nearest-neighbor correlation). Choosing k >= len(ProteinSequence) results in zero iterations and causes a ZeroDivisionError when computing the average; callers must ensure 0 <= k < len(ProteinSequence).
        - If AAP is empty or its dict(s) lack required keys for residues present in ProteinSequence, GetCorrelationFunction may raise KeyError, TypeError, or other exceptions; callers must supply AAP matching the amino acids in ProteinSequence.
        - If GetCorrelationFunction is not defined in the runtime environment, a NameError will occur when the function is invoked.
        - The function does not perform input normalization (such as uppercasing the sequence) or validation of amino-acid letters; such preprocessing should be performed by the caller if needed.
    
    Returns:
        float: The average sequence order correlation factor for gap k, rounded to three decimal places. This scalar can be used as a numeric feature in DeepPurpose protein encodings for downstream modeling (e.g., as one element of a feature vector used by DTI or protein-prediction models). If computation cannot proceed due to invalid k, malformed AAP, or missing helper functions, an exception will be raised instead of returning a value.
    """
    from DeepPurpose.pybiomed_helper import GetSequenceOrderCorrelationFactor
    return GetSequenceOrderCorrelationFactor(ProteinSequence, k, AAP)


################################################################################
# Source: DeepPurpose.pybiomed_helper.GetSequenceOrderCorrelationFactorForAPAAC
# File: DeepPurpose/pybiomed_helper.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def DeepPurpose_pybiomed_helper_GetSequenceOrderCorrelationFactorForAPAAC(
    ProteinSequence: str,
    k: int = 1
):
    """DeepPurpose.pybiomed_helper.GetSequenceOrderCorrelationFactorForAPAAC computes the sequence-order correlation factors used in the Amphiphilic Pseudo Amino Acid Composition (APAAC, type II PseAAC) encoding for proteins. The function calculates the average correlation between amino acid pairs separated by a gap of k residues using two physicochemical properties: hydrophobicity and hydrophilicity. These two averaged correlation values are commonly appended to composition-based protein descriptors in DeepPurpose for downstream tasks such as drug–target interaction (DTI) prediction, protein–protein interaction (PPI) prediction, and protein function prediction.
    
    Args:
        ProteinSequence (str): A pure protein sequence given as a Python string of one-letter amino acid codes. This sequence is the input whose sequence-order correlation factors are computed. The function expects standard amino-acid single-letter symbols; non-standard characters or gaps may cause the underlying correlation lookup (_GetCorrelationFunctionForAPAAC) to fail or raise an error.
        k (int): Gap (sequence separation) between residue pairs for which correlations are computed. The default value is 1, meaning adjacent residue pairs. The parameter controls the sequence-order distance used by APAAC (type II PseAAC) to capture local and semi-local physicochemical interactions; larger k measures correlations over longer sequence distances.
    
    Returns:
        list: A two-element list of floats [hydrophobicity_corr, hydrophilicity_corr]. Each element is the arithmetic mean (rounded to three decimal places) of the corresponding per-pair correlation values returned by _GetCorrelationFunctionForAPAAC across all valid pairs separated by k in ProteinSequence. The first element is the average hydrophobicity-based correlation and the second is the average hydrophilicity-based correlation. These values are intended to be concatenated with other APAAC components (e.g., amino acid composition) to form type II PseAAC vectors used by DeepPurpose encoders.
    
    Raises:
        ZeroDivisionError: If k is greater than or equal to the length of ProteinSequence, there are no valid residue pairs to average and the function will attempt to divide by zero. Users should ensure len(ProteinSequence) > k.
        Exception: Underlying failures from _GetCorrelationFunctionForAPAAC (for example, due to non-standard amino-acid symbols) will propagate; callers should validate ProteinSequence and handle exceptions as appropriate.
    
    Behavior and side effects:
        The function iterates over each residue index i from 0 to len(ProteinSequence) - k - 1, forms the pair (ProteinSequence[i], ProteinSequence[i + k]), and calls _GetCorrelationFunctionForAPAAC to obtain a two-value correlation for that pair (hydrophobicity and hydrophilicity). It accumulates these values, computes their means, rounds each mean to three decimal places, and returns them as a list. There are no external side effects (no file I/O or global state modification). The function is deterministic given the same inputs and relies on the discrete per-residue correlation mapping implemented in _GetCorrelationFunctionForAPAAC.
    """
    from DeepPurpose.pybiomed_helper import GetSequenceOrderCorrelationFactorForAPAAC
    return GetSequenceOrderCorrelationFactorForAPAAC(ProteinSequence, k)


################################################################################
# Source: DeepPurpose.pybiomed_helper.GetSequenceOrderCouplingNumber
# File: DeepPurpose/pybiomed_helper.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def DeepPurpose_pybiomed_helper_GetSequenceOrderCouplingNumber(
    ProteinSequence: str,
    d: int = 1,
    distancematrix: dict = {'GW': 0.923, 'GV': 0.464, 'GT': 0.272, 'GS': 0.158, 'GR': 1.0, 'GQ': 0.467, 'GP': 0.323, 'GY': 0.728, 'GG': 0.0, 'GF': 0.727, 'GE': 0.807, 'GD': 0.776, 'GC': 0.312, 'GA': 0.206, 'GN': 0.381, 'GM': 0.557, 'GL': 0.591, 'GK': 0.894, 'GI': 0.592, 'GH': 0.769, 'ME': 0.879, 'MD': 0.932, 'MG': 0.569, 'MF': 0.182, 'MA': 0.383, 'MC': 0.276, 'MM': 0.0, 'ML': 0.062, 'MN': 0.447, 'MI': 0.058, 'MH': 0.648, 'MK': 0.884, 'MT': 0.358, 'MW': 0.391, 'MV': 0.12, 'MQ': 0.372, 'MP': 0.285, 'MS': 0.417, 'MR': 1.0, 'MY': 0.255, 'FP': 0.42, 'FQ': 0.459, 'FR': 1.0, 'FS': 0.548, 'FT': 0.499, 'FV': 0.252, 'FW': 0.207, 'FY': 0.179, 'FA': 0.508, 'FC': 0.405, 'FD': 0.977, 'FE': 0.918, 'FF': 0.0, 'FG': 0.69, 'FH': 0.663, 'FI': 0.128, 'FK': 0.903, 'FL': 0.131, 'FM': 0.169, 'FN': 0.541, 'SY': 0.615, 'SS': 0.0, 'SR': 1.0, 'SQ': 0.358, 'SP': 0.181, 'SW': 0.827, 'SV': 0.342, 'ST': 0.174, 'SK': 0.883, 'SI': 0.478, 'SH': 0.718, 'SN': 0.289, 'SM': 0.44, 'SL': 0.474, 'SC': 0.185, 'SA': 0.1, 'SG': 0.17, 'SF': 0.622, 'SE': 0.812, 'SD': 0.801, 'YI': 0.23, 'YH': 0.678, 'YK': 0.904, 'YM': 0.268, 'YL': 0.219, 'YN': 0.512, 'YA': 0.587, 'YC': 0.478, 'YE': 0.932, 'YD': 1.0, 'YG': 0.782, 'YF': 0.202, 'YY': 0.0, 'YQ': 0.404, 'YP': 0.444, 'YS': 0.612, 'YR': 0.995, 'YT': 0.557, 'YW': 0.244, 'YV': 0.328, 'LF': 0.139, 'LG': 0.596, 'LD': 0.944, 'LE': 0.892, 'LC': 0.296, 'LA': 0.405, 'LN': 0.452, 'LL': 0.0, 'LM': 0.062, 'LK': 0.893, 'LH': 0.653, 'LI': 0.013, 'LV': 0.133, 'LW': 0.341, 'LT': 0.397, 'LR': 1.0, 'LS': 0.443, 'LP': 0.309, 'LQ': 0.376, 'LY': 0.205, 'RT': 0.808, 'RV': 0.914, 'RW': 1.0, 'RP': 0.796, 'RQ': 0.668, 'RR': 0.0, 'RS': 0.86, 'RY': 0.859, 'RD': 0.305, 'RE': 0.225, 'RF': 0.977, 'RG': 0.928, 'RA': 0.919, 'RC': 0.905, 'RL': 0.92, 'RM': 0.908, 'RN': 0.69, 'RH': 0.498, 'RI': 0.929, 'RK': 0.141, 'VH': 0.649, 'VI': 0.135, 'EM': 0.83, 'EL': 0.854, 'EN': 0.599, 'EI': 0.86, 'EH': 0.406, 'EK': 0.143, 'EE': 0.0, 'ED': 0.133, 'EG': 0.779, 'EF': 0.932, 'EA': 0.79, 'EC': 0.788, 'VM': 0.12, 'EY': 0.837, 'VN': 0.38, 'ET': 0.682, 'EW': 1.0, 'EV': 0.824, 'EQ': 0.598, 'EP': 0.688, 'ES': 0.726, 'ER': 0.234, 'VP': 0.212, 'VQ': 0.339, 'VR': 1.0, 'VT': 0.305, 'VW': 0.472, 'KC': 0.871, 'KA': 0.889, 'KG': 0.9, 'KF': 0.957, 'KE': 0.149, 'KD': 0.279, 'KK': 0.0, 'KI': 0.899, 'KH': 0.438, 'KN': 0.667, 'KM': 0.871, 'KL': 0.892, 'KS': 0.825, 'KR': 0.154, 'KQ': 0.639, 'KP': 0.757, 'KW': 1.0, 'KV': 0.882, 'KT': 0.759, 'KY': 0.848, 'DN': 0.56, 'DL': 0.841, 'DM': 0.819, 'DK': 0.249, 'DH': 0.435, 'DI': 0.847, 'DF': 0.924, 'DG': 0.697, 'DD': 0.0, 'DE': 0.124, 'DC': 0.742, 'DA': 0.729, 'DY': 0.836, 'DV': 0.797, 'DW': 1.0, 'DT': 0.649, 'DR': 0.295, 'DS': 0.667, 'DP': 0.657, 'DQ': 0.584, 'QQ': 0.0, 'QP': 0.272, 'QS': 0.461, 'QR': 1.0, 'QT': 0.389, 'QW': 0.831, 'QV': 0.464, 'QY': 0.522, 'QA': 0.512, 'QC': 0.462, 'QE': 0.861, 'QD': 0.903, 'QG': 0.648, 'QF': 0.671, 'QI': 0.532, 'QH': 0.765, 'QK': 0.881, 'QM': 0.505, 'QL': 0.518, 'QN': 0.181, 'WG': 0.829, 'WF': 0.196, 'WE': 0.931, 'WD': 1.0, 'WC': 0.56, 'WA': 0.658, 'WN': 0.631, 'WM': 0.344, 'WL': 0.304, 'WK': 0.892, 'WI': 0.305, 'WH': 0.678, 'WW': 0.0, 'WV': 0.418, 'WT': 0.638, 'WS': 0.689, 'WR': 0.968, 'WQ': 0.538, 'WP': 0.555, 'WY': 0.204, 'PR': 1.0, 'PS': 0.196, 'PP': 0.0, 'PQ': 0.228, 'PV': 0.244, 'PW': 0.72, 'PT': 0.161, 'PY': 0.481, 'PC': 0.179, 'PA': 0.22, 'PF': 0.515, 'PG': 0.376, 'PD': 0.852, 'PE': 0.831, 'PK': 0.875, 'PH': 0.696, 'PI': 0.363, 'PN': 0.231, 'PL': 0.357, 'PM': 0.326, 'CK': 0.887, 'CI': 0.304, 'CH': 0.66, 'CN': 0.324, 'CM': 0.277, 'CL': 0.301, 'CC': 0.0, 'CA': 0.114, 'CG': 0.32, 'CF': 0.437, 'CE': 0.838, 'CD': 0.847, 'CY': 0.457, 'CS': 0.176, 'CR': 1.0, 'CQ': 0.341, 'CP': 0.157, 'CW': 0.639, 'CV': 0.167, 'CT': 0.233, 'IY': 0.213, 'VA': 0.275, 'VC': 0.165, 'VD': 0.9, 'VE': 0.867, 'VF': 0.269, 'VG': 0.471, 'IQ': 0.383, 'IP': 0.311, 'IS': 0.443, 'IR': 1.0, 'VL': 0.134, 'IT': 0.396, 'IW': 0.339, 'IV': 0.133, 'II': 0.0, 'IH': 0.652, 'IK': 0.892, 'VS': 0.322, 'IM': 0.057, 'IL': 0.013, 'VV': 0.0, 'IN': 0.457, 'IA': 0.403, 'VY': 0.31, 'IC': 0.296, 'IE': 0.891, 'ID': 0.942, 'IG': 0.592, 'IF': 0.134, 'HY': 0.821, 'HR': 0.697, 'HS': 0.865, 'HP': 0.777, 'HQ': 0.716, 'HV': 0.831, 'HW': 0.981, 'HT': 0.834, 'HK': 0.566, 'HH': 0.0, 'HI': 0.848, 'HN': 0.754, 'HL': 0.842, 'HM': 0.825, 'HC': 0.836, 'HA': 0.896, 'HF': 0.907, 'HG': 1.0, 'HD': 0.629, 'HE': 0.547, 'NH': 0.78, 'NI': 0.615, 'NK': 0.891, 'NL': 0.603, 'NM': 0.588, 'NN': 0.0, 'NA': 0.424, 'NC': 0.425, 'ND': 0.838, 'NE': 0.835, 'NF': 0.766, 'NG': 0.512, 'NY': 0.641, 'NP': 0.266, 'NQ': 0.175, 'NR': 1.0, 'NS': 0.361, 'NT': 0.368, 'NV': 0.503, 'NW': 0.945, 'TY': 0.596, 'TV': 0.345, 'TW': 0.816, 'TT': 0.0, 'TR': 1.0, 'TS': 0.185, 'TP': 0.159, 'TQ': 0.322, 'TN': 0.315, 'TL': 0.453, 'TM': 0.403, 'TK': 0.866, 'TH': 0.737, 'TI': 0.455, 'TF': 0.604, 'TG': 0.312, 'TD': 0.83, 'TE': 0.812, 'TC': 0.261, 'TA': 0.251, 'AA': 0.0, 'AC': 0.112, 'AE': 0.827, 'AD': 0.819, 'AG': 0.208, 'AF': 0.54, 'AI': 0.407, 'AH': 0.696, 'AK': 0.891, 'AM': 0.379, 'AL': 0.406, 'AN': 0.318, 'AQ': 0.372, 'AP': 0.191, 'AS': 0.094, 'AR': 1.0, 'AT': 0.22, 'AW': 0.739, 'AV': 0.273, 'AY': 0.552, 'VK': 0.889}
):
    """GetSequenceOrderCouplingNumber computes the d-th rank sequence order coupling number for a protein sequence, producing a single numeric descriptor used in protein encoding for molecular modeling tasks (for example, drug-target interaction prediction, protein–protein interaction prediction, and protein function prediction within the DeepPurpose framework). The function implements the standard sequence order coupling number: it sums the squared pairwise distances between amino acids separated by exactly d positions in the sequence and returns the result rounded to three decimal places. This descriptor captures local sequence-order interactions that are commonly used in pseudo-amino-acid composition and other sequence-based feature sets for QSAR/DTI modeling.
    
    Args:
        ProteinSequence (str): A protein primary sequence given as a string of single-letter amino acid codes (e.g., "MTEYK..."). This sequence is used as the source of residue pairs; each pair is formed by taking a residue at position i and the residue at position i + d for all valid i. Practical significance: the sequence provides the biological input whose local order interactions are quantified for downstream deep learning or statistical models in DeepPurpose.
        d (int): The gap (rank) between two amino acids whose pairwise distance is included in the d-th rank coupling number. d should be a positive integer (the code is written assuming d >= 1). Typical use: d = 1 captures immediate neighbor coupling (local interactions), larger d capture longer-range sequence-order correlations. Behavior and defaults: default is 1. If d is greater than or equal to the sequence length, no pairs are summed and the function returns 0.000. Passing d < 1 produces undefined or unintended behavior because negative or zero gaps are not meaningful in the intended biological interpretation.
        distancematrix (dict): A mapping from two-letter amino-acid pair keys (concatenated single-letter codes, e.g., "AG", "YW") to numeric distance values (floats or ints). This matrix encodes empirical or computed physicochemical distances between residue types and is used to look up the distance for each pair (temp1 + temp2) before squaring and summing. Practical significance: the chosen distance matrix determines what biochemical or structural relationship is being captured by the coupling number. Default: a precomputed dictionary (commonly named _Distance1 in the codebase) that maps all standard residue pairs to numeric distances. Behavior and failure modes: keys must match the exact concatenation of the characters in ProteinSequence (case-sensitive); missing keys will raise KeyError. Values are expected to be numeric; non-numeric values will raise TypeError during arithmetic.
    
    Returns:
        float: The d-th rank sequence order coupling number as a floating-point numeric value rounded to three decimal places. This is the sum over i of (distancematrix[ProteinSequence[i] + ProteinSequence[i + d]])^2 for i = 0..(len(ProteinSequence) - d - 1). If the sequence is shorter than or equal to d, the function returns 0.000. Side effects: none (pure computation), but the function will raise exceptions on invalid inputs (for example, KeyError when a residue-pair key is missing from distancematrix, TypeError if distancematrix values are not numeric, or IndexError-like unintended results if d is not a positive integer).
    """
    from DeepPurpose.pybiomed_helper import GetSequenceOrderCouplingNumber
    return GetSequenceOrderCouplingNumber(ProteinSequence, d, distancematrix)


################################################################################
# Source: DeepPurpose.pybiomed_helper.GetSequenceOrderCouplingNumberGrant
# File: DeepPurpose/pybiomed_helper.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def DeepPurpose_pybiomed_helper_GetSequenceOrderCouplingNumberGrant(
    ProteinSequence: str,
    maxlag: int = 30,
    distancematrix: dict = {'GW': 0.923, 'GV': 0.464, 'GT': 0.272, 'GS': 0.158, 'GR': 1.0, 'GQ': 0.467, 'GP': 0.323, 'GY': 0.728, 'GG': 0.0, 'GF': 0.727, 'GE': 0.807, 'GD': 0.776, 'GC': 0.312, 'GA': 0.206, 'GN': 0.381, 'GM': 0.557, 'GL': 0.591, 'GK': 0.894, 'GI': 0.592, 'GH': 0.769, 'ME': 0.879, 'MD': 0.932, 'MG': 0.569, 'MF': 0.182, 'MA': 0.383, 'MC': 0.276, 'MM': 0.0, 'ML': 0.062, 'MN': 0.447, 'MI': 0.058, 'MH': 0.648, 'MK': 0.884, 'MT': 0.358, 'MW': 0.391, 'MV': 0.12, 'MQ': 0.372, 'MP': 0.285, 'MS': 0.417, 'MR': 1.0, 'MY': 0.255, 'FP': 0.42, 'FQ': 0.459, 'FR': 1.0, 'FS': 0.548, 'FT': 0.499, 'FV': 0.252, 'FW': 0.207, 'FY': 0.179, 'FA': 0.508, 'FC': 0.405, 'FD': 0.977, 'FE': 0.918, 'FF': 0.0, 'FG': 0.69, 'FH': 0.663, 'FI': 0.128, 'FK': 0.903, 'FL': 0.131, 'FM': 0.169, 'FN': 0.541, 'SY': 0.615, 'SS': 0.0, 'SR': 1.0, 'SQ': 0.358, 'SP': 0.181, 'SW': 0.827, 'SV': 0.342, 'ST': 0.174, 'SK': 0.883, 'SI': 0.478, 'SH': 0.718, 'SN': 0.289, 'SM': 0.44, 'SL': 0.474, 'SC': 0.185, 'SA': 0.1, 'SG': 0.17, 'SF': 0.622, 'SE': 0.812, 'SD': 0.801, 'YI': 0.23, 'YH': 0.678, 'YK': 0.904, 'YM': 0.268, 'YL': 0.219, 'YN': 0.512, 'YA': 0.587, 'YC': 0.478, 'YE': 0.932, 'YD': 1.0, 'YG': 0.782, 'YF': 0.202, 'YY': 0.0, 'YQ': 0.404, 'YP': 0.444, 'YS': 0.612, 'YR': 0.995, 'YT': 0.557, 'YW': 0.244, 'YV': 0.328, 'LF': 0.139, 'LG': 0.596, 'LD': 0.944, 'LE': 0.892, 'LC': 0.296, 'LA': 0.405, 'LN': 0.452, 'LL': 0.0, 'LM': 0.062, 'LK': 0.893, 'LH': 0.653, 'LI': 0.013, 'LV': 0.133, 'LW': 0.341, 'LT': 0.397, 'LR': 1.0, 'LS': 0.443, 'LP': 0.309, 'LQ': 0.376, 'LY': 0.205, 'RT': 0.808, 'RV': 0.914, 'RW': 1.0, 'RP': 0.796, 'RQ': 0.668, 'RR': 0.0, 'RS': 0.86, 'RY': 0.859, 'RD': 0.305, 'RE': 0.225, 'RF': 0.977, 'RG': 0.928, 'RA': 0.919, 'RC': 0.905, 'RL': 0.92, 'RM': 0.908, 'RN': 0.69, 'RH': 0.498, 'RI': 0.929, 'RK': 0.141, 'VH': 0.649, 'VI': 0.135, 'EM': 0.83, 'EL': 0.854, 'EN': 0.599, 'EI': 0.86, 'EH': 0.406, 'EK': 0.143, 'EE': 0.0, 'ED': 0.133, 'EG': 0.779, 'EF': 0.932, 'EA': 0.79, 'EC': 0.788, 'VM': 0.12, 'EY': 0.837, 'VN': 0.38, 'ET': 0.682, 'EW': 1.0, 'EV': 0.824, 'EQ': 0.598, 'EP': 0.688, 'ES': 0.726, 'ER': 0.234, 'VP': 0.212, 'VQ': 0.339, 'VR': 1.0, 'VT': 0.305, 'VW': 0.472, 'KC': 0.871, 'KA': 0.889, 'KG': 0.9, 'KF': 0.957, 'KE': 0.149, 'KD': 0.279, 'KK': 0.0, 'KI': 0.899, 'KH': 0.438, 'KN': 0.667, 'KM': 0.871, 'KL': 0.892, 'KS': 0.825, 'KR': 0.154, 'KQ': 0.639, 'KP': 0.757, 'KW': 1.0, 'KV': 0.882, 'KT': 0.759, 'KY': 0.848, 'DN': 0.56, 'DL': 0.841, 'DM': 0.819, 'DK': 0.249, 'DH': 0.435, 'DI': 0.847, 'DF': 0.924, 'DG': 0.697, 'DD': 0.0, 'DE': 0.124, 'DC': 0.742, 'DA': 0.729, 'DY': 0.836, 'DV': 0.797, 'DW': 1.0, 'DT': 0.649, 'DR': 0.295, 'DS': 0.667, 'DP': 0.657, 'DQ': 0.584, 'QQ': 0.0, 'QP': 0.272, 'QS': 0.461, 'QR': 1.0, 'QT': 0.389, 'QW': 0.831, 'QV': 0.464, 'QY': 0.522, 'QA': 0.512, 'QC': 0.462, 'QE': 0.861, 'QD': 0.903, 'QG': 0.648, 'QF': 0.671, 'QI': 0.532, 'QH': 0.765, 'QK': 0.881, 'QM': 0.505, 'QL': 0.518, 'QN': 0.181, 'WG': 0.829, 'WF': 0.196, 'WE': 0.931, 'WD': 1.0, 'WC': 0.56, 'WA': 0.658, 'WN': 0.631, 'WM': 0.344, 'WL': 0.304, 'WK': 0.892, 'WI': 0.305, 'WH': 0.678, 'WW': 0.0, 'WV': 0.418, 'WT': 0.638, 'WS': 0.689, 'WR': 0.968, 'WQ': 0.538, 'WP': 0.555, 'WY': 0.204, 'PR': 1.0, 'PS': 0.196, 'PP': 0.0, 'PQ': 0.228, 'PV': 0.244, 'PW': 0.72, 'PT': 0.161, 'PY': 0.481, 'PC': 0.179, 'PA': 0.22, 'PF': 0.515, 'PG': 0.376, 'PD': 0.852, 'PE': 0.831, 'PK': 0.875, 'PH': 0.696, 'PI': 0.363, 'PN': 0.231, 'PL': 0.357, 'PM': 0.326, 'CK': 0.887, 'CI': 0.304, 'CH': 0.66, 'CN': 0.324, 'CM': 0.277, 'CL': 0.301, 'CC': 0.0, 'CA': 0.114, 'CG': 0.32, 'CF': 0.437, 'CE': 0.838, 'CD': 0.847, 'CY': 0.457, 'CS': 0.176, 'CR': 1.0, 'CQ': 0.341, 'CP': 0.157, 'CW': 0.639, 'CV': 0.167, 'CT': 0.233, 'IY': 0.213, 'VA': 0.275, 'VC': 0.165, 'VD': 0.9, 'VE': 0.867, 'VF': 0.269, 'VG': 0.471, 'IQ': 0.383, 'IP': 0.311, 'IS': 0.443, 'IR': 1.0, 'VL': 0.134, 'IT': 0.396, 'IW': 0.339, 'IV': 0.133, 'II': 0.0, 'IH': 0.652, 'IK': 0.892, 'VS': 0.322, 'IM': 0.057, 'IL': 0.013, 'VV': 0.0, 'IN': 0.457, 'IA': 0.403, 'VY': 0.31, 'IC': 0.296, 'IE': 0.891, 'ID': 0.942, 'IG': 0.592, 'IF': 0.134, 'HY': 0.821, 'HR': 0.697, 'HS': 0.865, 'HP': 0.777, 'HQ': 0.716, 'HV': 0.831, 'HW': 0.981, 'HT': 0.834, 'HK': 0.566, 'HH': 0.0, 'HI': 0.848, 'HN': 0.754, 'HL': 0.842, 'HM': 0.825, 'HC': 0.836, 'HA': 0.896, 'HF': 0.907, 'HG': 1.0, 'HD': 0.629, 'HE': 0.547, 'NH': 0.78, 'NI': 0.615, 'NK': 0.891, 'NL': 0.603, 'NM': 0.588, 'NN': 0.0, 'NA': 0.424, 'NC': 0.425, 'ND': 0.838, 'NE': 0.835, 'NF': 0.766, 'NG': 0.512, 'NY': 0.641, 'NP': 0.266, 'NQ': 0.175, 'NR': 1.0, 'NS': 0.361, 'NT': 0.368, 'NV': 0.503, 'NW': 0.945, 'TY': 0.596, 'TV': 0.345, 'TW': 0.816, 'TT': 0.0, 'TR': 1.0, 'TS': 0.185, 'TP': 0.159, 'TQ': 0.322, 'TN': 0.315, 'TL': 0.453, 'TM': 0.403, 'TK': 0.866, 'TH': 0.737, 'TI': 0.455, 'TF': 0.604, 'TG': 0.312, 'TD': 0.83, 'TE': 0.812, 'TC': 0.261, 'TA': 0.251, 'AA': 0.0, 'AC': 0.112, 'AE': 0.827, 'AD': 0.819, 'AG': 0.208, 'AF': 0.54, 'AI': 0.407, 'AH': 0.696, 'AK': 0.891, 'AM': 0.379, 'AL': 0.406, 'AN': 0.318, 'AQ': 0.372, 'AP': 0.191, 'AS': 0.094, 'AR': 1.0, 'AT': 0.22, 'AW': 0.739, 'AV': 0.273, 'AY': 0.552, 'VK': 0.889}
):
    """Compute sequence order coupling numbers (tau) for a protein sequence using the Grantham chemical distance matrix.
    
    This function iterates lag = 1 .. maxlag and, for each lag, calls GetSequenceOrderCouplingNumber to compute the sequence order coupling number based on pairwise Grantham distances between amino acids separated by that lag. The returned values are commonly used as numerical protein descriptors in cheminformatics and bioinformatics feature engineering (for example, as input features for the DeepPurpose models for drug–target interaction (DTI), protein–protein interaction (PPI), and protein function prediction described in the repository README). The default behavior uses a Grantham distance dictionary (distancematrix) and a default maximum lag of 30, producing keys named "taugrant1", "taugrant2", ..., "taugrantN" in the output dict.
    
    Args:
        ProteinSequence (str): A pure protein amino-acid sequence (single-letter codes). This input is the primary biological sequence on which the sequence order coupling numbers are computed. In the DeepPurpose pipeline this string typically comes from protein targets used in DTI, PPI, or protein-function datasets. The function expects standard amino-acid single-letter characters; non-standard characters or gaps may lead to incorrect results from the underlying coupling-number computation.
        maxlag (int): The maximum lag (positive integer) to compute sequence order coupling numbers for. For each integer lag L in 1..maxlag the function computes one coupling number and stores it under the key "taugrantL". Default is 30. If maxlag <= 0 the function will perform no iterations and return an empty dict. Callers should ensure that the protein sequence length is greater than maxlag (see failure modes below) because coupling numbers for larger lags require pairs separated by that distance.
        distancematrix (dict): A mapping representing the Grantham chemical distance matrix between amino-acid pairs, where keys are two-letter amino-acid pairs (e.g., 'AG', 'LA') and values are numeric distances. This matrix provides the pairwise chemical difference used by GetSequenceOrderCouplingNumber to compute each tau value. The DeepPurpose codebase supplies a default Grantham-based dictionary; callers may pass an alternative distance dict with the same key convention to customize the chemical-distance metric.
    
    Returns:
        dict: A dictionary mapping string keys "taugrant1", "taugrant2", ..., "taugrant{maxlag}" to numeric coupling values (floats or numbers) computed by GetSequenceOrderCouplingNumber for the given protein and distance matrix. These values are sequence-order descriptors used as features in downstream machine-learning models (e.g., DTI, PPI, protein-function models). If maxlag <= 0 the returned dict will be empty.
    
    Behavior, side effects, and failure modes:
        This function is pure and has no side effects (it only computes and returns a dict). It internally calls GetSequenceOrderCouplingNumber for each lag; therefore, correctness depends on that helper function and on the provided distancematrix containing appropriate keys for all amino-acid pairs referenced in ProteinSequence. The function does not itself validate every character in ProteinSequence; if ProteinSequence contains non-standard letters or the distancematrix lacks required pair keys, the underlying computations may raise KeyError or produce incorrect numeric results. Also, if the length of ProteinSequence is less than or equal to maxlag, there will not be sufficient residue pairs for some lags and the underlying computation may raise an error or be undefined; callers should ensure len(ProteinSequence) > maxlag for meaningful results.
    """
    from DeepPurpose.pybiomed_helper import GetSequenceOrderCouplingNumberGrant
    return GetSequenceOrderCouplingNumberGrant(ProteinSequence, maxlag, distancematrix)


################################################################################
# Source: DeepPurpose.pybiomed_helper.GetSequenceOrderCouplingNumberSW
# File: DeepPurpose/pybiomed_helper.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def DeepPurpose_pybiomed_helper_GetSequenceOrderCouplingNumberSW(
    ProteinSequence: str,
    maxlag: int = 30,
    distancematrix: dict = {'GW': 0.923, 'GV': 0.464, 'GT': 0.272, 'GS': 0.158, 'GR': 1.0, 'GQ': 0.467, 'GP': 0.323, 'GY': 0.728, 'GG': 0.0, 'GF': 0.727, 'GE': 0.807, 'GD': 0.776, 'GC': 0.312, 'GA': 0.206, 'GN': 0.381, 'GM': 0.557, 'GL': 0.591, 'GK': 0.894, 'GI': 0.592, 'GH': 0.769, 'ME': 0.879, 'MD': 0.932, 'MG': 0.569, 'MF': 0.182, 'MA': 0.383, 'MC': 0.276, 'MM': 0.0, 'ML': 0.062, 'MN': 0.447, 'MI': 0.058, 'MH': 0.648, 'MK': 0.884, 'MT': 0.358, 'MW': 0.391, 'MV': 0.12, 'MQ': 0.372, 'MP': 0.285, 'MS': 0.417, 'MR': 1.0, 'MY': 0.255, 'FP': 0.42, 'FQ': 0.459, 'FR': 1.0, 'FS': 0.548, 'FT': 0.499, 'FV': 0.252, 'FW': 0.207, 'FY': 0.179, 'FA': 0.508, 'FC': 0.405, 'FD': 0.977, 'FE': 0.918, 'FF': 0.0, 'FG': 0.69, 'FH': 0.663, 'FI': 0.128, 'FK': 0.903, 'FL': 0.131, 'FM': 0.169, 'FN': 0.541, 'SY': 0.615, 'SS': 0.0, 'SR': 1.0, 'SQ': 0.358, 'SP': 0.181, 'SW': 0.827, 'SV': 0.342, 'ST': 0.174, 'SK': 0.883, 'SI': 0.478, 'SH': 0.718, 'SN': 0.289, 'SM': 0.44, 'SL': 0.474, 'SC': 0.185, 'SA': 0.1, 'SG': 0.17, 'SF': 0.622, 'SE': 0.812, 'SD': 0.801, 'YI': 0.23, 'YH': 0.678, 'YK': 0.904, 'YM': 0.268, 'YL': 0.219, 'YN': 0.512, 'YA': 0.587, 'YC': 0.478, 'YE': 0.932, 'YD': 1.0, 'YG': 0.782, 'YF': 0.202, 'YY': 0.0, 'YQ': 0.404, 'YP': 0.444, 'YS': 0.612, 'YR': 0.995, 'YT': 0.557, 'YW': 0.244, 'YV': 0.328, 'LF': 0.139, 'LG': 0.596, 'LD': 0.944, 'LE': 0.892, 'LC': 0.296, 'LA': 0.405, 'LN': 0.452, 'LL': 0.0, 'LM': 0.062, 'LK': 0.893, 'LH': 0.653, 'LI': 0.013, 'LV': 0.133, 'LW': 0.341, 'LT': 0.397, 'LR': 1.0, 'LS': 0.443, 'LP': 0.309, 'LQ': 0.376, 'LY': 0.205, 'RT': 0.808, 'RV': 0.914, 'RW': 1.0, 'RP': 0.796, 'RQ': 0.668, 'RR': 0.0, 'RS': 0.86, 'RY': 0.859, 'RD': 0.305, 'RE': 0.225, 'RF': 0.977, 'RG': 0.928, 'RA': 0.919, 'RC': 0.905, 'RL': 0.92, 'RM': 0.908, 'RN': 0.69, 'RH': 0.498, 'RI': 0.929, 'RK': 0.141, 'VH': 0.649, 'VI': 0.135, 'EM': 0.83, 'EL': 0.854, 'EN': 0.599, 'EI': 0.86, 'EH': 0.406, 'EK': 0.143, 'EE': 0.0, 'ED': 0.133, 'EG': 0.779, 'EF': 0.932, 'EA': 0.79, 'EC': 0.788, 'VM': 0.12, 'EY': 0.837, 'VN': 0.38, 'ET': 0.682, 'EW': 1.0, 'EV': 0.824, 'EQ': 0.598, 'EP': 0.688, 'ES': 0.726, 'ER': 0.234, 'VP': 0.212, 'VQ': 0.339, 'VR': 1.0, 'VT': 0.305, 'VW': 0.472, 'KC': 0.871, 'KA': 0.889, 'KG': 0.9, 'KF': 0.957, 'KE': 0.149, 'KD': 0.279, 'KK': 0.0, 'KI': 0.899, 'KH': 0.438, 'KN': 0.667, 'KM': 0.871, 'KL': 0.892, 'KS': 0.825, 'KR': 0.154, 'KQ': 0.639, 'KP': 0.757, 'KW': 1.0, 'KV': 0.882, 'KT': 0.759, 'KY': 0.848, 'DN': 0.56, 'DL': 0.841, 'DM': 0.819, 'DK': 0.249, 'DH': 0.435, 'DI': 0.847, 'DF': 0.924, 'DG': 0.697, 'DD': 0.0, 'DE': 0.124, 'DC': 0.742, 'DA': 0.729, 'DY': 0.836, 'DV': 0.797, 'DW': 1.0, 'DT': 0.649, 'DR': 0.295, 'DS': 0.667, 'DP': 0.657, 'DQ': 0.584, 'QQ': 0.0, 'QP': 0.272, 'QS': 0.461, 'QR': 1.0, 'QT': 0.389, 'QW': 0.831, 'QV': 0.464, 'QY': 0.522, 'QA': 0.512, 'QC': 0.462, 'QE': 0.861, 'QD': 0.903, 'QG': 0.648, 'QF': 0.671, 'QI': 0.532, 'QH': 0.765, 'QK': 0.881, 'QM': 0.505, 'QL': 0.518, 'QN': 0.181, 'WG': 0.829, 'WF': 0.196, 'WE': 0.931, 'WD': 1.0, 'WC': 0.56, 'WA': 0.658, 'WN': 0.631, 'WM': 0.344, 'WL': 0.304, 'WK': 0.892, 'WI': 0.305, 'WH': 0.678, 'WW': 0.0, 'WV': 0.418, 'WT': 0.638, 'WS': 0.689, 'WR': 0.968, 'WQ': 0.538, 'WP': 0.555, 'WY': 0.204, 'PR': 1.0, 'PS': 0.196, 'PP': 0.0, 'PQ': 0.228, 'PV': 0.244, 'PW': 0.72, 'PT': 0.161, 'PY': 0.481, 'PC': 0.179, 'PA': 0.22, 'PF': 0.515, 'PG': 0.376, 'PD': 0.852, 'PE': 0.831, 'PK': 0.875, 'PH': 0.696, 'PI': 0.363, 'PN': 0.231, 'PL': 0.357, 'PM': 0.326, 'CK': 0.887, 'CI': 0.304, 'CH': 0.66, 'CN': 0.324, 'CM': 0.277, 'CL': 0.301, 'CC': 0.0, 'CA': 0.114, 'CG': 0.32, 'CF': 0.437, 'CE': 0.838, 'CD': 0.847, 'CY': 0.457, 'CS': 0.176, 'CR': 1.0, 'CQ': 0.341, 'CP': 0.157, 'CW': 0.639, 'CV': 0.167, 'CT': 0.233, 'IY': 0.213, 'VA': 0.275, 'VC': 0.165, 'VD': 0.9, 'VE': 0.867, 'VF': 0.269, 'VG': 0.471, 'IQ': 0.383, 'IP': 0.311, 'IS': 0.443, 'IR': 1.0, 'VL': 0.134, 'IT': 0.396, 'IW': 0.339, 'IV': 0.133, 'II': 0.0, 'IH': 0.652, 'IK': 0.892, 'VS': 0.322, 'IM': 0.057, 'IL': 0.013, 'VV': 0.0, 'IN': 0.457, 'IA': 0.403, 'VY': 0.31, 'IC': 0.296, 'IE': 0.891, 'ID': 0.942, 'IG': 0.592, 'IF': 0.134, 'HY': 0.821, 'HR': 0.697, 'HS': 0.865, 'HP': 0.777, 'HQ': 0.716, 'HV': 0.831, 'HW': 0.981, 'HT': 0.834, 'HK': 0.566, 'HH': 0.0, 'HI': 0.848, 'HN': 0.754, 'HL': 0.842, 'HM': 0.825, 'HC': 0.836, 'HA': 0.896, 'HF': 0.907, 'HG': 1.0, 'HD': 0.629, 'HE': 0.547, 'NH': 0.78, 'NI': 0.615, 'NK': 0.891, 'NL': 0.603, 'NM': 0.588, 'NN': 0.0, 'NA': 0.424, 'NC': 0.425, 'ND': 0.838, 'NE': 0.835, 'NF': 0.766, 'NG': 0.512, 'NY': 0.641, 'NP': 0.266, 'NQ': 0.175, 'NR': 1.0, 'NS': 0.361, 'NT': 0.368, 'NV': 0.503, 'NW': 0.945, 'TY': 0.596, 'TV': 0.345, 'TW': 0.816, 'TT': 0.0, 'TR': 1.0, 'TS': 0.185, 'TP': 0.159, 'TQ': 0.322, 'TN': 0.315, 'TL': 0.453, 'TM': 0.403, 'TK': 0.866, 'TH': 0.737, 'TI': 0.455, 'TF': 0.604, 'TG': 0.312, 'TD': 0.83, 'TE': 0.812, 'TC': 0.261, 'TA': 0.251, 'AA': 0.0, 'AC': 0.112, 'AE': 0.827, 'AD': 0.819, 'AG': 0.208, 'AF': 0.54, 'AI': 0.407, 'AH': 0.696, 'AK': 0.891, 'AM': 0.379, 'AL': 0.406, 'AN': 0.318, 'AQ': 0.372, 'AP': 0.191, 'AS': 0.094, 'AR': 1.0, 'AT': 0.22, 'AW': 0.739, 'AV': 0.273, 'AY': 0.552, 'VK': 0.889}
):
    """DeepPurpose.pybiomed_helper.GetSequenceOrderCouplingNumberSW computes sequence-order coupling numbers for a protein sequence using the Schneider–Wrede physicochemical distance matrix and returns them as a dictionary of numeric features. In the DeepPurpose toolkit these coupling numbers are used as handcrafted protein descriptors that capture short-range sequence-order relationships; they are commonly used as input features for downstream models such as drug–target interaction (DTI) predictors, protein property predictors, and other protein-encoding pipelines described in the README.
    
    This function iterates lag values from 1 to maxlag and for each lag calls GetSequenceOrderCouplingNumber to compute the Schneider–Wrede coupling number for that lag. The result is a mapping of key names "tausw1", "tausw2", ..., "tausw{maxlag}" to floating-point coupling values that summarize pairwise physicochemical relationships between residues separated by the given lag.
    
    Args:
        ProteinSequence (str): A single amino-acid sequence string (one-letter codes) representing the protein to be encoded. This sequence is treated as a pure protein sequence (no gaps or non-standard characters). Practical significance: this input is the primary biological sequence from which the Schneider–Wrede sequence-order coupling features are computed for use as model input in DeepPurpose workflows (DTI, protein property prediction, PPI, etc.). The sequence length should be greater than the requested maxlag for meaningful coupling values; otherwise the underlying GetSequenceOrderCouplingNumber call(s) may raise an error.
        maxlag (int): Maximum lag (positive integer, default 30) up to which sequence-order coupling numbers are computed. For each integer lag L in 1..maxlag the function computes one coupling number that captures the average physicochemical distance between residues separated by L positions. Practical significance: larger maxlag captures longer-range sequence-order information but increases feature dimensionality and computation cost; set this based on protein length and modeling needs. Default behavior: when omitted, maxlag is 30. Failure modes: non-integer or negative values will result in errors from the implementation or from called helper functions.
        distancematrix (dict): A dictionary implementing the Schneider–Wrede physicochemical distance matrix (default is the module constant _Distance1). Keys are two-letter residue pair strings (for example 'AR', 'GK', etc.) and values are numeric distances (floats). Practical significance: this matrix defines the physicochemical distance measure used to compute coupling numbers; supplying a different valid distance dict allows computing coupling numbers under alternative distance definitions. Failure modes: if required residue-pair keys are missing or values are non-numeric, the computation will fail or raise an exception.
    
    Returns:
        dict: A dictionary mapping string keys to numeric coupling values. Keys are named "tausw1" through "tausw{maxlag}" in ascending lag order; each value is the floating-point sequence-order coupling number computed for that lag using the provided distancematrix. Practical significance: the returned dict is a compact set of handcrafted features that can be concatenated with other encodings or fed directly into classical or deep learning models in DeepPurpose. Side effects: none (pure computation). Error behavior: if ProteinSequence is too short for a requested lag, if distancematrix is malformed, or if input types are invalid, the function will propagate exceptions raised by GetSequenceOrderCouplingNumber (typically ValueError or TypeError) originating from invalid sequence length, invalid characters, or missing distance entries.
    """
    from DeepPurpose.pybiomed_helper import GetSequenceOrderCouplingNumberSW
    return GetSequenceOrderCouplingNumberSW(ProteinSequence, maxlag, distancematrix)


################################################################################
# Source: DeepPurpose.pybiomed_helper.GetSequenceOrderCouplingNumberTotal
# File: DeepPurpose/pybiomed_helper.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def DeepPurpose_pybiomed_helper_GetSequenceOrderCouplingNumberTotal(
    ProteinSequence: str,
    maxlag: int = 30
):
    """GetSequenceOrderCouplingNumberTotal computes sequence-order coupling numbers for a protein sequence by aggregating two established coupling-number computations (Schneider–Wrede style and Grant style) for lags 1..maxlag. In the DeepPurpose toolkit this function produces numeric sequence-order descriptors used as protein encodings for downstream tasks such as drug–target interaction (DTI) prediction, protein–protein interaction (PPI) prediction, and protein function prediction; these descriptors capture correlations of physicochemical properties between residues separated by specific sequence distances (lags) and are intended as features for machine learning models (for example, as part of DeepPurpose model input pipelines).
    
    Args:
        ProteinSequence (str): A pure protein sequence string composed of standard amino acid single-letter codes (e.g., "ACDEFGHIK..."). This argument is the primary biological input whose residue order and composition determine the computed coupling numbers. The sequence should not contain non-amino-acid characters (such as digits, whitespace, or punctuation). The sequence length is required to be larger than maxlag for full results; if it is not, the behavior depends on the downstream helper functions and may raise an error or produce only a subset of expected coupling values.
        maxlag (int): The maximum lag (positive integer, default 30) for which to compute sequence-order coupling numbers. A lag value L means computing coupling statistics between residues separated by L positions in the linear sequence. The default of 30 is chosen to capture medium-range sequence-order correlations commonly used in protein descriptor sets; increasing maxlag will produce more features and increase computation time and memory usage proportionally.
    
    Returns:
        dict: A dictionary containing the aggregated sequence-order coupling numbers computed for lags 1 through maxlag. The dictionary keys are string identifiers for each coupling descriptor produced by the two underlying methods (GetSequenceOrderCouplingNumberSW and GetSequenceOrderCouplingNumberGrant) and the values are numeric coupling measurements (typically floats). These descriptors are ready to be used as feature inputs for DeepPurpose encoding pipelines. If the input sequence is shorter than maxlag or contains invalid characters, the returned dictionary may be incomplete or the underlying helper functions may raise an exception; there are no other side effects (the function does not modify external state or files).
    """
    from DeepPurpose.pybiomed_helper import GetSequenceOrderCouplingNumberTotal
    return GetSequenceOrderCouplingNumberTotal(ProteinSequence, maxlag)


################################################################################
# Source: DeepPurpose.pybiomed_helper.GetSequenceOrderCouplingNumberp
# File: DeepPurpose/pybiomed_helper.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def DeepPurpose_pybiomed_helper_GetSequenceOrderCouplingNumberp(
    ProteinSequence: str,
    maxlag: int = 30,
    distancematrix: dict = {}
):
    """DeepPurpose.pybiomed_helper.GetSequenceOrderCouplingNumberp computes sequence order coupling numbers tau1..tauN for a protein sequence using a user-supplied amino-acid property distance matrix. These coupling numbers quantify correlations between residues separated by specific sequence lags and are used as sequence-order features in DeepPurpose workflows for drug-target interaction (DTI), protein–protein interaction (PPI), protein function prediction, virtual screening, and other protein-encoding tasks.
    
    Args:
        ProteinSequence (str): A pure protein primary sequence string composed of amino-acid single-letter codes. This is the input sequence whose sequence-order coupling numbers will be computed. The function assumes standard amino-acid codes; nonstandard characters or whitespace may cause downstream lookup or indexing errors when accessing the distancematrix.
        maxlag (int): The maximum lag (positive integer) to compute. The function computes coupling numbers for all integer lags from 1 up to maxlag inclusive and returns maxlag entries named "tau1" through "tau{maxlag}". Default is 30. Practically, maxlag should be smaller than the sequence length (see failure modes) and should be chosen to capture the desired-range residue correlations for downstream machine-learning feature sets.
        distancematrix (dict): A dictionary containing pairwise property distances for amino-acid residue pairs used to define the coupling between two positions. In typical use this dict holds 400 numeric entries corresponding to pairwise distances for the 20 standard amino acids (20 x 20 = 400). The function uses these values to compute the property-based correlation at each lag by delegating to GetSequenceOrderCouplingNumber for each lag. If the distancematrix is empty or missing required pair keys, lookups during computation will raise KeyError; therefore supply a complete pairwise distance mapping when using property-based coupling.
    
    Returns:
        dict: A dictionary mapping string keys "tau1", "tau2", ..., "tau{maxlag}" to numeric sequence order coupling numbers computed for each lag. Each value is the result of calling GetSequenceOrderCouplingNumber(ProteinSequence, lag, distancematrix) for the corresponding lag. The returned dict is intended for use as engineered features for machine-learning models in DeepPurpose (for example, concatenation with other encodings for DTI/compound/property prediction).
    
    Behavior, defaults, and failure modes:
    - By default, maxlag is 30 and distancematrix defaults to an empty dict; in practice, you should provide a meaningful distancematrix to obtain numeric coupling features.
    - The function iterates lags from 1 to maxlag and calls the helper GetSequenceOrderCouplingNumber for each lag; there are no other side effects (no file I/O or global state modification).
    - The function assumes len(ProteinSequence) > maxlag so that residue pairs separated by the maximum lag exist. If this assumption is violated (len(ProteinSequence) <= maxlag), the computation for larger lags may attempt invalid indexing or produce meaningless results; callers should validate sequence length before calling.
    - If distancematrix does not contain the expected pairwise entries for the amino-acid codes present in ProteinSequence, KeyError exceptions (or other lookup errors) may be raised by the underlying helper function. If argument types differ from the declared types, TypeError or other runtime exceptions may occur.
    - This routine is a deterministic feature-extraction helper used to produce interpretable sequence-order descriptors for downstream DeepPurpose pipelines (training, repurposing, virtual screening, etc.).
    """
    from DeepPurpose.pybiomed_helper import GetSequenceOrderCouplingNumberp
    return GetSequenceOrderCouplingNumberp(ProteinSequence, maxlag, distancematrix)


################################################################################
# Source: DeepPurpose.pybiomed_helper.GetSpectrumDict
# File: DeepPurpose/pybiomed_helper.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def DeepPurpose_pybiomed_helper_GetSpectrumDict(proteinsequence: str):
    """GetSpectrumDict computes a fixed-length spectrum descriptor vector (returned as a Python dict) that counts occurrences of every possible amino-acid 3-mer (tri-peptide) in a given protein sequence. This function is used in DeepPurpose as a simple protein encoding / feature extractor that produces a 8000-dimensional representation (one value per 3-mer) which can be fed to downstream models for tasks such as drug-target interaction (DTI) prediction, protein-protein interaction (PPI) prediction, protein function prediction, or other compound/protein modeling workflows described in the DeepPurpose README.
    
    The function iterates over a canonical list of 3-mer strings produced by Getkmers() and counts matches of each 3-mer in the provided sequence using Python's re.findall. The resulting dict maps each 3-mer key (the same string format returned by Getkmers) to an integer count reflecting how many times that 3-mer was found in the input sequence.
    
    Args:
        proteinsequence (str): A pure protein sequence given as a Python string of amino-acid single-letter codes (e.g., "MKT..."). This is the sequence to be encoded into the 3-mer spectrum descriptor. The function expects a string; if a non-string is provided, Python's re.findall will raise a TypeError. The function does not perform validation or normalization of characters beyond what the underlying regex matching does; characters not matching any 3-mer simply yield zero counts for those 3-mers.
    
    Returns:
        dict: A dictionary whose keys are the 3-mer strings produced by Getkmers() and whose values are integers equal to len(re.findall(kmer, proteinsequence)) for each kmer. The dict contains 8000 entries (one per possible 3-mer over the standard 20 amino acids) when Getkmers() returns the full 3-mer set; each value is the count of non-overlapping matches found by Python's re.findall for that 3-mer in the input sequence. This fixed-length mapping provides a deterministic, order-independent feature vector useful for downstream machine learning models in DeepPurpose.
    
    Behavior, side effects, defaults, and failure modes:
        - Behavior: The function performs no in-place modification of the input string and has no external side effects; it constructs and returns a new dict. It relies on a callable Getkmers() in the same module or namespace to supply the list of 3-mer strings; each k-mer is treated as a pattern passed to re.findall.
        - Matching semantics: Counts are produced by Python's re.findall for each pattern; by Python regex semantics, this yields non-overlapping matches unless the k-mer patterns returned by Getkmers() are written using regex constructs to enable overlapping matches.
        - Dependencies: Requires Getkmers() to be defined and import/re module available in the runtime. If Getkmers is undefined, a NameError will be raised. If Getkmers returns a list containing non-string values, behavior may be unpredictable and may raise TypeError.
        - Input constraints: The function assumes proteinsequence is a string of amino-acid single-letter codes. Unexpected characters will not crash the function but will typically result in zero counts for k-mers that do not match those characters.
        - Performance: Time complexity is proportional to (number of k-mers) × (cost of regex search on the sequence). For the canonical 8000 3-mers, this is substantial for very long sequences and may be a performance bottleneck in large-scale or high-throughput pipelines; callers should consider this when processing many or very long sequences.
        - Use in DeepPurpose: The returned dict is a deterministic, fixed-size representation compatible with DeepPurpose pipelines that accept k-mer / spectrum-style encodings for protein features in DTI, PPI, protein function prediction, virtual screening, and related tasks.
    """
    from DeepPurpose.pybiomed_helper import GetSpectrumDict
    return GetSpectrumDict(proteinsequence)


################################################################################
# Source: DeepPurpose.pybiomed_helper.NormalizeEachAAP
# File: DeepPurpose/pybiomed_helper.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def DeepPurpose_pybiomed_helper_NormalizeEachAAP(AAP: dict):
    """DeepPurpose.pybiomed_helper.NormalizeEachAAP centralizes and standardizes amino acid property indices for the 20 standard amino acids so they are on a common scale for downstream encoding and machine learning in the DeepPurpose toolkit (for example, protein encoding used in drug–target interaction, protein function prediction, and related tasks).
    
    Args:
        AAP (dict): A mapping containing the properties of 20 amino acids. Keys are expected to identify each of the 20 standard amino acids (for example, single-letter codes or another consistent identifier); values are numeric property values (int or float) for each amino acid. This dict is used as the source of raw amino-acid-specific indices that will be centralized and standardized across the 20 entries.
    
    Returns:
        dict: A dict with the same keys as the input AAP. Each value is the standardized property (float) computed by subtracting the mean of AAP.values() and dividing by the standard deviation of AAP.values(). The standard deviation is computed with ddof=0 (population standard deviation) via the module's internal _std function. The returned dict is intended for use as normalized amino-acid feature vectors when constructing protein encodings for DeepPurpose models.
    
    Behavior and practical significance:
        This function performs two statistical transformations commonly required before feeding biochemical features into machine learning models: centralization (subtracting the mean) and standardization (dividing by the standard deviation). Doing so places all amino-acid properties on a comparable scale, which improves numerical stability and training behavior of downstream PyTorch models in DeepPurpose (e.g., for DTI regression/classification, protein function prediction, or protein encodings used in repurposing and virtual screening). The mean and standard deviation are computed over the entire set of input AAP.values(); each amino-acid value j is transformed to (j - mean) / std.
    
    Side effects, defaults, and failure modes:
        The function expects exactly 20 entries in the input dict (one per standard amino acid). If len(AAP.values()) != 20, the function prints the message "You can not input the correct number of properities of Amino acids!" and does not perform the normalization. Because the implementation only defines and returns the normalized Result in the branch taken when there are exactly 20 entries, providing a dict with a number of entries other than 20 will lead to Result being undefined and the function raising an exception (e.g., UnboundLocalError or NameError) at return time. The function relies on the module-level helper functions _mean and _std; these must be available and behave as expected (with _std using ddof=0). Inputs should therefore be numeric and finite; non-numeric or NaN values will propagate through the arithmetic and may produce NaN or raise errors in the mean/std computations.
    """
    from DeepPurpose.pybiomed_helper import NormalizeEachAAP
    return NormalizeEachAAP(AAP)


################################################################################
# Source: DeepPurpose.utils.prauc_curve
# File: DeepPurpose/utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def DeepPurpose_utils_prauc_curve(
    y_pred: list,
    y_label: list,
    figure_file: str,
    method_name: str
):
    """DeepPurpose.utils.prauc_curve generates and saves a precision-recall (PR) curve plot for binary classification predictions produced by DeepPurpose models (e.g., drug–target interaction scoring, virtual screening, or repurposing pipelines). The function computes the precision and recall pairs from continuous prediction scores and binary ground-truth labels, annotates the plot with the average precision (area under the precision-recall curve), and saves the resulting figure to disk. In the DeepPurpose context this is used to evaluate binary DTI/DDI/PPI predictions or repurposing/virtual-screening scores where class imbalance makes PR curves more informative than ROC curves. The implementation uses sklearn.metrics.precision_recall_curve and average_precision_score and matplotlib for plotting; it does not perform thresholding or compute a no-skill baseline.
    
    Args:
        y_pred (list): Predicted scores for the positive class for each example. Must be a list of length n containing numeric scores (typically in the range (0, 1) representing probabilities or relative confidence) where larger values indicate higher confidence in the positive class. These scores are paired elementwise with y_label and are the values used to compute precision and recall at all thresholds.
        y_label (list): Ground-truth binary labels for each example. Must be a list of length n containing 0/1 values (0 = negative class, 1 = positive class). The order must correspond exactly to y_pred so that y_pred[i] is the prediction for the instance whose true label is y_label[i].
        figure_file (str): Filesystem path (including filename and extension, e.g., "results/pr_curve.png") where the generated PR curve figure will be saved. The function calls matplotlib.pyplot.savefig(figure_file). If the file already exists it will be overwritten if the running process has write permission; if the path is invalid or the process lacks permission, an exception (for example, an IOError/OSError) can be raised by matplotlib.
        method_name (str): Short label/name for the prediction method or model to display in the plot legend. The string is used to build the legend entry and is combined with the computed average precision to produce a label of the form "<method_name> (area = X.XX)".
    
    Behavior and side effects:
        The function computes precision and recall arrays from y_label and y_pred using sklearn.metrics.precision_recall_curve, computes average precision via sklearn.metrics.average_precision_score, and then draws a PR curve using matplotlib.pyplot.plot with line width 2. The plot is labeled with "Recall" (x-axis) and "Precision" (y-axis) using font size 14, given the title "Precision Recall Curve", and a legend showing the method_name and the average precision rounded to two decimal places. The figure is saved to the path provided in figure_file via plt.savefig(figure_file). The function does not close or clear the matplotlib figure after saving, so the current matplotlib state (open figure) remains and may affect subsequent plotting unless the caller clears or closes it. The function does not return any metrics besides saving the plot.
    
    Failure modes and constraints:
        - y_pred and y_label must be Python lists of equal length; mismatched lengths will cause sklearn to raise a ValueError.
        - y_label must contain only binary values (0 or 1); non-binary labels will cause incorrect computations or sklearn errors.
        - y_pred should contain numeric scores; non-numeric entries will raise errors.
        - The function assumes sklearn and matplotlib are installed and importable; ImportError will be raised otherwise.
        - Saving the figure may fail due to invalid path, missing directories, or filesystem permission errors; such I/O errors propagate from matplotlib.
        - The function does not compute or plot a no-skill baseline, F1 threshold markers, or confidence intervals; if these are required they must be added by the caller or an external utility.
    
    Reference:
        Implementation follows standard PR-curve computation patterns (see sklearn.metrics.precision_recall_curve) and is consistent with evaluation practices for imbalanced binary prediction problems in drug discovery and virtual screening.
    
    Returns:
        None: The function has no return value. Its primary side effect is producing and saving a PNG/SVG/etc. file at the given figure_file path that visualizes the precision-recall curve and average precision for the provided predictions and labels.
    """
    from DeepPurpose.utils import prauc_curve
    return prauc_curve(y_pred, y_label, figure_file, method_name)


################################################################################
# Source: DeepPurpose.utils.roc_curve
# File: DeepPurpose/utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def DeepPurpose_utils_roc_curve(
    y_pred: list,
    y_label: list,
    figure_file: str,
    method_name: str
):
    """Compute and save a Receiver Operating Characteristic (ROC) curve and its area under the curve (AUC) for binary prediction scores.
    
    This function takes predicted scores and binary ground-truth labels from a binary classification task common in DeepPurpose workflows (for example drug-target interaction (DTI) prediction, drug property screening, drug-drug interaction (DDI) prediction, protein-protein interaction (PPI) prediction, or virtual screening/repurposing). It computes the false positive rate (FPR) and true positive rate (TPR) using scikit-learn's roc_curve and auc utilities, plots the ROC curve with matplotlib, annotates the plot with the provided method_name and the computed AUC (formatted to two decimal places), adds a diagonal chance line, and saves the plotted figure to disk. The produced plot is useful for evaluating binary classifiers that output continuous prediction scores (probabilities or confidence scores) and for comparing different models or encodings in DeepPurpose experiments.
    
    Args:
        y_pred (list): Predicted scores for each instance. Each element should be a numeric score typically in the open interval (0, 1) representing predicted probability or confidence that the instance belongs to the positive class. In DeepPurpose use cases this is commonly the predicted binding/interaction probability or affinity-normalized score for a drug-target pair or other binary assay output.
        y_label (list): Ground-truth binary labels for each instance. Each element must be 0 or 1 where 1 indicates the positive class (e.g., observed interaction or active compound) and 0 indicates the negative class. The length of y_label must equal the length of y_pred and labels should correspond elementwise to y_pred in the same order.
        figure_file (str): Filesystem path (including filename and extension) where the ROC figure will be saved. This function calls matplotlib.pyplot.savefig(figure_file) and will overwrite an existing file at that path without prompting. Provide a writable path; common extensions supported by matplotlib (e.g., .png, .pdf, .svg) are acceptable.
        method_name (str): Short descriptive name for the model or method whose ROC is being plotted. This string is used in the figure legend and is combined with the computed AUC value (displayed as "method_name (area = 0.XX)") so that saved figures are self-descriptive when comparing models or encodings (for example "CNN+Transformer" or a pretrained model name).
    
    Behavior and side effects:
        This function converts the input lists to numpy arrays, computes FPR and TPR via sklearn.metrics.roc_curve, computes AUC via sklearn.metrics.auc, plots the ROC curve with a fixed line width and fontsize, draws a dashed diagonal reference line at y=x, sets axis limits ([0.0, 1.0] for x and [0.0, 1.05] for y), places the legend in the lower-right, and saves the final figure to the path provided by figure_file. The function produces no return value; its primary effect is writing the figure file to disk.
    
    Defaults and plotting details:
        The plot uses a line width of 2 for both the ROC curve and reference line and sets the plot title to "Receiver Operating Characteristic Curve" and axis labels to "False Positive Rate" and "True Positive Rate" with fontsize 14. The AUC displayed in the legend is formatted to two decimal places.
    
    Failure modes and exceptions:
        If y_pred and y_label lengths differ, scikit-learn's roc_curve will raise a ValueError; ensure they are the same length and aligned. If y_label contains values other than 0 or 1 or is not binary, roc_curve may raise an error or produce meaningless results. If y_pred values are outside a reasonable numeric range (e.g., not finite numbers), auc computation or plotting may fail. If matplotlib or scikit-learn are not available or figure_file points to an unwritable location (e.g., permission denied, nonexistent directory), an IOError or OSError may be raised when saving. This function does not perform extensive input validation beyond converting inputs to numpy arrays; callers should validate inputs beforehand in automated pipelines.
    
    Returns:
        None: The function does not return a Python value. Its observable effect is to create or overwrite the file at figure_file containing the plotted ROC curve and annotated AUC for the supplied predictions and labels.
    """
    from DeepPurpose.utils import roc_curve
    return roc_curve(y_pred, y_label, figure_file, method_name)


################################################################################
# Source: DeepPurpose.utils.smiles2mpnnfeature
# File: DeepPurpose/utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def DeepPurpose_utils_smiles2mpnnfeature(smiles: str):
    """DeepPurpose.utils.smiles2mpnnfeature converts a SMILES string into the set of padded torch tensors required as input features for the message-passing neural network (MPNN) encoder used across DeepPurpose (e.g., data_process -> mpnn_collate_func -> mpnn_feature_collate_func -> encoders.MPNN.forward). It parses the molecule, computes per-atom and per-bond feature vectors, builds directed-bond and bond-neighbor adjacency tensors, pads these arrays to the library's fixed maximum sizes (using MAX_ATOM, MAX_BOND, MAX_NB), and returns them together with a shape tensor that records the original (un-padded) atom and bond counts. This function is used in drug/compound encoding for tasks such as drug-target interaction prediction, virtual screening, and other molecular modeling workflows in DeepPurpose.
    
    Args:
        smiles (str): A single-molecule SMILES string. This is the canonical input representation for this utility: a text string encoding the molecule to be converted into MPNN features. The function uses get_mol(smiles) (RDKit-backed in this repository) to parse the SMILES into a molecular object, then extracts atomic and bond-level descriptors via atom_features(...) and bond_features(...). In the DeepPurpose pipeline, this function is applied element-wise to drug SMILES during data preprocessing (data_process) to produce tensors that downstream MPNN encoders consume.
    
    Returns:
        list: A list of five torch.Tensor objects (all converted to float) in the exact order returned by the function:
            fatoms: torch.Tensor. The atom feature matrix after padding. Each row corresponds to an atom feature vector computed by atom_features(atom). After internal zero-padding the returned fatoms has been extended to the library's MAX_ATOM rows; the original (un-padded) number of atoms is returned separately in shape_tensor. The feature dimensionality equals the per-atom feature length determined at runtime (in code this is fatoms.shape[1]; in the library's parsing-failure default this is 39).
            fbonds: torch.Tensor. The directed bond feature matrix after padding. For each directed bond (both directions for each chemical bond) the function concatenates the source-atom feature vector with bond_features(bond) to form each bond feature row. A dummy padding row is inserted at index 0 (consistent with the all_bonds list initialization) and the matrix is padded up to MAX_BOND rows. The returned fbonds has the runtime per-bond feature dimensionality (in the parsing-failure default this is 50).
            agraph: torch.Tensor. The atom-to-incoming-bond index tensor (padded). For each atom index a, agraph[a, i] lists the bond index (into fbonds/all_bonds) of the i-th incoming directed bond to atom a. This tensor is padded across the second dimension to MAX_NB. After padding it has shape (MAX_ATOM, MAX_NB) when MAX_ATOM padding is applied.
            bgraph: torch.Tensor. The bond-to-neighboring-bond index tensor (padded). For each directed bond index b1, bgraph[b1, i] lists the index of a neighboring bond b2 (incoming to the source atom of b1) excluding the bond that points back to the bond target (this supports message-passing updates that exclude immediate reverse messages). The second dimension is padded to MAX_NB. After padding it has shape (MAX_BOND, MAX_NB) when MAX_BOND padding is applied.
            shape_tensor: torch.Tensor. A 1x2 tensor holding the original, un-padded integer counts [Natom, Nbond] where Natom is the number of atoms and Nbond is the total number of directed bonds (including the dummy at index 0 in the library convention). This tensor lets downstream code know the true sizes before padding.
    
    Behavior, side effects, defaults, and failure modes:
        - Molecule parsing: The function calls get_mol(smiles) to obtain an RDKit molecule object; atom and bond features are then derived via atom_features(atom) and bond_features(bond) as implemented in the repository. If get_mol or feature extraction fails (e.g., invalid SMILES or RDKit parse error), the function catches the exception, prints the message "Molecules not found and change to zero vectors..", and proceeds to return tensors initialized to small zero-shaped defaults (fatoms zero tensor with shape (0,39), fbonds zero tensor with shape (0,50), agraph and bgraph zero tensors with second dimension equal to MAX_NB as in the code). These default shapes are then padded to MAX_ATOM / MAX_BOND by the same padding logic, so downstream code still receives tensors of the expected padded shapes.
        - Padding and fixed sizes: The function pads fatoms, fbonds, agraph, and bgraph to fixed sizes determined by the module-level constants MAX_ATOM, MAX_BOND, and MAX_NB. If the parsed molecule requires more atoms or bonds than those maxima permit, the function checks atoms_completion_num and bonds_completion_num and will raise an Exception instructing the user to increase MAX_ATOM (and analogously MAX_BOND) in the utils module and reinstall (this is the explicit failure mode when the molecule is larger than the library's configured capacity).
        - Indexing convention: The implementation constructs all_bonds with an initial dummy entry [(-1, -1)] so valid directed bond indices begin at 1; agraph and bgraph indices reference this convention. Downstream components (MPNN encoders) expect this indexing and padding convention.
        - Computational dependencies and environment: The function depends on torch and the repository's RDKit-backed get_mol, atom_features, bond_features functions, and the module-level constants ATOM_FDIM, BOND_FDIM, MAX_ATOM, MAX_BOND, and MAX_NB. If these are not available or misconfigured, the function may raise import or attribute errors. The function also performs tensor concatenation and stacking which may raise runtime errors if feature extraction returns inconsistent shapes.
        - Determinism and mutability: The function does not mutate global state; it returns new tensors. It prints a message on parse failure as a side effect.
    
    Practical significance in DeepPurpose:
        This utility bridges chemical string representations (SMILES) and the MPNN encoder used in DeepPurpose for tasks such as DTI prediction, virtual screening, and compound property prediction. The returned padded tensors are the exact inputs expected by the library's MPNN implementations and downstream data loaders/collate functions; correct usage of this function ensures that molecular batches are assembled with consistent shapes for GPU/CPU training and inference.
    """
    from DeepPurpose.utils import smiles2mpnnfeature
    return smiles2mpnnfeature(smiles)


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
