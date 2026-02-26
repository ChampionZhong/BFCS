"""
Regenerated Google-style docstrings for module 'useful_rdkit_utils'.
README source: others/readme/useful_rdkit_utils/README.md
Generated at: 2025-12-02T00:54:49.752185Z

Total functions: 8
"""


################################################################################
# Source: useful_rdkit_utils.ring_systems.create_ring_dictionary
# File: useful_rdkit_utils/ring_systems.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def useful_rdkit_utils_ring_systems_create_ring_dictionary(
    input_smiles: str,
    output_csv: str
):
    """useful_rdkit_utils.ring_systems.create_ring_dictionary: Read a whitespace-separated SMILES file, extract ring systems from each molecule using RDKit utilities, count frequency of each unique ring system (represented by an InChIKey), and write a CSV listing a representative SMILES, the InChI identifier, and the occurrence count for each ring system.
    
    Args:
        input_smiles (str): Path to the input SMILES file. The function expects a whitespace-separated text file where the first column is a SMILES string and the second column is a name/identifier. The file is read with pandas.read_csv(input_smiles, sep=" ", names=["SMILES", "Name"]). If the file cannot be read (for example, it does not exist or is not accessible), pandas will raise the corresponding IO/FileNotFoundError or ParserError. Each SMILES string is passed to RDKit.Chem.MolFromSmiles; entries that fail to parse (MolFromSmiles returns None) are silently skipped.
        output_csv (str): Path where the output CSV will be written. The function writes a CSV with columns in the following order: SMILES, InChI, Count (no index column). If the target path is not writable, the underlying pandas I/O will raise an appropriate exception (for example, PermissionError or an IO error).
    
    Returns:
        None: This function does not return a value. Instead it has the side effect of writing a CSV file at output_csv. The CSV contains one row per unique ring system found across all input molecules. For each unique ring system the CSV records a representative SMILES string (taken from the last encountered molecule that produced that ring system), the InChI identifier string produced by RDKit.Chem.MolToInchiKey (the code stores these identifiers under the column/index name "InChI"), and the frequency Count of how many input molecules contained that ring system. The output ordering is by descending Count because the implementation uses pandas.Series.value_counts().
    
    Behavior and implementation details:
        - This function is part of the useful_rdkit_utils collection of RDKit helper functions and is intended for cheminformatics tasks such as analyzing ring system prevalence across a dataset of molecules (as described in the package README and demo notebooks).
        - A RingSystemFinder instance is created and used to extract ring systems from each parsed RDKit molecule via ring_system_finder.find_ring_systems(mol, as_mols=True). The returned ring system fragments (RDKit Mol objects) are converted to InChI identifiers with Chem.MolToInchiKey and to SMILES with Chem.MolToSmiles.
        - The function accumulates all InChI identifiers across the dataset and counts how many times each appears. It also builds a mapping from each InChI identifier to a SMILES string; because a Python dict is used and the mapping is updated for each occurrence, the final SMILES stored for an InChI will be the last SMILES observed in the input that produced that InChI.
        - A progress bar is displayed during iteration using tqdm over the SMILES column (tqdm must be available in the environment for the progress bar to appear).
        - Input molecules that cannot be parsed by RDKit (MolFromSmiles returns None) are ignored; no exception is raised for invalid SMILES entries by this function.
        - The function relies on RDKit (Chem.MolFromSmiles, Chem.MolToInchiKey, Chem.MolToSmiles) and pandas; if these libraries are not available or if RingSystemFinder is not defined/imported in the runtime environment, import errors or NameError will be raised before or during execution.
    
    Failure modes and edge cases:
        - Missing or malformed input file: pandas.read_csv will raise FileNotFoundError/IOError/ParserError as appropriate.
        - Non-parsable SMILES: those lines are skipped silently; they do not contribute to the output counts or mapping.
        - If RingSystemFinder.find_ring_systems returns []) for a molecule, that molecule contributes no ring systems.
        - Multiple different SMILES that map to the same InChIKey will be collapsed to a single output row with Count equal to the total occurrences; the representative SMILES in the output is the last one encountered in the input sequence for that InChIKey.
        - Writing the output CSV may raise IO-related exceptions if the destination is not writable.
        - The function does not validate that the SMILES file uses exactly two columns beyond how it is read (sep=" " and names=["SMILES", "Name"]); users must ensure their file matches this format or pre-process it accordingly.
    """
    from useful_rdkit_utils.ring_systems import create_ring_dictionary
    return create_ring_dictionary(input_smiles, output_csv)


################################################################################
# Source: useful_rdkit_utils.ring_systems.get_min_ring_frequency
# File: useful_rdkit_utils/ring_systems.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def useful_rdkit_utils_ring_systems_get_min_ring_frequency(ring_list: list):
    """useful_rdkit_utils.ring_systems.get_min_ring_frequency returns the ring identifier and its minimum observed frequency from a ring-frequency list produced by RingSystemLookup.process_smiles, used in cheminformatics workflows that analyze ring systems with RDKit.
    
    Args:
        ring_list (list): A list produced by RingSystemLookup.process_smiles containing ring frequency records. Each element is expected to be a two-element sequence where the first element is a ring identifier (typically a string) and the second element is the observed frequency (typically an integer). This function will sort this list in-place by the second element (frequency) and therefore mutates the provided list. Provide an empty list for acyclic molecules (no ring systems); in that case the function returns the sentinel ["", -1].
    
    Returns:
        list: A two-element list [ring_identifier, minimum_frequency]. If ring_list is non-empty, this is the element with the smallest frequency (the element at index 0 after in-place sort). If ring_list is empty (e.g., acyclic molecules), returns the sentinel ["", -1] where the empty string denotes no ring and -1 denotes the absence of a valid frequency.
    
    Behavior and side effects:
        The function sorts ring_list in-place using the frequency value (second element of each record) as the sort key, so the original ordering of ring_list is lost. The returned value is either the first element of the (now sorted) ring_list or the sentinel ["", -1] when ring_list is empty. No copies of the input are made.
    
    Failure modes and errors:
        If ring_list is not a list or does not support in-place sorting, an AttributeError or TypeError may be raised. If elements of ring_list are not two-element sequences with a second element that can be compared (for sorting) or accessed, the function may raise IndexError, TypeError, or ValueError during sorting or key extraction. Ensure that ring_list conforms to the expected structure from RingSystemLookup.process_smiles to avoid these errors.
    
    Domain significance:
        In the useful_rdkit_utils package (a collection of RDKit helper functions), this function is used to identify the least-common ring system observed across a set of molecules processed by RingSystemLookup.process_smiles. The returned pair can be used in downstream filtering, reporting, or statistical analyses of ring system frequency within chemical datasets.
    """
    from useful_rdkit_utils.ring_systems import get_min_ring_frequency
    return get_min_ring_frequency(ring_list)


################################################################################
# Source: useful_rdkit_utils.seaborn_utils.set_sns_size
# File: useful_rdkit_utils/seaborn_utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def useful_rdkit_utils_seaborn_utils_set_sns_size(width: float, height: float):
    """useful_rdkit_utils.seaborn_utils.set_sns_size sets the default seaborn/matplotlib figure size used by subsequent plots in this useful_rdkit_utils package (for example in the demos and Jupyter notebooks included with the repository). The function calls seaborn.set(rc={'figure.figsize': (width, height)}) to update the global plotting rc parameters so that all following seaborn/matplotlib figures use the specified width and height in inches unless overridden.
    
    Args:
        width (float): Width of the figure in inches. In the context of this RDKit utilities package and its demos, this controls the horizontal size of plots produced by seaborn/matplotlib after this call. The value is passed directly to seaborn.set as the first element of the tuple assigned to the 'figure.figsize' rc parameter.
        height (float): Height of the figure in inches. In the context of this RDKit utilities package and its demos, this controls the vertical size of plots produced by seaborn/matplotlib after this call. The value is passed directly to seaborn.set as the second element of the tuple assigned to the 'figure.figsize' rc parameter.
    
    Returns:
        None: This function does not return a value. Its effect is to modify global plotting configuration by calling seaborn.set(rc={'figure.figsize': (width, height)}). Side effects: the seaborn/matplotlib rc parameters are changed globally for the running Python session and will affect all subsequent plots (in notebooks, scripts, and other functions that rely on matplotlib/seaborn) until they are changed again. Failure modes: a runtime ImportError will be raised if seaborn is not available in the environment; seaborn/matplotlib may raise TypeError or ValueError if the provided width or height cannot be interpreted as numeric values acceptable to matplotlib. The function performs no additional validation or return value.
    """
    from useful_rdkit_utils.seaborn_utils import set_sns_size
    return set_sns_size(width, height)


################################################################################
# Source: useful_rdkit_utils.useful_rdkit_utils.rd_set_image_size
# File: useful_rdkit_utils/useful_rdkit_utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def useful_rdkit_utils_useful_rdkit_utils_rd_set_image_size(x: int, y: int):
    """Set the image size used by RDKit's IPythonConsole for molecule structure rendering.
    
    This function configures the global rendering size used by RDKit's IPythonConsole (used in Jupyter notebooks and interactive IPython sessions) by assigning a two-element tuple (x, y) to IPythonConsole.molSize. In the context of this repository (a collection of useful RDKit utilities and demo notebooks), rd_set_image_size is used to control how large 2D/3D molecular depictions appear when RDKit renders molecules inline. The change is global for the running Python process and affects all subsequent RDKit IPythonConsole renderings until overridden.
    
    Args:
        x (int): X dimension for rendered structures. This integer is the first element of the tuple assigned to IPythonConsole.molSize and defines the horizontal image size used by RDKit's interactive renderer. In practice this controls the width of inline molecule depictions produced after calling this function.
        y (int): Y dimension for rendered structures. This integer is the second element of the tuple assigned to IPythonConsole.molSize and defines the vertical image size used by RDKit's interactive renderer. In practice this controls the height of inline molecule depictions produced after calling this function.
    
    Returns:
        None: This function does not return a value. Its purpose is to produce a side effect: it mutates rdkit.Chem.Draw.IPythonConsole.molSize in-place for the current Python process so that subsequent molecule renderings use the specified (x, y) dimensions.
    
    Behavior and failure modes:
        - The function performs an import from rdkit.Chem.Draw and assigns IPythonConsole.molSize = (x, y). If RDKit is not installed or the import fails, an ImportError will be raised.
        - If IPythonConsole lacks the molSize attribute or if assignment is not permitted, an AttributeError or TypeError may be raised by the RDKit code.
        - The function does not validate the numeric values of x or y beyond assigning them; providing values that are not appropriate for RDKit's renderer (for example extremely large integers or negative integers) may lead to unexpected rendering behavior or errors emitted by RDKit.
        - The change is global for the RDKit IPythonConsole object in the current interpreter session; call again to change the size or restart the session to reset to defaults.
    """
    from useful_rdkit_utils.useful_rdkit_utils import rd_set_image_size
    return rd_set_image_size(x, y)


################################################################################
# Source: useful_rdkit_utils.useful_rdkit_utils.smi2mol_with_errors
# File: useful_rdkit_utils/useful_rdkit_utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def useful_rdkit_utils_useful_rdkit_utils_smi2mol_with_errors(smi: str):
    """Parse a SMILES string using RDKit and return the parsed RDKit molecule together with any warnings or error text that RDKit wrote to standard error during parsing.
    
    This function is intended for cheminformatics workflows (see project README: a collection of useful RDKit functions) where callers need both the parsed molecule for downstream operations (fingerprinting, substructure search, property calculation) and the textual diagnostic messages RDKit emits when a SMILES is ambiguous, has valence/atom errors, or triggers warnings. The function temporarily redirects Python's sys.stderr to capture RDKit stderr output produced by Chem.MolFromSmiles, then restores sys.stderr before returning. If parsing succeeds, the returned molecule can be used in typical RDKit pipelines; if parsing fails the molecule value will be None and the returned error string may contain explanatory messages.
    
    Behavior notes, side effects, and failure modes:
    - The function redirects the global sys.stderr to an in-memory buffer (io.StringIO) to capture messages written to stderr by RDKit during Chem.MolFromSmiles(smi). After capturing, it restores sys.stderr to the original sys.__stderr__.
    - The captured text includes warnings and diagnostic messages that RDKit would normally print to the process standard error (for example, valence warnings, kekulization issues, or other parser diagnostics).
    - If there are no messages written to stderr during parsing, the returned error string will be an empty string.
    - The first return value is the RDKit molecule object produced by Chem.MolFromSmiles or None if parsing failed; callers should check for None before using RDKit molecule APIs.
    - Because the function manipulates the global sys.stderr, it is not thread-safe: concurrent threads that rely on sys.stderr may observe the temporary redirection. Also, if Chem.MolFromSmiles or other code raises an exception while sys.stderr is redirected, the function does not use a try/finally to guarantee restoration; in that exceptional case sys.stderr may remain redirected and global state could be left altered.
    - The function does not attempt to capture messages sent via Python logging or other streams; it only captures text written to sys.stderr.
    - The function does not perform input validation beyond passing the provided smi string to RDKit; if RDKit is not available or Chem.MolFromSmiles is not present, a NameError/ImportError may be raised by the runtime.
    
    Args:
        smi (str): A SMILES string representing a chemical structure. In cheminformatics practice this is the linear text notation that encodes atoms and bonds; the string is passed directly to rdkit.Chem.MolFromSmiles for parsing. Typical use is to provide canonical or literal SMILES from a dataset or user input that you want converted to an RDKit molecule for subsequent processing.
    
    Returns:
        tuple: A 2-tuple (mol, err) where:
            mol (rdkit.Chem.Mol or None): The RDKit molecule object returned by Chem.MolFromSmiles(smi) when parsing succeeds. This object is intended for typical downstream RDKit operations (property calculations, substructure matching, depiction). If parsing fails, this value will be None.
            err (str): A string containing the exact text captured from sys.stderr during the call to Chem.MolFromSmiles. This includes RDKit warnings and error messages that would normally be emitted to standard error; it may be an empty string if no messages were produced.
    """
    from useful_rdkit_utils.useful_rdkit_utils import smi2mol_with_errors
    return smi2mol_with_errors(smi)


################################################################################
# Source: useful_rdkit_utils.useful_rdkit_utils.smi2morgan_fp
# File: useful_rdkit_utils/useful_rdkit_utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def useful_rdkit_utils_useful_rdkit_utils_smi2morgan_fp(
    smi: str,
    radius: int = 2,
    nBits: int = 2048
):
    """Convert a SMILES string to an RDKit Morgan (circular) fingerprint bit vector.
    
    This function is used in cheminformatics workflows (similarity searching, clustering,
    and machine learning) to convert a molecule represented as a SMILES string into a fixed-length
    binary fingerprint that encodes presence/absence of circular substructures. Internally the function
    builds an RDKit Mol from the provided SMILES using Chem.MolFromSmiles and then computes a
    Morgan fingerprint as an RDKit bit vector via AllChem.GetMorganFingerprintAsBitVect.
    
    Args:
        smi (str): A SMILES string representing the molecule to be fingerprinted.
            Role and practical significance: input chemical structure encoded as a text string.
            The function passes this string to RDKit's Chem.MolFromSmiles to construct an RDKit Mol.
            If the SMILES is invalid or cannot be parsed by RDKit, Chem.MolFromSmiles returns None
            and this function will return None (see failure modes below). Provide canonical or
            non-canonical SMILES as available; no additional preprocessing is performed by this function.
        radius (int): Radius of the Morgan fingerprint (default: 2).
            Role and practical significance: controls the size of the circular atom neighborhoods
            used to generate hashed substructure identifiers. A larger radius encodes larger local
            environments (more chemical context) at the cost of potentially more collisions and
            greater sensitivity to small changes in structure. This value is passed directly to
            RDKit's GetMorganFingerprintAsBitVect as the radius parameter.
        nBits (int): Number of bits in the returned fingerprint bit vector (default: 2048).
            Role and practical significance: sets the fixed dimensionality of the binary fingerprint.
            The fingerprint is a binary (bit) vector of length nBits produced by hashing the circular
            substructure identifiers into this fixed-size space. Choose nBits to balance memory/compute
            costs and collision rate for downstream tasks.
    
    Returns:
        rdkit.DataStructs.cDataStructs.ExplicitBitVect or None: An RDKit Morgan fingerprint bit vector
        (a fixed-length binary vector) computed from the input SMILES, or None if the SMILES could
        not be parsed into an RDKit Mol. The returned object is the direct output of
        AllChem.GetMorganFingerprintAsBitVect and can be used with RDKit similarity, distance,
        and serialization utilities. No conversion to other types is performed by this function;
        callers who need numpy arrays or Python lists must convert the RDKit ExplicitBitVect themselves.
    
    Behavior, defaults, and failure modes:
        - If Chem.MolFromSmiles(smi) fails (invalid or unparsable SMILES), the function returns None.
        - Default parameters are radius=2 and nBits=2048, common choices for many cheminformatics
          applications, but can be adjusted to suit specific use cases.
        - The function relies on RDKit (Chem and AllChem). If RDKit is not available in the runtime
          environment, importing or calling the underlying RDKit routines will raise the corresponding
          ImportError/NameError from the module scope or when this function is invoked.
        - The function has no side effects on input data and does not modify global state; it only
          constructs RDKit objects and returns the fingerprint bit vector (or None).
    """
    from useful_rdkit_utils.useful_rdkit_utils import smi2morgan_fp
    return smi2morgan_fp(smi, radius, nBits)


################################################################################
# Source: useful_rdkit_utils.useful_rdkit_utils.smi2numpy_fp
# File: useful_rdkit_utils/useful_rdkit_utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def useful_rdkit_utils_useful_rdkit_utils_smi2numpy_fp(
    smi: str,
    radius: int = 2,
    nBits: int = 2048
):
    """Convert a SMILES string to an RDKit Morgan fingerprint and (internally) populate a numpy array of fingerprint bits.
    
    Args:
        smi (str): SMILES string for a single molecule. In the cheminformatics domain (see README), SMILES is a compact text representation of a molecule; this function parses that string with RDKit's Chem.MolFromSmiles to produce an RDKit Mol used to compute a fingerprint. If the SMILES cannot be parsed, the function will not compute a fingerprint and will return None.
        radius (int): Fingerprint radius for the Morgan fingerprint (default 2). In practice this controls how many bond hops around each atom are considered when constructing circular substructure keys; larger radii capture larger local environments and affect similarity and descriptor behavior used in tasks such as similarity searching, clustering, or machine learning feature generation.
        nBits (int): Number of fingerprint bits (default 2048). This is the length of the bit vector produced by the Morgan fingerprint procedure; it controls the dimensionality of the fingerprint representation used in downstream analyses (e.g., computing Tanimoto similarity or using as features for models).
    
    Returns:
        DataStructs.cDataStructs.ExplicitBitVect or None: The RDKit Morgan fingerprint object (the variable fp in the source). When a valid SMILES is provided, the function constructs an RDKit Mol, calls mol2morgan_fp(mol=mol, radius=radius, nBits=nBits) to obtain an RDKit ExplicitBitVect (the Morgan fingerprint), and returns that object. If the SMILES cannot be parsed by RDKit (Chem.MolFromSmiles returns None), the function returns None.
    
    Behavior and side effects:
        The function also allocates a numpy.ndarray (arr = np.zeros((0,), dtype=np.int8)) and calls DataStructs.ConvertToNumpyArray(fp, arr) to copy fingerprint bits into that array. That numpy array is created and populated only as an internal side effect and is not returned or exposed to the caller. Note that the original (short) docstring described the return as a numpy array; the implementation actually returns the RDKit fingerprint object and only populates a numpy array internally. Calling code that requires a numpy array of bits should explicitly create and pass a suitably sized numpy array and/or convert the returned RDKit ExplicitBitVect to numpy itself.
    
    Failure modes and notes:
        If Chem.MolFromSmiles fails to parse smi, the function returns None and does not attempt to compute a fingerprint. mol2morgan_fp and DataStructs.ConvertToNumpyArray may raise exceptions if RDKit internals fail (for example, if mol2morgan_fp raises due to unexpected molecule state); these exceptions propagate unless caught by the caller. The function uses the provided radius and nBits exactly as passed; changing these defaults will change fingerprint structure and downstream behavior in similarity calculations or ML feature sets.
    """
    from useful_rdkit_utils.useful_rdkit_utils import smi2numpy_fp
    return smi2numpy_fp(smi, radius, nBits)


################################################################################
# Source: useful_rdkit_utils.useful_rdkit_utils.taylor_butina_clustering
# File: useful_rdkit_utils/useful_rdkit_utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def useful_rdkit_utils_useful_rdkit_utils_taylor_butina_clustering(
    fp_list: list,
    cutoff: float = 0.35
):
    """Cluster a set of RDKit fingerprints using the Taylor-Butina algorithm (RDKit's
    Butina.ClusterData) driven by pairwise Tanimoto similarities computed with
    rdkit.DataStructs.BulkTanimotoSimilarity. This function is provided as part of
    the useful_rdkit_utils collection of RDKit helper utilities and is used to group
    molecular fingerprints into discrete clusters based on chemical similarity,
    useful in tasks such as dataset deduplication, scaffold analysis, or
    representative selection for cheminformatics workflows.
    
    Args:
        fp_list (list): A list of fingerprint objects. These must be RDKit-compatible
            fingerprint objects accepted by rdkit.DataStructs.BulkTanimotoSimilarity
            (for example, RDKit bit-vector fingerprints such as ExplicitBitVect).
            The function computes pairwise Tanimoto similarities between each
            fingerprint and all previous fingerprints in the list; the input list
            order therefore influences the clustering outcome and cluster numbering.
        cutoff (float): Distance cutoff value used by Butina.ClusterData when
            clustering. The implementation computes distances as 1.0 - Tanimoto
            similarity and passes those distances to Butina.ClusterData with
            isDistData=True, so this cutoff is a distance threshold (not a
            similarity). The default is 0.35, which corresponds to grouping items
            with Tanimoto similarity >= 0.65 (1 - 0.35). Choosing a smaller cutoff
            results in more, tighter clusters; larger cutoffs give fewer, broader
            clusters.
    
    Returns:
        numpy.ndarray: A one-dimensional numpy array of integer cluster ids with
        length equal to len(fp_list). Each element at index i is the integer id of
        the cluster assigned to the fingerprint at fp_list[i]. Cluster ids are
        assigned sequentially in the order returned by rdkit.ML.Cluster.Butina and
        start at 0. If fp_list is empty, an empty numpy array of dtype int is
        returned.
    
    Behavior and side effects:
        The function first computes pairwise Tanimoto similarities using
        rdkit.DataStructs.BulkTanimotoSimilarity for each fingerprint against all
        earlier fingerprints in the list, converts those similarities to distances
        (1 - similarity), and then calls rdkit.ML.Cluster.Butina.ClusterData with
        isDistData=True and the provided cutoff. The function allocates a numpy
        integer array (dtype=int) to store and return cluster assignments; no other
        external side effects occur (no files are written). Computational cost is
        roughly quadratic in the number of fingerprints because of the pairwise
        similarity calculations; performance may become a concern for very large
        fp_list inputs.
    
    Failure modes and validation:
        The function does not perform deep type validation. If elements of fp_list
        are not compatible with rdkit.DataStructs.BulkTanimotoSimilarity, that call
        will raise an RDKit/TypeError. If rdkit or numpy are not available or if
        rdkit.ML.Cluster.Butina.ClusterData is provided invalid data (for example,
        inconsistent distance array length relative to nfps), RDKit will raise
        appropriate exceptions. The caller should ensure fp_list contains valid RDKit
        fingerprint objects and that len(fp_list) is appropriate for clustering.
    """
    from useful_rdkit_utils.useful_rdkit_utils import taylor_butina_clustering
    return taylor_butina_clustering(fp_list, cutoff)


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
