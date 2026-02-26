"""
Regenerated Google-style docstrings for module 'datamol'.
README source: others/readme/datamol/README.md
Generated at: 2025-12-02T00:51:55.272027Z

Total functions: 13
"""


################################################################################
# Source: datamol.convert.from_selfies
# File: datamol/convert.py
# Category: valid
################################################################################

def datamol_convert_from_selfies(selfies: str, as_mol: bool = False):
    """datamol.convert.from_selfies converts a SELFIES representation of a molecule into a SMILES string or into an RDKit Mol object usable in Datamol pipelines.
    
    Args:
        selfies (str): SELFIES string encoding a molecular graph. In the datamol context this is the serialized, robust molecular representation produced or consumed by the selfies library; passing None is tolerated by this implementation and will cause the function to return None immediately. This parameter is the primary molecular input for conversion and is expected to be a textual SELFIES sequence.
        as_mol (bool, optional): Whether to return an RDKit Mol object instead of a SMILES string. Defaults to False. When True the function will first decode SELFIES to a SMILES string and then convert that SMILES to an rdkit.Chem.rdchem.Mol using datamol.to_mol, producing an RDKit Mol which Datamol and downstream RDKit-based operations expect.
    
    Returns:
        Union[str, rdkit.Chem.rdchem.Mol, NoneType]: If selfies is None, returns None. Otherwise, returns a SMILES string (type str) produced by decoding the SELFIES input when as_mol is False (default). If as_mol is True and the decoder returns a non-None SMILES, returns an rdkit.Chem.rdchem.Mol instance (RDKit molecule) created via datamol.to_mol; this is the object type used throughout Datamol for molecular processing (visualization, fingerprinting, conformer generation, etc.). If the SELFIES decoder returns None or decoding fails, the function will propagate that None (or raise an exception from the underlying decoder); likewise, conversion to Mol may raise errors if the decoded SMILES is invalid or if RDKit fails to parse it. The function has no external side effects beyond calling the selfies decoder and datamol.to_mol.
    """
    from datamol.convert import from_selfies
    return from_selfies(selfies, as_mol)


################################################################################
# Source: datamol.data.chembl_drugs
# File: datamol/data/__init__.py
# Category: fix_docstring
# Reason: Schema parsing failed: Cannot generate JSON schema for chembl_drugs because the docstring has no description for the argument 'as_df'
################################################################################

def datamol_data_chembl_drugs(as_df: bool = True):
    """datamol.data.chembl_drugs returns a curated dataset of approved drugs from ChEMBL packaged with datamol. The dataset contains approximately 2.5k entries (small molecules only) in SMILES format together with associated metadata fields used by cheminformatics workflows: year of first approval, ChEMBL molecule identifier (chembl id), molecule type, and preferred name (pref_name). This function provides a ready-to-use reference set of approved drugs for common tasks described in the datamol README such as fingerprinting, descriptor calculation, visualization, standardization, and dataset benchmarking.
    
    The data file was generated using the notebook Get_ChEMBL_Approved_Drugs.ipynb (https://github.com/datamol-io/datamol/notebooks/Get_ChEMBL_Approved_Drugs.ipynb) on 2023-10-18. That notebook queries the chembl_webresource_client API to collect ChEMBL IDs and metadata, filters for small molecules with valid SMILES and a reported first approval date, and stores the resulting table as a parquet file that is bundled with datamol and accessed at runtime.
    
    Behavior details and side effects: the function opens the packaged parquet data file "chembl_approved_drugs.parquet" via datamol's open_datamol_data_file utility with open_binary=True and reads it using pandas.read_parquet into a pandas.core.frame.DataFrame. If as_df is False, the function converts the DataFrame to a list of RDKit molecule objects by calling datamol's from_df conversion routine. This routine transforms SMILES strings in the DataFrame into rdkit.Chem.rdchem.Mol instances so the returned list can be directly used with datamol and RDKit APIs. The function performs no network requests at call time (the network work was done when the parquet was generated) but does perform local I/O to load the bundled resource.
    
    Failure modes and exceptions: calling this function may raise file- or IO-related exceptions if the packaged resource cannot be located, opened, or parsed (for example FileNotFoundError, OSError, or pandas errors related to reading parquet). If conversion to RDKit molecules is requested (as_df=False) and RDKit is not installed or importable, an ImportError may occur. If the parquet file is present but missing expected columns (for example no SMILES column) or contains corrupted data, the function may raise KeyError, ValueError, or pandas parsing errors. When converting SMILES strings to rdkit.Chem.rdchem.Mol objects, invalid or unparsable SMILES may result in None entries or molecules that fail RDKit sanitization depending on the behavior of datamol.from_df; callers should validate molecules before downstream use. The default behavior is safe for data exploration (as_df=True) because it returns the raw dataframe without requiring RDKit to be importable.
    
    Args:
        as_df (bool): If True (default), return the raw dataset as a pandas.core.frame.DataFrame. The DataFrame contains one row per approved small-molecule drug and includes columns for SMILES and metadata fields: year of first approval, chembl id, molecule type, and pref_name. Returning a DataFrame is useful for tabular inspection, filtering, joining with other datasets, and pandas-based I/O. If False, convert the DataFrame into a Python list of rdkit.Chem.rdchem.Mol objects using datamol.from_df and return that list; this is useful when the caller intends to perform RDKit-centric operations (fingerprints, 2D/3D manipulation, visualization) and expects RDKit molecule instances. Choosing False triggers the SMILES-to-Mol conversion step and therefore requires RDKit to be available and may raise conversion-related errors for invalid SMILES.
    
    Returns:
        Union[List[rdkit.Chem.rdchem.Mol], pandas.core.frame.DataFrame]: When as_df is True, returns a pandas.core.frame.DataFrame containing the assembled ChEMBL approved-drugs table with SMILES and metadata columns; when as_df is False, returns a Python list of rdkit.Chem.rdchem.Mol objects created from the DataFrame SMILES column. The returned object is intended for immediate use with datamol and RDKit workflows: DataFrame for tabular processing and reproducible dataset handling, or List[rdkit.Chem.rdchem.Mol] for molecule-centric cheminformatics operations.
    """
    from datamol.data import chembl_drugs
    return chembl_drugs(as_df)


################################################################################
# Source: datamol.data.chembl_samples
# File: datamol/data/__init__.py
# Category: fix_docstring
# Reason: Schema parsing failed: Cannot generate JSON schema for chembl_samples because the docstring has no description for the argument 'as_df'
################################################################################

def datamol_data_chembl_samples(as_df: bool = True):
    """datamol.data.chembl_samples returns a small curated sample of molecules drawn from the ChEMBL database (approximately 2,000 entries). This helper function provides an easy, fast source of real-world molecules for examples, tutorials, quick experiments, and tests within the datamol ecosystem. The sample data is read from the packaged resource file "chembl_samples.csv" (originally proposed by Patrick Walters at https://github.com/PatWalters/practical_cheminformatics_posts) using datamol's open_datamol_data_file helper, then loaded with pandas and optionally converted to RDKit Mol objects for downstream cheminformatics processing.
    
    Args:
        as_df (bool): If True (default), return the raw data loaded from the packaged CSV as a pandas.core.frame.DataFrame. The DataFrame contains the columns and metadata provided in the packaged "chembl_samples.csv" (for example, structural representations and identifiers as present in the source CSV) and is suitable for dataframe-based analysis, filtering, or persisting. If False, the function converts the DataFrame into a Python list of rdkit.Chem.rdchem.Mol objects by calling datamol.from_df; this is useful when the caller intends to perform RDKit-based molecule operations (fingerprinting, sanitization, 2D/3D processing, etc.). Note that conversion to rdkit.Chem.rdchem.Mol objects requires RDKit to be available in the environment; the conversion behavior (e.g., how invalid rows are handled) follows datamol.from_df semantics. The default value is True.
    
    Behavior and side effects:
        The function opens the packaged resource "chembl_samples.csv" using datamol.open_datamol_data_file and reads it with pandas.read_csv. No modification is made to the packaged file. If as_df is False, datamol.from_df is invoked on the loaded DataFrame to produce a list of rdkit.Chem.rdchem.Mol objects; this conversion may perform parsing of SMILES or other structural fields according to datamol.from_df and may produce None or drop entries for molecules that cannot be parsed. The function does not persist or write any files. It is intended for quick access to an example ChEMBL-derived dataset for interactive use, unit tests, and documentation examples.
    
    Failure modes and exceptions:
        If the packaged CSV file is not available or cannot be opened, an OSError or FileNotFoundError may be raised by open_datamol_data_file or the underlying file system. pandas.read_csv may raise pandas.errors.ParserError or related IO errors if the CSV is malformed. If as_df is False and RDKit is not installed or datamol.from_df fails to construct molecules, ImportError or conversion-specific exceptions may occur. Conversion may also silently produce None entries for unparseable structures depending on datamol.from_df behavior; callers who require strict conversion should validate the returned molecules.
    
    Returns:
        Union[List[rdkit.Chem.rdchem.Mol], pandas.core.frame.DataFrame]: If as_df is True, a pandas.core.frame.DataFrame containing the rows and columns from the packaged "chembl_samples.csv" (used for dataframe-oriented workflows and inspection). If as_df is False, a Python list of rdkit.Chem.rdchem.Mol objects produced by converting the DataFrame via datamol.from_df, suitable for direct RDKit-based cheminformatics operations.
    """
    from datamol.data import chembl_samples
    return chembl_samples(as_df)


################################################################################
# Source: datamol.data.freesolv
# File: datamol/data/__init__.py
# Category: fix_docstring
# Reason: Schema parsing failed: Cannot generate JSON schema for freesolv because the docstring has no description for the argument 'as_df'
################################################################################

def datamol_data_freesolv(as_df: bool = True):
    """datamol.data.freesolv returns the FreeSolv dataset packaged with datamol. The function loads the CSV file included with the datamol distribution ("freesolv.csv") and returns the dataset either as a pandas DataFrame (default) or as a list of RDKit Mol objects suitable for molecular processing with RDKit and datamol utilities. This dataset is provided as a small toy dataset for pedagogy and testing of datamol workflows (not for benchmarking, model training, or large-scale analysis).
    
    The loaded dataset contains 642 rows (molecules) and the following columns exactly as stored in the CSV: ['iupac', 'smiles', 'expt', 'calc'].
    The columns have the following practical meanings in the molecular data domain: 'iupac' is the IUPAC name of the molecule (string), 'smiles' is the SMILES string representation used to construct RDKit Mol objects (string), 'expt' is the experimental free energy value (numeric) and 'calc' is a calculated free energy value (numeric). These columns are useful for quick examples and unit tests when demonstrating datamol's IO, molecule conversion, and small-scale analysis features.
    
    Warning:
        This dataset is only intended for pedagogic examples and testing. It is not suitable for benchmarking, rigorous analysis, or model training.
    
    Args:
        as_df (bool): If True (default), return the dataset as a pandas.core.frame.DataFrame produced by pandas.read_csv on the packaged "freesolv.csv" file. The DataFrame preserves the original columns ['iupac', 'smiles', 'expt', 'calc'] and row order (one row per molecule). If False, the function converts the DataFrame to a list of RDKit molecule objects by calling datamol.from_df(data) and returns a Python list of rdkit.Chem.rdchem.Mol objects in the same order as the rows in the CSV. Use as_df=False when you want immediate RDKit Mol objects for downstream molecular processing (fingerprints, conformer generation, visualization) as shown in the datamol README examples.
    
    Returns:
        Union[List[rdkit.Chem.rdchem.Mol], pandas.core.frame.DataFrame]: If as_df is True, a pandas.core.frame.DataFrame with columns ['iupac', 'smiles', 'expt', 'calc'] and 642 rows. If as_df is False, a Python list of rdkit.Chem.rdchem.Mol objects obtained by converting the 'smiles' column using datamol.from_df; the list preserves the CSV row order.
    
    Behavior, side effects, defaults, and failure modes:
        The function opens the packaged data file via open_datamol_data_file("freesolv.csv") and reads it with pandas.read_csv. No external network access is required; the file is expected to be bundled with the datamol package. Default behavior returns a DataFrame (as_df=True). When as_df=False, datamol.from_df is invoked to parse SMILES strings into RDKit Mol objects; this conversion may perform RDKit parsing and any behavior or sanitization implemented by datamol.from_df applies (for example, invalid SMILES may result in None entries or raise errors depending on datamol.from_df configuration). The function does not write files or modify global state beyond allocating the returned DataFrame or list of Mol objects.
    
    Raises:
        FileNotFoundError: If the packaged "freesolv.csv" file cannot be located by open_datamol_data_file.
        pandas.errors.ParserError or pandas.errors.EmptyDataError: If pandas.read_csv fails to parse the CSV file.
        Exception: If as_df is False and datamol.from_df or RDKit fails to construct molecules from the 'smiles' strings (for example, due to invalid SMILES or RDKit sanitization/valence errors).
    """
    from datamol.data import freesolv
    return freesolv(as_df)


################################################################################
# Source: datamol.descriptors.compute.any_rdkit_descriptor
# File: datamol/descriptors/compute.py
# Category: fix_docstring
# Reason: Schema parsing failed: ("Couldn't parse this type hint, likely due to a custom class or object: ", typing.Callable)
################################################################################

def datamol_descriptors_compute_any_rdkit_descriptor(name: str):
    """Return an RDKit descriptor function by its attribute name from the standard RDKit descriptor modules.
    This helper is used in datamol (a thin layer on top of RDKit) to resolve a descriptor
    by name so callers can compute molecular descriptors/features for rdkit.Chem.Mol objects
    without importing RDKit descriptor modules directly. The function implements a lookup
    strategy: it first attempts to retrieve the attribute from rdkit.Chem.Descriptors and,
    if not found, from rdkit.Chem.rdMolDescriptors. This is useful in datamol pipelines
    that build descriptor-based feature vectors for machine learning, filtering, or
    chemoinformatics analyses.
    
    Args:
        name (str): The exact attribute name of the RDKit descriptor to resolve. This is
            case-sensitive and must match the attribute name defined in rdkit.Chem.Descriptors
            or rdkit.Chem.rdMolDescriptors (for example, "MolWt", "NumRotatableBonds",
            or "CalcTPSA"). The parameter represents the descriptor identifier that datamol
            callers provide when they want to compute a specific molecular feature.
    
    Returns:
        Callable: The descriptor callable object retrieved from RDKit. Practically, this is
            the raw RDKit function or object used to compute the descriptor value(s) for
            a molecule; in typical usage within datamol the returned callable accepts an
            rdkit.Chem.Mol instance (and possibly additional RDKit-specific parameters)
            and returns a numeric value (int or float) or descriptor-specific result.
            The returned callable has no datamol-side wrappers applied by this helper;
            callers should use it exactly as provided by RDKit.
    
    Behavior and failure modes:
        The function performs no side effects and does not modify input arguments or global
        state. If a descriptor with the given name exists in rdkit.Chem.Descriptors it is
        returned; otherwise rdkit.Chem.rdMolDescriptors is searched. If the name is not found
        in either module a ValueError is raised with the message "Descriptor {name} not found."
        Callers must ensure RDKit is available in the environment and that the provided name
        corresponds to a valid RDKit descriptor attribute. The callable returned may have
        descriptor-specific calling conventions (for example, some RDKit descriptor functions
        accept additional keyword arguments); consult RDKit documentation for the exact
        signature of the resolved descriptor when using it in datamol workflows.
    """
    from datamol.descriptors.compute import any_rdkit_descriptor
    return any_rdkit_descriptor(name)


################################################################################
# Source: datamol.fragment._assemble.build
# File: datamol/fragment/_assemble.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def datamol_fragment__assemble_build(
    ll_mols: list,
    max_n_mols: float,
    mode: str = "brics",
    frag_rxn: str = None,
    ADD_RNXS: list = []
):
    """Build a super molecule from lists of fragment pools by applying retrosynthetic/assembly reactions and yielding assembled RDKit Mol objects.
    
    Args:
        ll_mols (list): A list of fragment pools where each element is an iterable (commonly a list) of fragment objects. In Datamol (RDKit-first design), each fragment is expected to be an rdkit.Chem.Mol-like object representing a molecular fragment that will be used as building blocks. The function enumerates Cartesian products across the fragment pools via itertools.product(*ll_mols) so ll_mols defines which combinations of fragments are attempted for assembly.
        max_n_mols (float): Maximum number of unique assembled molecules to produce. The default is infinity (float("inf")), meaning no artificial upper bound. The function maintains an internal set of seen canonical/molecular identifiers (mSmi strings returned by the internal reaction runner) and stops yielding once len(seen) >= max_n_mols. Use this to limit runtime and memory when exploring large combinatorial fragment spaces.
        mode (str): Selection of the predefined reaction sets to apply. Accepts "brics" to use the library's BRICS retrosynthetic/assembly reactions, "rxn" to use the non-BRICS reaction set, or None to use both sets combined. The default is "brics". The chosen mode determines which module-level reaction collections (e.g., ALL_BRICS_RETRO, ALL_RXNS_RETRO and their associated type lists) are iterated and applied to fragment pairs.
        frag_rxn (str): Optional filter to restrict assembly to a single reaction type. If provided, frag_rxn is compared (after stripping surrounding double-quote characters) against the reaction type labels (elements of the module-level CUR_RXNS_TYPE list). When a match is found, only the corresponding reaction from the selected reaction set is used for all assembly attempts. Use this to reproduce or debug a single reaction’s behavior instead of trying the full reaction collection.
        ADD_RNXS (list): Additional reaction objects or a mapping of reaction-type-label -> reaction object to append to the selected reaction set. If a list is given (default []), synthetic type labels "RXN-0", "RXN-1", ... are generated for those values. If a dict is given, its keys are used as reaction type labels and its values as reaction objects. If ADD_RNXS is None, no extra reactions are appended. Note: the function signature uses a mutable default (empty list); to avoid surprises from shared mutable defaults across calls, explicitly pass a freshly created list or None when calling.
    
    Behavior, side effects, defaults, and failure modes:
        - The function is a generator: it yields assembled molecules (rdkit.Chem.Mol objects) as they are produced rather than returning a single collection. This supports streaming large combinatorial spaces without storing all results in memory.
        - For each combination of fragments produced by Cartesian product over ll_mols, the function attempts each reaction in the active reaction list (determined by mode and ADD_RNXS, and optionally filtered by frag_rxn). Reactions are applied via an internal helper (_run_at_all_rct) which is expected to return a sequence of (m, mSmi) tuples where m is the assembled rdkit.Chem.Mol and mSmi is a string identifier (commonly a canonical SMILES) used for uniqueness checks.
        - The function maintains an internal seen set of mSmi strings to ensure uniqueness. Only molecules whose mSmi is not already in seen are yielded; their mSmi values are then added to seen. Generation stops early when the number of seen unique molecules reaches max_n_mols.
        - Reaction application is guarded by a broad try/except: any Exception raised by _run_at_all_rct is caught and ignored (the failing reaction/fragment combination is skipped). This makes the generator robust to individual reaction failures but can silently hide underlying problems; when diagnosing issues, consider running reactions in isolation or inspecting the reaction objects.
        - If frag_rxn matches a reaction type label, CUR_RXNS is reduced to that single reaction and only it is tried; otherwise, all selected reactions are attempted in sequence for each fragment combination.
        - ADD_RNXS handling accepts either a list or a dict (as implemented): when a dict is provided, its keys become reaction-type labels and its values are appended to the reaction list. If ADD_RNXS is None, no extra reactions are appended.
        - The function relies on module-level reaction collections (e.g., ALL_BRICS_RETRO, ALL_RXNS_RETRO and their associated type lists) to exist and be compatible with _run_at_all_rct. Missing or incompatible reaction objects will lead to exceptions that are caught and skipped, possibly resulting in fewer or no yielded molecules.
        - Because the function yields rdkit.Chem.Mol objects produced by applying retrosynthetic/assembly reactions, the caller should be prepared to receive RDKit molecule objects and may want to further sanitize, standardize, or canonicalize them using Datamol/RDKit utilities (for example, to compute canonical SMILES used as mSmi).
    
    Returns:
        generator: A generator that yields assembled rdkit.Chem.Mol objects. Each yielded molecule is produced by applying one of the active reactions to a combination of fragments taken from ll_mols and corresponds to a unique mSmi identifier not previously yielded. The generator may terminate early when max_n_mols unique molecules have been produced or when all fragment combinations and reactions have been exhausted.
    """
    from datamol.fragment._assemble import build
    return build(ll_mols, max_n_mols, mode, frag_rxn, ADD_RNXS)


################################################################################
# Source: datamol.io.read_molblock
# File: datamol/io.py
# Category: valid
################################################################################

def datamol_io_read_molblock(
    molblock: str,
    sanitize: bool = True,
    strict_parsing: bool = True,
    remove_hs: bool = True,
    fail_if_invalid: bool = False
):
    """Read a single Mol block (MDL molfile block) and return an RDKit molecule object.
    
    This function is part of datamol's IO utilities and is used to convert a string containing a Mol block (for example the contents of an SDF entry) into an rdkit.Chem.rdchem.Mol object so downstream datamol workflows (which operate on RDKit Mol objects) can manipulate, standardize, or compute properties on the molecule. Note that potential molecule properties embedded in the Mol block are not read or preserved by this function.
    
    Args:
        molblock (str): String containing the Mol block (MDL molfile block) to parse. This is the raw text representation of a single molecule as found in SDF/MOL files. The function passes this string to RDKit's MolFromMolBlock parser.
        sanitize (bool): Whether to sanitize the molecule after parsing. When True (default), RDKit's sanitization routines are applied (aromaticity perception, valence checks, cleanup of atom/bond properties, etc.). Sanitization can modify the returned rdkit.Chem.rdchem.Mol to produce a chemically consistent representation; when False, the returned Mol may keep original atom/bond settings and may therefore contain valence or aromaticity inconsistencies that downstream code must handle.
        strict_parsing (bool): If set to True (default), the RDKit parser enforces stricter correctness checks on the Mol block content and may fail (return None) on malformed input. If set to False, the parser is more permissive and attempts to recover from certain formatting or correctness issues in the Mol block.
        remove_hs (bool): Whether to remove explicit hydrogens present in the Mol block (passed as RDKit removeHs argument). Default is True. Removing explicit hydrogens yields a molecule with implicit hydrogens handled by RDKit; keeping explicit hydrogens (set to False) preserves H atoms as separate atoms in the returned rdkit.Chem.rdchem.Mol.
        fail_if_invalid (bool): If set to True, the function raises a ValueError when parsing fails or the parser returns None (default False). When False, the function returns None on invalid or unparsable input instead of raising an exception. Use True when you need strict failure semantics in data pipelines.
    
    Returns:
        Optional[rdkit.Chem.rdchem.Mol]: An rdkit.Chem.rdchem.Mol instance representing the parsed molecule if parsing succeeds. If the parser fails or determines the molecule is invalid, returns None unless fail_if_invalid is True, in which case a ValueError is raised. There are no other side effects; molecule properties embedded in the Mol block are not extracted by this routine.
    """
    from datamol.io import read_molblock
    return read_molblock(molblock, sanitize, strict_parsing, remove_hs, fail_if_invalid)


################################################################################
# Source: datamol.io.read_pdbblock
# File: datamol/io.py
# Category: valid
################################################################################

def datamol_io_read_pdbblock(
    molblock: str,
    sanitize: bool = True,
    remove_hs: bool = True,
    flavor: int = 0,
    proximity_bonding: bool = True
):
    """datamol.io.read_pdbblock reads a PDB-format string and returns an RDKit molecule suitable for use with datamol and RDKit pipelines. This function wraps RDKit's MolFromPDBBlock to convert a PDB text block (for example the contents of a .pdb file or a PDB record returned by a remote service) into an rdkit.Chem.rdchem.Mol object that can be used for downstream operations in datamol such as sanitization, conformer handling, fingerprinting, visualization, and other molecular manipulations.
    
    Args:
        molblock (str): String containing the PDB block to parse. This should be the full text of a PDB file or record (ATOM/HETATM and related records). The function parses atomic coordinates and record information contained in this string to build the RDKit molecule representation.
        sanitize (bool): Whether to perform RDKit sanitization after parsing. Sanitization includes standard RDKit checks and operations such as valence validation, aromaticity perception, and implicit hydrogen assignment. Defaults to True. If sanitize is True and RDKit cannot successfully sanitize the parsed structure, the parse may fail (RDKit may return a falsy result); callers should validate the returned molecule before further use. If sanitize is False, the returned rdkit.Chem.rdchem.Mol may be chemically incomplete or contain valence/aromaticity issues and may require manual sanitization via datamol or RDKit utilities before downstream chemistry operations.
        remove_hs (bool): Whether to remove explicit hydrogen atoms found in the PDB block during parsing. Defaults to True. When True, explicit H atoms present in the input PDB text will be removed from the returned rdkit.Chem.rdchem.Mol; when False, explicit hydrogens are preserved. This affects downstream tasks that depend on explicit versus implicit hydrogens (for example, some topology or charge computations).
        flavor (int): Integer flavor flag passed directly to RDKit's MolFromPDBBlock flavor argument. Defaults to 0. This integer encodes RDKit-specific parsing options that influence how PDB records are interpreted; consult RDKit documentation for the meaning of specific flavor values. Invalid or unsupported flavor values are passed to RDKit and may change parsing behavior or cause parsing to fail.
        proximity_bonding (bool): Whether to enable RDKit's automatic proximity bonding based on atomic coordinates. Defaults to True. When True, RDKit will infer bonds between atoms based on their spatial proximity (useful when bond connectivity is missing or incomplete in the PDB). Enabling proximity bonding can create bonds that are geometrically plausible but not chemically intended (e.g., in noisy coordinate sets or multi-molecule PDBs), so verify the resulting topology when using this option.
    
    Returns:
        rdkit.Chem.rdchem.Mol: An RDKit molecule object representing the parsed PDB structure. This rdkit.Chem.rdchem.Mol is the standard molecule type used throughout datamol and RDKit and can be passed to datamol functions (conformer generation, fingerprinting, visualization, etc.). Note that RDKit parsing/sanitization may fail for malformed or incompatible PDB content; in such cases RDKit may return a falsy value (commonly None), so callers should check the returned object for validity before downstream use.
    """
    from datamol.io import read_pdbblock
    return read_pdbblock(molblock, sanitize, remove_hs, flavor, proximity_bonding)


################################################################################
# Source: datamol.mol.standardize_smiles
# File: datamol/mol.py
# Category: valid
################################################################################

def datamol_mol_standardize_smiles(smiles: str):
    """datamol.mol.standardize_smiles: Standardize a SMILES string using RDKit's SMILES standardizer and tautomeric canonicalization to produce a normalized, canonical SMILES suitable for downstream molecular processing tasks (deduplication, fingerprinting, storage, and comparison).
    
    Args:
        smiles (str): A SMILES string representing a chemical structure as accepted by RDKit. This function expects a textual SMILES input (for example "CCO" for ethanol). The input is interpreted by RDKit and may include stereochemistry, charges, isotopes, and explicit hydrogens; the exact handling of these features depends on the RDKit standardizer implementation. Pass the raw SMILES you wish to normalize prior to further processing with datamol (for example before calling dm.to_mol, dm.to_fp, or storing in a dataset).
    
    Returns:
        standard_smiles (str): The standardized SMILES produced by rdMolStandardize.StandardizeSmiles. This string is a deterministic, normalized representation of the input for a given RDKit version and standardizer configuration and is intended to be used for canonical comparisons, deduplication, indexing, and other downstream tasks in cheminformatics workflows. The function itself has no side effects (it does not modify global state or molecule objects); it simply returns the standardized SMILES. If RDKit cannot parse or standardize the provided SMILES (for example if the input is syntactically invalid or unsupported by the RDKit standardizer), RDKit may raise an exception or return a value indicating failure—callers should handle such errors and be aware that results can vary with RDKit version/configuration as documented in the datamol README.
    """
    from datamol.mol import standardize_smiles
    return standardize_smiles(smiles)


################################################################################
# Source: datamol.reactions._attachments.add_brackets_to_attachment_points
# File: datamol/reactions/_attachments.py
# Category: valid
################################################################################

def datamol_reactions__attachments_add_brackets_to_attachment_points(smiles: str):
    """Add brackets to attachment points in a SMILES string so that attachment tokens are explicitly bracketed for downstream processing.
    
    This function is part of datamol.reactions._attachments and is used to normalize the textual representation of attachment points in SMILES strings before those strings are parsed or used to build reaction templates. It finds occurrences of the module's attachment-point token that are not already enclosed in square brackets (the pattern is defined by ATTACHMENT_POINT_NO_BRACKETS_REGEXP) and replaces each with a bracketed form using the module-level ATTACHMENT_POINT_TOKEN. Example: "CC(C)CO*" becomes "CC(C)CO[*]".
    
    Args:
        smiles (str): A SMILES string potentially containing attachment point tokens. The function operates on the raw SMILES text (not on rdkit.Chem.Mol objects). It expects a Python string containing the SMILES representation. If a non-string value is passed, the underlying re.sub call will raise a TypeError.
    
    Behavior and side effects:
        The function performs a pure textual transformation using re.sub and the module-level regular expression ATTACHMENT_POINT_NO_BRACKETS_REGEXP together with ATTACHMENT_POINT_TOKEN to produce bracketed attachment points. It does not modify any external state, does not validate or sanitize the SMILES chemically, and does not convert the string to an RDKit molecule. The operation is idempotent: running it on a SMILES that already has bracketed attachment points will leave those tokens unchanged. If the input contains no matching attachment points, the original string is returned unchanged.
    
    Failure modes and notes:
        If the module-level constants ATTACHMENT_POINT_NO_BRACKETS_REGEXP or ATTACHMENT_POINT_TOKEN are missing or incorrectly defined, the function behavior depends on those definitions (e.g., no replacements may occur). Passing a non-str value will cause a TypeError from re.sub. This function should be used as a preprocessing/text-normalization step prior to SMILES parsing or reaction template generation; it does not guarantee that the resulting SMILES is chemically valid.
    
    Returns:
        str: The input SMILES string with attachment point tokens enclosed in square brackets. If no attachment points are matched, returns the original input string unchanged.
    """
    from datamol.reactions._attachments import add_brackets_to_attachment_points
    return add_brackets_to_attachment_points(smiles)


################################################################################
# Source: datamol.reactions._reactions.rxn_from_block
# File: datamol/reactions/_reactions.py
# Category: valid
################################################################################

def datamol_reactions__reactions_rxn_from_block(rxn_block: str, sanitize: bool = False):
    """datamol.reactions._reactions.rxn_from_block: Create and initialize an RDKit ChemicalReaction object from a reaction block string (RXN block). This function is intended for use in Datamol's reaction processing workflows (RDKit-first cheminformatics pipelines) to convert a textual RXN block into a ready-to-use reaction object that can be applied to rdkit.Chem.Mol instances or inspected for reactant/product templates.
    
    Args:
        rxn_block (str): A reaction block string in MDL RXN block format (the textual representation of a chemical reaction). In Datamol and RDKit contexts, this string encodes reactant and product connectivity and atom mappings and is the raw input produced by RXN files or generated programmatically. The function parses this string with RDKit's rdChemReactions.ReactionFromRxnBlock to build the reaction object. If the provided string is malformed or not a valid RXN block, RDKit parsing will fail and the function will raise an exception (see failure modes below).
        sanitize (bool): Whether to run RDKit sanitization during parsing (default False). When True, RDKit performs standard molecule sanitization steps (valence checks, aromaticity assignment, implicit/explicit hydrogen normalization, etc.) on the reaction templates as they are created. Sanitization can surface structural issues early but may also raise RDKit errors for invalid chemistry; keeping sanitize=False may succeed for blocks that need later custom handling. This parameter maps directly to the sanitize argument of rdChemReactions.ReactionFromRxnBlock.
    
    Returns:
        rdkit.Chem.rdChemReactions.ChemicalReaction: An initialized RDKit ChemicalReaction object representing the parsed reaction block. The returned object has had its Initialize() method invoked by this function, so reactant and product templates and internal reaction metadata are prepared for immediate use in Datamol workflows (for example, applying the reaction to rdkit.Chem.Mol reactants, analyzing atom mappings, or serializing the reaction). Callers should treat the returned object as ready-to-use but may optionally perform additional validation or sanitization depending on downstream requirements.
    
    Behavior and side effects:
        The function calls rdChemReactions.ReactionFromRxnBlock(rxnblock=rxn_block, sanitize=sanitize) to create the reaction and then calls Initialize() on the resulting object. Initialize() populates internal reaction templates and data structures; this is a necessary step before using the reaction to perform transformations. No files are written and no global state in Datamol is modified; the only side effect is the allocation and initialization of the returned RDKit ChemicalReaction object.
    
    Defaults:
        sanitize defaults to False to provide a conservative parsing behavior aligned with Datamol's goal of offering sensible defaults while allowing callers to opt into RDKit's sanitization when appropriate.
    
    Failure modes and exceptions:
        If rxn_block is not a valid RXN block or contains inconsistent chemistry, RDKit's ReactionFromRxnBlock or the subsequent Initialize() call may raise exceptions (for example, RDKit parsing errors or attribute errors if parsing returns None). Callers should validate input RXN content and catch exceptions from RDKit when using untrusted or programmatically generated RXN blocks.
    """
    from datamol.reactions._reactions import rxn_from_block
    return rxn_from_block(rxn_block, sanitize)


################################################################################
# Source: datamol.reactions._reactions.rxn_from_smarts
# File: datamol/reactions/_reactions.py
# Category: valid
################################################################################

def datamol_reactions__reactions_rxn_from_smarts(rxn_smarts: str):
    """datamol.reactions._reactions.rxn_from_smarts: Create and initialize an RDKit ChemicalReaction from a reaction SMARTS string for use in Datamol reaction processing pipelines.
    
    Args:
        rxn_smarts (str): Reaction SMARTS string describing the chemical transformation using RDKit reaction SMARTS syntax. This string is passed verbatim to rdkit.Chem.rdChemReactions.ReactionFromSmarts(SMARTS=...), so it must follow RDKit's SMARTS conventions for reactant and product patterns, atom mapping, and bond specifications. In the Datamol context this SMARTS is used to define transformation rules that can be applied to rdkit.Chem.Mol objects (for example via ChemicalReaction.RunReactants) during reaction enumeration, retrosynthesis workflows, or virtual chemical transformations.
    
    Returns:
        rdkit.Chem.rdChemReactions.ChemicalReaction: An initialized RDKit ChemicalReaction object created from the provided SMARTS and prepared for immediate use. The function calls rdChemReactions.ReactionFromSmarts with the given SMARTS and then calls Initialize() on the resulting reaction; Initialize() computes and finalizes internal templates and mappings so the returned ChemicalReaction is ready to run against reactant molecule tuples. The returned object is mutable and is the canonical RDKit ChemicalReaction type used throughout Datamol for applying and analyzing reactions.
    
    Behavior and side effects:
        The function delegates parsing to RDKit's ReactionFromSmarts and then invokes the reaction.Initialize() method. Initialize() mutates the ChemicalReaction object in place to compute internal data structures required for running the reaction (such as template matching and atom mappings). There is no additional sanitization or validation performed by Datamol beyond what RDKit performs.
    
    Failure modes and recommendations:
        If rxn_smarts is not a valid RDKit reaction SMARTS string, RDKit may raise a parsing error or produce a reaction object that will fail when used; callers should validate or catch RDKit exceptions when supplying untrusted SMARTS. The function assumes a correct SMARTS string and a compatible RDKit installation as documented in the Datamol README; mismatches in RDKit versions may affect SMARTS parsing behavior.
    """
    from datamol.reactions._reactions import rxn_from_smarts
    return rxn_from_smarts(rxn_smarts)


################################################################################
# Source: datamol.utils.fs.join
# File: datamol/utils/fs.py
# Category: fix_docstring
# Reason: Schema parsing failed: Cannot generate JSON schema for join because the docstring has no description for the argument 'paths'
################################################################################

def datamol_utils_fs_join(*paths: str):
    """Join path components using the filesystem separator determined by the first path component.
    
    This function is used throughout datamol for modern IO operations that must work across local and remote storage backends supported by fsspec (for example: local file system, s3, gcs). The first path component is used to determine which fsspec filesystem (mapper) to use and therefore which path separator (fs.sep) is applied when joining all components. Each provided component is converted to a string, has trailing forward slashes removed, and then the components are concatenated using the filesystem separator. The function does not check whether the resulting path exists nor does it normalize beyond stripping trailing "/" characters.
    
    Args:
        paths (str): One or more path components to join. These are the positional varargs passed to the function; each element will be converted via built-in str(), trailing forward slashes ("/") will be removed from each converted component, and the resulting components will be joined using the separator of the filesystem inferred from the first component. Typical values are fsspec-style paths such as "s3://bucket/folder", "gs://bucket/folder", or local paths like "/home/user/dir". Mixing components that belong to different filesystems is allowed syntactically but may produce an invalid or unintended combined path because only the first component determines the filesystem and separator.
    
    Returns:
        str: The joined path string formed by concatenating the processed components using the fs.sep of the filesystem resolved from the first component. Example outcome for inputs ("s3://bucket", "path", "to/file") will use the S3 separator (usually "/") and return "s3://bucket/path/to/file".
    
    Behavior, side effects, and failure modes:
        - The function calls get_mapper(source_path).fs internally to resolve the fsspec filesystem from the first component; any exceptions raised by get_mapper (for example, if the scheme is unknown or fsspec is not available) will propagate to the caller.
        - If no positional components are provided, the function will raise an IndexError when trying to access the first element; callers must provide at least one path component.
        - Only trailing forward slashes ("/") are removed from each component via rstrip("/"); other trailing separators or platform-specific nuances are not altered.
        - The function does not perform path normalization (beyond stripping trailing "/") nor does it verify existence or permissions on the resulting path; it purely composes a string according to the resolved filesystem separator.
        - Non-string inputs will be converted with str(), which may produce unexpected results if given complex objects; the function signature and intended use expect string-like path components consistent with fsspec-supported paths.
    """
    from datamol.utils.fs import join
    return join(*paths)


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
