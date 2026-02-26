"""
Regenerated Google-style docstrings for module 'chembl_downloader'.
README source: others/readme/chembl_downloader/README.md
Generated at: 2025-12-02T00:43:21.034293Z

Total functions: 2
"""


################################################################################
# Source: chembl_downloader.queries.get_assay_sql
# File: chembl_downloader/queries.py
# Category: fix_docstring
# Reason: Schema parsing failed: Cannot generate JSON schema for get_assay_sql because the docstring has no description for the argument 'assay_chembl_id'
################################################################################

def chembl_downloader_queries_get_assay_sql(assay_chembl_id: str):
    """Get the SQL query string to retrieve molecular structures and standardized activity measurements for a single ChEMBL assay.
    
    Args:
        assay_chembl_id (str): The ChEMBL assay identifier to query (the value stored in the ASSAYS.chembl_id column of the ChEMBL SQLite dump). In the domain of chembl_downloader this is the canonical assay id used to select all activities measured in that assay. This function interpolates this string directly into the SQL WHERE clause to restrict results to the specified assay.
    
    Behavior and practical significance: This function builds and returns a single SQL SELECT statement (as a Python str) designed for use against the packaged ChEMBL SQLite database that chembl_downloader downloads and manages (see the project README for how to obtain and open the SQLite dump). The generated SQL joins the MOLECULE_DICTIONARY, COMPOUND_STRUCTURES, ACTIVITIES, and ASSAYS tables to return COMPOUND_STRUCTURES.canonical_smiles, MOLECULE_DICTIONARY.chembl_id, ACTIVITIES.STANDARD_TYPE, ACTIVITIES.STANDARD_RELATION, ACTIVITIES.STANDARD_VALUE, and ACTIVITIES.STANDARD_UNITS for rows where ASSAYS.chembl_id equals the provided assay_chembl_id and where the activity measurement is present and reported as an equality (standard_value is not null, standard_relation is not null, and standard_relation = '='). The SQL is returned as a dedented multiline string suitable for passing to chembl_downloader.query() (which uses pandas.read_sql) or to a sqlite3 cursor.execute() in the typical usage patterns described in the README.
    
    Side effects, defaults, and failure modes: The function has no side effects and does not execute the SQL; it only returns the SQL text. Because the assay_chembl_id value is interpolated directly into the returned SQL using string formatting, providing untrusted or malformed assay_chembl_id values can lead to SQL syntax errors at execution time or, if used with an execution context that accepts multiple statements, could enable SQL injection. To avoid those risks, validate assay_chembl_id against expected ChEMBL assay identifier formats or use a parameterized query at execution time instead of this string when handling untrusted input. At execution time, the query will fail if the required tables or columns (ASSAYS, ACTIVITIES, MOLECULE_DICTIONARY, COMPOUND_STRUCTURES and the referenced columns) are not present in the SQLite database or if the database schema differs from the standard ChEMBL SQLite dumps that chembl_downloader provides.
    
    Returns:
        str: A multiline SQL SELECT statement (as a Python string) that, when executed against a standard ChEMBL SQLite dump, returns canonical SMILES, the molecule ChEMBL identifier, and standardized activity type, relation, value, and units for the specified assay. The string does not execute the query and does not append a terminating semicolon.
    """
    from chembl_downloader.queries import get_assay_sql
    return get_assay_sql(assay_chembl_id)


################################################################################
# Source: chembl_downloader.queries.get_document_molecule_sql
# File: chembl_downloader/queries.py
# Category: fix_docstring
# Reason: Schema parsing failed: Cannot generate JSON schema for get_document_molecule_sql because the docstring has no description for the argument 'document_chembl_id'
################################################################################

def chembl_downloader_queries_get_document_molecule_sql(document_chembl_id: str):
    """Get the SQL query string that returns all molecules mentioned in a ChEMBL document.
    
    This function constructs and returns a SQL SELECT statement (as a Python str) designed for use against the ChEMBL SQLite dump managed by this package. The returned SQL joins the DOCS, COMPOUND_RECORDS, MOLECULE_DICTIONARY, and COMPOUND_STRUCTURES tables to produce one row per distinct molecule mentioned in the specified document. The query selects the ChEMBL molecule identifier (MOLECULE_DICTIONARY.chembl_id), the human-readable compound name (COMPOUND_RECORDS.compound_name), and the machine-readable canonical SMILES string (COMPOUND_STRUCTURES.canonical_smiles). Typical downstream uses (as shown in the project README) include passing the returned string to chembl_downloader.query() or executing it via a connection/cursor to load results into a pandas.DataFrame or to process with RDKit for cheminformatics workflows (e.g., fingerprinting, substructure search).
    
    Behavior and important details:
    - The query uses SELECT DISTINCT to avoid duplicate molecule rows when the same molecule appears multiple times in the document.
    - The WHERE clause compares DOCS.chembl_id to the provided document_chembl_id value; the value is interpolated directly into the returned SQL and is placed inside single quotes in the WHERE clause.
    - This function does not execute the SQL; it only returns the SQL string. There are no side effects such as network or filesystem access.
    - The function assumes the standard ChEMBL SQLite schema with the tables DOCS, COMPOUND_RECORDS, MOLECULE_DICTIONARY, and COMPOUND_STRUCTURES present. The package README notes that most ChEMBL versions are compatible but that very early SQLite dumps may have caveats; callers should ensure they are querying against a compatible ChEMBL SQLite file.
    - Because the document identifier is interpolated into the SQL string, callers should ensure the provided document_chembl_id is a valid ChEMBL document identifier (a str matching the DOCS.chembl_id values in the database, e.g., "CHEMBLXXXX") and comes from a trusted source or is properly sanitized. If an identifier contains single quotes or other special characters, the produced SQL may be syntactically invalid or may behave unexpectedly; this also implies a risk of SQL injection if untrusted input is provided. For untrusted input, prefer using parameterized queries at execution time rather than relying on this helper to perform escaping.
    - If the specified document_chembl_id does not exist in the database, executing the returned query will produce an empty result set (zero rows) rather than raising an error.
    
    Args:
        document_chembl_id (str): The ChEMBL document identifier to filter by. This is the value compared to DOCS.chembl_id in the SQLite dump and identifies which document's referenced molecules will be returned. It must be supplied as a Python str and should match the identifiers used in the target ChEMBL SQLite file.
    
    Returns:
        str: A SQL query string that, when executed against a ChEMBL SQLite database with the expected schema, returns distinct rows of molecules mentioned in the specified document. Each row contains MOLECULE_DICTIONARY.chembl_id, COMPOUND_RECORDS.compound_name, and COMPOUND_STRUCTURES.canonical_smiles.
    """
    from chembl_downloader.queries import get_document_molecule_sql
    return get_document_molecule_sql(document_chembl_id)


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
