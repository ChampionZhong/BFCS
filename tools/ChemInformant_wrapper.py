"""
Regenerated Google-style docstrings for module 'ChemInformant'.
README source: others/readme/ChemInformant/README.md
Generated at: 2025-12-02T00:45:11.227343Z

Total functions: 5
"""


################################################################################
# Source: ChemInformant.api_helpers.get_batch_properties
# File: ChemInformant/api_helpers.py
# Category: valid
################################################################################

def ChemInformant_api_helpers_get_batch_properties(cids: list[int], props: list[str]):
    """Fetches multiple properties for a batch of PubChem CIDs in a single, paginated request and returns the raw CamelCase property dictionaries for each requested CID.
    
    This function is the low-level engine used by the higher-level get_properties() convenience API. It is intended for efficient bulk property retrieval from PubChem and performs the following domain-relevant tasks: constructs PubChem PUG-REST property requests for many compound IDs (CIDs), automatically follows PubChem pagination using the returned ListKey for large batches (>1000 compounds), and delegates network reliability behavior (rate limiting, retries, and any configured caching) to the internal helper _fetch_with_ratelimit_and_retry. Returned property dictionaries preserve PubChem's original CamelCase property names (for example, "MolecularWeight", "XLogP", "CanonicalSMILES") and include the "CID" key when available.
    
    Args:
        cids (list[int]): List of PubChem compound IDs (CIDs) to query. Each element must be an integer CID as used by PubChem. The function guarantees that the output dictionary includes an entry for every CID in this list; if a CID has no data or its lookup fails, that CID maps to an empty dictionary in the result. If this list is empty, the function short-circuits and returns an empty dictionary immediately (no network requests are made).
        props (list[str]): List of property names to request from PubChem, provided as exact PubChem API property names in CamelCase (for example, ["MolecularWeight", "XLogP", "CanonicalSMILES"]). These strings must match PubChem's property names exactly; the function does not perform snake_case-to-CamelCase translation. If this list is empty, the function short-circuits and returns an empty dictionary immediately.
    
    Returns:
        dict[int, dict[str, Any]]: A dictionary mapping each requested CID (int) to a dictionary of properties as returned by PubChem. Each inner dictionary contains keys that are the original CamelCase property names requested (and typically a "CID" key when PubChem returned it). For CIDs with no returned data or for CIDs whose lookup failed on the server side, the value is an empty dictionary. If the initial network fetch returns a non-dictionary response (indicating an unexpected API failure), the function returns a dictionary with every input CID present and each mapping to an empty dictionary.
    
    Behavior, side effects, and failure modes:
        - Network requests: The function constructs PubChem PUG-REST URLs using the global PUBCHEM_API_BASE and calls _fetch_with_ratelimit_and_retry to perform each HTTP request. That helper implements rate limiting, retry logic, and may use persistent caching per the package configuration; any exceptions raised by that helper (for example, after retry exhaustion or for fatal HTTP errors) will propagate to the caller unless handled externally.
        - Pagination: For large batches, PubChem may return a ListKey in the response. When a ListKey is present, the function will loop and fetch subsequent pages until no ListKey remains. Each pagination event prints a short informational message to standard error indicating the ListKey being followed (this is a deliberate side effect used for runtime visibility).
        - Input validation: The function does not coerce input types; cids should be integers and props should be exact CamelCase property strings. Supplying non-integer values in cids or misspelled property names in props will either lead to empty results for those entries or API-level errors propagated from the network helper.
        - Empty inputs: If either cids or props is empty, the function returns {} immediately and does not perform network activity.
        - Partial failures: If a paginated request fails (the paginated _fetch_with_ratelimit_and_retry call returns a non-dict), pagination stops and any successfully collected properties up to that point are used; CIDs with no collected data are returned as empty dictionaries.
        - Output format: The function returns the raw PubChem-style property dictionaries (CamelCase). Users who prefer snake_case, validated fields, or higher-level DataFrame/SQL outputs should call the public get_properties() API, which wraps this function and converts results to snake_case and/or structured outputs.
    
    Notes:
        - Intended for use in high-throughput workflows such as batch property retrieval for QSAR, virtual screening, and dataset assembly where preserving PubChem's original field names and efficient network usage are required.
        - This function is the internal backend of get_properties(); typical end users should call get_properties() for convenience, automatic snake_case conversion, and higher-level output formats.
    """
    from ChemInformant.api_helpers import get_batch_properties
    return get_batch_properties(cids, props)


################################################################################
# Source: ChemInformant.api_helpers.get_cas_for_cid
# File: ChemInformant/api_helpers.py
# Category: fix_docstring
# Reason: Schema parsing failed: ("Couldn't parse this type hint, likely due to a custom class or object: ", str | None)
################################################################################

def ChemInformant_api_helpers_get_cas_for_cid(cid: int):
    """ChemInformant.api_helpers.get_cas_for_cid fetches the primary CAS Registry Number for a single PubChem CID by querying the PUG-View JSON record for that compound and parsing the "Names and Identifiers" -> "Other Identifiers" -> "CAS" sections.
    
    Args:
        cid (int): The PubChem compound identifier (CID) to look up. In the ChemInformant workflow this integer identifies a single compound record in the PubChem database; callers supply a validated CID (for example, as produced or validated by get_properties() or by convenience lookup functions). This function constructs a PUG-View URL for the given CID and performs network I/O to retrieve the detailed compound record.
    
    Returns:
        str | None: The first CAS Registry Number found in the compound's PUG-View record as a hyphenated string (for example, "50-78-2"). Practically, the returned CAS is the canonical Chemical Abstracts Service identifier used to map PubChem entries to external chemical registries and literature. If the PUG-View record does not contain a CAS section, the "CAS" section has no usable StringWithMarkup entry, or the parsed value is not a string, the function returns None to indicate no CAS could be extracted.
    
    Behavior and side effects:
        This function issues a network request to PubChem's PUG-View endpoint (constructed as PUG_VIEW_BASE/compound/{cid}/JSON) and therefore performs I/O. The implementation delegates HTTP reliability behavior (rate limiting, retries, and any caching) to the internal helper _fetch_with_ratelimit_and_retry; callers should assume the call may be slower than property-API queries because full PUG-View compound records are larger and more deeply nested. The function inspects the returned JSON for the nested sections "Record" -> "Section" where "TOCHeading" equals "Names and Identifiers", then within that for "Other Identifiers" and finally for a "CAS" subsection; it returns the first StringWithMarkup[0]["String"] value if present and a string.
    
    Failure modes and propagation:
        If the internal fetch helper raises network-, HTTP-, or parsing-related exceptions, those exceptions are propagated to the caller (they are not swallowed here). If the response JSON structure differs from the expected nesting or the CAS entry is missing or malformed, the function will return None rather than raising. Because the function returns only the first CAS entry it finds, compounds with multiple CAS entries will yield only the primary/first-listed identifier.
    
    Notes:
        This helper is used internally by the higher-level get_properties() engine and the convenience function get_cas() to provide CAS lookup when requested; it is intended for single-CID lookups and may be used in batch workflows by calling code that manages parallelism, caching, and rate control.
    """
    from ChemInformant.api_helpers import get_cas_for_cid
    return get_cas_for_cid(cid)


################################################################################
# Source: ChemInformant.api_helpers.get_cids_by_name
# File: ChemInformant/api_helpers.py
# Category: fix_docstring
# Reason: Schema parsing failed: ("Couldn't parse this type hint, likely due to a custom class or object: ", list[int] | None)
################################################################################

def ChemInformant_api_helpers_get_cids_by_name(name: str):
    """Fetch PubChem Compound IDs (CIDs) that match a given chemical name.
    
    This helper queries the PubChem PUG REST "compound/name/.../cids/JSON" endpoint to resolve a human-readable chemical name into one or more PubChem integer CIDs. It is used internally by the library's higher-level functions (for example, get_properties()) to translate user-supplied names into PubChem identifiers suitable for subsequent property lookups, batch queries, and database integration. The function calls an internal network helper that implements persistent caching, rate-limiting, and automatic retry logic; the network helper may perform sleeps, retries, and cache lookups as part of a normal call.
    
    Args:
        name (str): The chemical name to search for on PubChem (for example, "aspirin" or "acetylsalicylic acid"). This argument is treated as a plain string and is URL-quoted before being sent to the PubChem REST API. Typical callers provide common names, trade names, or IUPAC names; do not pass a CID or a SMILES string to this function (those are handled elsewhere in the library).
    
    Returns:
        list[int] | None: A list of integer PubChem CIDs that match the provided name when the API returns a valid IdentifierList.CID array. Multiple integers can be returned when the name is ambiguous or maps to multiple stereoisomers, salts, or different registry entries (for example, some common names map to isomers or mixtures). Returns None when no matching CIDs are found, when the API response does not contain the expected IdentifierList.CID list, or when the internal fetch helper returns a non-dictionary result. Note that transient network errors may be retried by the internal helper; unrecoverable HTTP/network exceptions from the underlying request layer may propagate to the caller if retries are exhausted. The function has no other side effects beyond making the network request and consulting the request cache.
    """
    from ChemInformant.api_helpers import get_cids_by_name
    return get_cids_by_name(name)


################################################################################
# Source: ChemInformant.api_helpers.get_cids_by_smiles
# File: ChemInformant/api_helpers.py
# Category: fix_docstring
# Reason: Schema parsing failed: ("Couldn't parse this type hint, likely due to a custom class or object: ", list[int] | None)
################################################################################

def ChemInformant_api_helpers_get_cids_by_smiles(smiles: str):
    """Fetches PubChem Compound IDs (CIDs) that match a provided SMILES string using the PubChem PUG REST API and the package's network helpers.
    
    This function performs a network lookup against the PubChem PUG REST endpoint /compound/smiles/{smiles}/cids/JSON (the base URL is taken from the module-level PUBCHEM_API_BASE). The input SMILES string is percent-encoded before the request. The call is executed via the library's internal fetch helper (_fetch_with_ratelimit_and_retry), so the request benefits from the package's rate-limiting, retry logic, and persistent caching layers described in the project README. This function may return multiple integer CIDs when PubChem exposes several records for the same structural representation (for example, stereoisomers, different tautomeric or protonation states, or alternate registrations). It is intended primarily as an internal helper for SMILES-to-CID resolution and is used by get_properties() to map SMILES inputs to PubChem identifiers; end users are generally encouraged to call get_properties() for full property retrieval and batch handling.
    
    Args:
        smiles (str): The input SMILES string that describes the molecule's connectivity and stereochemistry. This must be a standard SMILES string (for example, "CC(=O)OC1=CC=CC=C1C(=O)O" for aspirin). The function does not perform SMILES validation beyond sending the encoded string to PubChem; malformed SMILES may lead to no matches or an error response from the API.
    
    Returns:
        list[int] | None: A list of PubChem CIDs (integers) that PubChem reports for the given SMILES when a valid JSON IdentifierList/CID array is returned. Returns None if no matching CIDs are found, if the API response does not contain an IdentifierList/CID array, or if an unexpected/non-dict response or network failure occurs (note that network failures may be subject to the library's retry logic implemented in _fetch_with_ratelimit_and_retry). The caller should treat None as "no resolvable CID(s) available" and handle it accordingly (for example, skipping the input or logging a status in batch workflows).
    """
    from ChemInformant.api_helpers import get_cids_by_smiles
    return get_cids_by_smiles(smiles)


################################################################################
# Source: ChemInformant.api_helpers.get_synonyms_for_cid
# File: ChemInformant/api_helpers.py
# Category: valid
################################################################################

def ChemInformant_api_helpers_get_synonyms_for_cid(cid: int):
    """ChemInformant.api_helpers.get_synonyms_for_cid: Retrieve all known synonyms (alternative names) for a PubChem compound identified by its CID.
    
    This function performs an HTTP request to the PubChem PUG REST synonyms endpoint for the given compound identifier and returns the list of names associated with that compound. In the ChemInformant data-acquisition workflow, this is used to normalize and enrich chemical records (for example, mapping brand names, systematic names, and common names to a single PubChem CID), to populate convenience functions such as get_synonyms() and to support downstream tasks in drug discovery, QSAR modeling, and compound annotation pipelines. The implementation constructs a URL of the form "{PUBCHEM_API_BASE}/compound/cid/{cid}/synonyms/JSON" and delegates network reliability concerns (rate limiting, retries, and caching) to the internal helper _fetch_with_ratelimit_and_retry; the function then parses the expected JSON structure InformationList -> Information -> [0] -> Synonym.
    
    Args:
        cid (int): The PubChem Compound ID (CID) to look up. This integer uniquely identifies a compound in PubChem and is the primary key used by ChemInformant to fetch authoritative metadata. The caller should supply a valid PubChem CID; when an invalid or non-existent CID is supplied, the function will not raise for that condition but will typically return an empty list (see failure modes below).
    
    Returns:
        list[str]: A list of synonym strings for the requested CID. The list is ordered with the most common/preferred names first when provided by PubChem; the first element is typically the preferred/common name. If the PubChem response is missing, malformed, or contains no synonyms, the function returns an empty list. The returned list is safe to use for display, downstream identifier matching, dataset enrichment, or as input to other ChemInformant convenience functions.
    
    Behavior and side effects:
        This function issues a network request to the PubChem REST API and therefore may incur network latency. It relies on the library's internal rate-limiting, retry, and caching mechanisms (via _fetch_with_ratelimit_and_retry) to improve robustness and avoid transient failures; these helpers may introduce blocking waits due to rate limiting or retries. Because the function performs I/O, it may be slower than pure in-memory operations and should be used accordingly in high-throughput pipelines (batching and the library's bulk endpoints are preferred for large jobs).
    
    Failure modes and error handling:
        The function returns an empty list when no synonyms are found or when the returned JSON does not match the expected structure (for example, missing InformationList/Information/Synonym keys). Network-related errors or unrecoverable HTTP client exceptions may propagate from the underlying fetch helper if they are not resolved by the retry logic; callers that require strict failure signaling should handle these exceptions at a higher level. The function does not validate the semantic correctness of returned names; downstream validation or deduplication may be necessary when integrating synonyms into datasets.
    
    Practical significance in the ChemInformant domain:
        Synonyms obtained from this function are used to standardize names across datasets, improve matching between proprietary inventories and public databases, populate human-readable labels in analysis-ready DataFrames, and support features such as CLI lookups and SQL exports. Because many cheminformatics workflows depend on consistent naming (e.g., merging external assay results with PubChem annotations), this function is a lightweight but important step in the ChemInformant extraction and enrichment pipeline.
    """
    from ChemInformant.api_helpers import get_synonyms_for_cid
    return get_synonyms_for_cid(cid)


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
