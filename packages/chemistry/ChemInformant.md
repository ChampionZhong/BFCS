# ChemInformant

[🔙 Back to Main Repo](../../../README.md) | [🔗 Original Repo](https://github.com/HzaCode/ChemInformant)

![Tool Count](https://img.shields.io/badge/Agent_Tools-5-blue?style=flat-square)
![Category](https://img.shields.io/badge/Category-Chemistry-green?style=flat-square)
![Status](https://img.shields.io/badge/Import_Test-Passed-success?style=flat-square)

## 📖 Overview

ChemInformant is a Python data acquisition engine for cheminformatics that provides robust, batch-friendly access to chemical databases (e.g., PubChem) to retrieve compounds, structures, and molecular properties for scientific workflows.

> **Note**: This documentation lists the **agent-ready wrapper functions** generated for this package. These functions have been strictly typed, docstring-enhanced, and tested for import stability within a standardized Apptainer environment.

## 🛠️ Available Agent Tools

Below is the list of **5** functions optimized for LLM tool-use.

| **Tool Name (Wrapper)**   | **Source**          | **File Path**     | **Arguments (Type)**        | **Description**                |
| ------------------------- | ------------------- | ----------------- | --------------------------- | ------------------------------ |
| `ChemInformant_api_helpers_get_batch_properties` | `ChemInformant.api_helpers.get_batch_properties` | `ChemInformant/api_helpers.py` | `cids: list[int], props: list[str]` | `Fetches multiple properties for a batch of PubChem CIDs in a single, paginated request and returns the raw CamelCase property dictionaries for each requested CID. This function is the low-level engine used by the higher-level get_properties() convenience API. It is intended for efficient bulk property retrieval from PubChem and performs the following domain-relevant tasks: constructs PubChem PUG-REST property requests for many compound IDs (CIDs), automatically follows PubChem pagination using the returned ListKey for large batches (>1000 compounds), and delegates network reliability behavior (rate limiting, retries, and any configured caching) to the internal helper _fetch_with_ratelimit_and_retry. Returned property dictionaries preserve PubChem's original CamelCase property names (for example, "MolecularWeight", "XLogP", "CanonicalSMILES") and include the "CID" key when available.` |
| `ChemInformant_api_helpers_get_cas_for_cid` | `ChemInformant.api_helpers.get_cas_for_cid` | `ChemInformant/api_helpers.py` | `cid: int` | `ChemInformant.api_helpers.get_cas_for_cid fetches the primary CAS Registry Number for a single PubChem CID by querying the PUG-View JSON record for that compound and parsing the "Names and Identifiers" -> "Other Identifiers" -> "CAS" sections.` |
| `ChemInformant_api_helpers_get_cids_by_name` | `ChemInformant.api_helpers.get_cids_by_name` | `ChemInformant/api_helpers.py` | `name: str` | `Fetch PubChem Compound IDs (CIDs) that match a given chemical name. This helper queries the PubChem PUG REST "compound/name/.../cids/JSON" endpoint to resolve a human-readable chemical name into one or more PubChem integer CIDs. It is used internally by the library's higher-level functions (for example, get_properties()) to translate user-supplied names into PubChem identifiers suitable for subsequent property lookups, batch queries, and database integration. The function calls an internal network helper that implements persistent caching, rate-limiting, and automatic retry logic; the network helper may perform sleeps, retries, and cache lookups as part of a normal call.` |
| `ChemInformant_api_helpers_get_cids_by_smiles` | `ChemInformant.api_helpers.get_cids_by_smiles` | `ChemInformant/api_helpers.py` | `smiles: str` | `Fetches PubChem Compound IDs (CIDs) that match a provided SMILES string using the PubChem PUG REST API and the package's network helpers. This function performs a network lookup against the PubChem PUG REST endpoint /compound/smiles/{smiles}/cids/JSON (the base URL is taken from the module-level PUBCHEM_API_BASE). The input SMILES string is percent-encoded before the request. The call is executed via the library's internal fetch helper (_fetch_with_ratelimit_and_retry), so the request benefits from the package's rate-limiting, retry logic, and persistent caching layers described in the project README. This function may return multiple integer CIDs when PubChem exposes several records for the same structural representation (for example, stereoisomers, different tautomeric or protonation states, or alternate registrations). It is intended primarily as an internal helper for SMILES-to-CID resolution and is used by get_properties() to map SMILES inputs to PubChem identifiers; end users are generally encouraged to call get_properties() for full property retrieval and batch handling.` |
| `ChemInformant_api_helpers_get_synonyms_for_cid` | `ChemInformant.api_helpers.get_synonyms_for_cid` | `ChemInformant/api_helpers.py` | `cid: int` | `ChemInformant.api_helpers.get_synonyms_for_cid: Retrieve all known synonyms (alternative names) for a PubChem compound identified by its CID. This function performs an HTTP request to the PubChem PUG REST synonyms endpoint for the given compound identifier and returns the list of names associated with that compound. In the ChemInformant data-acquisition workflow, this is used to normalize and enrich chemical records (for example, mapping brand names, systematic names, and common names to a single PubChem CID), to populate convenience functions such as get_synonyms() and to support downstream tasks in drug discovery, QSAR modeling, and compound annotation pipelines. The implementation constructs a URL of the form "{PUBCHEM_API_BASE}/compound/cid/{cid}/synonyms/JSON" and delegates network reliability concerns (rate limiting, retries, and caching) to the internal helper _fetch_with_ratelimit_and_retry; the function then parses the expected JSON structure InformationList -> Information -> [0] -> Synonym.` |

## ⚖️ License

Original Code License: MIT

Wrapper Code & Documentation: Apache-2.0

*This file was automatically generated on February 26, 2026.*
