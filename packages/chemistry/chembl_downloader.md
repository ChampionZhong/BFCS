# chembl_downloader

[🔙 Back to Main Repo](../../../README.md) | [🔗 Original Repo](https://github.com/cthoyt/chembl-downloader)

![Tool Count](https://img.shields.io/badge/Agent_Tools-2-blue?style=flat-square)
![Category](https://img.shields.io/badge/Category-Chemistry-green?style=flat-square)
![Status](https://img.shields.io/badge/Import_Test-Passed-success?style=flat-square)

## 📖 Overview

`chembl_downloader` provides reproducible utilities to automatically download, cache, extract, and open/query versioned ChEMBL database dumps (e.g., the SQLite dump) in Python.

> **Note**: This documentation lists the **agent-ready wrapper functions** generated for this package. These functions have been strictly typed, docstring-enhanced, and tested for import stability within a standardized Apptainer environment.

## 🛠️ Available Agent Tools

Below is the list of **2** functions optimized for LLM tool-use.

| **Tool Name (Wrapper)**   | **Source**          | **File Path**     | **Arguments (Type)**        | **Description**                |
| ------------------------- | ------------------- | ----------------- | --------------------------- | ------------------------------ |
| `chembl_downloader_queries_get_assay_sql` | `chembl_downloader.queries.get_assay_sql` | `chembl_downloader/queries.py` | `assay_chembl_id: str` | `Get the SQL query string to retrieve molecular structures and standardized activity measurements for a single ChEMBL assay.` |
| `chembl_downloader_queries_get_document_molecule_sql` | `chembl_downloader.queries.get_document_molecule_sql` | `chembl_downloader/queries.py` | `document_chembl_id: str` | `Get the SQL query string that returns all molecules mentioned in a ChEMBL document. This function constructs and returns a SQL SELECT statement (as a Python str) designed for use against the ChEMBL SQLite dump managed by this package. The returned SQL joins the DOCS, COMPOUND_RECORDS, MOLECULE_DICTIONARY, and COMPOUND_STRUCTURES tables to produce one row per distinct molecule mentioned in the specified document. The query selects the ChEMBL molecule identifier (MOLECULE_DICTIONARY.chembl_id), the human-readable compound name (COMPOUND_RECORDS.compound_name), and the machine-readable canonical SMILES string (COMPOUND_STRUCTURES.canonical_smiles). Typical downstream uses (as shown in the project README) include passing the returned string to chembl_downloader.query() or executing it via a connection/cursor to load results into a pandas.DataFrame or to process with RDKit for cheminformatics workflows (e.g., fingerprinting, substructure search). Behavior and important details: - The query uses SELECT DISTINCT to avoid duplicate molecule rows when the same molecule appears multiple times in the document. - The WHERE clause compares DOCS.chembl_id to the provided document_chembl_id value; the value is interpolated directly into the returned SQL and is placed inside single quotes in the WHERE clause. - This function does not execute the SQL; it only returns the SQL string. There are no side effects such as network or filesystem access. - The function assumes the standard ChEMBL SQLite schema with the tables DOCS, COMPOUND_RECORDS, MOLECULE_DICTIONARY, and COMPOUND_STRUCTURES present. The package README notes that most ChEMBL versions are compatible but that very early SQLite dumps may have caveats; callers should ensure they are querying against a compatible ChEMBL SQLite file. - Because the document identifier is interpolated into the SQL string, callers should ensure the provided document_chembl_id is a valid ChEMBL document identifier (a str matching the DOCS.chembl_id values in the database, e.g., "CHEMBLXXXX") and comes from a trusted source or is properly sanitized. If an identifier contains single quotes or other special characters, the produced SQL may be syntactically invalid or may behave unexpectedly; this also implies a risk of SQL injection if untrusted input is provided. For untrusted input, prefer using parameterized queries at execution time rather than relying on this helper to perform escaping. - If the specified document_chembl_id does not exist in the database, executing the returned query will produce an empty result set (zero rows) rather than raising an error.` |

## ⚖️ License

Original Code License: MIT

Wrapper Code & Documentation: Apache-2.0

*This file was automatically generated on February 26, 2026.*
