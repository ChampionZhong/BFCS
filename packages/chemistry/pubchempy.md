# pubchempy

[🔙 Back to Main Repo](../../../README.md) | [🔗 Original Repo](https://github.com/mcs07/PubChemPy)

![Tool Count](https://img.shields.io/badge/Agent_Tools-2-blue?style=flat-square)
![Category](https://img.shields.io/badge/Category-Chemistry-green?style=flat-square)
![Status](https://img.shields.io/badge/Import_Test-Passed-success?style=flat-square)

## 📖 Overview

PubChemPy is a Python library for programmatically querying the PubChem database to search and retrieve chemical compounds and properties, standardize structures, convert formats, and generate chemical depictions.

> **Note**: This documentation lists the **agent-ready wrapper functions** generated for this package. These functions have been strictly typed, docstring-enhanced, and tested for import stability within a standardized Apptainer environment.

## 🛠️ Available Agent Tools

Below is the list of **2** functions optimized for LLM tool-use.

| **Tool Name (Wrapper)**   | **Source**          | **File Path**     | **Arguments (Type)**        | **Description**                |
| ------------------------- | ------------------- | ----------------- | --------------------------- | ------------------------------ |
| `pubchempy_deprecated` | `pubchempy.deprecated` | `pubchempy.py` | `message: str` | `pubchempy.deprecated: Factory for a decorator that marks a function as deprecated in PubChemPy and emits a runtime warning when the deprecated function is called. This is intended for use by PubChemPy maintainers and contributors to signal that a public API (for example a function that performs PubChem lookups, conversions, or property retrievals) is obsolete and that callers should migrate to an alternative. The returned decorator wraps the original function, preserves its metadata, issues a PubChemPyDeprecationWarning via warnings.warn with stacklevel=2 at each call, and then invokes the original function so normal behavior and return values are preserved.` |
| `pubchempy_get_all_sources` | `pubchempy.get_all_sources` | `pubchempy.py` | `domain: str = "substance"` | `Return a list of all current depositors (source names) for the specified PubChem domain (for example, substances or assays). This function is a thin helper used by PubChemPy to query the PubChem REST API "sources" endpoint for metadata about who submitted records; it is useful in workflows that need to audit, filter, or cross-reference contributors to PubChem datasets.` |

## ⚖️ License

Original Code License: MIT

Wrapper Code & Documentation: Apache-2.0

*This file was automatically generated on February 26, 2026.*
