# drugbank_downloader

[🔙 Back to Main Repo](../../../README.md) | [🔗 Original Repo](https://github.com/cthoyt/drugbank_downloader)

![Tool Count](https://img.shields.io/badge/Agent_Tools-1-blue?style=flat-square)
![Category](https://img.shields.io/badge/Category-Medicine-green?style=flat-square)
![Status](https://img.shields.io/badge/Import_Test-Passed-success?style=flat-square)

## 📖 Overview

`drugbank_downloader` is a Python utility for programmatically downloading specific DrugBank releases (e.g., the full XML ZIP) with your credentials and caching them locally (via `pystow`) to avoid re-downloading.

> **Note**: This documentation lists the **agent-ready wrapper functions** generated for this package. These functions have been strictly typed, docstring-enhanced, and tested for import stability within a standardized Apptainer environment.

## 🛠️ Available Agent Tools

Below is the list of **1** functions optimized for LLM tool-use.

| **Tool Name (Wrapper)**   | **Source**          | **File Path**     | **Arguments (Type)**        | **Description**                |
| ------------------------- | ------------------- | ----------------- | --------------------------- | ------------------------------ |
| `drugbank_downloader_version_get_version` | `drugbank_downloader.version.get_version` | `drugbank_downloader/version.py` | `with_git_hash: bool = False` | `drugbank_downloader.version.get_version returns the current drugbank_downloader package version string and, optionally, a git commit hash appended to that version to provide more precise provenance.` |

## ⚖️ License

Original Code License: MIT

Wrapper Code & Documentation: Apache-2.0

*This file was automatically generated on February 26, 2026.*
