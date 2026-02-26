# mordred

[🔙 Back to Main Repo](../../../README.md) | [🔗 Original Repo](https://github.com/mordred-descriptor/mordred)

![Tool Count](https://img.shields.io/badge/Agent_Tools-1-blue?style=flat-square)
![Category](https://img.shields.io/badge/Category-Chemistry-green?style=flat-square)
![Status](https://img.shields.io/badge/Import_Test-Passed-success?style=flat-square)

## 📖 Overview

Mordred is a molecular descriptor calculator for cheminformatics that uses RDKit to compute a large set of 2D and 3D descriptors for molecules from SMILES/structure input via a Python API or command-line tool.

> **Note**: This documentation lists the **agent-ready wrapper functions** generated for this package. These functions have been strictly typed, docstring-enhanced, and tested for import stability within a standardized Apptainer environment.

## 🛠️ Available Agent Tools

Below is the list of **1** functions optimized for LLM tool-use.

| **Tool Name (Wrapper)**   | **Source**          | **File Path**     | **Arguments (Type)**        | **Description**                |
| ------------------------- | ------------------- | ----------------- | --------------------------- | ------------------------------ |
| `mordred__util_to_ordinal` | `mordred._util.to_ordinal` | `mordred/_util.py` | `n: int` | `Convert an integer to a short English ordinal string used for human-readable labels. This utility function in mordred._util is a small pure helper used by the mordred molecular descriptor calculator to produce human-friendly ordinal labels (for example when formatting descriptor positions, log messages, CLI output, or report fields). It maps the integers 1, 2 and 3 to the English words "first", "second" and "third" respectively; for any other integer it returns a numeric ordinal using the pattern "<n>-th" (for example "4-th" or "104-th"). The function has no side effects and does not modify external state.` |

## ⚖️ License

Original Code License: BSD-3-Clause

Wrapper Code & Documentation: Apache-2.0

*This file was automatically generated on February 26, 2026.*
