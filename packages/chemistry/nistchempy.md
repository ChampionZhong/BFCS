# nistchempy

[🔙 Back to Main Repo](../../../README.md) | [🔗 Original Repo](https://github.com/IvanChernyshov/NistChemPy)

![Tool Count](https://img.shields.io/badge/Agent_Tools-1-blue?style=flat-square)
![Category](https://img.shields.io/badge/Category-Chemistry-green?style=flat-square)
![Status](https://img.shields.io/badge/Import_Test-Passed-success?style=flat-square)

## 📖 Overview

nistchempy is an unofficial Python API and scraper for the NIST Chemistry WebBook that automates compound searching (including bypassing the 400-result limit) and extracts properties, coordinates, spectra (IR/THz/MS/UV‑Vis), and gas chromatography data.

> **Note**: This documentation lists the **agent-ready wrapper functions** generated for this package. These functions have been strictly typed, docstring-enhanced, and tested for import stability within a standardized Apptainer environment.

## 🛠️ Available Agent Tools

Below is the list of **1** functions optimized for LLM tool-use.

| **Tool Name (Wrapper)**   | **Source**          | **File Path**     | **Arguments (Type)**        | **Description**                |
| ------------------------- | ------------------- | ----------------- | --------------------------- | ------------------------------ |
| `nistchempy_requests_fix_html` | `nistchempy.requests.fix_html` | `nistchempy/requests.py` | `html: str` | `Fixes common, known typos in HTML source returned by the NIST Chemistry WebBook and returns a cleaned HTML string for use by NistChemPy parsing routines.` |

## ⚖️ License

Original Code License: MIT

Wrapper Code & Documentation: Apache-2.0

*This file was automatically generated on February 26, 2026.*
