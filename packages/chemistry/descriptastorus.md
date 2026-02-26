# descriptastorus

[🔙 Back to Main Repo](../../../README.md) | [🔗 Original Repo](https://github.com/bp-kelley/descriptastorus)

![Tool Count](https://img.shields.io/badge/Agent_Tools-3-blue?style=flat-square)
![Category](https://img.shields.io/badge/Category-Chemistry-green?style=flat-square)
![Status](https://img.shields.io/badge/Import_Test-Passed-success?style=flat-square)

## 📖 Overview

descriptastorus is a cheminformatics utility that builds and serves fast, indexed stores of RDKit-based molecular descriptors and properties for machine-learning workflows, with tools to generate, append, and validate descriptor data for new molecules.

> **Note**: This documentation lists the **agent-ready wrapper functions** generated for this package. These functions have been strictly typed, docstring-enhanced, and tested for import stability within a standardized Apptainer environment.

## 🛠️ Available Agent Tools

Below is the list of **3** functions optimized for LLM tool-use.

| **Tool Name (Wrapper)**   | **Source**          | **File Path**     | **Arguments (Type)**        | **Description**                |
| ------------------------- | ------------------- | ----------------- | --------------------------- | ------------------------------ |
| `descriptastorus_MolFileIndex_MakeSDFIndex` | `descriptastorus.MolFileIndex.MakeSDFIndex` | `descriptastorus/MolFileIndex.py` | `filename: str, dbdir: str` | `descriptastorus.MolFileIndex.MakeSDFIndex creates an on-disk index for an SDF (Structure-Data File) so that individual molecules in the SDF can be accessed randomly by row number. This function is used by the DescriptaStorus project to provide fast random access to indexed molecule files (see README: "fast random access to indexed molecule files"). The index maps sequential row numbers to byte offsets in the SDF file and stores those offsets in a raw DescriptaStorus store located at the provided dbdir. The returned MolFileIndex object uses SDFNameGetter as the name extraction function so the index can be used with the rest of the DescriptaStorus API (for example, to call getMol, getName, or to iterate descriptors).` |
| `descriptastorus_descriptors_DescriptorGenerator_MakeGenerator` | `descriptastorus.descriptors.DescriptorGenerator.MakeGenerator` | `descriptastorus/descriptors/DescriptorGenerator.py` | `generator_names: list` | `descriptastorus.descriptors.DescriptorGenerator.MakeGenerator creates a combined descriptor generator by looking up one or more named descriptor generator factories in the DescriptorGenerator.REGISTRY and returning either a single generator or a Container that composes multiple generators. This function is used by consumers of the descriptastorus library (for example, code that needs RDKit2D, Morgan3Counts, or combinations thereof) to obtain a callable descriptor generator that will produce the feature vectors described in the README (the generator.process(smiles) convention where the first element is a boolean success flag and the remaining elements are the descriptor values).` |
| `descriptastorus_descriptors_QED_ads` | `descriptastorus.descriptors.QED.ads` | `descriptastorus/descriptors/QED.py` | `x: float, a: float, b: float, c: float, d: float, e: float, f: float, dmax: float` | `descriptastorus.descriptors.QED.ads computes an asymmetric double-sigmoid (ADS) mathematical transform used in descriptor generation to convert a single raw scalar property into a scaled contribution value for descriptor scoring and storage in DescriptaStorus. This function implements the ADS expression used by descriptor modules (for example, QED-style component scoring) to produce a smoothly varying, bounded-like contribution from an input property. In the DescriptaStorus project this kind of transform is applied when building descriptor rows for storage and subsequent machine learning tasks: it maps a raw property value x through two logistic (sigmoid) components combined with amplitude and offset parameters, and then normalizes by dmax to produce the final scalar that is stored in descriptor arrays.` |

## ⚖️ License

Original Code License: BSD-3-Clause

Wrapper Code & Documentation: Apache-2.0

*This file was automatically generated on February 26, 2026.*
