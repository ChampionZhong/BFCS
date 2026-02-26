# chemprop

[🔙 Back to Main Repo](../../../README.md) | [🔗 Original Repo](https://github.com/chemprop/chemprop)

![Tool Count](https://img.shields.io/badge/Agent_Tools-4-blue?style=flat-square)
![Category](https://img.shields.io/badge/Category-Chemistry-green?style=flat-square)
![Status](https://img.shields.io/badge/Import_Test-Passed-success?style=flat-square)

## 📖 Overview

Chemprop is a Python package for training and using message-passing neural network models to predict molecular (and reaction) properties from chemical structures.

> **Note**: This documentation lists the **agent-ready wrapper functions** generated for this package. These functions have been strictly typed, docstring-enhanced, and tested for import stability within a standardized Apptainer environment.

## 🛠️ Available Agent Tools

Below is the list of **4** functions optimized for LLM tool-use.

| **Tool Name (Wrapper)**   | **Source**          | **File Path**     | **Arguments (Type)**        | **Description**                |
| ------------------------- | ------------------- | ----------------- | --------------------------- | ------------------------------ |
| `chemprop_utils_utils_make_mol` | `chemprop.utils.utils.make_mol` | `chemprop/utils/utils.py` | `smi: str, keep_h: bool = False, add_h: bool = False, ignore_stereo: bool = False, reorder_atoms: bool = False` | `chemprop.utils.utils.make_mol builds an RDKit molecule (Chem.Mol) from a SMILES string for use in Chemprop's molecular featurization and model pipelines. This function parses the SMILES with configurable handling of explicit hydrogens, optional addition of hydrogens, optional removal of stereochemical information, and optional reordering of atoms by atom map numbers. It is used throughout Chemprop to convert string representations of molecules into RDKit Mol objects that downstream message-passing neural network code expects.` |
| `chemprop_utils_v1_to_v2_convert_hyper_parameters_v1_to_v2` | `chemprop.utils.v1_to_v2.convert_hyper_parameters_v1_to_v2` | `chemprop/utils/v1_to_v2.py` | `model_v1_dict: dict` | `chemprop.utils.v1_to_v2.convert_hyper_parameters_v1_to_v2 converts a saved Chemprop v1 model dictionary into a hyper_parameters dictionary formatted for Chemprop v2. This function is used during migration from Chemprop v1 to v2 (see the repository README and v1→v2 transition notes) to transform model hyperparameters, metric/loss identifiers, message-passing block dimensions, aggregation and predictor configuration into the v2 structure expected by the v2 training and inference codepaths.` |
| `chemprop_utils_v1_to_v2_convert_model_dict_v1_to_v2` | `chemprop.utils.v1_to_v2.convert_model_dict_v1_to_v2` | `chemprop/utils/v1_to_v2.py` | `model_v1_dict: dict` | `Converts a Chemprop v1 checkpoint dictionary (as loaded from a .pt file) into the v2 checkpoint dictionary format expected by Chemprop v2 and its PyTorch Lightning-based loading routines. This function is used during migration of trained molecular property prediction models from Chemprop v1 to Chemprop v2 so that v2 code can inspect and restore model parameters and hyperparameters produced by v1.` |
| `chemprop_utils_v1_to_v2_convert_state_dict_v1_to_v2` | `chemprop.utils.v1_to_v2.convert_state_dict_v1_to_v2` | `chemprop/utils/v1_to_v2.py` | `model_v1_dict: dict` | `Converts a saved Chemprop v1 model dictionary (checkpoint) into a Chemprop v2 state dictionary suitable for loading into v2 model code. This utility is used during migration from Chemprop v1 to v2 (see README v1 -> v2 transition notes) for message passing neural networks applied to molecular property prediction; it remaps parameter key names and copies/reshapes certain metadata (for example, per-target output scaling for regression) so that v1 checkpoints can be consumed by v2 code.` |

## ⚖️ License

Original Code License: MIT

Wrapper Code & Documentation: Apache-2.0

*This file was automatically generated on February 26, 2026.*
