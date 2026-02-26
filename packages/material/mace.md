# mace

[🔙 Back to Main Repo](../../../README.md) | [🔗 Original Repo](https://github.com/ACEsuit/mace)

![Tool Count](https://img.shields.io/badge/Agent_Tools-7-blue?style=flat-square)
![Category](https://img.shields.io/badge/Category-Material-green?style=flat-square)
![Status](https://img.shields.io/badge/Import_Test-Passed-success?style=flat-square)

## 📖 Overview

MACE is a PyTorch-based toolkit for training and deploying fast, accurate equivariant message-passing neural network interatomic potentials (force fields) to predict energies and forces for atomistic simulations.

> **Note**: This documentation lists the **agent-ready wrapper functions** generated for this package. These functions have been strictly typed, docstring-enhanced, and tested for import stability within a standardized Apptainer environment.

## 🛠️ Available Agent Tools

Below is the list of **7** functions optimized for LLM tool-use.

| **Tool Name (Wrapper)**   | **Source**          | **File Path**     | **Arguments (Type)**        | **Description**                |
| ------------------------- | ------------------- | ----------------- | --------------------------- | ------------------------------ |
| `mace_calculators_lammps_mliap_mace_timer` | `mace.calculators.lammps_mliap_mace.timer` | `../../../../../usr/local/lib/python3.10/contextlib.py` | `name: str, enabled: bool = True` | `mace.calculators.lammps_mliap_mace.timer: Context manager that measures wall-clock elapsed time for a block of code and emits a logging.info entry with the measured duration in milliseconds. This utility is intended for use in the MACE codebase (for example in LAMMPS/MLIAP calculator integration, training, evaluation, preprocessing, or performance debugging) to label and record how long specific operations take (neighbor list construction, model forward passes, data preprocessing, etc.). It uses Python's high-resolution time.perf_counter() to measure elapsed real time and formats the log message as "Timer - {name}: {elapsed_ms:.3f} ms".` |
| `mace_tools_run_train_utils_combine_datasets` | `mace.tools.run_train_utils.combine_datasets` | `mace/tools/run_train_utils.py` | `datasets: list, head_name: str` | `Combine multiple datasets which might be of different types and return a single object suitable for use by MACE training/evaluation code. This utility is used by the MACE training pipeline (for example in run_train.py and preprocessing scripts) to merge multiple data sources that represent configurations for a given model "head" into one combined dataset object or list. It supports inputs that are Python lists (e.g., lists of raw configuration dicts or ASE Atoms generated during preprocessing) and PyTorch Dataset objects (e.g., preprocessed HDF5-backed Dataset or TensorDataset). The function attempts safe fallbacks when inputs are mixed types, logs informative messages using head_name to aid debugging, and may perform conversions that materialize dataset items into memory.` |
| `mace_tools_scripts_utils_get_config_type_weights` | `mace.tools.scripts_utils.get_config_type_weights` | `mace/tools/scripts_utils.py` | `ct_weights: str` | `Parse a command-line config_type_weights string into a mapping of configuration type names to numeric weights used to weight dataset entries (loss contributions) during MACE training and evaluation.` |
| `mace_tools_tables_utils_custom_key` | `mace.tools.tables_utils.custom_key` | `mace/tools/tables_utils.py` | `key: str` | `mace.tools.tables_utils.custom_key returns a two-element sort key that prioritizes the "train" and "valid" entries when sorting the keys of a data-loader or results dictionary used during MACE training and evaluation. This ensures that the training set and validation set are evaluated (and therefore reported or plotted) before other datasets (for example test or per-config-type results) in scripts such as the training loop (run_train.py / mace_run_train) and evaluation utilities (mace_eval_configs), where deterministic ordering of dataset evaluation and logging is important.` |
| `mace_tools_torch_geometric_seed_seed_everything` | `mace.tools.torch_geometric.seed.seed_everything` | `mace/tools/torch_geometric/seed.py` | `seed: int` | `mace.tools.torch_geometric.seed.seed_everything sets the global random seed for Python's random module, NumPy, and PyTorch (including all CUDA devices accessible from the current process). This function is used throughout MACE to improve reproducibility of experiments such as model initialization, data shuffling and splitting, on-line preprocessing, and stochastic training procedures described in the README (training, evaluation, preprocessing, and distributed training workflows).` |
| `mace_tools_torch_geometric_utils_download_url` | `mace.tools.torch_geometric.utils.download_url` | `mace/tools/torch_geometric/utils.py` | `url: str, folder: str, log: bool = True` | `Downloads the content of a URL to a local folder and returns the local file path.` |
| `mace_tools_torch_geometric_utils_extract_zip` | `mace.tools.torch_geometric.utils.extract_zip` | `mace/tools/torch_geometric/utils.py` | `path: str, folder: str, log: bool = True` | `mace.tools.torch_geometric.utils.extract_zip extracts a ZIP archive from the filesystem to a target folder. This utility is used in the MACE codebase (torch_geometric utilities) to unpack archived resources such as preprocessed datasets, example inputs, or model artifacts that may be distributed as .zip files for preprocessing, on-line data loading, training, or evaluation workflows. The function opens the ZIP file at the given path for reading and calls zipfile.ZipFile.extractall to write the archive contents under the destination folder. The primary practical significance is to make the files contained in an archive available on disk for subsequent preprocessing (for example, the preprocessing scripts and training workflows described in the README), model loading, or evaluation steps.` |

## ⚖️ License

Original Code License: MIT

Wrapper Code & Documentation: Apache-2.0

*This file was automatically generated on February 26, 2026.*
