# guacamol

[🔙 Back to Main Repo](../../../README.md) | [🔗 Original Repo](https://github.com/BenevolentAI/guacamol)

![Tool Count](https://img.shields.io/badge/Agent_Tools-5-blue?style=flat-square)
![Category](https://img.shields.io/badge/Category-Chemistry-green?style=flat-square)
![Status](https://img.shields.io/badge/Import_Test-Passed-success?style=flat-square)

## 📖 Overview

GuacaMol is an open-source Python benchmarking suite for evaluating *de novo* molecular design (molecular generative) models on distribution-learning and goal-directed molecule generation tasks.

> **Note**: This documentation lists the **agent-ready wrapper functions** generated for this package. These functions have been strictly typed, docstring-enhanced, and tested for import stability within a standardized Apptainer environment.

## 🛠️ Available Agent Tools

Below is the list of **5** functions optimized for LLM tool-use.

| **Tool Name (Wrapper)**   | **Source**          | **File Path**     | **Arguments (Type)**        | **Description**                |
| ------------------------- | ------------------- | ----------------- | --------------------------- | ------------------------------ |
| `guacamol_goal_directed_score_contributions_uniform_specification` | `guacamol.goal_directed_score_contributions.uniform_specification` | `guacamol/goal_directed_score_contributions.py` | `*top_counts: int` | `guacamol.goal_directed_score_contributions.uniform_specification creates a ScoreContributionSpecification that assigns equal weight to each specified top-x contribution used in goal-directed benchmark scoring. This function is part of the GuacaMol benchmarking suite for de novo molecular design (see README). In the goal-directed benchmarks, scoring functions may compute contributions from the top-x matching subcomponents or predictions; this helper builds a specification listing those top-x values each paired with the same weight (1.0). The returned ScoreContributionSpecification can be passed to the goal-directed scoring machinery to indicate that all listed top-x contributions should be treated with equal importance.` |
| `guacamol_utils_data_download_if_not_present` | `guacamol.utils.data.download_if_not_present` | `guacamol/utils/data.py` | `filename: str, uri: str` | `Download a file from a URI to a local path if the local file does not already exist. This utility is used by guacamol data ingestion workflows (for example the get_data script) to ensure required dataset files used in GuacaMol benchmarking (training/validation/test splits derived from ChEMBL and published on Figshare) are present on disk before further processing. The function checks for an existing file at the provided local path and, if absent, streams the remote resource to that path while displaying a progress bar. It does not perform content validation (for example, MD5 checksum verification) or atomic move semantics; callers should verify file integrity separately if required for reproducibility.` |
| `guacamol_utils_data_remove_duplicates` | `guacamol.utils.data.remove_duplicates` | `guacamol/utils/data.py` | `list_with_duplicates: list` | `Removes duplicate elements from a list while preserving the original ordering of the first occurrences. This function is used in the GuacaMol data-processing pipeline (for example when preparing standardized SMILES training/validation/test sets) to ensure that each molecule or entry appears only once while keeping the original ordering used in the source file. For duplicates, the first occurrence is kept and any later occurrences are ignored. The operation is non-destructive with respect to the input: the input list is not modified and a new list is returned. The function relies on a Python set for membership checks, so elements are expected to be hashable (see Failure modes below). The implementation provides O(n) average-time complexity and O(n) additional memory where n is the length of the input list.` |
| `guacamol_utils_math_arithmetic_mean` | `guacamol.utils.math.arithmetic_mean` | `guacamol/utils/math.py` | `values: List[float]` | `guacamol.utils.math.arithmetic_mean computes the arithmetic mean of a list of float values. This function is a small pure utility used in the GuacaMol benchmarking toolkit for de novo molecular design to aggregate numeric results such as per-molecule scores, per-run metrics, or other scalar quantities that must be averaged when evaluating distribution-learning or goal-directed generation methods.` |
| `guacamol_utils_math_geometric_mean` | `guacamol.utils.math.geometric_mean` | `guacamol/utils/math.py` | `values: List[float]` | `Compute the geometric mean of a list of numeric values. This function is a small numeric utility used in the GuacaMol benchmarking codebase to aggregate multiplicative scores (for example, combining per-objective or per-component scores in goal-directed benchmark scoring). It converts the input sequence to a NumPy array, computes the product of all elements, and returns the n-th root of that product (product ** (1/len(values))). The operation is commutative with respect to the input order and has no side effects on external state.` |

## ⚖️ License

Original Code License: MIT

Wrapper Code & Documentation: Apache-2.0

*This file was automatically generated on February 26, 2026.*
