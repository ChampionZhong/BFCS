# dscribe

[🔙 Back to Main Repo](../../../README.md) | [🔗 Original Repo](https://github.com/SINGROUP/dscribe)

![Tool Count](https://img.shields.io/badge/Agent_Tools-7-blue?style=flat-square)
![Category](https://img.shields.io/badge/Category-Material-green?style=flat-square)
![Status](https://img.shields.io/badge/Import_Test-Passed-success?style=flat-square)

## 📖 Overview

DScribe is a Python library that converts atomic structures into fixed-size numerical descriptors (fingerprints) such as SOAP, ACSF, and Coulomb matrices for materials-science tasks like machine learning, similarity analysis, and visualization.

> **Note**: This documentation lists the **agent-ready wrapper functions** generated for this package. These functions have been strictly typed, docstring-enhanced, and tested for import stability within a standardized Apptainer environment.

## 🛠️ Available Agent Tools

Below is the list of **7** functions optimized for LLM tool-use.

| **Tool Name (Wrapper)**   | **Source**          | **File Path**     | **Arguments (Type)**        | **Description**                |
| ------------------------- | ------------------- | ----------------- | --------------------------- | ------------------------------ |
| `dscribe_descriptors_mbtr_check_geometry` | `dscribe.descriptors.mbtr.check_geometry` | `dscribe/descriptors/mbtr.py` | `geometry: dict` | `Used to validate MBTR geometry settings before computing Many-Body Tensor Representation (MBTR) descriptors. This function checks that the provided geometry configuration dictionary contains a "function" key and that its value is one of the allowed geometry functions used by MBTR k-body terms. In the context of the DScribe library, MBTR (Many-body Tensor Representation) converts atomic structures into fixed-size numerical fingerprints for machine learning and similarity analysis in materials science. Validating the geometry function here ensures that the MBTR descriptor will compute the intended k-body contribution (k = 1, 2, or 3) and prevents silent misconfiguration that would produce incorrect descriptors or runtime errors later in the descriptor pipeline.` |
| `dscribe_descriptors_mbtr_check_grid` | `dscribe.descriptors.mbtr.check_grid` | `dscribe/descriptors/mbtr.py` | `grid: dict` | `dscribe.descriptors.mbtr.check_grid validates MBTR grid settings and enforces basic consistency rules used by the Many-Body Tensor Representation (MBTR) descriptor in DScribe. This function is used before constructing MBTR fingerprints (fixed-size numerical descriptors for atomic structures) to ensure the provided grid dictionary contains the required entries that define the discretization range and resolution for the descriptor.` |
| `dscribe_descriptors_mbtr_check_weighting` | `dscribe.descriptors.mbtr.check_weighting` | `dscribe/descriptors/mbtr.py` | `k: int, weighting: dict, periodic: bool` | `dscribe.descriptors.mbtr.check_weighting validates weighting settings for the Many-Body Tensor Representation (MBTR) descriptor in DScribe. It checks that the provided weighting dictionary contains a supported weighting function for the requested MBTR degree k (1, 2, or 3), that all required additional parameters for that function are present and not contradictory (for example, not providing both 'scale' and 'r_cut'), and that periodic systems have an appropriate non-unity weighting when required. This validation is used before constructing MBTR fingerprints (fixed-size numerical descriptors of atomic structures) so that downstream descriptor creation, machine learning, or analysis tasks receive consistent and well-specified weighting behavior.` |
| `dscribe_utils_geometry_get_adjacency_matrix` | `dscribe.utils.geometry.get_adjacency_matrix` | `dscribe/utils/geometry.py` | `radius: float, pos1: numpy.ndarray, pos2: numpy.ndarray = None, output_type: str = "coo_matrix"` | `Calculates a sparse adjacency matrix of pairwise Euclidean distances for points within a specified cutoff radius using a k-d tree for efficient neighbor search. This function is used throughout DScribe to build neighbor lists and distance-based adjacency information for atomic-structure descriptors (for example SOAP, ACSF, MBTR), where only interatomic pairs within a cutoff radius contribute to the descriptor. By delegating neighbor search to scipy.spatial.cKDTree.sparse_distance_matrix the function attains approximately O(n log n) scaling for large point sets, making it suitable for datasets of atoms or centers in high-throughput materials-science workflows.` |
| `dscribe_utils_species_get_atomic_numbers` | `dscribe.utils.species.get_atomic_numbers` | `dscribe/utils/species.py` | `species: list` | `Return ordered unique atomic numbers for a list of chemical species. This utility is used throughout DScribe to normalize user-provided species lists into a canonical form of atomic numbers that descriptors (for example SOAP, CoulombMatrix, MBTR) expect when constructing fixed-size fingerprints of atomic structures. The function accepts either a sequence of atomic numbers or a sequence of chemical element symbols and returns a sorted one-dimensional numpy array containing the unique atomic numbers present in the input. The function performs input validation, rejects non-iterable inputs and single strings, converts symbols to numbers when needed, checks for negative integers, and raises informative ValueError exceptions for malformed input.` |
| `dscribe_utils_species_symbols_to_numbers` | `dscribe.utils.species.symbols_to_numbers` | `dscribe/utils/species.py` | `symbols: list` | `Convert a sequence of chemical element symbols into their corresponding atomic numbers for use in DScribe descriptors. This function is used throughout DScribe to prepare atomic species input for numerical descriptor generation (for example, SOAP, CoulombMatrix, ACSF, MBTR). It looks up each chemical symbol in the ASE atomic_numbers mapping (ase.data.atomic_numbers) and returns a numpy integer array where each element is the atomic number corresponding to the symbol at the same position in the input. The order of the output matches the input order and the resulting array can be passed directly to descriptor constructors and creation routines that expect atomic numbers.` |
| `dscribe_utils_stats_system_stats` | `dscribe.utils.stats.system_stats` | `dscribe/utils/stats.py` | `system_iterator: list` | `dscribe.utils.stats.system_stats: Compute aggregated statistics over a collection of atomic systems for use in DScribe descriptor construction and dataset analysis.` |

## ⚖️ License

Original Code License: Apache-2.0

Wrapper Code & Documentation: Apache-2.0

*This file was automatically generated on February 26, 2026.*
