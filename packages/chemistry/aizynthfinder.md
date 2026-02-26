# aizynthfinder

[🔙 Back to Main Repo](../../../README.md) | [🔗 Original Repo](https://github.com/MolecularAI/aizynthfinder)

![Tool Count](https://img.shields.io/badge/Agent_Tools-3-blue?style=flat-square)
![Category](https://img.shields.io/badge/Category-Chemistry-green?style=flat-square)
![Status](https://img.shields.io/badge/Import_Test-Passed-success?style=flat-square)

## 📖 Overview

aizynthfinder is a retrosynthetic planning tool that uses Monte Carlo tree search guided by neural-network reaction-template policies to break target molecules into purchasable precursor routes.

> **Note**: This documentation lists the **agent-ready wrapper functions** generated for this package. These functions have been strictly typed, docstring-enhanced, and tested for import stability within a standardized Apptainer environment.

## 🛠️ Available Agent Tools

Below is the list of **3** functions optimized for LLM tool-use.

| **Tool Name (Wrapper)**   | **Source**          | **File Path**     | **Arguments (Type)**        | **Description**                |
| ------------------------- | ------------------- | ----------------- | --------------------------- | ------------------------------ |
| `aizynthfinder_utils_math_rectified_linear_unit` | `aizynthfinder.utils.math.rectified_linear_unit` | `aizynthfinder/utils/math.py` | `x: numpy.ndarray` | `aizynthfinder.utils.math.rectified_linear_unit returns the element-wise Rectified Linear Unit (ReLU) activation of a NumPy array. In the AiZynthFinder codebase this function is used to introduce a simple, computationally efficient non-linearity in neural network computations (for example in expansion policy or filter policy networks that guide the retrosynthetic Monte Carlo tree search). The function maps each input element to itself when it is greater than zero and to zero otherwise, producing a non-negative array that can induce sparsity and improve training/stability of downstream policy networks.` |
| `aizynthfinder_utils_math_sigmoid` | `aizynthfinder.utils.math.sigmoid` | `aizynthfinder/utils/math.py` | `x: numpy.ndarray` | `aizynthfinder.utils.math.sigmoid computes the logistic (sigmoid) activation function used to squash real-valued scores into a bounded range for downstream decision making in AiZynthFinder's neural-network-driven policy components. This function implements the element-wise logistic sigmoid 1 / (1 + exp(-x)). In the AiZynthFinder retrosynthetic planning workflow (see README), policy networks produce raw scores for candidate reaction templates; applying this sigmoid converts those raw scores into normalized confidence-like values that can be used to rank or weight suggestions during the Monte Carlo tree search.` |
| `aizynthfinder_utils_math_softmax` | `aizynthfinder.utils.math.softmax` | `aizynthfinder/utils/math.py` | `x: numpy.ndarray` | `Compute column-wise softmax of the input scores and return normalized probabilities. This function converts raw scores (logits) into non-negative values normalized per column by applying the exponential function and dividing by the column-wise sum. In the AiZynthFinder retrosynthetic planning workflow, it is typically used to transform the output scores from an expansion policy neural network into probabilities that guide the Monte Carlo tree search when selecting precursor reactions. The implementation performs the operation exp(x) / sum(exp(x), axis=0) using NumPy without additional numerical stabilization.` |

## ⚖️ License

Original Code License: MIT

Wrapper Code & Documentation: Apache-2.0

*This file was automatically generated on February 26, 2026.*
