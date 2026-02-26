# torchio

[🔙 Back to Main Repo](../../../README.md) | [🔗 Original Repo](https://github.com/TorchIO-project/torchio)

![Tool Count](https://img.shields.io/badge/Agent_Tools-3-blue?style=flat-square)
![Category](https://img.shields.io/badge/Category-Medicine-green?style=flat-square)
![Status](https://img.shields.io/badge/Import_Test-Passed-success?style=flat-square)

## 📖 Overview

TorchIO is a PyTorch-based library for loading, preprocessing, and augmenting 3D medical images (e.g., MRI/CT) to build efficient deep learning pipelines for medical imaging research.

> **Note**: This documentation lists the **agent-ready wrapper functions** generated for this package. These functions have been strictly typed, docstring-enhanced, and tested for import stability within a standardized Apptainer environment.

## 🛠️ Available Agent Tools

Below is the list of **3** functions optimized for LLM tool-use.

| **Tool Name (Wrapper)**   | **Source**          | **File Path**     | **Arguments (Type)**        | **Description**                |
| ------------------------- | ------------------- | ----------------- | --------------------------- | ------------------------------ |
| `torchio_utils_get_batch_images_and_size` | `torchio.utils.get_batch_images_and_size` | `torchio/utils.py` | `batch: dict` | `Get names of image entries in a batch and the number of channels (size) per image.` |
| `torchio_utils_get_subjects_from_batch` | `torchio.utils.get_subjects_from_batch` | `torchio/utils.py` | `batch: dict` | `torchio.utils.get_subjects_from_batch: Reconstruct a list of torchio.data.Subject instances from a collated minibatch dictionary produced by a SubjectsLoader.` |
| `torchio_visualization_get_num_bins` | `torchio.visualization.get_num_bins` | `torchio/visualization.py` | `x: numpy.ndarray` | `Get the optimal number of bins for a histogram using the Freedman–Diaconis rule. This function implements the Freedman–Diaconis heuristic to choose a bin width that balances histogram resolution and variability of the data. In the context of TorchIO's visualization tools for 3D medical images (for example, plotting intensity histograms of MRI scans to inspect intensity distributions and augmentation effects), this routine computes an integer number of bins intended to minimize the integral of the squared difference between the histogram (relative frequency density) and the underlying theoretical probability density (see the Freedman–Diaconis rule). Concretely, the function computes the 25th and 75th percentiles q25 and q75 of the input values (percentiles are computed over all elements of x, i.e., the input is effectively flattened), forms the interquartile range IQR = q75 - q25, computes the bin width as 2 * IQR * n**(-1/3) where n is the number of samples len(x), and returns round((x.max() - x.min()) / bin_width) as the number of bins.` |

## ⚖️ License

Original Code License: Apache-2.0

Wrapper Code & Documentation: Apache-2.0

*This file was automatically generated on February 26, 2026.*
