# bioemu

[🔙 Back to Main Repo](../../../README.md) | [🔗 Original Repo](https://github.com/microsoft/bioemu)

![Tool Count](https://img.shields.io/badge/Agent_Tools-4-blue?style=flat-square)
![Category](https://img.shields.io/badge/Category-Biology-green?style=flat-square)
![Status](https://img.shields.io/badge/Import_Test-Passed-success?style=flat-square)

## 📖 Overview

**bioemu** provides inference code and pretrained weights for Microsoft’s Biomolecular Emulator, enabling users to sample an approximate equilibrium ensemble of 3D protein monomer structures from an amino-acid sequence (using ColabFold-based MSA/embeddings under the hood).

> **Note**: This documentation lists the **agent-ready wrapper functions** generated for this package. These functions have been strictly typed, docstring-enhanced, and tested for import stability within a standardized Apptainer environment.

## 🛠️ Available Agent Tools

Below is the list of **4** functions optimized for LLM tool-use.

| **Tool Name (Wrapper)**   | **Source**          | **File Path**     | **Arguments (Type)**        | **Description**                |
| ------------------------- | ------------------- | ----------------- | --------------------------- | ------------------------------ |
| `bioemu_get_embeds_shahexencode` | `bioemu.get_embeds.shahexencode` | `bioemu/get_embeds.py` | `s: str` | `bioemu.get_embeds.shahexencode computes and returns the lowercase hexadecimal SHA-256 digest of a given input string. In the BioEmu codebase this helper is intended for creating deterministic, filesystem-safe identifiers and cache keys used during embedding and MSA generation (for example, names derived from protein sequences, single-sequence FASTA content, MSA query URLs, or local paths used by the Colabfold setup described in the README).` |
| `bioemu_hpacker_setup_setup_hpacker_ensure_hpacker_install` | `bioemu.hpacker_setup.setup_hpacker.ensure_hpacker_install` | `bioemu/hpacker_setup/setup_hpacker.py` | `envname: str = "hpacker", repo_dir: str = "/mnt/petrelfs/zhongzhanping/.hpacker"` | `Ensures that the HPacker tool and its runtime dependencies are installed inside a conda environment named by envname. This function is used by BioEmu's side-chain reconstruction and MD-relaxation pipeline to prepare an isolated conda environment containing hpacker and its dependencies (the hpacker conda environment is required before running bioemu.sidechain_relax). The function locates the conda installation, checks whether the target environment already exists, and if not invokes the hpacker installation script to create and populate the environment.` |
| `bioemu_seq_io_check_protein_valid` | `bioemu.seq_io.check_protein_valid` | `bioemu/seq_io.py` | `seq: str` | `Checks that an input protein sequence string consists only of the canonical 20 IUPAC single-letter amino acid codes used by BioEmu. This function is a lightweight validator used throughout the BioEmu codebase (for example, prior to sampling structure ensembles and before embedding or MSA processing) to ensure that downstream components receive canonical protein sequences. It iterates over each character in the provided sequence string and verifies membership in the module-level IUPACPROTEIN set (the standard 20 amino acid single-letter codes). The function does not modify its input, does not access files, and is intended only for validating sequence contents; it does not parse FASTA files or accept sequence metadata.` |
| `bioemu_utils_format_npz_samples_filename` | `bioemu.utils.format_npz_samples_filename` | `bioemu/utils.py` | `start_id: int, num_samples: int` | `Format a canonical filename for a batch of sample records saved as a NumPy .npz file. This function is used by the BioEmu sampling pipeline to produce deterministic, parseable filenames for saved batches of generated protein-structure samples. The produced filename encodes the zero-padded start index of the first sample in the batch and the computed one-past-last (exclusive upper bound) index for the batch, allowing downstream tools and users to infer the global sample index range contained in the file.` |

## ⚖️ License

Original Code License: MIT

Wrapper Code & Documentation: Apache-2.0

*This file was automatically generated on February 26, 2026.*
