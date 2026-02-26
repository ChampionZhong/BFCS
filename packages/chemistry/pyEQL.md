# pyEQL

[🔙 Back to Main Repo](../../../README.md) | [🔗 Original Repo](https://github.com/KingsburyLab/pyEQL)

![Tool Count](https://img.shields.io/badge/Agent_Tools-4-blue?style=flat-square)
![Category](https://img.shields.io/badge/Category-Chemistry-green?style=flat-square)
![Status](https://img.shields.io/badge/Import_Test-Passed-success?style=flat-square)

## 📖 Overview

pyEQL is a Python package for modeling aqueous electrolyte (water chemistry) solutions via a `Solution` object that lets you add solutes and compute species and bulk properties (e.g., activities, diffusion coefficients, density, conductivity, osmotic pressure) using built-in thermodynamic/property data and configurable modeling methods (including concentrated-solution models like Pitzer).

> **Note**: This documentation lists the **agent-ready wrapper functions** generated for this package. These functions have been strictly typed, docstring-enhanced, and tested for import stability within a standardized Apptainer environment.

## 🛠️ Available Agent Tools

Below is the list of **4** functions optimized for LLM tool-use.

| **Tool Name (Wrapper)**   | **Source**          | **File Path**     | **Arguments (Type)**        | **Description**                |
| ------------------------- | ------------------- | ----------------- | --------------------------- | ------------------------------ |
| `pyEQL_activity_correction_get_activity_coefficient_guntelberg` | `pyEQL.activity_correction.get_activity_coefficient_guntelberg` | `pyEQL/activity_correction.py` | `ionic_strength: float, z: int = 1, temperature: str = "25 degC"` | `Return the activity coefficient of a solute in the parent aqueous solution using the Guntelberg approximation. This function is used within pyEQL to compute an ionic activity correction for aqueous electrolyte solutions (affecting speciation, effective concentrations, and other derived bulk properties like conductivity and osmotic pressure) by estimating the mean ionic activity coefficient on the molal (mol/kg) scale.` |
| `pyEQL_equilibrium_alpha` | `pyEQL.equilibrium.alpha` | `pyEQL/equilibrium.py` | `n: int, pH: float, pKa_list: list` | `pyEQL.equilibrium.alpha computes the acid-base distribution coefficient alpha_n for an acid at a given pH. This function is used in aqueous chemistry speciation routines (for example in pyEQL Solution objects and related calculations) to determine the fraction of a total acid pool present in a specific deprotonation state. The computation follows the classical formulation from Stumm & Morgan (Aquatic Chemistry) using pKa values (negative base-10 logarithms of Ka) and the hydrogen ion activity [H+] = 10**(-pH). The function sorts pKa values, constructs the sequence of terms corresponding to each protonation state, and returns the fraction (term_n / sum_of_all_terms). The result is used downstream for species-specific properties (activities, transport coefficients) and bulk properties derived from speciation.` |
| `pyEQL_utils_format_solutes_dict` | `pyEQL.utils.format_solutes_dict` | `pyEQL/utils.py` | `solute_dict: dict, units: str` | `pyEQL.utils.format_solutes_dict formats a dictionary of solutes into the string-valued form expected by the pyEQL Solution class for constructing an aqueous electrolyte Solution from specified solute amounts.` |
| `pyEQL_utils_interpret_units` | `pyEQL.utils.interpret_units` | `pyEQL/utils.py` | `unit: str` | `pyEQL.utils.interpret_units translates commonly used environmental unit abbreviations (for example, "ppm") into strings that the pint library can understand and use in pyEQL's units-aware calculations for aqueous solution properties. This function is used throughout pyEQL when parsing user-provided unit strings (for concentrations, amounts, and other solution properties) so they can be passed to a pint UnitRegistry for numeric conversions and arithmetic. It provides a small, explicit mapping for a handful of common environmental shorthand notations to practical units used in water chemistry modeling. The mapping is case-sensitive and limited to the explicit keys implemented; unrecognized inputs are returned unchanged so callers can decide how to handle them.` |

## ⚖️ License

Original Code License: LGPLv3

Wrapper Code & Documentation: Apache-2.0

*This file was automatically generated on February 26, 2026.*
