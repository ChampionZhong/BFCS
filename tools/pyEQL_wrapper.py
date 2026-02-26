"""
Regenerated Google-style docstrings for module 'pyEQL'.
README source: others/readme/pyEQL/README.md
Generated at: 2025-12-02T00:48:09.693511Z

Total functions: 4
"""


################################################################################
# Source: pyEQL.activity_correction.get_activity_coefficient_guntelberg
# File: pyEQL/activity_correction.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def pyEQL_activity_correction_get_activity_coefficient_guntelberg(
    ionic_strength: float,
    z: int = 1,
    temperature: str = "25 degC"
):
    """Return the activity coefficient of a solute in the parent aqueous solution using the Guntelberg approximation. This function is used within pyEQL to compute an ionic activity correction for aqueous electrolyte solutions (affecting speciation, effective concentrations, and other derived bulk properties like conductivity and osmotic pressure) by estimating the mean ionic activity coefficient on the molal (mol/kg) scale.
    
    Args:
        ionic_strength (Quantity): The ionic strength of the parent solution expressed as a pint.Quantity on the molal scale (mol/kg). The function expects a Quantity with a .magnitude attribute (the code uses ionic_strength.magnitude). The ionic strength is the driving variable for the Guntelberg approximation and determines the strength of electrostatic interactions that reduce effective solute activity. Validity: the approximation is intended for ionic strength I < 0.1 mol/kg; if ionic_strength.magnitude > 0.1 the function will still compute a value but emits a warning because accuracy degrades.
        z (int): The formal charge of the solute (including sign), e.g., +1 for Na+, -1 for Cl-. Defaults to 1. The charge appears squared in the Guntelberg expression and therefore strongly influences the magnitude of the activity correction (multiplies the ln γ term by z**2).
        temperature (str, Quantity, optional): Temperature of the solution used to compute the Debye parameter A^γ via the helper _debye_parameter_activity. Accepts a string like "25 degC" (the default) or a pint.Quantity carrying temperature units. The temperature controls the dielectric and thermal factors in A^γ; supplying an incorrectly typed object that _debye_parameter_activity cannot parse may raise an error.
    
    Returns:
        Quantity: A pint.Quantity with unit "dimensionless" containing the mean molal (mol/kg scale) ionic activity coefficient γ for the solute. This is computed as exp(ln γ) where ln γ = A^γ * z^2 * sqrt(I) / (1 + sqrt(I)), and A^γ is obtained from _debye_parameter_activity(temperature). Side effects: if ionic_strength.magnitude > 0.1 the function logs a warning that the Guntelberg approximation exceeds its recommended range; if inputs are not provided as the expected types (for example a plain float without .magnitude) the function may raise AttributeError or TypeError when attempting to access attributes or perform unit-aware operations.
    """
    from pyEQL.activity_correction import get_activity_coefficient_guntelberg
    return get_activity_coefficient_guntelberg(ionic_strength, z, temperature)


################################################################################
# Source: pyEQL.equilibrium.alpha
# File: pyEQL/equilibrium.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def pyEQL_equilibrium_alpha(n: int, pH: float, pKa_list: list):
    """pyEQL.equilibrium.alpha computes the acid-base distribution coefficient alpha_n for an acid at a given pH. This function is used in aqueous chemistry speciation routines (for example in pyEQL Solution objects and related calculations) to determine the fraction of a total acid pool present in a specific deprotonation state. The computation follows the classical formulation from Stumm & Morgan (Aquatic Chemistry) using pKa values (negative base-10 logarithms of Ka) and the hydrogen ion activity [H+] = 10**(-pH). The function sorts pKa values, constructs the sequence of terms corresponding to each protonation state, and returns the fraction (term_n / sum_of_all_terms). The result is used downstream for species-specific properties (activities, transport coefficients) and bulk properties derived from speciation.
    
    Args:
        n (int): The number of protons that have been lost by the desired form of the acid (the subscript in alpha_n). For domain context, n=0 corresponds to the fully protonated form, n=1 to the singly deprotonated form (e.g., HCO3- for carbonic acid when n=1), etc. This integer selects which alpha fraction to return and must be non-negative and less than or equal to the number of dissociable protons implied by pKa_list.
        pH (float or int): The solution pH used to compute hydrogen ion activity via [H+] = 10**(-pH). This controls the protonation equilibrium and therefore the partitioning among protonation states; pH may be provided as an integer or float. Practical significance: small changes in pH near pKa values produce large changes in alpha values and are important for speciation in natural and engineered waters.
        pKa_list (list of floats or ints): The acid dissociation constants expressed as pKa = -log10(Ka) for each dissociation step of the acid, provided as a sequence of numbers. The list is sorted internally (ascending) before computation so the caller need not pre-sort. The length of pKa_list defines the number of dissociable protons (num_protons = len(pKa_list)). There must be at least n pKa values (i.e., len(pKa_list) >= n) and at least one pKa value is required. These values are used to compute Ka = 10**(-pKa) and thus the multiplicative terms in the distribution coefficient formula.
    
    Behavior, side effects, defaults, and failure modes:
        The function implements the classical denominator terms D_j = (product_{k=1..j} Ka_k) * [H+]^{num_protons-j} for j = 0..num_protons, where Ka_k = 10**(-pKa_list[k-1]) and [H+] = 10**(-pH). The returned value is alpha_n = D_n / sum_{j=0..num_protons} D_j. The pKa_list is sorted in ascending order before constructing these terms to ensure consistent ordering of dissociation steps if the caller provides unsorted input.
        Side effects: the implementation prints the computed internal list of terms to standard output (terms_list) and emits a logger.debug call; callers should be aware of this debug output. The logger message in the implementation may contain an unformatted string.
        Validation and errors: the function raises ValueError if pKa_list is empty (no pKa values supplied) or if len(pKa_list) < n (insufficient pKa entries to represent the requested deprotonation index). If invalid types are supplied (non-integer n, non-numeric pH or pKa entries), Python may raise TypeError or ValueError during numeric operations; callers should ensure types match the documented forms (n as int, pH as float or int, pKa_list as list of floats/ints).
        Numerical considerations: results are floating-point fractions in [0, 1] and the sum of alpha_n over n=0..num_protons equals 1 within floating-point tolerance. For very large or small pH/pKa differences, intermediate terms may underflow or overflow floating-point range, producing values extremely close to 0 or 1; callers requiring extended precision should handle this externally.
    
    Returns:
        float: The fraction (between 0 and 1 inclusive, subject to floating-point precision) of the total acid present in the specified n-deprotonated form at the supplied pH. This value is dimensionless and directly usable in speciation calculations (e.g., to partition total acid concentration into species concentrations).
    """
    from pyEQL.equilibrium import alpha
    return alpha(n, pH, pKa_list)


################################################################################
# Source: pyEQL.utils.format_solutes_dict
# File: pyEQL/utils.py
# Category: fix_docstring
# Reason: Schema parsing failed: Cannot generate JSON schema for format_solutes_dict because the docstring has no description for the argument 'solute_dict'
################################################################################

def pyEQL_utils_format_solutes_dict(solute_dict: dict, units: str):
    """pyEQL.utils.format_solutes_dict formats a dictionary of solutes into the string-valued form expected by the pyEQL Solution class for constructing an aqueous electrolyte Solution from specified solute amounts.
    
    Args:
        solute_dict (dict): A mapping of solute identifiers to their numeric amounts. In the pyEQL context these keys are chemical species labels used by Solution (for example "Na+" or "Cl-") and the values are numeric quantities (e.g., int or float) representing the amount of each solute in the same units. Example form: {"Na+": 0.5, "Cl-": 0.9}. The function requires that all numeric values in this dictionary are expressed in the same physical units (see units). If solute_dict is not a dict, the function raises a TypeError.
        units (str): A units string to append to every numeric value in solute_dict to produce a quantity string understood by Solution and pyEQL's units-aware calculations (pint-compatible unit strings such as "mol/kg", "mol/L", "mg/L", etc.). This argument must be a string; the function performs no unit parsing or validation itself beyond string concatenation, so the caller should supply a units string compatible with downstream pyEQL/pint usage.
    
    Returns:
        dict: A new dictionary with the same keys and insertion order as solute_dict, where each value has been converted to a string of the form "<value> <units>" using Python's str() representation of the original numeric value (implemented as f"{value!s} {units}"). The returned mapping is intended to be passed directly to pyEQL.Solution to populate a Solution with the specified solutes and amounts.
    
    Raises:
        TypeError: If solute_dict is not a dict. The function does not raise for non-numeric values in solute_dict; such values will be converted to strings and may cause downstream errors when the returned dictionary is used to construct a Solution. The function also does not validate the units string; invalid or incompatible unit strings may lead to errors later in pyEQL when units are interpreted by pint or Solution. There are no other side effects; the input dictionary is not modified and a new dictionary is returned.
    """
    from pyEQL.utils import format_solutes_dict
    return format_solutes_dict(solute_dict, units)


################################################################################
# Source: pyEQL.utils.interpret_units
# File: pyEQL/utils.py
# Category: valid
################################################################################

def pyEQL_utils_interpret_units(unit: str):
    """pyEQL.utils.interpret_units translates commonly used environmental unit abbreviations (for example, "ppm") into strings that the pint library can understand and use in pyEQL's units-aware calculations for aqueous solution properties.
    
    This function is used throughout pyEQL when parsing user-provided unit strings (for concentrations, amounts, and other solution properties) so they can be passed to a pint UnitRegistry for numeric conversions and arithmetic. It provides a small, explicit mapping for a handful of common environmental shorthand notations to practical units used in water chemistry modeling. The mapping is case-sensitive and limited to the explicit keys implemented; unrecognized inputs are returned unchanged so callers can decide how to handle them.
    
    Args:
        unit (str): The input unit string to translate. This should be the exact string provided by the user or upstream code (case-sensitive). Typical expected inputs include environmental shorthand such as "ppm", "ppb", "ppt", and "m" (where "m" here denotes molal). The function treats these specific lowercase strings as special cases and maps them to pint-compatible equivalents: "m" -> "mol/kg" (molality), "ppm" -> "mg/L", "ppb" -> "ug/L", and "ppt" -> "ng/L". If the caller provides a unit string not listed here or with different casing (for example "PPM" or "M"), the function will not perform a translation and will simply return the original string.
    
    Returns:
        str: A unit string suitable for use with pint when possible. For recognized shorthand inputs the returned string is the mapped pint-compatible unit (see mapping above) which pyEQL uses for concentration and mass/volume conversions in solution property calculations. If the input is not one of the recognized keys, the original input string is returned unchanged. Note that this function does not validate that the returned string is a valid pint unit; callers should pass the returned string to a pint UnitRegistry and handle any pint parsing errors themselves. The function has no side effects and does not raise exceptions on unrecognized inputs.
    """
    from pyEQL.utils import interpret_units
    return interpret_units(unit)


from typing import Dict, Any


def get_tools() -> Dict[str, Dict[str, Any]]:
    """Extract JSON schemas for all functions in this module."""
    import sys
    import os
    
    # Add project root to path to import our json_schema module
    # Try multiple possible paths
    possible_paths = [
        os.path.join(os.path.dirname(__file__), '..', '..', 'utils'),
        '/app/utils',
        '/app/project/utils',
    ]
    
    json_schema_path = None
    for path in possible_paths:
        abs_path = os.path.abspath(path)
        if os.path.exists(os.path.join(abs_path, 'json_schema.py')):
            if abs_path not in sys.path:
                sys.path.insert(0, abs_path)
            json_schema_path = abs_path
            break
    
    if json_schema_path:
        from json_schema import get_json_schema
    else:
        # Fallback to transformers if our module not found
        from transformers.utils import get_json_schema
    
    tools = {}
    failed_count = 0
    
    for name, func in get_lib().items():
        try:
            tools[name] = get_json_schema(func)
        except Exception as e:
            failed_count += 1
            # Only print first few errors to avoid spam
            if failed_count <= 3:
                print(f"Failed to get schema for {name}: {type(e).__name__}: {e}", file=sys.stderr)
            continue
    
    if failed_count > 0:
        print(f"Warning: Failed to extract schemas for {failed_count} out of {len(get_lib())} functions", file=sys.stderr)
    
    return tools


def get_lib():
    """Get all functions defined in this module."""
    import inspect
    global_vars = inspect.currentframe().f_globals
    
    functions = {
        name: obj for name, obj in global_vars.items()
        if inspect.isfunction(obj) and obj.__module__ == __name__
    }
    functions.pop("get_lib", None)
    functions.pop("get_tools", None)
    return functions
