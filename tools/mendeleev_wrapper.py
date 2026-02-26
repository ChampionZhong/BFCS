"""
Regenerated Google-style docstrings for module 'mendeleev'.
README source: others/readme/mendeleev/README.md
Generated at: 2025-12-02T00:58:39.191804Z

Total functions: 23
"""


from typing import List

################################################################################
# Source: mendeleev.econf.get_l
# File: mendeleev/econf.py
# Category: fix_docstring
# Reason: Schema parsing failed: Cannot generate JSON schema for get_l because the docstring has no description for the argument 'subshell'
################################################################################

def mendeleev_econf_get_l(subshell: str):
    """Return the orbital angular momentum quantum number for a given subshell label.
    
    This function is part of the electronic configuration utilities in the mendeleev.econf module and is used when converting spectroscopic subshell labels (used throughout the package and its electronic configuration tutorials) into the corresponding orbital angular momentum quantum number l. The implementation performs a case-insensitive membership check against the module-level ORBITALS sequence and returns the index of the matching label; this index is the integer l used in electronic-structure related computations, selection-rule checks, and when constructing or parsing electronic configurations.
    
    Args:
        subshell (str): Subshell label in spectroscopic notation provided as a string (for example labels used throughout mendeleev electronic-configuration code). The value is treated case-insensitively (the function lowercases the input before lookup) and must be one of the allowed labels defined by the module-level ORBITALS sequence. This parameter represents the spectroscopic subshell whose orbital angular momentum quantum number (l) is requested; supplying an unrecognized label will trigger an error described below.
    
    Returns:
        int: The orbital angular momentum quantum number l corresponding to the provided subshell label. Concretely, this is the index of the subshell label in the module-level ORBITALS sequence (i.e., the function returns ORBITALS.index(subshell.lower())). The returned integer is used by other parts of the mendeleev package for electronic configuration handling and related computations.
    
    Raises:
        ValueError: If the provided subshell string (after lowercasing) is not present in the module-level ORBITALS sequence. The raised ValueError contains a message indicating the invalid label and listing the allowed subshell labels (the message is constructed in the form 'wrong subshell label: "<subshell>", should be one of: <allowed labels>'). No other side effects occur; the function is pure and deterministic.
    """
    from mendeleev.econf import get_l
    return get_l(subshell)


################################################################################
# Source: mendeleev.econf.get_spin_strings
# File: mendeleev/econf.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def mendeleev_econf_get_spin_strings(sodict: dict, average: bool = True):
    """Get per-spin occupation strings for valence subshells.
    
    Constructs two parallel sequences that represent the occupation of individual spin-orbitals (alpha and beta spins) across the valence subshells described in sodict. This function is intended for use in mendeleev's electronic-configuration utilities when generating per-spin occupation patterns for valence electrons, for example when preparing data for display, further processing of element electronic configurations, or converting to array form for numerical routines. This should be called for valence only; the function expands each subshell according to its degeneracy as returned by the subshell_degeneracy helper.
    
    Args:
        sodict (dict): Mapping that describes valence subshell occupations. Each key is a tuple (n, orb) where n is the principal quantum number and orb is the subshell label (for example 's', 'p', 'd', 'f'), and each value is a dictionary-like object occ with numeric entries occ["alpha"] and occ["beta"] giving the number of alpha and beta electrons assigned to that subshell. The function expects occ["alpha"] and occ["beta"] to be non-negative numbers not exceeding the subshell degeneracy returned by subshell_degeneracy(orb). The insertion order of sodict determines the order of subshells in the output, and within each subshell the output contains one entry per spin-orbital (i.e., degeneracy copies) expanded consecutively.
        average (bool): If True (the default), produce averaged occupations distributed evenly over the degenerate spin-orbitals: each spin-orbital in a subshell receives occ["alpha"]/nss for alpha and occ["beta"]/nss for beta, where nss is the subshell degeneracy. If False, produce explicit integer occupancy strings for each spin-orbital using 1.0 for occupied and 0.0 for unoccupied: the first occ["alpha"] entries are 1.0 and the remaining (nss - occ["alpha"]) entries are 0.0 for alpha, and likewise for beta. Use average=True for fractional/mean occupations, and average=False for explicit occupation bit-strings.
    
    Behavior, side effects, defaults, and failure modes:
        The function does not modify sodict or its nested occ dictionaries; it builds and returns new sequences. The default average=True yields floating-point occupancy values that may be non-integer. If average=False the returned values are 1.0 and 0.0 floats representing occupied and unoccupied spin-orbitals. The function relies on subshell_degeneracy(orb) to obtain the integer degeneracy nss for each subshell; nss must be a positive integer. Expected failure modes include TypeError if sodict is not a mapping or if occ does not support indexing by "alpha" and "beta", KeyError if occ lacks the "alpha" or "beta" keys, and TypeError or ValueError if occ["alpha"] or occ["beta"] are not numeric or are negative or exceed nss (these conditions will cause incorrect list construction or runtime errors). The function assumes it is called for valence subshells only; calling it with core subshells or with inconsistent occupation numbers may produce semantically incorrect results.
    
    Returns:
        tuple: A pair (alphas, betas) of lists of numeric occupancy values (floats) representing the spin strings for alpha and beta spins respectively. Each list length equals the total number of spin-orbitals obtained by summing degeneracies over all subshells in sodict. Values are floating-point: when average=True they are fractional occupancies (occ["alpha"]/nss or occ["beta"]/nss), when average=False they are 1.0 or 0.0 indicating occupied or unoccupied spin-orbitals. These sequences are compatible with conversion to numpy arrays if array semantics (for example, vectorized numerical operations) are required.
    """
    from mendeleev.econf import get_spin_strings
    return get_spin_strings(sodict, average)


################################################################################
# Source: mendeleev.econf.print_spin_occupations
# File: mendeleev/econf.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def mendeleev_econf_print_spin_occupations(sodict: dict, average: bool = True):
    """Pretty-print spin occupations for electronic configurations.
    
    This function formats and prints the spin-resolved electron occupations for each subshell/orbital given a mapping of occupations. It is intended for use in the electronic-configuration tools of the mendeleev package (see the "Electronic Configuration" tutorial in the README) where human-readable representations of alpha and beta spin occupations are needed for elements, ions, or computed configurations. The function uses subshell degeneracy information (via subshell_degeneracy) to expand or average occupations across degenerate orbitals and prints a one-line alpha and one-line beta representation for each orbital. It also returns the formatted strings so callers can capture the same formatted output programmatically (for example, to build tables or further visualizations).
    
    Args:
        sodict (dict): Mapping from orbital identity to occupation counts. Each key is a tuple (n, orb) where n is the principal quantum number and orb is the orbital label (e.g., '1s', '3d'); each value is a dict with numeric entries "alpha" and "beta" giving the total number of alpha and beta electrons in that subshell. This structure is produced by the econf modules that compute electronic occupations. The function iterates sodict.items() in the dict's iteration order and formats each entry in that order.
        average (bool = True): Controls the formatting mode. If True (default), the total alpha and beta occupations for the subshell are divided evenly among the degenerate orbitals and each orbital occupation is printed as a fixed-width floating-point number with 10.8f precision (i.e., ten characters with eight digits after the decimal point). If False, occupations are displayed as discrete occupation markers using 3.1f formatting: the function will create [1] entries for occupied orbitals and [0] entries for unoccupied orbitals according to integer counts occ["alpha"] and occ["beta"], and then format those as "1.0" or "0.0". Use average=True when you want averaged fractional occupations per degenerate orbital (common in computed or fractional-occupation contexts) and average=False when you want an explicit integer occupancy pattern across the degenerate slots.
    
    Behavior and side effects:
        The function prints two lines per subshell during iteration: one prefixed "<orb> alpha: " and one prefixed "<orb> beta : ", where <orb> is the orbital label from the sodict key. The printed alpha and beta lines show comma-separated formatted numbers as described above. The function relies on the subshell_degeneracy(orb) helper to determine how many degenerate orbitals correspond to the subshell label; that degeneracy is used either to replicate an averaged value or to expand integer occupations into per-orbital entries. The printed output is sent to standard output using print(), so callers who want to suppress printing should capture or redirect stdout or use the returned lists instead.
    
    Failure modes and errors:
        If sodict does not follow the expected structure (for example, keys are not (n, orb) tuples or values do not contain "alpha" and "beta"), the function may raise KeyError or TypeError when accessing occ["alpha"] / occ["beta"] or unpacking items. If subshell_degeneracy(orb) returns 0 or an unexpected non-positive integer, a ZeroDivisionError or ValueError may be raised during averaging or formatting. Non-numeric occupation values can raise TypeError during numeric operations or formatting. The function does not validate orbital labels beyond passing them to subshell_degeneracy, so invalid labels may cause that helper to raise an error.
    
    Returns:
        tuple: A pair (alphas, betas) where alphas is a list of strings and betas is a list of strings. Each element in alphas and betas corresponds to the formatted, comma-separated occupation string for the respective subshell in the same iteration order as sodict.items(). These returned lists contain the exact strings that were printed for alpha and beta occupations and can be used for programmatic consumption, logging, or downstream formatting.
    """
    from mendeleev.econf import print_spin_occupations
    return print_spin_occupations(sodict, average)


################################################################################
# Source: mendeleev.econf.shell_capactity
# File: mendeleev/econf.py
# Category: valid
################################################################################

def mendeleev_econf_shell_capactity(shell: str):
    """mendeleev.econf.shell_capactity: Compute the maximum number of electrons that can occupy a named atomic shell label used in electronic configuration calculations.
    
    This function is part of the econf (electronic configuration) utilities in the mendeleev package, which provides a Pythonic API for periodic table data and assists in building electronic configurations for atoms and ions (see README tutorials on electronic configuration). The capacity is computed from the principal quantum number n using the physical rule N = 2 * n**2, where n is the shell number (1 for K, 2 for L, etc.). The principal quantum number n is determined by locating the provided shell label in the module-level SHELLS list and using its index (n = SHELLS.index(shell.upper()) + 1). The function accepts case-insensitive shell labels such as "K", "L", "M", ... and returns an integer electron capacity for that shell.
    
    Args:
        shell (str): Shell label to query, e.g. "K", "L", "M". This argument is interpreted case-insensitively; the implementation converts the provided string to upper case before lookup. The value must exactly match one of the labels present in the module-level SHELLS list (in the expected order corresponding to increasing principal quantum number). The label identifies the atomic shell for which the electron capacity is requested and is typically used when allocating electrons across shells while constructing an element's electronic configuration.
    
    Returns:
        int: Number of electrons that can occupy the specified shell. This is the integer value N = 2 * n**2 where n is the principal quantum number determined from SHELLS (n = SHELLS.index(shell.upper()) + 1). The return value is suitable for use in electronic configuration algorithms and validations within the mendeleev package.
    
    Behavior, side effects, defaults, and failure modes:
        This function is deterministic and has no side effects (it does not modify global state or external resources). It does not accept numeric principal quantum numbers or labels not present in SHELLS; only string labels that match entries in SHELLS are valid. If the provided shell label (after upper-casing) is not found in SHELLS, the function raises a ValueError. The raised ValueError carries a message of the form 'wrong shell label: "<provided>", should be one of: <comma-separated SHELLS>' which lists the accepted labels to aid debugging. No default shell is assumed; the caller must supply a valid shell label.
    """
    from mendeleev.econf import shell_capactity
    return shell_capactity(shell)


################################################################################
# Source: mendeleev.econf.subshell_capacity
# File: mendeleev/econf.py
# Category: fix_docstring
# Reason: Schema parsing failed: Cannot generate JSON schema for subshell_capacity because the docstring has no description for the argument 'subshell'
################################################################################

def mendeleev_econf_subshell_capacity(subshell: str):
    """mendeleev.econf.subshell_capacity: Return the maximum number of electrons that can occupy a specified electronic subshell.
    
    This function accepts a subshell label string (as used in electronic configuration notation in the mendeleev package, e.g. the subshells that appear in strings like "[Ne] 3s2 3p2") and returns the integer electron capacity of that subshell. The capacity is computed as twice the subshell degeneracy (2 * degeneracy), where "degeneracy" corresponds to the number of distinct magnetic quantum states (m_l orbitals) in the subshell. This value is used throughout the mendeleev package when constructing, validating, or manipulating electronic configurations for elements and ions in the periodic table API.
    
    Args:
        subshell (str): The electronic subshell label identifying the orbital type whose capacity is requested. This is the same kind of string used by the package for electronic configurations (for example commonly 's', 'p', 'd', 'f' possibly combined with principal quantum number components elsewhere in the package). The function treats this argument as a pure identifier string and does not mutate it.
    
    Returns:
        int: The maximum number of electrons that can occupy the specified subshell. Concretely this is 2 * subshell_degeneracy(subshell). Typical returned values are small positive integers (for example, 2 for an s subshell, 6 for p, 10 for d, 14 for f) and the return value is deterministic and has no side effects.
    
    Behavior, side effects, defaults, and failure modes:
        This is a pure, deterministic function with no side effects: it does not modify global state or perform I/O. It delegates the computation of orbital degeneracy to the internal subshell_degeneracy routine and multiplies that result by 2. If the provided subshell string is not recognized by the internal degeneracy lookup or parser, an error raised by the underlying subshell_degeneracy implementation will propagate to the caller (i.e., invalid or malformed subshell identifiers will not return a meaningful capacity). Callers should validate or handle such errors when supplying subshell strings extracted from external sources.
    """
    from mendeleev.econf import subshell_capacity
    return subshell_capacity(subshell)


################################################################################
# Source: mendeleev.econf.subshell_degeneracy
# File: mendeleev/econf.py
# Category: fix_docstring
# Reason: Schema parsing failed: Cannot generate JSON schema for subshell_degeneracy because the docstring has no description for the argument 'subshell'
################################################################################

def mendeleev_econf_subshell_degeneracy(subshell: str):
    """mendeleev.econf.subshell_degeneracy returns the degeneracy (number of magnetic sublevels/orbitals) associated with an atomic electronic subshell label.
    
    This utility is used in the mendeleev package when working with electronic configurations (see the project's electronic configuration tutorials and API). Internally it obtains the azimuthal quantum number l for the provided subshell label (by calling get_l) and computes the degeneracy as 2*l + 1, which corresponds to the number of distinct magnetic quantum number (m_l) values and therefore the number of orbitals in that subshell. This value is commonly used when parsing or validating electronic configurations, building periodic-table visualizations of orbital occupancy, or computing the maximum number of electrons that a subshell can hold (which is 2 * degeneracy when spin is included).
    
    Args:
        subshell (str): A string identifying an atomic subshell whose degeneracy is requested. Typical, conventional labels are strings such as '1s', '2p', '3d', '4f' where the numeric prefix is the principal quantum number and the letter denotes the orbital type. The function does not modify the input. The provided string is passed to the internal get_l(subshell) helper to determine the azimuthal quantum number l; therefore the accepted formats and semantics of subshell strings follow those supported by get_l.
    
    Returns:
        int: The degeneracy of the given subshell computed as 2*l + 1, where l is the azimuthal quantum number derived from the subshell string. This integer equals the number of magnetic sublevels (m_l values) and the number of orbitals in that subshell. To obtain the maximum electron capacity of the subshell including spin, multiply this return value by 2.
    
    Behavior and failure modes:
        This function is pure and has no side effects. It relies on get_l(subshell) to parse the subshell string and determine l; if the input string is not a valid subshell label according to get_l, the exception raised by get_l (for example a parsing or value error) will propagate to the caller. There are no default values. The function always returns an integer when successful.
    """
    from mendeleev.econf import subshell_degeneracy
    return subshell_degeneracy(subshell)


################################################################################
# Source: mendeleev.electronegativity.allred_rochow
# File: mendeleev/electronegativity.py
# Category: valid
################################################################################

def mendeleev_electronegativity_allred_rochow(zeff: float, radius: float):
    """Calculate the electronegativity of an atom according to the Allred and Rochow definition.
    
    This function implements the mathematical core of the Allred & Rochow electronegativity scale as used in the mendeleev package's "Electronegativity scales" utilities. The returned value is the proportional Allred–Rochow electronegativity computed as zeff / radius**2. In the context of mendeleev, zeff is typically derived from nuclear screening constants or other effective nuclear charge estimates available in the package data tables, and radius should be a corresponding atomic-size value (for example an atomic or covalent radius drawn from the size-related properties in mendeleev). The numerical value and its comparability to literature Allred–Rochow numbers depend on using consistent radius values and units with the source of zeff.
    
    Args:
        zeff (float): Effective nuclear charge for the atom. This is a dimensionless measure of the net positive charge experienced by the valence electrons after accounting for electron screening. In mendeleev this value is commonly obtained from nuclear screening constants (Slater/Clementi) or computed effective charges and serves as the numerator in the Allred–Rochow formula.
        radius (float): Value of the radius used in the denominator of the Allred–Rochow expression. In practice this should be an atomic-size measure (for example a covalent or atomic radius available in mendeleev's size-related properties). The value is squared in the formula, so units must be consistent with those used to derive zeff when comparing results to published Allred–Rochow electronegativities.
    
    Returns:
        float: The Allred–Rochow electronegativity in the proportional form computed as zeff / radius**2. This returned float represents the scale value produced by this formula; to match published numeric Allred–Rochow values a dataset-specific scaling or consistent unit choice for radius may be required.
    
    Raises:
        ZeroDivisionError: If radius is zero and Python floating-point division semantics are used, a ZeroDivisionError will be raised. If NumPy types are used for the inputs, division by zero may instead produce an infinite (inf) or NaN result and emit a runtime warning.
        ValueError: The function does not itself validate that radius is positive; negative radius values are physically invalid for atomic radii but will produce a numeric result because the radius is squared. Callers should validate inputs before calling if strict physical constraints are required.
    
    Side effects:
        None. This function is pure: it does not modify its inputs or any external state.
    """
    from mendeleev.electronegativity import allred_rochow
    return allred_rochow(zeff, radius)


################################################################################
# Source: mendeleev.electronegativity.cottrell_sutton
# File: mendeleev/electronegativity.py
# Category: valid
################################################################################

def mendeleev_electronegativity_cottrell_sutton(zeff: float, radius: float):
    """mendeleev.electronegativity.cottrell_sutton computes the Cottrell–Sutton electronegativity (the Allred & Rochow style) for an atom from an effective nuclear charge and an atomic radius. This function is part of the mendeleev package electronegativity scales (Cottrell & Sutton entry) and is used to derive a scalar electronegativity value for comparing elements, visualizing periodic trends, and supplying computed properties for the periodic table API.
    
    Args:
        zeff (float): Effective nuclear charge for the atom. In the context of mendeleev this value represents the screened nuclear charge experienced by valence electrons (z* or Zeff) used in electronegativity model calculations. It should be provided as a floating point numeric value consistent with how Zeff is computed or stored in the caller's workflow; nonphysical negative values are not meaningful for the chemical interpretation.
        radius (float): A characteristic radius used in the Cottrell–Sutton formula. This is a floating point value representing an atomic/ionic length scale (the code expects the radius in the same length units used elsewhere in the caller's dataset). The radius must be nonzero and, for meaningful chemical results, positive; zero or negative values will produce invalid mathematical arguments to the square root.
    
    Returns:
        float: The computed electronegativity according to the Cottrell–Sutton / Allred & Rochow relation, computed as sqrt(zeff / radius). The returned value is a Python float (produced via numpy.sqrt in the implementation) that can be used directly for comparisons, plotting, or storing in element property tables.
    
    Behavior and failure modes:
        The function is a pure computation with no side effects. It deterministically returns numpy.sqrt(zeff / radius). If radius is zero a division-by-zero will occur (numpy will emit warnings and produce inf or raise depending on numpy settings). If zeff/radius is negative, numpy.sqrt will produce NaN (and may emit a runtime warning); the function does not perform additional validation or raise exceptions for these conditions. Callers should validate inputs (ensure zeff >= 0 and radius > 0) when they require physically meaningful electronegativity values.
    """
    from mendeleev.electronegativity import cottrell_sutton
    return cottrell_sutton(zeff, radius)


################################################################################
# Source: mendeleev.electronegativity.generic
# File: mendeleev/electronegativity.py
# Category: valid
################################################################################

def mendeleev_electronegativity_generic(
    zeff: float,
    radius: float,
    rpow: float = 1,
    apow: float = 1
):
    """mendeleev.electronegativity.generic: Calculate an element's electronegativity using a general power-law formula that combines an effective nuclear charge and an atomic radius. This function implements the formula chi = (Z_eff / r**rpow)**apow where Z_eff is the effective nuclear charge and r is a radius (typically a covalent radius from the mendeleev data tables). It is used within the mendeleev package to derive electronegativity-like, dimensionless indices from elemental properties and can be applied when implementing or comparing different electronegativity scales that depend on effective charge and size.
    
    Args:
        zeff (float): Effective nuclear charge (Z_eff) for the element. In the domain of the mendeleev package this value is expected to represent the screened nuclear charge experienced by valence electrons and should be a non-negative floating-point number. Passing negative or non-physical values may produce invalid results (NaN or inf) or runtime warnings when combined with non-integer exponents.
        radius (float): Radius value for the element (r). In typical usage this is a covalent radius drawn from the package data (see README "Size related properties"). The radius must be provided as a positive float; radius values of zero or negative values are physically meaningless and will lead to division-by-zero, infinities, NaNs, or runtime warnings from the underlying numpy operations.
        rpow (float): Power to raise the radius to (beta in the equation). This parameter controls how strongly the radius attenuates the effective nuclear charge; the default value is 1. Provide this as a floating-point exponent. Non-integer or negative exponents are permitted mathematically but may amplify sensitivity to small/zero radius values and can produce non-finite results for non-positive radii.
        apow (float): Power to raise the fraction to (alpha in the equation). This parameter controls the overall scaling of the computed electronegativity; the default value is 1. Provide this as a floating-point exponent. Non-integer apow applied to negative bases (which can occur if zeff or the fraction is negative) may produce invalid results.
    
    Returns:
        float: The computed electronegativity (χ) as a floating-point, dimensionless value defined by χ = (zeff / radius**rpow)**apow. The result is intended as a unitless index comparable across elements in the context of mendeleev electronegativity scales. If inputs are non-physical (for example radius <= 0 or negative zeff with non-integer exponents) the function may return numpy.inf, numpy.nan, or produce runtime warnings; callers should validate inputs to avoid these conditions.
    
    Behavior and side effects:
        This is a pure numerical function with no external side effects; it uses numpy.power for exponentiation and returns a floating-point result. Default parameter values (rpow=1, apow=1) yield the simple ratio zeff/radius. The function does not perform unit conversion—radius and zeff must be provided in the units/definitions consistent with the intended scale. The function does not intentionally raise Python exceptions for zero or invalid numeric inputs; instead, numpy may emit runtime warnings and return IEEE floating-point infinities or NaNs. Validate inputs before calling if deterministic exceptions are required.
    """
    from mendeleev.electronegativity import generic
    return generic(zeff, radius, rpow, apow)


################################################################################
# Source: mendeleev.electronegativity.gordy
# File: mendeleev/electronegativity.py
# Category: valid
################################################################################

def mendeleev_electronegativity_gordy(zeff: float, radius: float):
    """mendeleev.electronegativity.gordy: Compute the Gordy electronegativity of an atom using effective nuclear charge and atomic radius.
    
    Computes a simple Gordy-style electronegativity estimate by dividing the effective nuclear charge by a radius value. This function is part of the mendeleev package electronegativity implementations (see the README electronegativity scales entry for Gordy) and is used by higher-level code and visualizations to produce an electronegativity value for comparing elements and illustrating periodic trends. The result is a scalar float proportional to zeff/radius; the function performs a direct numeric division and does not modify inputs or external state. Users must ensure the radius supplied corresponds to the same radius convention and units used elsewhere in their analysis (for example one of the atomic radius values available in mendeleev data); inconsistent radius units will change the numerical scale of the returned electronegativity.
    
    Args:
        zeff (float): Effective nuclear charge for the atom. In the mendeleev context this is the net positive charge "felt" by valence electrons after screening by inner electrons and is used as the numerator in the Gordy estimate. Pass the numeric effective nuclear charge value computed or retrieved from mendeleev data or other source.
        radius (float): Radius value to use as the denominator in the Gordy estimate. Typically this is an atomic or ionic radius value obtained from mendeleev properties (the caller must ensure the chosen radius type and units are appropriate and consistent with zeff). This value must be a non-zero float for a meaningful result.
    
    Returns:
        float: A floating-point electronegativity estimate on the Gordy-like scale equal to zeff / radius. The returned value is deterministic and intended for relative comparisons and plotting of periodic trends within the mendeleev framework.
    
    Raises:
        ZeroDivisionError: If radius is zero, since the implementation performs a direct division.
        TypeError: If the provided zeff or radius cannot be used in numeric division (for example if they are non-numeric objects).
    """
    from mendeleev.electronegativity import gordy
    return gordy(zeff, radius)


################################################################################
# Source: mendeleev.electronegativity.interpolate_property
# File: mendeleev/electronegativity.py
# Category: fix_docstring
# Reason: Schema parsing failed: Cannot generate JSON schema for interpolate_property because the docstring has no description for the argument 'poly_deg'
################################################################################

def mendeleev_electronegativity_interpolate_property(
    x: int,
    x_ref: List[int],
    y_ref: List[float],
    poly_deg: int = 1
):
    """Interpolate or extrapolate a numeric property for an element using reference element indices and property values.
    
    Args:
        x (int): The target independent variable value for which the property will be estimated. In the domain of the mendeleev package this represents an element identifier on the numeric axis used for the reference data (for example an atomic number or any integer index used consistently in x_ref). The function evaluates the property at this single integer x by either interpolating within the provided reference range or by extrapolating beyond it.
        x_ref (List[int]): A list of integer independent-variable reference points corresponding to known elements or ordered element indices. These are the x-coordinates of the known data points (for example atomic numbers for which the property in y_ref is known). The list must be non-empty and its values define the inclusive interval used to decide whether to interpolate (x within min(x_ref) .. max(x_ref)) or extrapolate (x outside that interval). The routine converts this sequence to a numpy array internally.
        y_ref (List[float]): A list of floating-point dependent-variable reference values corresponding one-to-one with x_ref. Each entry is the known property value for the element/index in the same position in x_ref (for example electronegativity values, radii, energies, etc.). The lengths of x_ref and y_ref must match; otherwise numpy functions called internally (np.interp or np.polyfit) will raise an error. The sequence is converted to a numpy array internally.
        poly_deg (int): Degree of the polynomial used for extrapolation beyond the provided data points. Default is 1 which selects a linear polynomial fit for extrapolation. This integer controls np.polyfit called on a small slice of the reference data when x lies outside the inclusive range defined by x_ref. Choosing a poly_deg larger than the number of reference points used for the fit or an inappropriate value may cause numpy to emit warnings or raise exceptions (for example from np.polyfit); the caller is responsible for selecting a sensible degree given the available data.
    
    Returns:
        float: A scalar numeric estimate of the property at x. If x is between min(x_ref) and max(x_ref) (inclusive) the function uses linear interpolation via numpy.interp and returns the interpolated value for x. If x is outside that inclusive range the function selects a slice of three reference points from the end closest to x (the first three if x < min(x_ref), or the last three if x > max(x_ref)), fits a polynomial of degree poly_deg to that slice using numpy.polyfit, evaluates the fitted polynomial at x, and returns that value. Note: numpy is used internally and may return numpy scalar types; the returned value represents a single floating-point estimate of the property.
    
    Behavior, defaults and failure modes:
        The function does not modify global state; it only reads the provided sequences and uses numpy routines to compute the result. For interpolation the algorithm is linear and uses numpy.interp. For extrapolation it uses numpy.polyfit on up to three reference points as described above and evaluates the resulting polynomial with numpy.poly1d.
        If x_ref is empty, not numeric, or x_ref and y_ref have unequal lengths, numpy calls will raise an exception (for example AttributeError, ValueError, or a numpy-specific error). If poly_deg is too large relative to the number of points available for fitting, numpy.polyfit may raise a ValueError or produce a RankWarning; the caller should choose poly_deg appropriate for the available data. The function expects the inputs to follow the types in the signature; passing other types may lead to implicit conversion by numpy or to runtime errors.
    """
    from mendeleev.electronegativity import interpolate_property
    return interpolate_property(x, x_ref, y_ref, poly_deg)


################################################################################
# Source: mendeleev.electronegativity.li_xue
# File: mendeleev/electronegativity.py
# Category: fix_docstring
# Reason: Schema parsing failed: Cannot generate JSON schema for li_xue because the docstring has no description for the argument 'ionization_energy'
################################################################################

def mendeleev_electronegativity_li_xue(
    ionization_energy: float,
    radius: float,
    valence_pqn: int
):
    """mendeleev.electronegativity.li_xue computes the electronegativity of an atom or ion according to the Li & Xue definition used in the mendeleev package. It implements the formula used in the source code:
    n_effective(valence_pqn, source="zhang") * sqrt(ionization_energy / RY) * 100.0 / radius, where RY is the Rydberg energy constant imported in this module and n_effective(...) is computed with the "zhang" prescription.
    
    Args:
        ionization_energy (float): First ionization energy (numeric). This value must be expressed in the same energy units as the module-level RY constant (the Rydberg energy) because the implementation divides ionization_energy by RY prior to taking the square root. In the Li & Xue electronegativity model this term represents the ionization energy contribution to orbital binding and therefore to electronegativity. Passing a negative ionization_energy will raise a ValueError from math.sqrt. The caller is responsible for ensuring the value is in the correct units and is physically meaningful (non-negative).
        radius (float): Effective atomic or ionic radius as a numeric scalar. This radius is used as the denominator in the Li & Xue formula, so it must be non-zero; a zero radius will raise a ZeroDivisionError. Provide the radius in the same length units consistently used elsewhere in the codebase (for example, the same units returned by element attributes such as crystal_radius or ionic_radius) because the formula scales inversely with radius. A larger radius decreases the computed electronegativity in this model.
        valence_pqn (int): Valence principal quantum number (n) for the atom's valence shell. This integer is passed to n_effective(valence_pqn, source="zhang") to compute an effective principal quantum number following the "zhang" prescription used in this implementation. The effective quantum number scales the overall result; valence_pqn should be a positive integer consistent with the element's valence shell (for example, 1, 2, 3, ...). Supplying a non-integer or nonsensical quantum number may produce an incorrect result or a downstream error from n_effective.
    
    Returns:
        float: The Li & Xue electronegativity value computed for the provided inputs. Higher returned values indicate greater electronegativity according to the Li & Xue scale as implemented in this package. The return value is a pure numeric computation with no side effects. Possible failure modes include ValueError if ionization_energy is negative (square root of negative), ZeroDivisionError if radius is zero, and TypeError if arguments are of incompatible types. The function relies on the module-level constant RY and the n_effective(...) helper (called with source="zhang"); inconsistencies between the units of ionization_energy or radius and those expected by these helpers will produce physically incorrect results without additional runtime errors.
    """
    from mendeleev.electronegativity import li_xue
    return li_xue(ionization_energy, radius, valence_pqn)


################################################################################
# Source: mendeleev.electronegativity.martynov_batsanov
# File: mendeleev/electronegativity.py
# Category: valid
################################################################################

def mendeleev_electronegativity_martynov_batsanov(ionization_energies: List[float]):
    """mendeleev.electronegativity.martynov_batsanov: Compute the Martynov & Batsanov electronegativity value (χ_MB) for an element by taking the square root of the arithmetic mean of its valence ionization energies. This implementation is used within the mendeleev package as one of the available electronegativity scales (see the package electronegativity scales including "Martynov & Batsanov" in the README) and is intended to consume the same ionization energy data that the library exposes (for example the element.ionization_energies property).
    
    The function implements the formula χ_MB = sqrt((1 / n_v) * sum_{k=1..n_v} I_k) where n_v is the number of valence electrons (the length of the provided list) and I_k are the ionization energies provided in ionization_energies.
    
    Args:
        ionization_energies (List[float]): Ionization energies for the valence electrons. Each entry must be a numeric float value representing an individual ionization potential for a valence electron. Typical usage is to pass the list of valence ionization energies obtained from mendeleev data tables or an element object's ionization_energies attribute. The order of values is not significant because the arithmetic mean is taken; the number of entries defines n_v (the number of valence electrons) used in the formula.
    
    Returns:
        float: The Martynov & Batsanov electronegativity value χ_MB computed as the square root of the arithmetic mean of the supplied ionization energies. The returned float is computed via numpy operations (np.sqrt and np.mean) and therefore will follow numpy's numeric semantics.
    
    Behavior, defaults, and failure modes:
        This function is pure and has no side effects. It converts the input list to a numpy array and computes its mean, then returns the square root of that mean. If ionization_energies is an empty list, numpy.mean produces a NaN and a RuntimeWarning may be emitted; the function will then return NaN. If the array conversion encounters non-numeric types, numpy will raise a TypeError or produce an array containing object dtype which will typically lead to an error when computing the mean. If the arithmetic mean of the provided values is negative (which is not physically expected for ionization energies), np.sqrt will return NaN (no complex numbers are produced). The function does not perform unit validation: callers must ensure values are expressed in consistent energy units compatible with other mendeleev data they use. Computationally, the function runs in O(n) time where n is the length of ionization_energies.
    """
    from mendeleev.electronegativity import martynov_batsanov
    return martynov_batsanov(ionization_energies)


################################################################################
# Source: mendeleev.electronegativity.mulliken
# File: mendeleev/electronegativity.py
# Category: valid
################################################################################

def mendeleev_electronegativity_mulliken(ionization_energy: float, electron_affinity: float):
    """mendeleev.electronegativity.mulliken computes the absolute (Mulliken) electronegativity for an element or species by averaging its ionization energy and electron affinity. This function is used in the mendeleev package to provide one of the standardized electronegativity scales (Mulliken) described in the project README and is useful for comparing tendencies of elements to attract electrons when analyzing periodic trends, generating periodic table visualizations, or computing derived chemical descriptors.
    
    Args:
        ionization_energy (float): The ionization energy I of the element or species. This is the energy required to remove an electron and is provided as a numeric energy value. The function accepts None at runtime (the implementation checks for None), in which case the function returns None because ionization energy is required to define the Mulliken value. Practically, supply ionization_energy in the same energy units as electron_affinity (for example both in eV or both in kJ/mol) so the average has a consistent physical meaning. Providing a non-numeric type will typically raise a TypeError during arithmetic operations.
        electron_affinity (float): The electron affinity A of the element or species. This is the energy change when an electron is added. If electron_affinity is None, the function falls back to using only the ionization_energy and returns half of that value (I / 2). As with ionization_energy, supply this value in the same units as ionization_energy; supplying a non-numeric type will typically raise a TypeError.
    
    Behavior:
        The Mulliken electronegativity χ is calculated as the arithmetic mean of ionization energy and electron affinity:
        χ = (I + A) / 2
        where I is ionization_energy and A is electron_affinity. If electron_affinity is missing (None), the implementation returns I / 2 as a pragmatic fallback. If ionization_energy is None, the function cannot compute a meaningful value and returns None. There are no external side effects; the function performs a pure calculation. Errors from invalid arithmetic (for example, passing incompatible types) propagate as standard Python exceptions (TypeError, ValueError) raised by the Python runtime.
    
    Returns:
        Optional[float]: The computed Mulliken electronegativity value (numeric) when ionization_energy is provided. Returns None if ionization_energy is None. If electron_affinity is None but ionization_energy is provided, returns ionization_energy * 0.5. The returned numeric value has the same energy-unit basis as the input arguments and serves as the Mulliken electronegativity used in periodic-trend analyses and visualizations within the mendeleev package.
    """
    from mendeleev.electronegativity import mulliken
    return mulliken(ionization_energy, electron_affinity)


################################################################################
# Source: mendeleev.electronegativity.n_effective
# File: mendeleev/electronegativity.py
# Category: valid
################################################################################

def mendeleev_electronegativity_n_effective(n: int, source: str = "slater"):
    """mendeleev.electronegativity.n_effective returns the effective principal quantum number (n*) used by the mendeleev package for approximate atomic orbital and electronegativity calculations. This scalar value is taken from published parameter sets (Slater or Zhang) and is used in empirical formulas that require an effective principal quantum number to represent screening and orbital contraction effects in atoms and ions.
    
    This function looks up a published effective principal quantum number for a given principal quantum number and source. The implementation currently supports two named sources: 'slater' and 'zhang'. Slater values are taken from J. A. Pople and D. L. Beveridge, "Approximate Molecular Orbital Theory", McGraw-Hill, 1970. Zhang values are taken from Zhang, Y. (1982). Electronegativities of elements in valence states and their applications. Inorganic Chemistry, 21(11), 3886–3889. The function performs no I/O and has no side effects beyond returning a numeric value or None; it will raise an exception if an unknown source is requested.
    
    Args:
        n (int): Principal quantum number. This is the hydrogen-like shell index (1 for K-shell, 2 for L-shell, etc.) used to select the effective principal quantum number from the chosen published dataset. Valid integer n values are determined by the selected source: for 'slater' the implemented keys are 1, 2, 3, 4, 5, 6; for 'zhang' the implemented keys are 1, 2, 3, 4, 5, 6, 7. If an n value outside the implemented keys is supplied for a supported source, the function returns None (no side effects).
        source (str): Identifier of the published dataset to use; must be one of 'slater' or 'zhang'. Default is 'slater'. 'slater' selects values from Pople & Beveridge (1970) used for Slater-type screening approximations; 'zhang' selects values from Zhang (1982) used in empirical electronegativity assignments. If an unknown source string is provided, the function raises a ValueError listing the available sources.
    
    Returns:
        Optional[float]: The effective principal quantum number (n*) as a floating-point value when a mapping for the given n exists in the chosen source. If the chosen source is valid but does not define a value for the provided n, the function returns None to indicate no available parameter. If the source argument itself is not recognized, the function raises ValueError rather than returning.
    """
    from mendeleev.electronegativity import n_effective
    return n_effective(n, source)


################################################################################
# Source: mendeleev.electronegativity.nagle
# File: mendeleev/electronegativity.py
# Category: valid
################################################################################

def mendeleev_electronegativity_nagle(nvalence: int, polarizability: float):
    """mendeleev.electronegativity.nagle: Compute the Nagle electronegativity from valence electron count and dipole polarizability.
    
    This function implements the Nagle definition of electronegativity used in the mendeleev package's collection of electronegativity scales. It is used to derive an element's electronegativity value from two elemental descriptors: the number of valence electrons and the dipole polarizability. The returned scalar is suitable for comparing relative electronegativities across elements in periodic trends, data tables, visualizations, and downstream analyses provided by mendeleev.
    
    Args:
        nvalence (int): Number of valence electrons for the atom. This integer represents the count of electrons in the outermost shells that participate in bonding and chemical interactions; it is a primary descriptor in empirical electronegativity models such as Nagle's. Supplying a negative integer is non-physical and may lead to non-meaningful results.
        polarizability (float): Dipole polarizability of the atom. This floating-point scalar represents the ease with which the electron cloud of the atom is distorted by an external electric field and is the denominator in Nagle's formula. The value must be non-zero and in a physically meaningful (positive) range for a valid electronegativity; passing zero will raise a ZeroDivisionError and passing negative or otherwise non-physical values can produce NaN or complex results due to the fractional power operation.
    
    Returns:
        float: Electronegativity value on the Nagle scale computed as (nvalence / polarizability) ** (1/3). The return is a single floating-point scalar (dimensionless in the implementation) representing the Nagle electronegativity for the provided inputs. No side effects occur. Failure modes: if polarizability == 0.0 a ZeroDivisionError will be raised by the division; if polarizability or nvalence are negative or non-physical, numpy's fractional-power behavior may produce NaN or complex values (and may emit runtime warnings).
    """
    from mendeleev.electronegativity import nagle
    return nagle(nvalence, polarizability)


################################################################################
# Source: mendeleev.electronegativity.sanderson
# File: mendeleev/electronegativity.py
# Category: fix_docstring
# Reason: Schema parsing failed: Cannot generate JSON schema for sanderson because the docstring has no description for the argument 'radius'
################################################################################

def mendeleev_electronegativity_sanderson(radius: float, noble_gas_radius: float):
    """mendeleev.electronegativity.sanderson computes Sanderson's electronegativity for an element using the ratio of a hypothetical noble-gas radius and the element's radius raised to the third power; this function is part of the mendeleev package's set of electronegativity scales used to analyse periodic trends and to supply values for visualization and data tables described in the project's README.
    
    Args:
        radius (float): Radius value for the element whose Sanderson electronegativity is being calculated. This is a linear measure of atomic size (must be provided in the same units as noble_gas_radius). There is no default; the caller must supply a finite floating-point value. If radius is zero a ZeroDivisionError will occur; if radius is non-finite (NaN or ±inf) the result will propagate that non-finite value. Physically, radius should be a positive float representing an atomic or ionic radius taken from the dataset or a computed value used within the mendeleev framework.
        noble_gas_radius (float): Radius value of a hypothetical noble gas having the same atomic number as the element for which electronegativity is calculated. This value is used as the normalization reference (AD_ng in Sanderson's formulation) and must be expressed in the same units as radius. It should be a finite float; non-finite inputs will propagate through the computation. In the context of the mendeleev package this parameter is typically taken from the dataset or computed to represent the reference noble-gas atomic size used when constructing the Sanderson electronegativity scale.
    
    Returns:
        float: The Sanderson electronegativity (unitless) computed as (noble_gas_radius / radius) ** 3. The returned value is a relative, dimensionless measure: larger values indicate higher electronegativity on Sanderson's scale for the given input radii. This function has no side effects; it performs a pure numerical calculation and returns the computed float.
    """
    from mendeleev.electronegativity import sanderson
    return sanderson(radius, noble_gas_radius)


################################################################################
# Source: mendeleev.fetch.fetch_ionic_radii
# File: mendeleev/fetch.py
# Category: valid
################################################################################

def mendeleev_fetch_fetch_ionic_radii(radius: str = "ionic_radius"):
    """Fetch a pandas.DataFrame of ionic radii for all elements and ions available in the package data.
    
    This function, mendeleev.fetch.fetch_ionic_radii, is used by the mendeleev package to retrieve size-related properties (ionic radii and crystal radii) from the internal "ionicradii" data table (part of the mendeleev data assets). It is intended for downstream analysis and visualization of element and ion size trends (for example, comparing radii across coordination numbers or plotting periodic trends) and returns a pivoted table keyed by atomic number and ionic charge with coordination numbers as columns.
    
    Args:
        radius (str): The radius column to return from the underlying "ionicradii" table. Must be either 'ionic_radius' (the default) to obtain ionic radii as reported in the data source, or 'crystal_radius' to obtain crystal radii when available. The parameter selects which numeric radius values (as provided in the mendeleev data assets) will populate the pivoted table.
    
    Returns:
        pandas.core.frame.DataFrame: A pandas DataFrame produced by pivoting the internal "ionicradii" table. The returned DataFrame has a MultiIndex in the rows with levels ("atomic_number", "charge") where "atomic_number" is the element atomic number and "charge" is the ionic charge. The columns correspond to coordination numbers (taken from the "coordination" column in the source table). The cell values are the numeric radii selected by the radius argument. If multiple source rows map to the same (atomic_number, charge, coordination) combination, pandas.pivot_table will aggregate them using the default aggregation (mean), and missing combinations will be represented as NaN.
    
    Raises:
        ValueError: If radius is not one of the accepted string identifiers: 'ionic_radius' or 'crystal_radius'. In that case the function raises a ValueError indicating the radius name was not found and listing the available options (the function uses the exact message constructed in the implementation). The function may also propagate exceptions raised by the underlying fetch_table call (for example, if the internal data table cannot be loaded), which callers should handle as appropriate.
    
    Behavior and side effects:
        The function calls fetch_table("ionicradii") to load the raw ionic radii table from the package data assets; this is the only side effect (data access) performed. No files are written. The returned DataFrame is a view constructed by pivot_table and is suitable for immediate analysis or visualization within the mendeleev API (for example, joining with Element objects or plotting coordination-dependent radius trends). Default behavior uses 'ionic_radius' which yields ionic radii commonly used for ionic size comparisons; choose 'crystal_radius' when crystal-specific radii are required.
    """
    from mendeleev.fetch import fetch_ionic_radii
    return fetch_ionic_radii(radius)


################################################################################
# Source: mendeleev.mendeleev.get_attribute_for_all_elements
# File: mendeleev/mendeleev.py
# Category: fix_docstring
# Reason: Schema parsing failed: Cannot generate JSON schema for get_attribute_for_all_elements because the docstring has no description for the argument 'attribute'
################################################################################

def mendeleev_mendeleev_get_attribute_for_all_elements(attribute: str):
    """Return a list of values for a single attribute for all elements stored in the mendeleev element database. This function is used within the mendeleev package (a Pythonic periodic table of elements) to extract one column of element data (for example "atomic_number", "name", "atomic_weight", "thermal_conductivity" or any other property documented in the README data sections) from the underlying SQL-backed element table and return those values in a simple Python list ordered by increasing atomic number. Practical uses include preparing a column of values for bulk analysis, plotting periodic trends, building custom periodic-table visualizations, or exporting a single property for all elements.
    
    Args:
        attribute (str): The attribute name to retrieve from every Element row in the database. This is the literal attribute name on the Element ORM class (for example one of the properties listed in the README such as "atomic_number", "name", "atomic_weight", "symbol", "isotopes", "thermal_conductivity", etc.). The function uses getattr(Element, attribute) to build the query, so the string must exactly match an existing attribute defined on the Element class; the attribute identifies which column/property of the Element table will be returned. This parameter controls which single column of element data is extracted from the database and determines the type of objects contained in the returned list (for example int for atomic_number, str for name, float for atomic_weight, or user-defined objects for composite properties such as isotopes).
    
    Returns:
        List: A Python list containing one entry per element present in the database, where each entry is the value of the requested attribute for that element. The values are returned in ascending order of Element.atomic_number (so the first list item corresponds to atomic number 1, the second to atomic number 2, and so on). The concrete element count and types of list items depend on the database contents and the requested attribute; missing values in the database will be returned as the corresponding Python value (for example None) as stored.
    
    Behavior and side effects:
        The function opens a database connection by calling get_engine() and creating a Session, executes a query selecting the requested attribute for all Element rows, orders results by Element.atomic_number, and materializes the results into a Python list. Side effects include establishing and closing a database session; no modifications are made to the database. Performance depends on the database backend and the number of elements in the table.
    
    Failure modes:
        If attribute does not correspond to a valid attribute on the Element class, getattr(Element, attribute) will raise an AttributeError before the query is executed. If the database engine cannot be obtained or the Session cannot connect to the underlying database, a database-related exception will be raised by the underlying engine/session implementation. If the query succeeds but some elements lack a value for the attribute, those entries will appear in the returned list as the stored representation of missing data (for example None).
    """
    from mendeleev.mendeleev import get_attribute_for_all_elements
    return get_attribute_for_all_elements(attribute)


################################################################################
# Source: mendeleev.mendeleev.ids_to_attr
# File: mendeleev/mendeleev.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def mendeleev_mendeleev_ids_to_attr(ids: list, attr: str = "atomic_number"):
    """Convert element identifiers (atomic numbers, atomic symbols, or English element names) to a list of attribute values for the corresponding Element objects from the mendeleev periodic-table API.
    
    This function is used in the mendeleev package to map user-supplied element identifiers into concrete element properties that are commonly used when accessing the periodic table data programmatically (for example in the CLI, tutorials, or when preparing data for pandas/bokeh visualizations). It resolves identifiers via the package's element() helper, obtains the requested attribute from each resolved Element using getattr, and returns the attribute values in a list preserving the input order. The default attribute is "atomic_number", which yields a list of atomic numbers for the provided identifiers.
    
    Args:
        ids (list, tuple, int or str): One or multiple element identifiers to resolve. Accepted identifier forms are documented in the mendeleev README and include atomic numbers (int), atomic symbols (str, e.g., "Fe"), and English element names (str, e.g., "Iron"). If a list or tuple is provided, each item may be any supported identifier type and the function returns a list with one entry per item. If a single int or str is provided, the function returns a single-element list containing the requested attribute for that element. This parameter is the primary input used to select which element properties are returned and may be mixed (e.g., [26, "Fe", "Oxygen"]) to obtain attributes for multiple elements in one call.
    
        attr (str): The name of the attribute to retrieve from each resolved Element object. This must be the exact attribute name as exposed on mendeleev.Element instances (for example, "atomic_number", "name", "symbol", "thermal_conductivity", "isotopes", etc., as listed in the package documentation). The default value is "atomic_number". The attribute is obtained via Python's getattr, so its meaning and type follow how the attribute is implemented on the Element object (it may be a scalar like an int/float or a complex object such as a list of Isotope objects).
    
    Returns:
        list: A list containing the requested attribute values for each resolved element, in the same order as the input identifiers. If a single identifier was provided, a one-element list is returned. The list elements are the raw attribute values returned by the Element instances (for example, integers for "atomic_number", strings for "name", lists for "isotopes").
    
    Behavior, defaults, and failure modes:
        The function delegates identifier resolution to element(); therefore any exceptions raised by element() for unknown or invalid identifiers (for example when an identifier cannot be resolved to an Element) are propagated to the caller. If attr is not a valid attribute on the resolved Element object, getattr will raise AttributeError which is propagated. There are no side effects: the function does not modify Element objects or any global state. Tuples are treated the same as lists. The function always returns a list (never a bare scalar), and the default attribute returned when attr is omitted is "atomic_number".
    """
    from mendeleev.mendeleev import ids_to_attr
    return ids_to_attr(ids, attr)


################################################################################
# Source: mendeleev.models.fetch_by_group
# File: mendeleev/models.py
# Category: fix_docstring
# Reason: Schema parsing failed: The type hint tuple[list[Any]] is a Tuple with a single element, which we do not automatically convert to JSON schema as it is rarely necessary. If this input can contain more than one element, we recommend using a List[] type instead, or if it really is a single element, remove the Tuple[] wrapper and just pass the element directly.
################################################################################

def mendeleev_models_fetch_by_group(properties: List[str], group: int = 18):
    """mendeleev.models.fetch_by_group retrieves specified Element attributes for every element in a given periodic-table group.
    
    Args:
        properties (List[str]): One or more attribute names of the Element model to retrieve for each element in the group. Typical attributes are those listed in the package README under "Basic properties" and "Physical properties" (for example, "atomic_number", "name", "atomic_weight", "melting_point", "ionization_energies"). The function also accepts a single attribute as a str (the implementation checks isinstance(properties, str) and will convert it to a one-item list). The order of attributes in this sequence determines the order of values in each returned row. If "atomic_number" is not included in this sequence it will be automatically prepended so that results include the atomic number and are ordered reliably.
        group (int): Periodic-table group number used to filter elements (Element.group_id == group). This corresponds to the chemical group/column in the periodic table; the default value is 18 (the noble gases). Use this parameter to request data for all elements that belong to the specified group.
    
    Returns:
        tuple[list[Any]]: A sequence (the return value produced by SQLAlchemy Query.all()) where each item corresponds to one element in the requested group and contains the requested attribute values in the same order as the properties argument. Concretely, the outer sequence is the collection of rows for all group members ordered by atomic_number (ascending), and each row is a sequence (tuple) of attribute values (which can be of varying Python types depending on the attribute, hence Any). If no elements match the group filter an empty sequence is returned.
    
    Behavior and side effects:
        This function opens a database session using get_session(), constructs a SQLAlchemy query by converting each requested property name to getattr(Element, prop), filters elements by Element.group_id == group, orders results by "atomic_number", executes the query with .all(), closes the session, and returns the fetched rows. Because the function constructs attribute access at runtime, requesting a property name that is not an attribute of the Element model will raise an AttributeError. Database connectivity issues or SQLAlchemy execution errors will propagate (for example OperationalError or InvalidRequestError) and are not suppressed. The function ensures the session is closed after the query completes (normal return path); if an exception occurs during query construction or execution the session may still be closed by higher-level session management but callers should be prepared to handle exceptions.
    
    Practical significance:
        Use this function when you need tabular data for all members of a chemical group (for example to populate a pandas DataFrame, feed a visualization that compares properties across a group, or compute periodic trends). Because "atomic_number" is always included and results are ordered by atomic_number, the output is suitable for tasks that rely on element ordering within the group.
    """
    from mendeleev.models import fetch_by_group
    return fetch_by_group(properties, group)


################################################################################
# Source: mendeleev.models.with_uncertainty
# File: mendeleev/models.py
# Category: valid
################################################################################

def mendeleev_models_with_uncertainty(value: float, uncertainty: float, digits: int = 5):
    """mendeleev.models.with_uncertainty formats a numeric value together with its measurement uncertainty into a human-readable string using scientific notation conventions commonly used in the mendeleev package (for printing element properties, isotope masses, atomic weights and other numeric material/chemical properties in CLI, tables and web views).
    
    Args:
        value (float): The measured or computed numeric value to format. In the source code this is treated as a float but the function also explicitly accepts None for the case where no value is available; if value is None and uncertainty is not None the function will raise an exception when attempting to format. The value is the primary quantity shown to users in periodic-table outputs, reports and visualizations.
        uncertainty (float): The absolute uncertainty of the value, given as a float. The function accepts None to indicate an unknown/absent uncertainty (in which case the function falls back to fixed-point formatting using the digits parameter). If uncertainty is 0.0 the function treats the quantity as exact and returns a fixed-point representation. If uncertainty is a positive non-zero float the function computes the number of significant digits to display using scientific rounding; if uncertainty is negative (other than -0.0) a ValueError from the underlying math.log10 call will be raised. This parameter represents measurement or tabulated uncertainty associated with element or isotope properties in mendeleev data views.
        digits (int): The number of digits after the decimal point to print when uncertainty is None or equals 0.0. The default is 5. This parameter must be an integer appropriate for Python string-format precision; negative values or non-integer types will raise a ValueError or TypeError from the formatting operation. In mendeleev contexts this controls fallback display precision for properties that lack an associated uncertainty.
    
    Returns:
        str: A formatted string suitable for human-readable output. If both value and uncertainty are None the function returns the literal string "None" (so callers can display missing data consistently). If uncertainty is None or equals 0.0 the return value is a fixed-point decimal formatted with the given digits (for example "12.34500" when digits is 5). If uncertainty is a positive non-zero float the return value uses a compact notation where the main value is shown with an appropriate number of decimal places and the scaled uncertainty is shown in parentheses as an integer representing the uncertainty in the last printed digits (for example "1.2345(67)"). This representation is intended for human-facing displays of element properties and preserves the significant-figure convention used in the mendeleev project.
    
    Behavior, defaults and failure modes:
        - If value is None and uncertainty is None the function returns the string "None" (no exception).
        - If uncertainty is None or uncertainty == 0.0 the function returns a fixed-point formatted string using the digits parameter.
        - If uncertainty is a positive non-zero float the function computes digits_to_round = -int(math.floor(math.log10(uncertainty))) and returns "{value:.{digits_to_round}f}({scaled_uncertainty:.0f})" where scaled_uncertainty = uncertainty * 10**digits_to_round. This yields the conventional parenthesized-uncertainty notation used in scientific tables.
        - If uncertainty is negative (and not exactly -0.0) the internal math.log10 call will raise a ValueError (math domain error). Callers should ensure uncertainty is non-negative.
        - If value is None but uncertainty is provided (not None and not 0.0), formatting will attempt to format None as a float and a TypeError or ValueError will be raised; callers should avoid passing value=None together with a numeric uncertainty.
        - If digits is negative or not an integer the Python format operation may raise a ValueError or TypeError; callers should pass a non-negative int when relying on the fixed-point fallback.
        - No side effects occur; the function only returns a string and does not modify external state.
    """
    from mendeleev.models import with_uncertainty
    return with_uncertainty(value, uncertainty, digits)


################################################################################
# Source: mendeleev.utils.coeffs
# File: mendeleev/utils.py
# Category: valid
################################################################################

def mendeleev_utils_coeffs(a: int, b: int = 2):
    """mendeleev.utils.coeffs: Compute integer stoichiometric coefficients for a binary compound from two oxidation states.
    
    Computes the smallest integer ratio of atoms for two elements (or ions) required to balance their charges when forming a binary compound. This function is used within the mendeleev package to convert integer oxidation states (as found in element data and periodic-table-based computations) into stoichiometric coefficients by computing the least common multiple (LCM) of the two oxidation states and dividing by each oxidation state. The implementation uses lcm = abs(a * b) // math.gcd(a, b) and returns lcm // a, lcm // b. The returned pair corresponds to the multiplicities of the first and second element, in that order, as determined directly from the provided oxidation states.
    
    Args:
        a (int): Oxidation state of the first element. This integer represents the formal charge per atom used to determine charge balance in the resulting compound. Negative values represent negatively charged ions (anions) and positive values represent positively charged ions (cations). The value must be a non-zero integer; passing zero will lead to a division-by-zero error in the current implementation. Non-integer types (e.g., float, str) will raise a TypeError because the underlying math.gcd operation requires integers.
        b (int): Oxidation state  of the second element. Same semantic meaning as `a`. Defaults to 2. The default is provided for convenience when one oxidation state is commonly +2; callers should supply an explicit integer oxidation state for accurate results.
    
    Returns:
        Tuple[int, int]: A tuple of two integers (coeff_a, coeff_b). coeff_a is the integer coefficient for the first element (corresponding to `a`) and coeff_b is the integer coefficient for the second element (corresponding to `b`). These integers are produced by dividing the LCM of the two oxidation states by each oxidation state. Note that because the function preserves the sign of the input oxidation states in the division, returned coefficients may be negative if the corresponding oxidation state is negative. For conventional stoichiometric counts used in chemical formulas (non-negative atom counts), take the absolute values of the returned integers. No side effects occur.
    
    Raises:
        ZeroDivisionError: If either `a` or `b` is zero (or both are zero), the computation involves division by zero and will raise a ZeroDivisionError.
        TypeError: If `a` or `b` is not an integer, the underlying math.gcd call will raise a TypeError.
    
    Examples of intended use:
        - Given oxidation states a = 2 and b = -3 (e.g., Mg2+ and N3-), the function returns (3, -2); the conventional positive atom counts for a neutral Mg–N compound are (3, 2) (take absolute values).
        - Use this function when deriving empirical formula ratios from integer oxidation states provided by mendeleev element data.
    """
    from mendeleev.utils import coeffs
    return coeffs(a, b)


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
