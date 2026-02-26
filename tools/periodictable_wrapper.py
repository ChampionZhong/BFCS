"""
Regenerated Google-style docstrings for module 'periodictable'.
README source: others/readme/periodictable/README.rst
Generated at: 2025-12-02T01:23:12.058589Z

Total functions: 21
"""


import numpy

################################################################################
# Source: periodictable.activation.sorted_activity
# File: periodictable/activation.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def periodictable_activation_sorted_activity(activity_pair: list):
    """periodictable.activation.sorted_activity returns a new list of activity pairs sorted first by the isotope identifier and then by the daughter product identifier. This function is used in the package's neutron activation reporting routines (see README: activation calculations based on Shleien (1998)) to produce a stable, human- and machine-readable ordering of activation results so that entries for the same isotope and their daughter products are grouped together for summary and display.
    
    Args:
        activity_pair (list): An iterable container (typically a list) of activity-pair records used by the activation routines. Each element of activity_pair is expected to be an indexable sequence (for example, a tuple or list) whose first element (x[0]) is an object exposing attributes named isotope and daughter. The isotope attribute is used as the primary sort key and the daughter attribute as the secondary sort key. The elements themselves are not modified; the function only reads x[0].isotope and x[0].daughter to determine ordering. If activity_pair is empty, an empty list is returned.
    
    Returns:
        list: A new list containing the same elements as activity_pair but sorted in ascending order by the tuple (x[0].isotope, x[0].daughter). The sorting is stable (equal keys preserve the original relative order) and does not mutate the input list. The returned list is suitable for iteration and further processing by activation-reporting or summarization code that expects activity pairs grouped by isotope and daughter product.
    
    Behavior, side effects, defaults, and failure modes:
        This function performs a pure transformation: it produces and returns a new sorted list and has no side effects on the input container or its elements. It uses Python's built-in sorted() with a key function that accesses x[0].isotope and x[0].daughter.
        If activity_pair is not iterable, sorted() will raise TypeError. If an element is not indexable (has no element at index 0) or if x[0] lacks the attributes isotope or daughter, an AttributeError or IndexError may be raised at runtime. If isotope or daughter attribute values are of types that cannot be compared with each other using the default Python ordering semantics, a TypeError may be raised during sorting. The time complexity is O(n log n) for n = len(activity_pair), as governed by Python's sorted implementation.
    """
    from periodictable.activation import sorted_activity
    return sorted_activity(activity_pair)


################################################################################
# Source: periodictable.core.get_data_path
# File: periodictable/core.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def periodictable_core_get_data_path(data: str):
    """periodictable.core.get_data_path locates and returns the filesystem directory path for the tables belonging to a named extension used by the periodictable package. This function is used by extension modules (for example the 'xsf' extension) to find the packaged data files that provide element tables and supporting data for mass, density, x-ray and neutron scattering, and neutron activation calculations described in the package README.
    
    Args:
        data (string): Name of the extension data directory to locate. This is the directory name as used by extension modules (for example, 'xsf' for the xsf extension). The function treats this name as a literal subdirectory to search for in a series of candidate locations; it does not validate or normalize the name beyond joining it to candidate base paths.
    
    Returns:
        string: Filesystem path to the found data directory. The returned value is a string representing the path to the directory that contains the requested extension's data files (tables) used by the periodictable package and its extensions.
    
    Behavior and side effects:
        The function attempts to locate the named data directory by checking, in this order, specific filesystem locations and environment configuration:
        1. If the environment variable PERIODICTABLE_DATA is set, the function joins that environment path with the provided data name and checks whether the resulting path is an existing directory. This allows users or deployers to override where data files are stored. If the environment path exists but is not a directory the function raises RuntimeError with the message 'Path in environment PERIODICTABLE_DATA not a directory'. The function does not create or modify the filesystem when using this option; it only reads the environment and checks filesystem metadata.
        2. The function checks for a subdirectory named by data inside the package directory (the directory containing periodictable.core.__file__). This is the normal location when data files are installed as part of the Python package.
        3. The function checks next to the Python executable (sys.executable) for a directory named 'periodictable-data' and then the requested subdirectory inside it. This supports deployments where a companion 'periodictable-data' tree is placed alongside an application executable or a zipped application.
        4. For macOS applications packaged with py2app, the function checks a Resources path relative to the executable (../Resources/periodictable-data/<data>). This supports the common py2app layout where resources are placed in Contents/Resources while the executable is in Contents/MacOS.
        These checks only read filesystem state and the environment; they have no other side effects (they do not create, modify, or delete files or directories).
    
    Failure modes:
        If the PERIODICTABLE_DATA environment variable points to a non-directory path the function raises RuntimeError('Path in environment PERIODICTABLE_DATA not a directory').
        If none of the candidate locations contain a directory named by data, the function raises RuntimeError('Could not find the periodic table data files').
        The function assumes a POSIX/Windows filesystem semantics as provided by os.path and does not perform additional normalization, verification of file contents, or version checks of the data directory; callers should handle the RuntimeError exceptions to provide fallback behavior or user-friendly error messages.
    """
    from periodictable.core import get_data_path
    return get_data_path(data)


################################################################################
# Source: periodictable.cromermann.fxrayatstol
# File: periodictable/cromermann.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def periodictable_cromermann_fxrayatstol(symbol: str, stol: float, charge: int = None):
    """Calculate x-ray scattering factors at a specified sin(theta)/lambda (stol) for an element or ion using the Cromer–Mann form-factor data in the periodictable package.
    
    This function is used in x-ray scattering calculations (see package README) to obtain the atomic scattering factor f(sin(theta)/lambda) for a resolved element or ion symbol. It resolves and normalizes the provided element/ion symbol, optionally applies an explicit integer ion charge override, looks up the Cromer–Mann parameters for that species via getCMformula, and evaluates the form factor at the provided stol value(s) by calling the resulting object's atstol method. The returned value is intended for use in computing x-ray scattering amplitudes and cross sections in crystallography and related applications.
    
    Args:
        symbol (str): Symbol of an element or ion to evaluate, e.g., "Ca" or "Ca2+". This input may include ionic suffixes such as "+", "-", "3+" etc. If the symbol ends with a single sign character and no trailing digit (for example "Na+" or "Cl-"), the function will normalize it to an explicit 1 charge (e.g., "Na1+", "Cl1-") before lookup. The resolved symbol determines which Cromer–Mann parameters are used for the scattering factor lookup and thus directly controls the physical species whose scattering is returned.
        stol (float or sequence of float): The value(s) of sin(theta)/lambda at which to evaluate the x-ray form factor, given in inverse angstroms (1/Å). The function accepts a scalar float for a single evaluation or a sequence (e.g., list or numpy array) of floats for vectorized evaluation; when a sequence is supplied, the function will return a numpy.ndarray of matching shape containing the scattering factor for each stol. These values are forwarded to the underlying cmf.atstol(stol) routine which computes the Cromer–Mann form factor.
        charge (int): Optional integer ion charge override. If provided (not None), this integer will override any ionic suffix present in symbol. The function strips trailing digits and sign characters from symbol and then appends an explicit charge suffix corresponding to this integer (formatted so that 2 becomes "2+", -1 becomes "1-", etc.). If charge is None (the default), the function uses any valence suffix already present in symbol (after the normalization rule described above).
    
    Returns:
        float or numpy.ndarray: The computed x-ray atomic scattering factor(s) f at the supplied stol position(s). A scalar float is returned for a scalar stol input; a numpy.ndarray is returned for sequence inputs and will contain one value per input stol element. The returned scattering factors are dimensionless quantities used directly in x-ray scattering amplitude and cross-section calculations.
    
    Behavior, defaults, and failure modes:
        - Symbol normalization: If charge is provided, it overrides any suffix on symbol. If charge is None and symbol ends with '+' or '-' with no preceding digit, a "1" is inserted before the sign. These normalization steps produce the lookup symbol used by getCMformula.
        - Lookup and evaluation: The function calls getCMformula(resolved_symbol) to obtain a Cromer–Mann form factor object and then evaluates cmf.atstol(stol). Any errors raised by getCMformula (for example if the element/ion is unknown) or by cmf.atstol (for example if stol values are outside the supported range for the tabulated form factors) will propagate to the caller; the function does not swallow these exceptions.
        - Side effects: The function performs no persistent side effects; it is purely a lookup-and-evaluate computation. Underlying caching or lookup mechanisms used by getCMformula are external to this function.
        - Input types: The function documents and expects the types described above (symbol as str; stol as float or sequence of floats in 1/Å; charge as int or None). It does not perform implicit unit conversion; stol must be provided in inverse angstroms.
    """
    from periodictable.cromermann import fxrayatstol
    return fxrayatstol(symbol, stol, charge)


################################################################################
# Source: periodictable.cromermann.getCMformula
# File: periodictable/cromermann.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def periodictable_cromermann_getCMformula(symbol: str):
    """Obtain the Cromer–Mann analytic form-factor formula and its fitted coefficients for a specified chemical element and return them as a CromerMannFormula instance.
    
    This function is used by the periodictable package to provide the set of Cromer–Mann parameters required for X-ray scattering and form-factor calculations. The returned CromerMannFormula instance encapsulates the numerical coefficients and any metadata needed by other modules in this package to compute atomic scattering factors and related quantities used in crystallography and X-ray physics.
    
    Args:
        symbol (str): Chemical element symbol identifying the element whose Cromer–Mann parameters are requested. This should be the standard element symbol string (for example "Fe" for iron, "O" for oxygen). The symbol is used as a key into the module's internal cache of Cromer–Mann formulas.
    
    Behavior and side effects:
        If the module-level cache of Cromer–Mann formulas (_cmformulas) is empty when this function is called, the function will lazily initialize that cache by calling _update_cmformulas(), which loads or constructs the available Cromer–Mann formula entries. This is a side effect: the first call may perform I/O or data parsing indirectly (depending on _update_cmformulas implementation) and populate in-memory data used by subsequent calls. Subsequent calls return entries from the populated cache without repeating the initialization work.
    
    Failure modes and errors:
        If symbol is not present among the loaded Cromer–Mann entries, this function will raise a KeyError when attempting to return _cmformulas[symbol]. If a non-string value is passed as symbol, a TypeError may occur elsewhere in indexing or during cache population; callers should pass a str. If the lazy initialization fails (for example, because required data files are missing or unreadable), _update_cmformulas() may raise an exception which will propagate to the caller.
    
    Returns:
        CromerMannFormula: An instance of CromerMannFormula representing the Cromer–Mann analytic form-factor expression and its fitted coefficients for the requested element. This object is intended for direct use in the package's X-ray scattering and form-factor computations.
    """
    from periodictable.cromermann import getCMformula
    return getCMformula(symbol)


################################################################################
# Source: periodictable.fasta.D2Omatch
# File: periodictable/fasta.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def periodictable_fasta_D2Omatch(Hsld: float, Dsld: float):
    """periodictable.fasta.D2Omatch computes the percent D2O in solvent required to match the neutron scattering length density (SLD) of a sample.
    This function is used in neutron scattering contrast matching (for example small-angle neutron scattering or neutron reflectometry) to find the solvent D2O concentration that makes the neutron SLD of a mixed hydrogenated/deuterated sample equal to the neutron SLD of a H2O/D2O solvent mixture. It assumes linear mixing of SLDs between hydrogenated and deuterated forms of the sample and between H2O and D2O solvent, and uses the module constants H2O_SLD and D2O_SLD (water SLD values evaluated at 20 °C) for solvent SLDs. The deuterated sample SLD (Dsld) is defined to include exchangeable labile protons replaced by deuterons. Note: this function was deprecated in periodictable 1.5.3; prefer periodictable.nsf.D2O_match(formula) for newer code paths.
    
    Args:
        Hsld (float): Neutron scattering length density (SLD) of the hydrogenated form of the material. This is the SLD for the sample containing protium (H) where labile protons have not been exchanged. The value must be expressed in the same units as the module-level H2O_SLD and D2O_SLD constants (the function does not convert units). Hsld represents the SLD contribution of the sample end-member that mixes linearly with its deuterated counterpart when varying sample deuteration.
        Dsld (float): Neutron scattering length density (SLD) of the deuterated form of the material. This SLD is for the sample form where all exchangeable labile protons have been replaced by deuterons (D). As with Hsld, Dsld must be given in the same units as H2O_SLD and D2O_SLD. Dsld represents the SLD of the fully (with respect to labile protons) deuterated sample end-member.
    
    Returns:
        float: The computed percentage (0–100 scale) of D2O in the solvent mixture required to match the neutron SLD of the sample. The numeric result is obtained from the linear mixing relation:
            percent = 100 * (H2O_SLD - Hsld) / (Dsld - Hsld + H2O_SLD - D2O_SLD)
        where H2O_SLD and D2O_SLD are module constants for pure H2O and pure D2O SLDs at 20 °C. The returned value is interpreted as the percent D2O in the solvent; values between 0 and 100 are physically meaningful as percent composition of the H2O/D2O solvent mixture. Values below 0 or above 100 may be returned when no pure H2O/D2O mixture can match the sample SLD; in particular, values greater than 100 indicate that even 100% D2O solvent does not reach the SLD needed and an additional contrast agent would be required to increase solvent SLD further.
    
    Behavior and failure modes:
        This function performs a purely numerical calculation with no side effects. It assumes linear mixing of SLDs between hydrogenated and deuterated sample forms and between H2O and D2O solvent. If the denominator of the formula is zero (i.e., Dsld - Hsld + H2O_SLD - D2O_SLD == 0), a ZeroDivisionError will be raised; this corresponds physically to a degenerate case where changing sample or solvent composition cannot change the relative SLD required for matching. If inputs are not numeric floats, the operation may raise the usual Python TypeError/ValueError during arithmetic. The function does not clamp or otherwise bound the returned percentage; callers should interpret out-of-range results (less than 0 or greater than 100) as indicating that a simple H2O/D2O mixture cannot achieve a match without additional contrast agents or a different sample composition.
    
    Deprecation and historical note:
        Deprecated in periodictable 1.5.3. Use periodictable.nsf.D2O_match(formula) for newer implementations. Change in 1.5.3: the D2O SLD constant was corrected, which will change computed match points compared to older releases.
    """
    from periodictable.fasta import D2Omatch
    return D2Omatch(Hsld, Dsld)


################################################################################
# Source: periodictable.formulas.count_elements
# File: periodictable/formulas.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def periodictable_formulas_count_elements(compound: str, by_isotope: bool = False):
    """Element composition of a molecule or formula string.
    
    This function parses a chemical formula and returns the total count of each element present in the formula. It is used throughout the periodictable package to compute elemental composition needed for mass, density, x-ray and neutron scattering calculations, and for rough activation estimates described in the project README. By default the function sums counts across isotopes and ionization levels to give a per-element tally; an optional flag keeps isotopic identities separate while still summing across ionization levels.
    
    Args:
        compound (str): Chemical formula to analyze. This must be a string in a format accepted by periodictable.formulas.formula(); examples used in the package include simple formulas (e.g. "H2O"), formulas with isotopic specification and charged fragments. The function calls formula(compound) internally and therefore accepts the same syntax and semantics as that parser. If a non-conforming string is provided, parsing errors raised by the formula parser will propagate out of this function.
        by_isotope (bool): If False (the default), the returned counts are keyed by the underlying element (isotopes are collapsed to their element) so the mapping represents total atoms of each chemical element, summed across isotopes and ionization levels. If True, isotopic identities returned by the formula parser are preserved as distinct keys (so counts are separated by isotope) but ionization levels are still resolved to their underlying isotopic or elemental identity. This flag controls whether the composition is reported at elemental resolution (by_isotope=False) or isotopic resolution (by_isotope=True).
    
    Returns:
        dict: A mapping from element or isotope objects (the same fragment objects produced by periodictable.formulas.formula().atoms) to numeric counts. When by_isotope is False the keys are the underlying element objects and the values are the total number of atoms of that element in the formula. When by_isotope is True the keys preserve isotopic identity (so different isotopes of the same element appear as separate keys) and the values are the counts for each isotope. Counts are accumulated across all occurrences in the formula (and across ionization levels, which are resolved to their underlying isotope/element before accumulation).
    
    Raises:
        Exception: Any exception raised by the underlying formula parsing or fragment utilities (for example when compound is not a valid formula or has unsupported syntax) is propagated; this function does not perform additional validation or exception translation.
    
    Side effects:
        None. The function does not modify global state or the input; it only computes and returns the composition dictionary.
    """
    from periodictable.formulas import count_elements
    return count_elements(compound, by_isotope)


################################################################################
# Source: periodictable.magnetic_ff.formfactor_0
# File: periodictable/magnetic_ff.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def periodictable_magnetic_ff_formfactor_0(j0: tuple, q: numpy.ndarray):
    """periodictable.magnetic_ff.formfactor_0: Compute the scattering potential for the magnetic form factor j0 at the supplied q values for use in neutron and x-ray scattering calculations.
    
    This function evaluates the analytic form used in the package to represent the magnetic form factor labeled j0. In the context of the periodictable package (which provides scattering and cross-section data for x-ray and neutron calculations), the returned scattering potential is a scalar or array giving the form factor contribution at each supplied momentum-transfer-like value q. The implementation converts q to an internal squared variable s_sq = (q/(4*pi))**2 and evaluates the sum of three Gaussian-like exponential terms plus a constant offset using coefficients supplied in j0.
    
    Args:
        j0 (tuple): A 7-tuple of numeric coefficients (A, a, B, b, C, c, D) that define the j0 form factor. A, B, and C are multiplicative amplitudes for the three exponential terms; a, b, and c are the corresponding exponential decay coefficients applied to s_sq; D is a constant offset added to the sum. These coefficients come from the package's magnetic form factor data tables and are used directly in the expression A*exp(-a*s_sq) + B*exp(-b*s_sq) + C*exp(-c*s_sq) + D. The tuple must have exactly seven elements; otherwise Python will raise a ValueError during unpacking.
        q (numpy.ndarray): Array-like input of scalar q values (e.g., momentum-transfer magnitudes used in scattering computations). The function calls numpy.asarray(q) internally, so q may be a numpy.ndarray, list, tuple, or scalar convertible to a numpy array. The computation is performed elementwise and preserves the shape of the converted array; if q is a scalar the result will be a scalar-like numpy value. Invalid numeric entries in q (NaN, +/-Inf, non-numeric types that cannot be converted) will propagate or raise an error from numpy.asarray or the subsequent arithmetic.
    
    Returns:
        numpy.ndarray: The scattering potential values computed for each element of q using the j0 coefficients. The return has the same shape as numpy.asarray(q); for scalar q the returned object is a numpy scalar. Each output value equals A*exp(-a*s_sq) + B*exp(-b*s_sq) + C*exp(-c*s_sq) + D with s_sq computed as (q/(4*pi))**2. These values are intended for use in higher-level neutron or x-ray scattering and cross-section calculations within the periodictable package.
    
    Behavior and failure modes:
        The function is pure and has no side effects; it only converts q to a numpy array and evaluates the expression. If j0 does not contain exactly seven items, unpacking will raise a ValueError. If q cannot be converted to a numpy array (for example, because it contains incompatible objects), numpy.asarray will raise a TypeError. Numeric overflow or underflow in the exponential evaluation may occur for extremely large or small coefficient values or q, producing Inf/0 results according to IEEE floating-point rules; such results will propagate to callers.
    """
    from periodictable.magnetic_ff import formfactor_0
    return formfactor_0(j0, q)


################################################################################
# Source: periodictable.magnetic_ff.formfactor_n
# File: periodictable/magnetic_ff.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def periodictable_magnetic_ff_formfactor_n(jn: tuple, q: numpy.ndarray):
    """periodictable.magnetic_ff.formfactor_n returns the scattering potential for a magnetic form factor defined by the coefficient tuple jn at the supplied momentum-transfer array q. This function is used by the periodictable package's neutron/magnetic scattering routines to evaluate a parametric magnetic form factor expression (commonly found in published tables and internal data files) as a function of momentum transfer. The implementation computes s_sq = (q / (4*pi))**2 and returns s_sq multiplied by a weighted sum of three exponential terms plus a constant term, reproducing the standard analytic form used for neutron magnetic form factors.
    
    Args:
        jn (tuple): A 7-tuple of coefficients (A, a, B, b, C, c, D) that define the magnetic form factor. A, B, C, D are amplitude coefficients and a, b, c are positive exponential decay parameters in the analytic model. These coefficients are typically taken from periodictable magnetic form factor data (e.g., tabulated values collected by neutron scattering data sources) and determine the shape and magnitude of the scattering potential returned by this function. The tuple must contain exactly seven elements; otherwise unpacking will raise a ValueError.
        q (numpy.ndarray): Array of momentum-transfer magnitudes at which to evaluate the form factor. This argument is converted to a NumPy array via numpy.asarray(q) inside the function, so scalar inputs or array-like objects are accepted and will be cast to numpy.ndarray. Elements of q are treated as real numbers representing the magnitude of momentum transfer and are broadcasted in the usual NumPy manner; negative or NaN values will propagate into the result according to NumPy arithmetic rules.
    
    Returns:
        numpy.ndarray: An array of the same shape as the broadcasted input q containing the computed scattering potential values for the provided form factor coefficients. The returned values are computed elementwise as s_sq * (A * exp(-a*s_sq) + B * exp(-b*s_sq) + C * exp(-c*s_sq) + D) where s_sq = (q/(4*pi))**2 and exp() denotes the elementwise exponential. For q values near zero the result scales as q**2 because of the leading s_sq multiplier.
    
    Behavior, defaults, and failure modes:
        This function has no side effects and does not modify its inputs. It relies on numeric NumPy operations and returns NumPy floats. If jn does not have exactly seven elements, a ValueError arises on tuple unpacking. If q cannot be converted to a numeric numpy.ndarray (for example, if it contains non-numeric objects), numpy.asarray will produce an array that may cause downstream TypeError or ValueError during arithmetic. The function does not perform unit validation; callers are responsible for supplying q in the units expected by the surrounding code or data tables used to produce jn. Numerical overflow or underflow follow NumPy rules for floating-point arithmetic (e.g., very large exponent arguments may underflow to zero).
    """
    from periodictable.magnetic_ff import formfactor_n
    return formfactor_n(jn, q)


################################################################################
# Source: periodictable.nsf.D2O_match
# File: periodictable/nsf.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def periodictable_nsf_D2O_match(compound: str, **kw):
    """Find the D2O contrast match point for the compound in the context of neutron
    scattering calculations provided by the periodictable package. This function
    computes the fraction of deuterium oxide (D2O) in an H2O/D2O solvent mixture
    needed to match the sample's neutron scattering length density (SLD), and
    returns the SLD value at that match point. It is used in neutron contrast
    variation experiments and calculations where matching the solvent SLD to the
    sample SLD is required to minimize or eliminate scattering from the sample
    for contrast-matching studies.
    
    Args:
        compound (str): Chemical composition of the sample. This is a formula
            string parsed by the package's formula parser (formulas.formula).
            Typical inputs are element symbols and stoichiometry (for example,
            "C6H12O6" or a polymer repeat unit). The parser may consult keyword
            arguments in kw (see below) to determine density, isotopic content,
            or which element table to use. The chemical formula is used to derive
            the sample SLD contributions from hydrogen and deuterium in the
            compound for the contrast-match calculation.
        kw (dict): Additional keyword arguments forwarded to the internal parser
            and SLD routines. Recognized keys include:
            wavelength or energy: choose whether neutron wavelength or energy is
                used for energy-dependent scattering lengths.
            density, natural_density, name, table: passed through to
                formulas.formula when parsing the compound to control assumed
                sample density, whether natural isotopic abundances are used,
                an explicit compound name, and which element/isotope table to
                consult. These control how the sample SLD is computed and thus
                affect the computed D2O fraction. Any other keywords accepted by
                formulas.formula, _D2O_slds, or D2O_sld may also be provided and
                will be forwarded.
    
    Behavior and calculation details:
        The function calls the internal helper _D2O_slds(compound, **kw) to obtain
        neutron SLDs for pure H2O, pure D2O, and the sample expressed with H and D
        substitutions (Hsld and Dsld). It then solves the linear mixture equation
        equating the SLD of a sample containing a fraction f of deuterated sample
        and (1-f) hydrogenated sample to the SLD of a solvent that is a fraction
        x of D2O and (1-x) of H2O. Algebraically this yields the fraction of D2O
        in the solvent required to match the sample SLD:
            D2O_fraction = (SLD(H2O) - SLD(Hsample)) /
                           (SLD(Dsample) - SLD(Hsample) + SLD(H2O) - SLD(D2O))
        The routine computes D2O_fraction (returned as a unitless float in the
        range [0, 1] for physically meaningful mixtures) and the SLD at the
        match point by linearly mixing Dsld and Hsld via the mix_values helper.
        The returned SLD value is the scalar SLD associated with the match point
        (the first component returned by mix_values), with units and reference
        convention consistent with the package's neutron SLD functions.
    
    Defaults and side effects:
        There are no persistent side effects; the function performs pure
        calculations and returns values. Default behavior uses the package's
        standard element/isotope tables and density assumptions unless overridden
        via kw. The D2O_fraction computed is a unitless fraction (0–1) of solvent
        that should be prepared as that fraction of D2O in H2O; if the calculated
        value exceeds 1 (or is negative) it indicates that pure D2O (or pure H2O)
        alone cannot reach the required SLD and an additional contrast agent or
        sample deuteration scheme would be required in practice.
    
    Failure modes and exceptions:
        If parsing the compound fails (invalid formula or incompatible kw), the
        underlying parser (formulas.formula or _D2O_slds) will raise an exception
        (for example ValueError). If the algebraic denominator in the fraction
        calculation is zero (for example when the relevant SLD differences cancel
        exactly), a ZeroDivisionError may be raised. The function does not
        validate that the returned fraction lies within [0, 1]; callers should
        interpret values outside that range as indicating the match point lies
        beyond pure D2O or H2O and that additional contrast agents or sample
        modification are required.
    
    Returns:
        tuple: A two-element tuple containing:
            D2O_fraction (float): Unitless fraction of D2O in the H2O/D2O solvent
                mixture required to match the sample SLD. Physically meaningful
                values lie in [0, 1]; values outside that range indicate the
                match point requires solvent compositions beyond pure H2O or pure
                D2O.
            SLD (float): The scattering length density (SLD) at the match point.
                The numeric SLD is returned using the package's neutron SLD
                conventions (units and sign are those produced by D2O_sld and
                mix_values).
    """
    from periodictable.nsf import D2O_match
    return D2O_match(compound, **kw)


################################################################################
# Source: periodictable.nsf.fix_number
# File: periodictable/nsf.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def periodictable_nsf_fix_number(str: str):
    """periodictable.nsf.fix_number converts a numeric string from NSF/periodictable data files into a plain Python float by removing uncertainty notation and a few special characters used in tabulated element data.
    
    This function is used throughout the periodictable.nsf data-handling code to normalise numeric fields found in tabular element and isotope data (for example neutron scattering lengths, cross sections, densities or activation-related values). It strips characters used in the source files ('<' for upper limits and '*' as an annotation), removes parenthetical uncertainty notation such as "35.24(2)", and returns only the central numeric value so downstream calculations (neutron scattering, x-ray scattering, or activation estimates described in the package README) receive a simple float without embedded uncertainty information.
    
    Args:
        str (str): A string representation of a numeric value as found in NSF/periodictable input data. Acceptable inputs include plain numeric text ("12.34"), numbers with parenthetical uncertainty ("35.24(2)"), values marked as upper limits ("<1e-6"), and strings that include an asterisk annotation ("35.24(2)*"). The argument must be a Python str; empty strings are treated as missing values and will be mapped to 0. The function removes all '<' and '*' characters before parsing and delegates parsing and uncertainty interpretation to periodictable.util.parse_uncertainty.
    
    Returns:
        float: The parsed numeric value with any uncertainty removed (the central value). For "35.24(2)*" this function returns 35.24; for "<1e-6" it returns 1e-6; for empty or otherwise missing string values it returns 0.0. The returned float is intended for use in numerical calculations in the periodictable package (e.g., scattering cross sections, densities, activation estimates).
    
    Raises:
        ValueError: If the input string cannot be parsed by periodictable.util.parse_uncertainty, a ValueError or other parsing exception raised by that utility will propagate. No side effects occur (the function is pure and returns a new float).
    """
    from periodictable.nsf import fix_number
    return fix_number(str)


################################################################################
# Source: periodictable.nsf.mix_values
# File: periodictable/nsf.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def periodictable_nsf_mix_values(a: tuple, b: tuple, fraction: float):
    """periodictable.nsf.mix_values: Mix two tuples of floating-point values according to the fractional contribution of the first tuple.
    
    This function computes a pointwise linear combination of two equal-meaning sequences of floating-point measurements (for example, arrays of mass, density, or x-ray/neutron scattering values used in the periodictable package) by weighting each paired element by fraction for the first sequence and (1 - fraction) for the second sequence. It is typically used in the domain of material and scattering-property composition or interpolation when forming weighted averages of per-energy or per-isotope value tuples.
    
    Args:
        a (tuple): Tuple of numeric (floating-point) values representing the first set of per-element measurements or properties (for example, per-energy scattering factors, mass contributions, or densities). The sequence order defines how elements are paired with b; no reordering is performed. Values are used as aj in the computation aj * fraction + bj * (1 - fraction).
        b (tuple): Tuple of numeric (floating-point) values representing the second set of per-element measurements or properties to be mixed with a. The sequence order must correspond to a for meaningful pairwise mixing. Values are used as bj in the computation aj * fraction + bj * (1 - fraction).
        fraction (float): Scalar floating-point weight specifying the fractional contribution of a to the result. A value of 1.0 yields a tuple equal to a (subject to truncation noted below), and 0.0 yields a tuple equal to b. Internally used as the multiplier for a in aj * fraction + bj * (1 - fraction).
    
    Returns:
        tuple: A new tuple containing the pointwise mixed floating-point results computed as aj * fraction + bj * (1 - fraction) for each paired element (aj, bj). The length of the returned tuple is the length of the shortest input tuple because Python's zip is used to pair elements; no padding or extrapolation of length is performed.
    
    Behavior, side effects, defaults, and failure modes:
        This function performs purely functional, elementwise arithmetic and has no side effects (it does not modify the input tuples). Pairing of elements uses Python's zip, so if a and b have different lengths the output length is min(len(a), len(b)). If fraction is outside the range [0, 1], the arithmetic still executes and yields the corresponding linear extrapolation of values (this is a mathematical consequence, not an enforced constraint). If either a or b is not iterable or contains non-numeric types that do not support multiplication and addition with floats, the function will raise a TypeError (or another exception produced by the underlying numeric operations). No validation of numeric ranges, tuple lengths alignment, or type coercion is performed by the function itself; callers in the periodictable context should ensure inputs are appropriate arrays of floating-point values representing compatible physical quantities.
    """
    from periodictable.nsf import mix_values
    return mix_values(a, b, fraction)


################################################################################
# Source: periodictable.nsf.neutron_composite_sld
# File: periodictable/nsf.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def periodictable_nsf_neutron_composite_sld(materials: list, wavelength: float = 1.798):
    """periodictable.nsf.neutron_composite_sld creates a composite neutron scattering length density
    (SLD) calculator for a collection of materials drawn from the periodictable package. The
    returned calculator computes the coherent real SLD, the magnitude of the imaginary SLD
    (which should be interpreted as the negative imaginary component in the complex SLD),
    and the incoherent SLD contribution for a composite made from the supplied materials.
    This function is used in neutron scattering and reflectometry contexts (as described
    in the package README) to combine material fragments or large molecules into a single
    effective SLD so that contrast curves, fits, and other sample-level scattering
    calculations can be performed efficiently.
    
    Args:
        materials (list): List of material objects or formula descriptors accepted by
            the periodictable package that define each fragment of the composite. Each
            material entry is queried for its composition, molar mass, coherent scattering
            length(s), and total scattering cross section(s) using the package's neutron
            data tables. The list order determines the ordering of input weights passed
            to the returned calculator. Practical significance: supply the constituent
            chemical formulas or material objects (e.g., periodictable Formulas/Materials)
            for the composite; table lookups and per-material partial sums are precomputed
            so subsequent SLD calculations are fast.
        wavelength (float): Probe wavelength in angstroms (Å). This may be a scalar
            float (the default ABSORPTION_WAVELENGTH ≈ 1.798 Å) or a sequence/array of
            floats to compute wavelength-dependent SLDs. When a sequence is provided the
            returned calculator produces vector-valued SLDs (one value per input
            wavelength) and handles energy-dependent scattering information for certain
            elements. Practical significance: select the neutron wavelength(s) used in
            the experiment or model; units are angstroms and the function preserves the
            scalar-versus-sequence behaviour (scalar input yields scalar outputs, array
            input yields vector outputs).
    
    Returns:
        function: A calculator function with signature f(weights, density=1) -> (sld_re, sld_im, sld_inc).
            sld_re (float or ndarray): Coherent real part of the composite SLD in units of 1e-6/Å^2
                (returned as a scalar when wavelength is scalar and as an array when wavelength
                is an array or when energy-dependent elements produce wavelength dependence).
                This value represents the real scattering length density contribution used
                in neutron reflectometry and scattering calculations.
            sld_im (float or ndarray): Non-negative magnitude of the imaginary part of the
                composite SLD in 1e-6/Å^2. The package follows the convention that the
                complex SLD is sld_re - i*sld_im, so the returned sld_im should be applied
                with a negative imaginary sign when reconstructing the complex SLD.
            sld_inc (float or ndarray): Incoherent SLD contribution in 1e-6/Å^2 computed from
                the incoherent cross section. This quantity is always non-negative and
                represents the incoherent scattering contribution to the total SLD.
            The returned calculator takes:
                weights: A sequence (array-like) of numeric weights with length equal to
                    the number of entries in the original materials list. Weights specify
                    the relative amount (relative number of formula units or fragments)
                    of each material in the composite; only relative proportions matter
                    because the absolute scale is set by density. Typical use is non-negative
                    floats. Practical significance: provide composition fractions or counts
                    for each fragment when assembling a composite; the calculator multiplies
                    these weights by precomputed per-material quantities to get composite SLDs.
                density: Numeric scalar (default 1) giving the mass density of the composite
                    in g/cm^3. The code uses molar mass (g/mol) divided by density (g/cm^3)
                    to compute a cell volume and hence number density (1/Å^3), so density
                    must be supplied in g/cm^3 for correct units. If multiple independent
                    compositions are being evaluated, supply an appropriate density for each
                    composition (the implementation treats density as the overall scale and
                    ignores per-material densities stored on individual materials).
            Behavior and side effects: The materials list is processed once when creating
                the calculator: per-material atom counts, molar masses, coherent scattering
                lengths, and total cross sections are looked up and partially summed for
                efficiency. The returned calculator performs only array arithmetic and
                returns SLD components with the units and conventions described above.
            Failure modes and edge cases: If molar_mass * density == 0 the calculator
                returns (0, 0, 0) to represent vacuum or an empty/zero-density composite.
                If the provided weights vector has a length or shape incompatible with the
                precomputed per-material arrays, numpy will raise a broadcasting or shape
                error (e.g., ValueError). Negative weights are not explicitly forbidden by
                the code but represent physically unusual compositions and may produce
                uninterpretable results. No other side effects or external state changes
                occur; the function does not raise custom exceptions for domain errors.
            Performance note: Table lookups and partial sums are precomputed at calculator
                construction time, so repeated calls to the returned function are efficient
                even for large molecule fragments or many wavelengths.
    """
    from periodictable.nsf import neutron_composite_sld
    return neutron_composite_sld(materials, wavelength)


################################################################################
# Source: periodictable.nsf.neutron_energy
# File: periodictable/nsf.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def periodictable_nsf_neutron_energy(wavelength: numpy.ndarray):
    """periodictable.nsf.neutron_energy converts a neutron wavelength (given in Angstroms) to the corresponding kinetic energy in millielectronvolts (meV). This function is used in the periodictable package's neutron scattering calculations (see README), where neutron wavelengths measured or specified for scattering, cross-section, or activation estimates must be converted to energies for subsequent physics formulae or lookups in neutron data tables.
    
    Args:
        wavelength (numpy.ndarray or float): Neutron wavelength(s) in Angstrom (Å). This may be a scalar Python float, a NumPy scalar, a NumPy ndarray, or any array-like sequence that can be converted with numpy.asarray. The function treats the value(s) as physical wavelengths: it computes energy element-wise for vector inputs and for scalar inputs returns a NumPy scalar. Physically, wavelength must be positive; negative values are accepted by the numeric routine (because the formula uses lambda**2) but are not physically meaningful and should be avoided. Units are required to be Angstrom; providing values in other length units will produce incorrect energies.
    
    Returns:
        numpy.ndarray or float: Neutron kinetic energy corresponding to the input wavelength(s), expressed in millielectronvolts (meV). For array-like inputs, a NumPy ndarray of the same shape is returned containing the energy for each wavelength. For scalar inputs, a NumPy scalar (e.g., numpy.float64) is returned. The conversion implements the standard non-relativistic relation E = h^2 / (2 m_n λ^2) with constants chosen so that wavelength in Angstrom yields energy in meV (the implementation uses a precomputed ENERGY_FACTOR). This value is suitable for use in neutron scattering cross-section calculations and in lookups against energy-dependent scattering data (e.g., the Atomic Institute neutron data booklet referenced in the package README).
    
    Behavior, side effects, defaults, and failure modes:
        The function performs element-wise arithmetic using numpy.asarray(wavelength)**2 and has no side effects (it does not modify global state or the input objects). It does not perform unit conversion beyond assuming input is in Angstrom. If wavelength is zero, the result is a division-by-zero yielding numpy.inf and NumPy may emit a runtime warning. Non-numeric inputs that cannot be converted to a numeric NumPy array will raise an exception (typically TypeError or ValueError from numpy.asarray or the subsequent arithmetic). The function uses a non-relativistic kinetic energy expression; at very short wavelengths / high energies where relativistic corrections are required, this function will not apply those corrections and thus will underestimate the true relativistic kinetic energy.
    """
    from periodictable.nsf import neutron_energy
    return neutron_energy(wavelength)


################################################################################
# Source: periodictable.nsf.neutron_wavelength
# File: periodictable/nsf.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def periodictable_nsf_neutron_wavelength(energy: numpy.ndarray):
    """periodictable.nsf.neutron_wavelength: Convert neutron energy (meV) to neutron wavelength (Å) for use in neutron scattering calculations.
    
    Converts a neutron energy value or array of energies, expressed in millielectronvolts (meV), to the corresponding neutron wavelength in angstroms (Å) using the non-relativistic relationship between kinetic energy and wavelength suitable for thermal and cold neutron scattering work. This function is used within the periodictable package's neutron-scattering utilities (nsf) to compute wavelengths needed for scattering cross section, scattering length density, and other neutron-beam calculations that rely on wavelength as an input parameter. The conversion is based on the formula λ = sqrt(h^2 / (2 m_n E)), where h is the Planck constant and m_n is the neutron mass; the implementation uses a module-level ENERGY_FACTOR equal to h^2 / (2 m_n) with units chosen so that E is given in meV and the result is in Å.
    
    Args:
        energy (numpy.ndarray): Neutron energy values in millielectronvolts (meV). This argument provides the kinetic energy(s) of neutrons for which the wavelength will be computed. The function accepts a numpy.ndarray containing one or more energies; a scalar energy may be provided as a 0-D numpy array (numpy.asarray is applied internally). The array's numeric dtype should represent positive energy values in meV; negative or zero energies are outside the physical domain for this classical conversion and will produce NaN/inf or runtime warnings when evaluated.
    
    Returns:
        numpy.ndarray: Wavelength(s) corresponding to the input energy values, expressed in angstroms (Å). The returned object is a numpy result produced by sqrt(ENERGY_FACTOR / numpy.asarray(energy)) and therefore has the same shape as the input array (for scalar 0-D input this may be a numpy scalar). Typical use: supplying an array of energies yields an array of wavelengths for element-specific neutron scattering computations in the periodictable package.
    
    Behavior, defaults, and failure modes:
        - Units: energy must be in meV; output is in Å. This unit convention matches the neutron data and scattering routines in the periodictable package and the README description of neutron scattering support.
        - Formula: uses λ = sqrt(h^2 / (2 m_n E)) implemented via a precomputed ENERGY_FACTOR = h^2 / (2 m_n) so that users pass energy in meV and receive Å.
        - Domain: energy values must be positive. energy == 0 results in division by zero producing +inf and a runtime warning from NumPy; negative energy values produce NaN (not-a-number) due to the square root of a negative argument and may emit warnings.
        - Types: input is converted with numpy.asarray; non-numeric or incompatible array contents will raise a TypeError or produce invalid values when numpy operations are applied.
        - Side effects: none (pure function); it does not modify the input array in-place.
        - Precision: numerical precision and rounding follow NumPy floating-point semantics and the precision of the input array dtype.
    """
    from periodictable.nsf import neutron_wavelength
    return neutron_wavelength(energy)


################################################################################
# Source: periodictable.nsf.neutron_wavelength_from_velocity
# File: periodictable/nsf.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def periodictable_nsf_neutron_wavelength_from_velocity(velocity: float):
    """periodictable.nsf.neutron_wavelength_from_velocity converts a neutron velocity (speed) to its de Broglie wavelength for use in neutron scattering and related calculations.
    
    This function implements the physical relation lambda = h/(m_n v) (wavelength equals Planck's constant divided by the neutron momentum m_n v) and, in the package implementation, returns VELOCITY_FACTOR / velocity where VELOCITY_FACTOR encodes the Planck constant, neutron mass, and the conversion to Angstrom units. In the context of this package (periodictable), this conversion is used in neutron scattering calculations (for example, computing wavelength-dependent scattering lengths, cross sections, Bragg conditions, and scattering length density) as described in the project README and the neutron data sources it relies on.
    
    Args:
        velocity (float or vector): Neutron velocity or speeds, expressed in metres per second (m/s). Provide a single floating-point value for a scalar velocity or a vector of velocities to obtain elementwise wavelengths. The argument must represent SI velocities; the function performs simple arithmetic division and does not perform automatic unit conversion beyond this expectation.
    
    Returns:
        float or vector: Wavelength(s) corresponding to the input velocity(ies), expressed in Angstrom (Å). If a scalar float is provided, a scalar float is returned; if a vector is provided, a vector of the same shape is returned. The returned value is computed as lambda = h / (m_n * v) and therefore represents the de Broglie wavelength used in neutron scattering calculations.
    
    Raises:
        ZeroDivisionError: If velocity is zero (or contains zero elements), division by zero will occur.
        TypeError: If velocity is not a numeric type or not a vector-like numeric container compatible with division, a TypeError (or an error from the underlying numeric operations) may be raised.
    
    Behavior and failure modes:
        The function is pure and has no side effects. It expects physically meaningful velocities (positive, non-zero) expressed in m/s; negative input values are mathematically accepted but yield negative wavelengths (which indicate direction in the signed arithmetic) and are not physically meaningful as wavelengths. The function does not validate units or enforce positivity; callers are responsible for providing correctly scaled SI velocities.
    """
    from periodictable.nsf import neutron_wavelength_from_velocity
    return neutron_wavelength_from_velocity(velocity)


################################################################################
# Source: periodictable.plot.table_plot
# File: periodictable/plot.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def periodictable_plot_table_plot(
    data: dict,
    form: str = "line",
    label: str = None,
    title: str = None
):
    """Plot periodic table data using element symbols positioned by element number and a numeric value.
    
    This function is part of the periodictable package and is typically used to visualize a per-element scalar property (for example, atomic mass, density, x-ray or neutron scattering values, or approximate activation estimates described in the package README). It places each element's chemical symbol at a horizontal position given by Element.number and a vertical position given by the numeric value, using matplotlib.pyplot text artists and axis controls.
    
    Args:
        data (dict): Mapping of Element -> float. Keys must be Element objects from the periodictable package (objects providing integer attribute number and string attribute symbol). Values are numeric scalar properties (floats) to plot for each element. Values that are None are skipped when placing text, but all values in the dict are used to compute the vertical axis limits; if the mapping is empty or contains non-numeric values (including None), computing min/max can raise ValueError or TypeError. Practical significance: supply the element-to-value mapping produced when examining element properties such as mass, density, scattering lengths, or cross sections so they can be inspected visually.
        form (str): Layout specifier for the table. The documented options are "line" or "grid"; the default is "line". In the current implementation only "line" is acted upon: when form == "line" the function places element symbols at x = Element.number and sets the x-axis limits to 0..100, draws each symbol with a rounded bounding box (boxstyle "round", line width 1, black edge, pale gray fill), and computes y-axis limits from the data with a 5% margin around the min/max. If form is any other value (including "grid"), the function performs no plotting (no warning is emitted). Practical significance: choose "line" to produce a simple symbol-vs-value scatter laid out by atomic number; be aware that "grid" is accepted by name but not implemented in this version.
        label (str): Y-axis label text to set via matplotlib.pyplot.ylabel. If None (default) no ylabel is set. Practical significance: provide a human-readable label describing the numeric values (for example "scattering length (fm)" or "density (g/cm^3)").
        title (str): Plot title text to set via matplotlib.pyplot.title. If None (default) no title is set. Practical significance: supply a descriptive title for the figure (for example "Neutron scattering lengths by element").
    
    Returns:
        None: This function does not return a value. Side effects: it modifies the current matplotlib state (adds text artists for each element, sets x-axis limits to 0..100, computes and sets y-axis limits to min..max with a 5% margin, and optionally sets the x-axis label to "Element number", the y-axis label, and the figure title). It does not create a new Figure explicitly nor call plt.show(); callers must manage figure creation, display, or saving as needed.
    
    Failure modes and notes:
        - If data is empty, min(values) / max(values) will raise ValueError.
        - If any value in data.values() is None or a non-numeric type, computing min/max may raise TypeError; although None values are not placed as text, they are still included in the list used for min/max.
        - If keys in data are not periodictable Element objects with integer .number and string .symbol attributes, AttributeError or TypeError may occur when accessing those attributes or when passing them to matplotlib.
        - The horizontal axis is fixed to the range 0 to 100 in this implementation; element.number values outside this range will be plotted but may be off-screen.
        - The function relies on matplotlib.pyplot; ensure a valid matplotlib backend is configured in the execution environment.
    """
    from periodictable.plot import table_plot
    return table_plot(data, form, label, title)


################################################################################
# Source: periodictable.util.cell_volume
# File: periodictable/util.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def periodictable_util_cell_volume(
    a: float = None,
    b: float = None,
    c: float = None,
    alpha: float = None,
    beta: float = None,
    gamma: float = None
):
    """Compute the unit cell volume from crystallographic lattice parameters used throughout the periodictable package for density, scattering, and activation-related calculations.
    
    Args:
        a (float): Lattice spacing a in Angstroms (|Ang|). This is the primary, required edge length of the unit cell. If a is None the function raises TypeError. This parameter represents the physical distance between repeating lattice points along the a crystallographic axis and is required to compute the cell volume used e.g. in mass density and scattering-length-density calculations.
        b (float): Lattice spacing b in Angstroms (|Ang|). If None, b defaults to the value of a (treats the cell as having b = a). This parameter represents the cell edge length along the b axis and is used directly in the multiplicative part of the volume formula.
        c (float): Lattice spacing c in Angstroms (|Ang|). If None, c defaults to the value of a (treats the cell as having c = a). This parameter represents the cell edge length along the c axis and is used directly in the multiplicative part of the volume formula.
        alpha (float): Angle alpha in degrees (|deg|), the angle between the b and c edges. If None, alpha defaults to 90 degrees (the code treats missing alpha as cos(alpha)=0). The provided angle is interpreted in degrees and converted to radians internally before taking the cosine; this affects the geometric factor under the square root in the volume formula.
        beta (float): Angle beta in degrees (|deg|), the angle between the a and c edges. If None, beta defaults to the value of alpha (so beta = alpha when alpha is provided, otherwise beta = 90 degrees). The provided angle is interpreted in degrees and converted to radians internally.
        gamma (float): Angle gamma in degrees (|deg|), the angle between the a and b edges. If None, gamma defaults to the value of alpha (so gamma = alpha when alpha is provided, otherwise gamma = 90 degrees). The provided angle is interpreted in degrees and converted to radians internally.
    
    This function implements the standard general-cell volume formula used in crystallography:
    V = a * b * c * sqrt(1 - cos^2(alpha) - cos^2(beta) - cos^2(gamma) + 2*cos(alpha)*cos(beta)*cos(gamma)),
    where cos(…) values are computed from the supplied angles (degrees -> radians conversion performed with math.radians). Behavior notes: a is required; b and c default to a when omitted; alpha defaults to 90 degrees when omitted (implemented as cos(alpha)=0); beta and gamma default to alpha when omitted. There are no external side effects (no I/O, no global state changes). All lengths are in Angstroms and angles in degrees; the output is in cubic Angstroms.
    
    Returns:
        V (float): Cell volume in cubic Angstroms (|Ang^3|). This scalar value is the geometric volume of the crystallographic unit cell and is used by the package for converting between mass and density, calculating scattering length densities, and other lattice-dependent properties.
    
    Raises:
        TypeError: If the required parameter a is missing (None) or if provided parameters are of an invalid/non-numeric type such that arithmetic or trig operations cannot be performed.
        ValueError: If the combination of lattice parameters leads to a negative argument under the square root (non-physical or inconsistent angles/lengths), causing a math domain error when computing the square root.
    """
    from periodictable.util import cell_volume
    return cell_volume(a, b, c, alpha, beta, gamma)


################################################################################
# Source: periodictable.util.parse_uncertainty
# File: periodictable/util.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def periodictable_util_parse_uncertainty(s: str):
    """Parse a numeric value with an uncertainty notation and return the numeric value and its 1-sigma uncertainty.
    
    This function is used throughout the periodictable package to interpret numeric fields that include uncertainties in data tables (for example isotopic masses, densities, and scattering factors used in x-ray and neutron calculations). It accepts several common textual forms produced in tabulated scientific data and converts them into a canonical (value, uncertainty) pair of Python floats so downstream code can perform arithmetic, unit conversions, and statistical combination of uncertainties.
    
    Args:
        s (str): Input string containing a numeric value with optional uncertainty. Accepted forms are:
            - A bare numeric value, e.g. "23.0035", which is interpreted as value with zero uncertainty.
            - The parenthetical form "value(unc)", e.g. "23.0035(12)", "23(1)", "23.0(1.0)" or "23(1.0)". The substring inside the parentheses is interpreted as the uncertainty. If the uncertainty has no decimal point but the value does, the parser aligns the significance by inserting leading zeros into the uncertainty (so "23.0035(12)" yields uncertainty 0.0012). Any characters after the closing parenthesis are ignored when extracting the uncertainty.
            - A bracketed nominal form "[nominal]" which is equivalent to a bare value with zero uncertainty.
            - A bracketed range "[low,high]" which denotes a uniform (rectangular) distribution between low and high; the function returns the midpoint as the nominal value and the equivalent 1-sigma uncertainty computed as (high - low) / sqrt(12), matching the standard conversion from a rectangular distribution to its 1-sigma width.
          The input must be a string. Exponential notation containing an exponent after a parenthetical uncertainty (for example "1.032(4)E10") is not supported by this parser and will not be interpreted as intended.
    
    Returns:
        tuple: A pair (value, uncertainty) where value is the nominal numeric value and uncertainty is the 1-sigma uncertainty, both returned as Python floats. For a plain numeric input or a bracketed nominal form the uncertainty is 0. For a bracketed range the returned uncertainty is (high - low) / sqrt(12). If the input string is empty (""), the function returns (None, None) to indicate missing data rather than treating it as 0 +/- infinity.
    
    Behavior, defaults, and failure modes:
        The function performs no I/O and has no side effects. It uses Python float conversions to produce numeric results; malformed numeric text will raise the usual Python exceptions (for example ValueError) originating from float() conversions. The parser does not accept exponential notation attached to parenthetical uncertainties and will not correctly parse strings like "1.032(4)E10". If the input contains parentheses, only the text between the first "(" and the first following ")" is used as the uncertainty. The rectangular-range conversion to a 1-sigma equivalent is provided to enable statistical combination of uncertainties in downstream calculations (for example propagation of uncertainties in computed cross sections and activation estimates used by the package).
    """
    from periodictable.util import parse_uncertainty
    return parse_uncertainty(s)


################################################################################
# Source: periodictable.xsf.index_of_refraction
# File: periodictable/xsf.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def periodictable_xsf_index_of_refraction(
    compound: str,
    density: float = None,
    natural_density: float = None,
    energy: float = None,
    wavelength: float = None
):
    """Calculates the X-ray index of refraction for a given chemical compound using tabulated atomic scattering factors.
    
    Args:
        compound (str): Chemical formula or formula initializer that specifies the material whose X-ray optical properties are to be computed. In the periodictable package this is used to identify element composition and isotope information so atomic scattering factors (f1, f2) can be looked up. The compound string controls which elements and stoichiometry are passed to the underlying xray_sld routine to compute the material scattering length density.
        density (float): Mass density of the compound in g/cm^3. If None, xray_sld will use the package default or any density implied by the compound definition; providing an explicit value overrides those defaults and affects the computed index because the scattering contribution scales with mass density.
        natural_density (float): Mass density in g/cm^3 for the compound assuming naturally occurring isotope abundances. This parameter is forwarded to xray_sld and is relevant when the user needs the density corresponding to natural isotope composition rather than any enriched or user-specified composition. If None, no special natural-density override is applied.
        energy (float): X-ray energy in keV. If provided, the function converts this energy to wavelength using periodictable.xsf.xray_wavelength and uses that wavelength to compute the index of refraction. When both energy and wavelength are supplied, energy takes precedence (energy is converted to wavelength and the provided wavelength is overwritten).
        wavelength (float): X-ray wavelength in Angstroms. This is used directly to compute the index if energy is not supplied. The function requires at least one of energy or wavelength to be provided; units must be keV for energy and Angstrom for wavelength as expected by the periodictable X-ray scattering routines.
    
    Behavior and side effects:
        The function computes atomic scattering factors (f1, f2) by calling xray_sld(compound, density=density, natural_density=natural_density, wavelength=wavelength) after ensuring a wavelength is available. If energy is not None, wavelength is set from energy via xray_wavelength(energy). If neither energy nor wavelength is provided, the function raises an AssertionError with the message "scattering calculation needs energy or wavelength". The calculation implements the standard relation used by the LBL/X-ray databases (xdb.lbl.gov / henke.lbl.gov): n = 1 - wavelength**2/(2*pi) * (f1 + i*f2) * 1e-6, where wavelength is in Angstroms and f1/f2 are the real and imaginary atomic scattering factors gathered from the periodictable data tables. No file I/O or external network access is performed by this function itself; errors raised by xray_wavelength or xray_sld (for example invalid formula syntax or missing data) propagate to the caller.
    
    Returns:
        float or vector: Unitless index of refraction of the material at the given X-ray energy or wavelength. The return value follows the complex convention n = 1 - delta + i*beta (so values may be complex-valued when absorption is present). The shape (scalar or vector) matches the input (energy/wavelength) shape when those inputs are arrays/vectors supported by the underlying routines.
    """
    from periodictable.xsf import index_of_refraction
    return index_of_refraction(compound, density, natural_density, energy, wavelength)


################################################################################
# Source: periodictable.xsf.xray_energy
# File: periodictable/xsf.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def periodictable_xsf_xray_energy(wavelength: numpy.ndarray):
    """periodictable.xsf.xray_energy: Convert an X-ray wavelength (measured in Angstrom) to photon energy (keV) for use in X-ray scattering and related calculations in the periodictable package.
    
    Converts input X-ray wavelength(s) to photon energy using the relation E = h c / λ and internal physical constants provided by the module (Planck constant, speed of light, and electron volt conversion). This function is used by the periodictable package's X-ray scattering and cross-section utilities to translate a wavelength given in Angstrom into an energy value in kilo-electronvolts (keV), which is a standard energy unit in X-ray physics and scattering calculations described in the project README and original source docstring.
    
    Args:
        wavelength (numpy.ndarray): Input wavelength(s) of X-rays in Angstrom (Å). The function accepts a scalar numeric value (float) or array-like/vector inputs that are convertible to a NumPy array; the signature annotates the parameter as numpy.ndarray and the implementation calls numpy.asarray on the provided value. In the physical domain, this represents the photon wavelength used in X-ray scattering and spectroscopy. Typical usage is to pass a single wavelength (float) or an array of wavelengths for batch conversion. The function does not modify the input in-place; it reads the numeric values and computes corresponding energies.
    
    Returns:
        float or numpy.ndarray: Photon energy or energies in kilo-electronvolts (keV). The return has the same shape semantics as the input: a scalar float for a scalar input, or a NumPy ndarray for vector inputs. The returned energy values are suitable for downstream calculations in X-ray scattering, cross-section lookups, or plotting routines that require energy in keV.
    
    Behavior, defaults, and failure modes:
        The conversion uses E = h c / λ with h (Planck constant) and c (speed of light) taken from the module and a conversion factor to produce keV. Internally the code performs arithmetic after calling numpy.asarray(wavelength), so non-numeric inputs that cannot be converted to a NumPy array will raise a TypeError. A wavelength value of zero will cause a division-by-zero condition, which will produce an infinity (inf) result and may emit a NumPy runtime warning; negative wavelength values are numerically processed but are not physically meaningful for photon wavelengths and will yield negative energy values. There are no side effects or global state changes; the function is pure and deterministic for given numeric inputs.
    """
    from periodictable.xsf import xray_energy
    return xray_energy(wavelength)


################################################################################
# Source: periodictable.xsf.xray_wavelength
# File: periodictable/xsf.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def periodictable_xsf_xray_wavelength(energy: numpy.ndarray):
    """Convert X-ray photon energy to wavelength for use in x-ray scattering and optics calculations.
    
    This function, periodictable.xsf.xray_wavelength, converts photon energy values expressed in kilo-electronvolts (keV) to wavelengths expressed in Angstroms (Å). It is used in the periodictable package’s x-ray scattering and optics modules (xsf) where wavelength is required for computing scattering factors, cross sections, or optical properties derived from energy-dependent datasets (for example, values sourced from the LBL Center for X-ray Optics as used by this package). The conversion implements the physical relation λ = h c / E using the module’s Planck constant, electron volt, and speed of light constants, and returns numeric wavelength values that follow the input shape semantics of numpy.asarray.
    
    Args:
        energy (numpy.ndarray): Photon energy value(s) in kilo-electronvolts (keV). This argument is the input energy to convert; it may be a NumPy array of any shape (the function calls numpy.asarray on the input so array-like sequences are accepted). Scalar numeric values (e.g., a Python float) are accepted in practice and will be converted to a zero-dimensional NumPy array internally, but the documented type is numpy.ndarray to match the function signature. Each element represents an X-ray photon energy in keV; the practical significance is that these energies feed downstream x-ray scattering and cross-section computations in the periodictable library.
    
    Returns:
        numpy.ndarray: Wavelength value(s) in Angstroms (Å) corresponding to the input energy value(s). The returned array has the same shape as numpy.asarray(energy). For scalar inputs the return may be a NumPy scalar/0-d array. The numeric relationship implemented is λ = h c / E with h as the Planck constant (J·s), c as the speed of light (m/s), and E converted from keV to joules via the module electron-volt constant; a factor of 1e7 is applied to convert meters to Angstroms.
    
    Behavior and failure modes:
        The function performs elementwise division and will propagate NumPy semantics: zero energy values lead to infinities and may emit NumPy runtime warnings; negative energies produce negative wavelengths according to the formula (physically non-meaningful for real photons and therefore should be avoided); non-numeric inputs that cannot be converted by numpy.asarray will raise an error (for example, a TypeError). There are no side effects (no global state is modified); the function is pure in that it returns a new array derived from the input. Constants used come from the periodictable module and are in SI units as required by the conversion formula.
    """
    from periodictable.xsf import xray_wavelength
    return xray_wavelength(energy)


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
