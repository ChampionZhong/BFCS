"""
Regenerated Google-style docstrings for module 'chempy'.
README source: others/readme/chempy/README.rst
Generated at: 2025-12-02T01:55:41.044337Z

Total functions: 45
"""


import numpy

################################################################################
# Source: chempy._util.intdiv
# File: chempy/_util.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def chempy__util_intdiv(p: int, q: int):
    """Integer division which rounds toward zero.
    
    Performs integer division of two integers p (numerator) and q (denominator) and returns the integer quotient with truncation toward zero. This differs from Python's floor division operator (//), which rounds toward negative infinity for negative operands; intdiv corrects that behavior so that results are the mathematical truncation of the exact quotient. In the ChemPy codebase this function is useful in contexts that require integer arithmetic with truncation semantics, for example when scaling or normalizing integer stoichiometric coefficients, distributing discrete counts, or computing signed integer indices where rounding toward zero is the intended domain behavior. The function has no side effects and its result is deterministic and constant-time for typical Python integer operations.
    
    Args:
        p (int): The integer numerator to be divided. In chemical computing contexts this often represents an integer count, coefficient, or signed difference that must be divided by an integer factor.
        q (int): The integer denominator by which p is divided. In chemical contexts this can represent a scaling factor, group size, or divisor for normalizing coefficients. Must be non-zero; passing zero will cause a ZeroDivisionError.
    
    Returns:
        int: The integer quotient of p divided by q with rounding toward zero (i.e., truncation). For positive results this equals p // q; for negative results it returns the truncated value nearest to zero, not the floored value. For example, intdiv(3, 2) == 1 and intdiv(-3, 2) == -1.
    
    Raises:
        ZeroDivisionError: If q is zero, the underlying integer division operation will raise this exception.
    """
    from chempy._util import intdiv
    return intdiv(p, q)


################################################################################
# Source: chempy._util.prodpow
# File: chempy/_util.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def chempy__util_prodpow(bases: list, exponents: numpy.ndarray):
    """chempy._util.prodpow computes the elementwise product of several numeric bases raised to corresponding exponents. In the context of ChemPy this is typically used to form mass-action style products such as ∏_i [A_i]^{ν_i} encountered in equilibrium expressions and rate laws, where "bases" are the factors (for example concentrations or activity coefficients) and "exponents" are the stoichiometric or power coefficients.
    
    This function converts the provided exponents to a NumPy array and then computes bases ** exponents using NumPy broadcasting rules, finally multiplying along the last axis. For example, prodpow([2, 3], np.array([[0, 1], [1, 2]])) yields array([3, 18]) corresponding to [2**0 * 3**1, 2**1 * 3**2].
    
    Args:
        bases (list): A sequence (Python list) of numeric base values. Each element represents a factor to be raised to an exponent (for example, a concentration or other multiplicative term in a chemical expression). The length of bases must match the size of the last axis of exponents or be broadcastable to that shape according to NumPy broadcasting rules.
        exponents (numpy.ndarray): Array of numeric exponents (stoichiometric coefficients, power terms, etc.). This parameter is converted internally via numpy.asarray(exponents). The last axis of this array indexes the exponents that pair with the entries of bases; any leading axes produce multiple independent results.
    
    Returns:
        numpy.ndarray: The elementwise product of bases raised to the corresponding exponents, computed by first evaluating bases ** exponents and then reducing (multiplying) along the last axis (np.multiply.reduce(..., axis=-1)). The returned array has shape exponents.shape[:-1]; when exponents is 1-D the result is a 0-D NumPy scalar. The dtype of the result follows NumPy's type promotion rules based on the input types.
    
    Behavior and failure modes:
        - The function applies NumPy broadcasting rules when evaluating bases ** exponents. If shapes are incompatible for broadcasting, NumPy will raise a ValueError.
        - exponents is converted with numpy.asarray, so array-like inputs are accepted in practice, but the signature documents numpy.ndarray.
        - If inputs contain non-numeric types a TypeError or other exceptions from NumPy operations may be raised.
        - Mathematical domain issues can occur (for example, negative bases with non-integer exponents), which can produce NaNs, runtime warnings, or complex results depending on NumPy dtypes and values.
        - There are no side effects: the inputs are not modified in-place by this function.
    """
    from chempy._util import prodpow
    return prodpow(bases, exponents)


################################################################################
# Source: chempy.chemistry.equilibrium_quotient
# File: chempy/chemistry.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def chempy_chemistry_equilibrium_quotient(concs: numpy.ndarray, stoich: list):
    """chempy.chemistry.equilibrium_quotient: Calculate the equilibrium quotient Q for a chemical equilibrium from per-substance concentrations and stoichiometric coefficients. In chemical equilibrium modeling (see ChemPy README examples for equilibria and pH calculations), the equilibrium quotient is the product of each species concentration raised to its stoichiometric coefficient; this function computes that product for a single concentration vector or for multiple concentration sets (batched) and is therefore used when evaluating reaction quotients, comparing to equilibrium constants K, or computing pH and speciation.
    
    This function accepts a numpy.ndarray of concentrations or any object with a 1-D semantics (no ndim attribute or ndim == 1) representing a single set of per-substance concentrations. If a 2-D numpy.ndarray is provided, it is interpreted as a collection of independent concentration sets with shape (n_sets, n_substances) and the function returns a numpy.ndarray of length n_sets with the quotient for each set. The stoichiometric coefficients are applied elementwise: each concentration is raised to the power given by the corresponding stoichiometric coefficient and all terms are multiplied together. No in-place modification of inputs is performed.
    
    Args:
        concs (numpy.ndarray): Per-substance concentration data used to form the equilibrium quotient. For a single equilibrium calculation, provide a 1-D array-like object of length N where N is the number of chemical species. For multiple independent calculations (batched evaluation), provide a 2-D numpy.ndarray with shape (n_sets, N) where each row is a set of concentrations for the N species; the implementation transposes such a 2-D array internally and returns one quotient per row. Objects without an ndim attribute are treated as 1-D. Values are interpreted as numeric concentrations (e.g., molar), and typical usage in ChemPy is to compare the returned quotient to an equilibrium constant K when solving equilibria or predicting pH/speciation.
        stoich (list): Iterable of stoichiometric coefficients (integers) of length N, one coefficient per species, describing the exponent applied to each species concentration in the quotient. Positive coefficients correspond to species in the product side of a reaction and negative coefficients to reactants (consistent with ChemPy Equilibrium conventions). The function uses these coefficients directly as exponents in concentration**stoich_element.
    
    Returns:
        float or numpy.ndarray: The computed equilibrium quotient Q. For a 1-D input (single concentration set) a scalar float-like value is returned. For a 2-D input (batched concentration sets) a 1-D numpy.ndarray of length n_sets is returned, where each element is the quotient for the corresponding input row. The numeric value represents the product over species of [C_i]**nu_i and can be compared directly to an equilibrium constant K to assess reaction direction or used in root-finding when solving equilibrium systems.
    
    Behavior and failure modes:
        - If concs has no attribute ndim or concs.ndim == 1 the function treats concs as a single concentration vector and returns a scalar. If concs is a numpy.ndarray with ndim > 1 it is treated as shape (n_sets, n_substances) and the function returns an array of quotients for each set.
        - The function does not validate that len(stoich) equals the number of species inferred from concs; Python's zip is used to pair stoichiometric coefficients with concentrations, so if lengths differ the operation will silently truncate to the shorter sequence and produce an unintended result without raising an exception.
        - Exponentiation and multiplication follow NumPy broadcasting and arithmetic rules. Supplying zeros together with negative stoichiometric coefficients will lead to division by zero (numpy will emit warnings and yield inf/NaN values), and very large exponents or extreme concentration values can cause overflow/underflow or loss of precision. Users should ensure inputs are in appropriate units and ranges (e.g., molar concentrations) and handle or check for NaN/inf results as needed.
        - There are no side effects; inputs are not modified. The function will raise exceptions only as produced by NumPy operations (for example, on incompatible shapes that prevent broadcasting during exponentiation/multiplication).
    """
    from chempy.chemistry import equilibrium_quotient
    return equilibrium_quotient(concs, stoich)


################################################################################
# Source: chempy.kinetics.arrhenius.fit_arrhenius_equation
# File: chempy/kinetics/arrhenius.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def chempy_kinetics_arrhenius_fit_arrhenius_equation(
    T: float,
    k: numpy.ndarray,
    kerr: numpy.ndarray = None,
    linearized: bool = False,
    constants: dict = None,
    units: dict = None
):
    """Curve fitting of the Arrhenius equation to measured rate-constant data.
    
    This function is used within chemical kinetics workflows to estimate the Arrhenius parameters
    for a reaction from measured rate constants at given temperatures. In the domain of physical
    chemistry (see README: "Arrhenius & Eyring equation"), the Arrhenius equation is
    k(T) = A * exp(-Ea / (R * T)), where A is the pre-exponential factor and Ea is the
    activation energy. fit_arrhenius_equation calls an internal fitter (_fit) with the
    Arrhenius model and provides transforms so that the fitted parameter vector p corresponds
    to the linearized model coefficients p[0] = ln(A) and p[1] = -Ea / R. The provided
    parameter transforms convert the fitted p into physically meaningful Arrhenius parameters:
    A = exp(p[0]) and Ea = -p[1] * R, where R is obtained via _get_R(constants, units).
    
    Args:
        T (float): Temperature at which the rate constants k were measured. In typical
            kinetics use-cases this is the absolute temperature in kelvin(s). Although the
            parameter is declared as float in the signature, the internal fitter expects T
            values that correspond elementwise to entries in k (i.e., a single float for a
            single measurement or an array-like of temperatures matching the shape of k).
            The temperature values are used in the model either directly (non-linear fit)
            or via the transform 1/T (linearized fit).
        k (numpy.ndarray): Array of measured rate constants corresponding to T. These are
            the dependent variable values k(T) to be fitted to the Arrhenius expression.
            Values must be positive when using the linearized option because the natural
            logarithm of k is taken; negative or zero values will cause a failure when
            linearized=True. Typical units are s^-1 or concentration^-1 time^-1 depending
            on reaction order; units are handled only insofar as they affect interpretation
            of returned parameters and the gas constant lookup via constants/units.
        kerr (numpy.ndarray = None): Optional array of one-sigma uncertainties (errors)
            associated with each element of k. When provided, these are interpreted as
            pointwise measurement uncertainties and used by the internal fitter to weight
            the fit (in _fit). If None (the default), the fitter assumes equal weighting
            for all points. The shape of kerr, if provided, must match k.
        linearized (bool = False): If True, the fitter performs a linear least-squares fit
            on the transformed variables x = 1/T and y = ln(k), using the internal
            transformations lambda T, k: 1/T and lambda T, k: np.log(k). This is equivalent
            to fitting ln(k) = ln(A) + (-Ea/R) * (1/T) and returns parameters consistent
            with that linear model. If False (default), the fitter performs a non-linear
            fit of the Arrhenius equation directly. Use of linearized=True is faster and
            numerically simpler but requires k>0 and assumes the error structure is
            appropriate for a log transform; linearized=False is more flexible but may
            require good initial guesses and is subject to non-linear fitting convergence
            issues.
        constants (dict = None): Optional dictionary used by the internal helper _get_R
            to obtain the gas constant R consistent with the units in which Ea should be
            expressed. If None, _get_R is invoked with the default behavior defined in the
            package; if a dictionary is provided it should contain the necessary entries
            for resolving R according to the project's constants/units conventions. An
            absent or incorrect dictionary may lead to an error when converting the fitted
            coefficient -p[1] into Ea via -p[1] * R.
        units (dict = None): Optional units mapping passed to _get_R together with
            constants to ensure consistent unit handling for the activation energy Ea.
            This argument follows the package conventions for unit dictionaries. If units
            is None, default unit-handling inside _get_R will be used. Supplying an
            inappropriate units mapping can lead to unit-mismatch errors or incorrect
            numeric scaling of Ea.
    
    Returns:
        object: The return value is the raw result produced by the internal _fit routine.
        The returned structure contains the fitted parameter vector p (whose elements for
        the Arrhenius fit are p[0] = ln(A) and p[1] = -Ea / R), and also includes whatever
        additional diagnostics/covariance information _fit provides (for example covariance
        estimates, fit uncertainties and solver info). The function-supplied parameter
        transforms convert p into the physically meaningful Arrhenius parameters:
        A = exp(p[0]) (pre-exponential factor) and Ea = -p[1] * R (activation energy,
        with R resolved via _get_R(constants, units)). Users should inspect the returned
        object to extract the fitted vector, transformed Arrhenius parameters, covariance
        and any diagnostic flags.
    
    Behavior, defaults, and failure modes:
        - The function delegates the actual numerical work to an internal _fit routine
          and supplies model and transform functions tailored for the Arrhenius law.
        - Default behavior is non-linear fitting (linearized=False). Setting linearized=True
          performs an ordinary linear least-squares fit on (1/T, ln(k)).
        - When linearized=True, k values must be strictly positive; otherwise np.log(k)
          will produce NaNs or raise errors and the fit will fail.
        - The shapes of T, k and kerr (if provided) must be compatible for elementwise
          operations; mismatched lengths will cause the internal fitter to raise an error.
        - If kerr is provided it is treated as pointwise one-sigma uncertainties for weighting;
          if kerr contains zeros or NaNs the weighting will be invalid and may cause the
          fit to fail or produce unreliable parameter estimates.
        - The conversion from the fitted linear coefficient p[1] to activation energy Ea uses
          the gas constant R resolved by _get_R(constants, units). If R cannot be resolved
          from the provided constants/units, _get_R may raise an exception and the function
          will fail.
        - Numerical issues common to least-squares and non-linear optimization apply:
          ill-conditioned matrices, lack of convergence, or insufficient data-range in 1/T
          can lead to large parameter uncertainties or fit failure. The caller should check
          the diagnostic information returned by _fit to assess success and parameter reliability.
    
    Side effects:
        - This function has no external side effects (it does not modify global state).
        - It returns the result produced by the internal fitter; any conversion of fitted
          coefficients to A and Ea is available via the provided parameter-transform logic
          documented above but users must extract those transformed values from the _fit
          return structure.
    """
    from chempy.kinetics.arrhenius import fit_arrhenius_equation
    return fit_arrhenius_equation(T, k, kerr, linearized, constants, units)


################################################################################
# Source: chempy.kinetics.eyring.fit_eyring_equation
# File: chempy/kinetics/eyring.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def chempy_kinetics_eyring_fit_eyring_equation(
    T: float,
    k: numpy.ndarray,
    kerr: numpy.ndarray = None,
    linearized: bool = False,
    constants: dict = None,
    units: dict = None
):
    """Curve fit of the Eyring equation to experimental rate-constant data to obtain linear-fit parameters
    and derived activation parameters used in chemical kinetics (activation enthalpy and entropy).
    This function is used in physical/chemical kinetics contexts (see ChemPy README section "Arrhenius & Eyring equation")
    to fit the Eyring form k(T) = (kB T / h) exp(DeltaS‡/R) exp(-DeltaH‡/(R T)) to measured rate constants
    and to provide the correspondence between the fitted linear model and the thermochemical parameters.
    
    Args:
        T (float): Temperature at which the rate constants were measured. This is the absolute temperature
            (Kelvin) associated with the k values passed in the k array. The function uses T in the
            1/T transform (x = 1/T) applied when fitting the linearized Eyring expression.
        k (numpy.ndarray): Array of measured rate constants k(T) corresponding to the temperature T.
            These are the observed rate constants (e.g., in s^-1 or other appropriate kinetic units).
            The routine forms the transformed dependent variable y = ln(k / T) when performing the
            linearized fit, so all elements of k must be strictly positive for the logarithm to be defined.
        kerr (numpy.ndarray, optional): Optional array of uncertainties (standard errors) for the k values.
            If provided, these are used as weights in the fitting procedure implemented by the internal
            helper _fit. The length and ordering must correspond to k. Default is None, meaning no explicit
            per-point errors are supplied (unweighted fit or internally determined weighting).
        linearized (bool): If True, perform a weighted linear regression on the transformed variables
            x = 1/T and y = ln(k / T) (analytical linear least-squares on the linearized Eyring form).
            If False (default), perform the default fitting strategy used by the internal _fit helper,
            which may use a non-linear fitting procedure on the original form. Use linearized=True to
            obtain the standard linear interpretation where the fitted parameter vector p = [a, b]
            satisfies ln(k/T) ≈ a + b*(1/T).
        constants (dict, optional): Optional dictionary of physical constants to override defaults used
            by the routine. This dictionary is passed to internal helpers _get_R and _get_kB_over_h to
            determine the gas constant R and the ratio kB/h (Boltzmann constant over Planck constant).
            If None (default), built-in default physical constants are used. Providing this dict allows
            using alternative constant values or units systems for domain-specific analyses.
        units (dict, optional): Optional dictionary of unit conversion information passed together with
            constants to the internal helpers. This allows the function to interpret or convert the
            provided k and T values consistently with a particular units convention. If None (default),
            no unit overrides are applied and the function assumes conventional SI-like usage.
    
    Returns:
        object: The raw result returned by the internal helper _fit. At minimum the returned object
            contains the optimized fit parameter vector p (ordered as p[0] = intercept a, p[1] = slope b)
            for the linearized relation ln(k/T) ≈ a + b*(1/T). The code also provides the explicit
            mapping from the fitted linear parameters to activation parameters:
            DeltaH‡ = -b * R and DeltaS‡ = R * (a - ln(kB/h)), where R and ln(kB/h) are determined
            from the provided constants/units (via _get_R and _get_kB_over_h). Users should inspect the
            returned object to obtain covariance, fit statistics, or any additional fields produced by _fit.
    
    Behavior, defaults, and failure modes:
        - The function computes R = _get_R(constants, units) and ln(kB/h) = log(_get_kB_over_h(constants, units))
          to relate fitted linear parameters to physical activation enthalpy and entropy.
        - For linearized=True the function fits the transformed variables x = 1/T and y = ln(k/T).
          For this transformation to be valid, T must be non-zero and finite, and all k values must be > 0.
          If any k <= 0 or T == 0, a ValueError or a floating-point error may occur (log or division by zero).
        - If kerr is supplied, its length and ordering must match k; otherwise the underlying _fit call may raise
          an exception for inconsistent array shapes.
        - The constants and units arguments, if provided, must be compatible with the internal helpers
          _get_R and _get_kB_over_h (they are forwarded directly); invalid or incomplete dictionaries can
          cause those helpers to raise KeyError/TypeError.
        - No file or global-state side effects are performed; the function only computes and returns the
          fit result from _fit. Exceptions raised originate from numpy, math.log on invalid input, or the
          internal _fit/_get_* helpers if fitting or constant lookup fails.
    """
    from chempy.kinetics.eyring import fit_eyring_equation
    return fit_eyring_equation(T, k, kerr, linearized, constants, units)


################################################################################
# Source: chempy.kinetics.integrated.binary_irrev
# File: chempy/kinetics/integrated.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def chempy_kinetics_integrated_binary_irrev(
    t: numpy.ndarray,
    kf: float,
    prod: float,
    major: float,
    minor: float,
    backend: str = None
):
    """chempy.kinetics.integrated.binary_irrev: Analytic product transient for an irreversible 2-to-1 bimolecular reaction.
    
    Computes the time-dependent concentration of the product for an irreversible second-order reaction where two reactant molecules (one major and one minor species) form a single product. This function implements the closed-form integrated rate expression used in the chempy kinetics utilities and integrated-rate fitting routines. The implementation uses a backend arithmetic library obtained via get_backend(backend) (defaulting to the numpy backend when backend is None) so the result can be numeric (array-like or scalar) or symbolic (e.g. when using a SymPy backend) depending on the inputs and backend.
    
    Args:
        t (float, Symbol or array_like): Time variable at which the product concentration is evaluated. For numeric backends this may be a scalar or an array-like sequence of time points; for symbolic backends this may be a symbolic time symbol, enabling algebraic manipulation of the analytic solution. The returned value has the same kind (scalar, array-like or symbolic) as permitted by the chosen backend and inputs.
        kf (number or Symbol): Forward (bimolecular) rate constant for the irreversible reaction. In typical kinetic modelling this is a non-negative numeric rate constant; when supplied as a Symbol with a symbolic backend the returned expression will contain this symbol.
        prod (number or Symbol): Initial concentration of the product/complex present at t = 0. This term is added to the analytic transient and permits modelling systems that already contain some product initially.
        major (number or Symbol): Initial concentration of the more abundant reactant at t = 0. This parameter appears in the closed-form expression that determines the time scale and asymptotic behaviour of product formation.
        minor (number or Symbol): Initial concentration of the less abundant reactant at t = 0. The difference (major - minor) appears multiplicatively in the exponent of the analytic expression; when minor equals zero the formula involves a division by zero and is therefore singular.
        backend (module or str): Backend module or backend identifier used to perform arithmetic and exponentiation. If None (the default) the function uses the numpy backend; passing a string such as 'sympy' or a module object selects a symbolic backend (e.g. SymPy) which yields symbolic expressions when inputs are symbolic. The backend must provide an exp implementation and standard arithmetic semantics; otherwise a TypeError or AttributeError may be raised.
    
    Returns:
        float, Symbol or array_like: The product concentration at time t. The concrete return type depends on the provided inputs and chosen backend: numeric inputs with a numeric backend produce a numeric scalar or array-like result; symbolic inputs with a symbolic backend produce a symbolic expression. The returned value represents the analytic solution of the irreversible 2-to-1 reaction and can be used for plotting transients, parameter estimation, or analytical manipulation in kinetics studies.
    
    Behavior and side effects:
        This function is pure (no side effects) and computes the analytic expression using the selected backend's arithmetic. When backend is None the default numeric behaviour corresponds to numpy semantics. No internal state is modified.
    
    Failure modes and cautions:
        The formula includes divisions by the quantity (major/minor - exp(-kf*(major - minor)*t)) and by minor in the major/minor ratio. If minor == 0 a division-by-zero will occur. For some parameter combinations and times the denominator may vanish (leading to a singularity) and raise a ZeroDivisionError or produce infinities/NaNs under numeric backends. If the backend does not provide the required exp function or does not support arithmetic with the supplied input types, a TypeError or AttributeError may be raised. Physically, rate constants and concentrations are normally non-negative; supplying values that violate physical assumptions may produce mathematically valid but non-physical results.
    """
    from chempy.kinetics.integrated import binary_irrev
    return binary_irrev(t, kf, prod, major, minor, backend)


################################################################################
# Source: chempy.kinetics.integrated.binary_irrev_cstr
# File: chempy/kinetics/integrated.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def chempy_kinetics_integrated_binary_irrev_cstr(
    t: numpy.ndarray,
    k: float,
    r: float,
    p: float,
    fr: float,
    fp: float,
    fv: float,
    n: int = 1,
    backend: str = None
):
    """Analytic solution for the transient concentrations in a continuous stirred-tank reactor (CSTR)
    undergoing the irreversible bimolecular reaction 2 A -> n B.
    
    This function evaluates a closed-form (analytic) solution of the ordinary differential
    equations that describe a CSTR with a feed and perfect mixing for the reaction 2 A -> n B.
    The solution is useful when you need rapid, direct evaluation of the time-dependent
    concentrations of the reactant A and product B without performing numerical integration.
    The implementation follows the symbolic derivation (see comments in source) and returns
    concentrations evaluated at the supplied times using the chosen numeric/symbolic backend.
    
    Args:
        t (array_like): Time points at which to evaluate the solution. May be a scalar or
            an array-like sequence; units (for example seconds) must be consistent with the
            rate constant k and the feed-rate-to-volume ratio fv. The returned concentration
            values correspond to these same time points.
        k (float_like): Reaction rate constant for the irreversible bimolecular step 2 A -> n B.
            For a reaction 2 A -> n B the physical units are typically concentration^-1 time^-1
            (e.g., M^-1 s^-1). This parameter controls the speed of conversion from reactant A
            to product B and appears in denominators in the analytic expressions (division by k
            will occur), so k must not be zero for a finite numeric result.
        r (float_like): Initial concentration of reactant A in the reactor at t = 0. This value
            sets the starting concentration profile for the analytic solution and must be given
            in the same concentration units as fr, fp and p.
        p (float_like): Initial concentration of product B in the reactor at t = 0. This is the
            starting concentration of B and must use the same units as r, fr and fp.
        fr (float_like): Concentration of reactant A in the feed stream. This models the inlet
            concentration of A supplied continuously to the CSTR and is used together with fv
            (feed rate / tank volume) to set the driving term for the inlet flow in the ODEs.
        fp (float_like): Concentration of product B in the feed stream. This models any B that
            may be present in the inlet and is combined with fv to determine the inlet contribution
            to the reactor concentration of B.
        fv (float_like): Feed rate divided by reactor volume (flow-per-volume ratio). Physically
            this is the volumetric flow rate divided by tank volume and has units of inverse time
            (e.g., s^-1). It controls the residence time and dilution in the CSTR and must be
            provided in the same time units as those used for t and k.
        n (int): Stoichiometric coefficient for product B produced per reaction event (default 1).
            In the chemical context this is the integer number of B formed by the reaction 2 A -> n B.
            The implementation accepts an integer and uses it algebraically in the analytic formulae.
        backend (module or str): Backend providing the numeric or symbolic primitives used to
            evaluate the closed-form expressions. Default is 'numpy' (numeric evaluation). Other
            options include a symbolic backend such as 'sympy' or a module that provides functions
            used internally (sqrt, exp, tanh, atanh/arctanh, cos, etc.). If a string is given,
            it is resolved to a backend module (see source usage of get_backend). The choice of
            backend determines whether the outputs are numeric arrays (numpy) or symbolic expressions
            (sympy). If backend is None the function will behave as with the default backend.
    
    Behavior, defaults, and side effects:
        This function computes a closed-form, time-dependent solution for concentrations of A and B
        in a well-mixed CSTR with constant feed concentrations fr and fp and constant feed rate fv.
        It does not perform numerical integration and has no side effects: inputs are not mutated
        and no global state is changed. The default backend performs numeric evaluation with numpy,
        producing numeric scalars/arrays that mirror the shape of t. Using a symbolic backend (for
        example sympy) yields symbolic expressions. The default stoichiometric coefficient is n = 1.
    
    Failure modes and warnings:
        - k == 0 will produce a division by zero when evaluating the analytic expressions (x0 = 1/k
          in the implementation). Do not call the function with k equal to zero if a finite numeric
          result is required.
        - The formulae use square roots (sqrt) and inverse hyperbolic tangent (atanh/arctanh). For
          numeric backends (numpy), passing values that make arguments of sqrt negative or arguments
          of atanh outside the domain [-1, 1] will yield NaNs or complex numbers depending on numpy
          configuration. For symbolic backends (sympy), these expressions will be represented symbolically
          and may simplify differently.
        - All concentration-like inputs (r, p, fr, fp) and fv must use consistent units; otherwise the
          numeric results will be physically meaningless.
        - The backend module must provide the mathematical functions used by the implementation;
          otherwise attribute errors may be raised when evaluating the expressions.
    
    Returns:
        length-2 tuple: A two-element tuple (A_conc, B_conc) containing the concentrations of the
            reactant A and product B evaluated at the supplied times t. The returned elements have
            the same numeric/symbolic nature as determined by the chosen backend (e.g., numpy arrays
            when using the numpy backend, or sympy expressions when using sympy). Units of the returned
            concentrations match the units of the input concentrations r, p, fr, and fp. The ordering
            is (reactant_concentration, product_concentration).
    """
    from chempy.kinetics.integrated import binary_irrev_cstr
    return binary_irrev_cstr(t, k, r, p, fr, fp, fv, n, backend)


################################################################################
# Source: chempy.kinetics.integrated.binary_rev
# File: chempy/kinetics/integrated.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def chempy_kinetics_integrated_binary_rev(
    t: float,
    kf: float,
    kb: float,
    prod: float,
    major: float,
    minor: float,
    backend: str = None
):
    """chempy.kinetics.integrated.binary_rev computes the analytic time-dependent concentration of the product (complex) for a reversible 2-to-1 reaction (A + B <-> AB) using the closed-form solution derived for second-order forward (bimolecular) association and first-order backward (unimolecular) dissociation kinetics.
    
    This function is used in chemical kinetics modeling and data analysis (for example when comparing to transient experimental measurements or when supplying an analytic integrated rate expression to fitting routines). It evaluates the same symbolic expression used for derivations in the project (see _integrated.ipynb) but dispatches arithmetic to a numeric or symbolic backend via get_backend(backend), enabling both numeric evaluation (e.g. with NumPy) and symbolic algebra (e.g. with SymPy).
    
    Args:
        t (float, Symbol or array_like): Time at which to evaluate the product concentration. In a physical chemistry context this represents elapsed time with units compatible with the rate constants; when using numeric backends t may be a scalar or an array of times to produce a time series of concentrations. For symbolic backends t may be a Symbol to obtain an algebraic expression.
        kf (number or Symbol): Forward (bimolecular) rate constant for the association A + B -> AB. Physically this has units consistent with concentration^-1 time^-1 (e.g. M^-1 s^-1) and must be expressed in units compatible with t, prod, major, and minor. If provided as a Symbol, a symbolic expression for the transient is returned.
        kb (number or Symbol): Backward (unimolecular) rate constant for the dissociation AB -> A + B. Physically this has units of time^-1 (e.g. s^-1). Must be in consistent units with t and kf.
        prod (number or Symbol): Initial concentration of the complex AB at time zero. This value sets the starting amount of product (complex) and has the same concentration units as major and minor. For typical kinetic experiments prod is often zero but the analytic expression supports nonzero initial complex.
        major (number or Symbol): Initial concentration of the more abundant reactant (the "major" reactant) at time zero. This parameter represents one reactant pool in the bimolecular forward step; the analytic form assumes two distinct reactant pools of different initial abundances with this being the larger.
        minor (number or Symbol): Initial concentration of the less abundant reactant (the "minor" reactant) at time zero. This parameter represents the second reactant pool in the bimolecular forward step; the analytic form assumes a major/minor labeling so the solution is applicable when initial reactant concentrations are unequal.
        backend (module or str): Optional. Backend to use for arithmetic and special functions. Default behavior is to use the NumPy-based backend for numeric evaluation; passing a symbolic backend such as SymPy (module or the string 'sympy') yields a symbolic expression. The backend must support the operations used (addition, multiplication, sqrt, exp). If None is passed, the function uses the default numeric backend (NumPy) as documented in the project.
    
    Returns:
        float, Symbol or array_like: The product (complex AB) concentration evaluated at time t according to the analytic solution for reversible second-order kinetics. The concrete return type and numeric/symbolic behavior follow the provided inputs and chosen backend: numeric inputs with a numeric backend yield floating-point scalars or arrays (array_like) matching the shape of t; symbolic inputs or a symbolic backend yield a symbolic expression (Symbol or composed symbolic expression). The returned concentration uses the same concentration units as the prod, major, and minor arguments and is computed without side effects.
    
    Notes on behavior and failure modes:
        - Units must be consistent across t, kf, kb, and concentrations; the function does not perform unit conversions. Use the chempy.units helpers elsewhere in the package to convert to unitless quantities before calling if needed.
        - If kf equals zero the formula contains division by kf and will raise an error or produce an undefined result; for purely first-order reversible kinetics a different analytic expression should be used.
        - The algebra involves a square root; when the argument under the square root is negative numeric backends will produce complex-valued results, while symbolic backends may retain a symbolic sqrt. This reflects the mathematical solution and should be interpreted accordingly in physical contexts.
        - No mutable global state or I/O side effects are produced; behavior depends only on the arguments and the selected backend.
    """
    from chempy.kinetics.integrated import binary_rev
    return binary_rev(t, kf, kb, prod, major, minor, backend)


################################################################################
# Source: chempy.kinetics.integrated.pseudo_irrev
# File: chempy/kinetics/integrated.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def chempy_kinetics_integrated_pseudo_irrev(
    t: float,
    kf: float,
    prod: float,
    major: float,
    minor: float,
    backend: str = None
):
    """chempy.kinetics.integrated.pseudo_irrev: Analytic product transient for an irreversible pseudo first-order reaction used in ChemPy's integrated kinetics utilities.
    
    Computes the time-dependent product concentration for an irreversible, bimolecular → unimolecular reaction treated under the pseudo-first-order approximation. In this approximation the more abundant reactant (major) is assumed to remain effectively constant, so the bimolecular forward reaction with rate constant kf reduces to first-order kinetics with an effective rate k_eff = major * kf. The function returns the analytic expression prod + minor * (1 - exp(-major * kf * t)), which represents the product (or complex) concentration at time t given the initial concentrations and rate constant. This closed-form expression is useful in kinetics modeling, parameter estimation and fitting of integrated rate laws as described in the ChemPy README's kinetics examples.
    
    Args:
        t (float, Symbol or array_like): Time at which to evaluate the product concentration. In practical use this may be a scalar float for a single time point, a sequence/array of time points for waveform evaluation (numpy arrays are supported by the default backend), or a symbolic time variable (Symbol) when constructing analytic expressions with e.g. sympy. Units are not interpreted by this function; pass unitless numeric values or use compatible unit handling externally (see ChemPy units utilities).
        kf (number or Symbol): Forward bimolecular rate constant for the reaction. In the pseudo-first-order reduction this combines with the major reactant concentration to form an effective first-order rate. Typical units are (concentration^-1 time^-1) for bimolecular kf; when combined with major (concentration) the product major * kf has units of time^-1. kf may be a numeric value for numerical evaluation or a Symbol for symbolic algebra.
        prod (number or Symbol): Initial concentration of the product/complex (the species whose transient is reported) at time t = 0. This is the baseline concentration to which the generated product from the minor reactant adds. Use a numeric value for numeric evaluation or a Symbol for symbolic manipulation.
        major (number or Symbol): Initial concentration of the more abundant reactant. In the pseudo-first-order approximation this concentration is treated as effectively constant and multiplies kf to give the effective first-order rate constant (major * kf). This parameter should be positive in physical models; the function does not enforce sign constraints.
        minor (number or Symbol): Initial concentration of the less abundant reactant that is consumed to form product. The term minor * (1 - exp(-major * kf * t)) represents the fraction of this initial minor pool converted to product over time. Use a numeric value for numeric evaluation or a Symbol for symbolic manipulation.
        backend (module or str): Backend providing basic mathematical operations used to evaluate the expression. Default is 'numpy' when backend is None (i.e., get_backend(backend) will return the numpy-backed API), but can be e.g. the sympy module when constructing symbolic expressions. The backend must provide an exp function (called as backend.exp) and support arithmetic between the provided inputs; an incompatible backend or one lacking exp will raise an AttributeError or similar when called.
    
    Returns:
        float, Symbol or array_like: The product concentration at time t, computed as prod + minor * (1 - backend.exp(-major * kf * t)). The return type mirrors the input types and backend: a scalar float for scalar numeric inputs with the numpy backend, an array_like (e.g. numpy.ndarray) for array-like t with the numpy backend, or a symbolic Expression (Symbol or composed symbolic expression) when using a symbolic backend such as sympy.
    
    Behavior and side effects:
        This is a pure function with no side effects: it does not mutate its inputs or global state. It calls get_backend(backend) to obtain the numerical/symbolic backend (defaulting to numpy) and uses the backend's exp implementation. No unit conversion is performed; callers should supply unitless or pre-converted numeric values. The function does not validate physical constraints such as non-negative concentrations or physically meaningful units.
    
    Failure modes and numerical notes:
        Passing a backend that lacks an exp function, or passing inputs incompatible with the chosen backend's arithmetic, will raise exceptions (AttributeError, TypeError, or backend-specific errors). For large values of major * kf * t the exponential may underflow to zero (producing prod + minor) or overflow for negative large arguments depending on backend numeric limits. Symbolic evaluation requires a symbolic backend (e.g., sympy) and symbolic input types; mixing incompatible numeric and symbolic types without a compatible backend may fail. The function does not perform broadcasting checks beyond what the backend provides; mismatched array shapes may raise errors from the backend.
    """
    from chempy.kinetics.integrated import pseudo_irrev
    return pseudo_irrev(t, kf, prod, major, minor, backend)


################################################################################
# Source: chempy.kinetics.integrated.pseudo_rev
# File: chempy/kinetics/integrated.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def chempy_kinetics_integrated_pseudo_rev(
    t: numpy.ndarray,
    kf: float,
    kb: float,
    prod: float,
    major: float,
    minor: float,
    backend: str = None
):
    """Analytic product transient for a reversible pseudo-first-order reaction.
    
    Computes the time-dependent concentration of the product/complex for a reversible
    bimolecular forward and unimolecular backward reaction under the pseudo-first-order
    assumption (one reactant, "major", is present in large excess and treated as constant).
    This routine is used in ChemPy's kinetics/integrated rate expressions to produce an
    explicit analytic expression that can be evaluated for numeric time arrays (e.g. for
    plotting or fitting) or as a symbolic expression when a symbolic backend is selected.
    The expression returned corresponds to the solution of A + B <-> C with forward
    rate kf (bimolecular) and backward rate kb (unimolecular) when [A] ~ major (constant).
    
    Args:
        t (numpy.ndarray): Time points at which to evaluate the product concentration.
            In practice this is typically a one-dimensional array of monotonically increasing
            time values used for transient simulation or fitting to experimental kinetic data.
            The units of t must be consistent with the units of the rate constants (kf, kb).
        kf (float): Forward (bimolecular) rate constant. This parameter represents the
            bimolecular rate coefficient for formation of the complex (C) from reactants (A + B).
            Its units are typically concentration^-1 time^-1 (e.g. M^-1 s^-1) so that kf * major
            has units of time^-1 under the pseudo-first-order assumption. kf directly controls
            the effective forward first-order rate kf * major used in the analytic solution.
        kb (float): Backward (unimolecular) rate constant. This parameter represents the
            first-order dissociation rate of the complex back to reactants (C -> A + B),
            with units of time^-1 (e.g. s^-1). kb competes with the effective forward rate
            kf * major to determine the approach to equilibrium and the transient time scale.
        prod (float): Initial concentration of the complex (product) at t = 0. This is the
            starting concentration of C used in the analytic transient expression. The units
            must match those of major and minor (e.g. mol/L) so that returned concentrations
            are in the same concentration units.
        major (float): Initial concentration of the more abundant reactant (the one assumed
            to be in large excess and treated as constant). Under the pseudo-first-order
            assumption the forward term is approximated as kf * major (first-order with respect
            to the complex). major sets the effective forward first-order rate and thereby
            strongly influences the transient kinetics and equilibrium position.
        minor (float): Initial concentration of the less abundant reactant. In the analytic
            expression this initial minor concentration appears in the product of kf * major * minor
            which determines the equilibrium concentration term when t -> infinity.
            The units must be consistent with prod and major.
        backend (str): Optional string selecting the numerical or symbolic backend used to
            evaluate the analytic expression. Default is None, which selects the default backend
            (typically 'numpy' via get_backend). Common values include 'numpy' for numeric
            evaluation of arrays and 'sympy' for symbolic expressions; the choice controls which
            exponential and arithmetic routines are used and whether a symbolic expression may be
            returned. If backend is a string naming a supported backend, the function obtains
            the backend via get_backend(backend).
    
    Behavior, defaults, side effects, and failure modes:
        The function evaluates the closed-form solution
        (-kb * prod + kf * major * minor + (kb * prod - kf * major * minor) * exp(-t * (kb + kf * major)))
        / (kb + kf * major)
        using the selected backend's exp and arithmetic. Returned values therefore represent
        instantaneous product concentration at each entry of t and carry the same concentration
        units as prod, major, and minor.
        The function has no side effects (it does not modify inputs or global state) and is
        deterministic for given inputs. If backend is None or not specified, the function
        requests the default backend (typically 'numpy') from get_backend; supplying 'sympy'
        or another supported backend results in a symbolic expression when symbolic types are
        used for the inputs.
        Failure modes to consider:
        - Division by zero occurs if kb + kf * major == 0; callers must ensure the denominator
          is non-zero to avoid runtime or symbolic exceptions.
        - Input shapes must be compatible with the backend's broadcasting rules; when using
          numpy, t, kf, kb, prod, major, and minor should be scalars or arrays with broadcastable
          shapes. Incompatible shapes will raise backend-specific errors.
        - If a symbolic backend is selected but numeric inputs are supplied (or vice versa),
          evaluation errors may arise; ensure backend choice matches the desired numeric or
          symbolic behavior.
        - Units are not checked by this function; users must supply inputs in a consistent unit
          system (time, concentration, rate constants) to obtain physically meaningful results.
    
    Returns:
        numpy.ndarray or scalar or symbolic expression: The product (complex) concentration
        evaluated at each time point in t according to the reversible pseudo-first-order analytic
        solution. The return type follows the selected backend and input types: with the default
        numeric backend this is typically a numpy.ndarray (or scalar if t is scalar); with a
        symbolic backend a sympy expression may be returned. The numeric values have the same
        concentration units as the prod, major, and minor inputs and reflect the transient
        approach to the equilibrium value kf * major * minor / (kb + kf * major).
    """
    from chempy.kinetics.integrated import pseudo_rev
    return pseudo_rev(t, kf, kb, prod, major, minor, backend)


################################################################################
# Source: chempy.kinetics.integrated.unary_irrev_cstr
# File: chempy/kinetics/integrated.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def chempy_kinetics_integrated_unary_irrev_cstr(
    t: numpy.ndarray,
    k: float,
    r: float,
    p: float,
    fr: float,
    fp: float,
    fv: float,
    backend: str = None
):
    """chempy.kinetics.integrated.unary_irrev_cstr: Analytic solution for a first-order irreversible reaction A -> B in a continuously stirred tank reactor (CSTR).
    
    This function returns the time-dependent, closed-form concentrations for the reactant (A) and product (B) for the first-order irreversible reaction A -> B in a CSTR. It is intended for chemical kinetics modelling, analytical validation of ODE solver results, parameter estimation or integrated rate expression evaluation described in the ChemPy kinetics utilities. The implementation uses a small set of algebraic combinations and exponentials so that the result can be produced either numerically (default numpy backend) or symbolically (e.g., sympy) by supplying a compatible backend.
    
    Args:
        t (array_like): Time point or array of time points at which to evaluate the analytic solution. Typical use is a sequence of times returned by an ODE integrator or a grid for plotting. The backend exponential function is applied to t; the returned concentration arrays or expressions will have the same shape as t (or will broadcast accordingly). t = 0 yields the initial concentrations r and p.
        k (float_like): First-order rate constant for the irreversible reaction A -> B. Physically this has units of inverse time and competes with the dilution rate fv in determining dynamics and steady state. Must be numeric (or backend-compatible symbolic) and used directly in exponentials and the algebraic steady-state expressions.
        r (float_like): Initial concentration of the reactant A at time t = 0. Used as the starting condition for the analytic expression; the function returns r for the reactant component when t equals zero.
        p (float_like): Initial concentration of the product B at time t = 0. Used as the starting condition for the analytic expression; the function returns p for the product component when t equals zero.
        fr (float_like): Concentration of reactant A in the feed stream entering the CSTR. Together with fv this determines the feed-driven steady-state contribution to the reactant concentration (the steady state reactant concentration equals fr*fv/(fv + k) when exponential transients have decayed).
        fp (float_like): Concentration of product B in the feed stream entering the CSTR. Contributes to the long-time (steady-state) product concentration through dilution and reaction terms.
        fv (float_like): Feed rate divided by reactor volume (often called dilution rate); has units of inverse time. It appears additively with k (fv + k) and therefore fv + k must not be zero (see Failure modes). Physically, fv governs how quickly feed concentrations replace reactor contents and thus influences both transient decay rates and steady-state values.
        backend (module or str): Backend providing numeric/symbolic operations (notably exp). Default is 'numpy' (the function selects numpy by default). Alternatively pass a module like sympy or a string recognized by the library backend selector; when a symbolic backend is used the returned values are symbolic expressions rather than numeric arrays.
    
    Behavior, defaults and failure modes:
        The function computes closed-form expressions using algebraic combinations and exponentials. At t = 0 the returned concentrations equal the supplied initial conditions r and p. As t -> infinity the exponential terms vanish and the reactant concentration approaches fr*fv/(fv + k); the product concentration approaches the steady-state expression obtained by setting transient exponentials to zero in the returned formula. The default backend is numpy when backend is None or the string 'numpy'; in that case numeric numpy arrays are returned. If a symbolic backend such as sympy is supplied, the function returns symbolic expressions that use the backend's exp and arithmetic functions.
        Failure modes include division by zero if fv + k == 0 (this produces a ZeroDivisionError in numeric backends or an undefined expression in symbolic backends). Supplying a backend that does not implement the operations used (for example, an object without an exp function) will raise AttributeError/TypeError. Non-numeric or incompatible types for t, k, r, p, fr, fp, or fv that cannot be handled by the chosen backend will result in backend-specific errors. The function has no side effects.
    
    Returns:
        tuple: length-2 tuple (reactant_conc, product_conc). Each element is either a numeric array (when using a numeric backend such as numpy) or a backend-compatible symbolic expression (when using a symbolic backend). The returned elements have the same shape as t (or follow standard broadcasting rules for the chosen backend) and represent the time-dependent concentrations of reactant A and product B computed from the analytic CSTR solution.
    """
    from chempy.kinetics.integrated import unary_irrev_cstr
    return unary_irrev_cstr(t, k, r, p, fr, fp, fv, backend)


################################################################################
# Source: chempy.printing.numbers.roman
# File: chempy/printing/numbers.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def chempy_printing_numbers_roman(num: int):
    """chempy.printing.numbers.roman converts a positive Python integer to its canonical uppercase Roman numeral representation using common subtractive notation (e.g., 4 -> "IV", 9 -> "IX"). This utility is used by the chempy printing/formatting utilities when a human-readable Roman numeral label is required (for example, in generated text or simple labels in documentation and output related to chemical data).
    
    The implementation iterates over fixed Roman tokens and corresponding integer values (M, CM, D, CD, C, XC, L, XL, X, IX, V, IV, I) and greedily consumes the input integer to build the output string. The function is pure (no side effects), deterministic, and returns a new string.
    
    Args:
        num (int): The integer to convert to a Roman numeral. This must be a Python int and is interpreted as a non-negative whole number of units to represent in Roman numerals. In the context of ChemPy, typical uses are small positive integers used for numbering or labeling; the function accepts values >= 1 for conventional Roman numerals. Passing zero will yield an empty string because there is no Roman numeral for zero. Passing negative integers or non-int types is not supported and leads to undefined or erroneous results (see Failure modes).
    
    Returns:
        str: The Roman numeral representation of the input integer as an uppercase ASCII string using standard subtractive notation. For example, 4 -> "IV", 17 -> "XVII". For integers larger than typical Roman numeral conventions (e.g., > 3999), the function will produce repeated "M" characters for thousands (e.g., 4000 -> "MMMM"), as the implementation uses a simple greedy algorithm rather than an extended notation; this behavior is deliberate and consistent with the algorithm.
    
    Failure modes and notes:
        - The function expects a Python int. Passing non-int types (floats, numpy integers that are not Python int, strings, etc.) may raise TypeError when the implementation attempts sequence multiplication or integer division, or otherwise produce incorrect results.
        - The behavior for num <= 0 is undefined for conventional Roman numeral semantics; num == 0 returns an empty string, and negative integers can produce unexpected output due to integer floor-division semantics in the implementation. Callers should validate input and only pass positive integers when conventional Roman numerals are required.
        - There are no side effects; the function does not modify external state.
        - Performance is O(1) with respect to the number of token/value pairs (a fixed small constant), and O(k) with respect to the length k of the returned string (number of characters in the Roman numeral).
    """
    from chempy.printing.numbers import roman
    return roman(num)


################################################################################
# Source: chempy.symmetry.representations.print_header
# File: chempy/symmetry/representations.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def chempy_symmetry_representations_print_header(group: str):
    """Print the header line for a character table corresponding to a molecular point group.
    
    This function is used in the chempy.symmetry.representations module to produce the header row of a character table for a given point group in Schoenflies notation (used broadly in molecular symmetry, group theory in physical/inorganic chemistry, and representation theory). The header lists the symmetry operation classes in the order required by the character table and prefixes operation symbols with their multiplicity when the class contains more than one equivalent operation (for example, an entry with multiplicity 2 will be rendered as "2X" if the symbol for that class is "X"). The function obtains the operation symbols and multiplicities from the module-level lookup tables headers and column_coeffs using the lowercase form of the provided group key.
    
    Args:
        group (str): Point group name in Schoenflies notation (e.g., 'C2v'). The value is treated case-insensitively by converting it with str.lower() before lookup. This argument identifies which point group's character-table header to print; the exact lowercase key must exist in the representations module's internal lookup dictionaries.
    
    Returns:
        None: The function has no return value. As a side effect it prints a single header line to standard output (stdout) with the class labels separated by single spaces. Each printed token is either the operation symbol (from headers) or the symbol prefixed by its multiplicity (from column_coeffs) when that multiplicity is not 1.
    
    Behavior and side effects:
    - The function looks up symbols = headers[group.lower()] and numbers = column_coeffs[group.lower()] and constructs a list of header tokens by iterating the length of numbers; for each index i it uses symbols[i] and numbers[i] to form either "symbols[i]" (if numbers[i] == 1) or "numbers[i] + symbols[i]" (if numbers[i] != 1). It then calls print(*header) to emit a single space-separated line.
    - The printed header is intended to indicate the order and multiplicity of symmetry operation classes for the specified point group, which chemists and physicists use when reading or producing character tables for molecular symmetry analyses.
    
    Failure modes and exceptions:
    - KeyError is raised if group.lower() is not a key in the internal lookup dictionaries headers or column_coeffs (i.e., the point group is unknown to the module).
    - TypeError will occur if the provided group is not a string and therefore does not support the .lower() method.
    - IndexError or other exceptions may arise if the internal lookup tables are inconsistent (for example, if the symbols list and numbers list for a key have mismatched lengths). These indicate internal data problems rather than incorrect user input.
    
    Usage note:
    - This function prints directly to stdout and is intended for human-readable display of character-table headers; if programmatic access to the header tokens is required, callers should query the internal lookup tables directly or modify the module to expose a non-printing accessor.
    """
    from chempy.symmetry.representations import print_header
    return print_header(group)


################################################################################
# Source: chempy.symmetry.representations.print_mulliken
# File: chempy/symmetry/representations.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def chempy_symmetry_representations_print_mulliken(group: str):
    """Print Mulliken symbols of the irreducible representations for a molecular point group.
    
    This function is part of ChemPy's symmetry utilities and is used to display the Mulliken notation labels for irreducible representations associated with a point group given in Schoenflies notation (for example, "C2v"). Mulliken symbols are the standard labels used in molecular symmetry and group theory to identify irreducible representations; they are commonly used in spectroscopy, vibrational analysis, and quantum-chemical point-group classification to determine mode symmetries and selection rules. The function looks up a module-level mapping named "mulliken" keyed by lower-cased Schoenflies strings and prints the symbols in the order stored in that mapping.
    
    Args:
        group (str): Point group in Schoenflies notation (e.g., 'C2v'). This argument is case-insensitive: the function calls group.lower() before lookup. The value must be a string corresponding to a key in the module's internal mapping of Mulliken symbols.
    
    Returns:
        None: This function does not return a value. Side effect: it prints the Mulliken symbols to standard output (sys.stdout) separated by spaces and followed by a newline (the same behavior as Python's built-in print called with print(*sequence)). The printed order is the order of elements in the internal mapping for the requested group.
    
    Behavior and failure modes:
        The function performs a dictionary lookup using mulliken[group.lower()] and then prints the resulting sequence. If the provided group string does not match any key in the internal mapping, a KeyError will be raised. If a non-string object is supplied (an object without a lower() method), an AttributeError will be raised when the function attempts to call lower(). No additional validation or normalization is performed beyond lower-casing; callers must provide a valid Schoenflies point-group name present in the module's mapping.
    """
    from chempy.symmetry.representations import print_mulliken
    return print_mulliken(group)


################################################################################
# Source: chempy.symmetry.representations.print_table
# File: chempy/symmetry/representations.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def chempy_symmetry_representations_print_table(group: str):
    """chempy.symmetry.representations.print_table prints a formatted character table for a molecular point group given in Schoenflies notation.
    
    This function is used in the ChemPy symmetry utilities to display the character table (symmetry operation headers and Mulliken irreducible-representation labels with their characters) for a point group commonly encountered in molecular and inorganic chemistry. The output is intended for human inspection (e.g., in interactive sessions, documentation, or debugging) and helps chemists and computational chemistry users reason about orbital symmetries, selection rules, and spectroscopic transitions. The function looks up pre-defined module-level dictionaries (headers, mulliken, column_coeffs, row_coeffs, tables) to build the printed table.
    
    Args:
        group (str): Point group in Schoenflies notation (for example, "C2v"). This argument is case-insensitive: the function immediately lowercases the input internally. The string must match one of the keys present in the module-level dictionaries used by the function (headers, mulliken, column_coeffs, tables); otherwise a KeyError will be raised.
    
    Behavior and side effects:
        The function lowercases the provided group name and uses the module-level mappings:
        - headers[group] to obtain symmetry operation symbols for table column headers,
        - mulliken[group] to obtain Mulliken irreducible-representation labels for table rows,
        - column_coeffs[group] to obtain multiplicity coefficients for header columns, and
        - row_coeffs (if present for the group) to repeat Mulliken labels when imaginary or degenerate representations occur,
        - tables[group] to obtain the character values for each Mulliken row and column.
        It composes header entries by prefixing a numeric coefficient to a symbol when the coefficient is not 1 (e.g., "2C2" if the coefficient for a C2 operation is 2). If row_coeffs contains an entry for the group, Mulliken labels will be duplicated according to those coefficients; otherwise the Mulliken list is used as-is. For the special case when group == "c1" the function appends a single row in a distinct format ([rows, 1]) as implemented in the source.
        The final table is formatted and printed to standard output using tabulate(..., tablefmt='rounded_grid'), so the primary observable effect is printing the formatted table to stdout. Nothing is returned.
    
    Failure modes and notes:
        - If any required module-level mapping (headers, mulliken, column_coeffs, tables) does not contain the requested group key, a KeyError will be raised.
        - If the tabulate function or any of the expected module-level variables are missing or malformed, a NameError or other exception may be raised.
        - The function does not validate the semantic correctness of the point-group name beyond dictionary membership; it relies on the predefined data in the CheMPy symmetry module.
        - This routine is intended for display only and is not meant to return machine-readable table objects.
    
    Returns:
        None: The function does not return a value. Its observable effect is to print a human-readable character table for the specified point group to standard output using a rounded-grid table format.
    """
    from chempy.symmetry.representations import print_table
    return print_table(group)


################################################################################
# Source: chempy.symmetry.salcs.calc_salcs_func
# File: chempy/symmetry/salcs.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def chempy_symmetry_salcs_calc_salcs_func(
    ligands: list,
    group: str,
    symbols: list,
    mode: str = "vector",
    to_dict: bool = False
):
    """chempy.symmetry.salcs.calc_salcs_func calculates symmetry-adapted linear combinations (SALCs) for a set of ligand atomic orbitals using the symmetry functions from a point group's character table.
    
    This function is used in molecular and ligand-field symmetry analysis (as provided by the ChemPy package) to convert geometric information about ligands or outer atoms into symbolic linear combinations of atomic-orbital contributions that transform according to the symmetry functions of a specified point group. The computed SALCs are useful when constructing molecular orbitals, assigning orbital symmetry labels, and analyzing how ligand positions couple to central-atom orbitals in group-theoretical treatments of molecules and crystals. Internally the function optionally converts angle pairs to Cartesian vectors, evaluates each symmetry function from an internal character-table lookup, normalizes the resulting weights, and maps numeric weights onto the supplied SymPy symbols to produce symbolic expressions.
    
    Args:
        ligands (list): Positions of ligand atomic orbitals. If mode is 'vector' this must be a list (or nested list) of Cartesian coordinates [x, y, z] for each ligand in the same units (unitless directional coordinates are expected). If mode is 'angle' this must be a list of pairs [azimuthal, polar] giving spherical-coordinate angles in degrees where azimuthal is the angle measured from the positive x-axis in the xy-plane and polar is the angle measured from the positive z-axis. These positions determine the direction cosines used by the internal symmetry functions to compute each ligand's contribution to a SALC.
        group (str): Point group in Schoenflies notation (for example 'C2v' or 'D3h'); lookup is case-insensitive. The string is used to index the package's internal character-table/symmetry-function dictionary (symmetry_func_dict). If the provided group is not present in that dictionary a KeyError will be raised by the lookup.
        symbols (list): Sequence of SymPy symbols (e.g., returned by sympy.symbols) representing the outer ligands or atomic orbitals that will be combined. Each symbol corresponds positionally to one ligand entry in ligands; the function maps computed numeric weights for each ligand onto these symbols to produce the final symbolic SALCs. A mismatch between the number of symbols and the number of ligand positions may lead to an error in the final mapping routine.
        mode (str): Either 'vector' or 'angle' (default: 'vector'). When 'vector' the entries in ligands are treated as Cartesian [x, y, z] coordinates. When 'angle' the entries are treated as spherical-coordinate angle pairs [azimuthal, polar] in degrees and are converted to Cartesian direction vectors before symmetry-function evaluation. An invalid value for mode causes an Exception with message "Invalid mode input: must be 'angle' or 'vector'".
        to_dict (bool): If False (default) the function returns a list of SALCs in the order of the internal symmetry functions for the requested point group. If True the function (via the @return_dict decorator applied in the module) returns a dictionary form instead; the dictionary maps the internal symmetry-function identifiers (the keys used by the character-table lookup) to the corresponding SALC expression(s). This flag changes only the output container format and does not alter the computed SALC expressions themselves.
    
    Returns:
        list or dict: By default (to_dict is False) a list of SymPy expressions and numeric placeholders representing the weight and sign of each atomic-orbital contribution to each SALC. The list preserves the order of symmetry functions as fetched from the internal character table: entries can be integers 0 or 1 for trivial functions, SymPy expressions for evaluated SALCs, or nested lists when multiple SALCs correspond to the same irreducible representation. If to_dict is True the return value is a dict (as described above) whose values are the same kinds of SymPy expressions or nested lists. Note that multiple symmetry functions belonging to the same irreducible representation can produce identical or redundant SALCs; normalization is applied before mapping weights to symbols so SALCs that cancel to zero will be returned as 0.
    
    Behavior, side effects, defaults, and failure modes:
        The function is pure computational and produces no I/O side effects. Default behavior is to treat ligand positions as Cartesian vectors (mode='vector') and to return results as a list (to_dict=False). Processing steps: if mode == 'angle' angles are converted to vectors via _angles_to_vectors; symmetry functions for the requested group are retrieved from symmetry_func_dict; each symmetry function is evaluated using _eval_sym_func (with symmetry functions equal to 0 or 1 handled as direct 0/1 placeholders); the set of raw SALC weights is normalized via _normalize_salcs; and numeric weights are converted to symbolic linear combinations with the provided symbols via _weights_to_symbols. The function will raise Exception("Invalid mode input: must be 'angle' or 'vector'") for an invalid mode, and will raise a KeyError if group is not found in the internal symmetry function dictionary. Other errors can arise from a mismatch between the number of provided symbols and ligand positions or from unexpected values produced by the internal helper functions; these will propagate the underlying exception types. Normalization can produce zero SALCs when contributions cancel.
    """
    from chempy.symmetry.salcs import calc_salcs_func
    return calc_salcs_func(ligands, group, symbols, mode, to_dict)


################################################################################
# Source: chempy.symmetry.salcs.calc_salcs_projection
# File: chempy/symmetry/salcs.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def chempy_symmetry_salcs_calc_salcs_projection(
    projection: list,
    group: str,
    to_dict: bool = False
):
    """Return SALCs (symmetry-adapted linear combinations) using the projection operator method for a specified molecular point group.
    
    This function implements the standard projection-operator approach used in molecular symmetry analysis (as in the ChemPy README examples and symmetry utilities). Given the results of applying each symmetry operation to a set of atomic or ligand orbitals (represented by SymPy symbols), calc_salcs_projection constructs the SALCs for every irreducible representation of the specified point group. The practical use is to transform a set of orbital basis functions (one symbol per ligand/outer atom) into linear combinations that transform according to irreducible representations, which is a common step when building molecular orbital models, assigning basis functions in ligand field theory, or setting up symmetry-adapted basis sets for chemical kinetics or equilibrium calculations.
    
    Behavior summary:
    - Expects projection to be the sequence of orbital images produced by applying the group's symmetry operations to a single orbital (one result per group operation), where each element is a SymPy symbol representing a ligand/outer-atom orbital.
    - Uses the internal character/table data for the point group (Schoenflies notation, case-insensitive) to form projection-operator sums for each irreducible representation, multiplies those characters by the provided projection entries, and sums to produce each SALC.
    - Calls an internal normalization routine (_normalize_salcs_expr) before returning results so the returned SALC expressions have been put into the expected simplified/normalized form used by the rest of the symmetry module.
    - If to_dict is True the function returns a mapping from irreducible-representation labels (strings like 'A1', 'E', etc., drawn from the group's table) to the corresponding SALC expression; otherwise it returns the SALCs in the irreducible-representation iteration order used by the group's internal table.
    - Note: for certain high-symmetry point groups mentioned in the original domain documentation, the projection operator method can produce only one SALC for some representations (for example E and T point groups the canonical projection yields a single SALC entry); this is preserved by the implementation.
    
    Args:
        projection (List, tuple, or array of SymPy symbols): Results of projection operations for the symmetry operations of the point group. Each item is a SymPy symbol that names the orbital or ligand coordinate being tracked; the sequence must be ordered to match the symmetry operations used internally for the specified group. In chemical practice this represents tracking a single orbital located on a ligand or outer atom through each symmetry operation so that the projection operator can form SALCs.
        group (str): Point group in Schoenflies notation (for example 'C2v', 'c3v'). This argument is case-insensitive and selects the character table and irreducible-representation ordering used to combine the projection entries into SALCs. The value must correspond to a group known to the module's internal tables.
        to_dict (bool = False): If False (default), the function returns the SALCs as a list (or nested list) in the order of irreducible representations for the chosen group; if True, the function returns a dictionary mapping irreducible-representation labels (strings) to the corresponding SALC expression. This flag controls output format only; it does not change the computed SALC content.
    
    Returns:
        List or nested list of strings of the SALCs for each irreducible representation: Each entry corresponds to the symmetry-adapted linear combination computed for the matching irreducible representation in the group's internal table ordering. For irreducible representations that do not admit a SALC from the provided projection vector the entry is 0. If to_dict=True the function returns a dictionary mapping irreducible-representation labels to these SALC expressions instead of a list.
    
    Failure modes and side effects:
    - Raises a KeyError (or similar lookup error) if group (case-insensitive) is not present in the module's internal tables for point groups.
    - A mismatch between the length/ordering of projection and the group's expected symmetry-operation ordering can produce incorrect SALCs or raise broadcasting/shape errors during internal array multiplication; ensure projection is constructed by tracking the same orbital across the group's symmetry operations in the module's convention.
    - The function does not modify its inputs; its only observable effect is the return of the computed SALC expressions (or dictionary when to_dict=True).
    - The function relies on internal helper routines and normalization; errors raised by those routines (for example from invalid SymPy symbols or unsupported internal table entries) will propagate to the caller.
    
    Practical significance:
    - Use this function when you have labeled ligand or outer-atom orbitals as SymPy symbols and the result of applying each symmetry operation to one orbital (the projection sequence), and you need the symmetry-adapted linear combinations (SALCs) for assigning orbitals to irreducible representations, reducing Hamiltonians by symmetry, or constructing symmetry-adapted basis sets in computational chemistry workflows.
    """
    from chempy.symmetry.salcs import calc_salcs_projection
    return calc_salcs_projection(projection, group, to_dict)


################################################################################
# Source: chempy.units.concatenate
# File: chempy/units.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def chempy_units_concatenate(arrays: list, **kwargs):
    """chempy.units.concatenate concatenates multiple arrays that carry a unit (quantities)
    while preserving and returning a single array with the unit of the first input.
    
    This function is a patched version of numpy.concatenate adapted for the chempy.units
    module (which wraps the quantities package). It is intended for chemical-domain
    workflows where arrays represent measured or computed physical quantities
    (e.g., time series in seconds, concentration arrays in molar) and it makes it
    convenient and safe to join such arrays without losing or mismatching units.
    Behavior: the unit of arrays[0] is taken as the target unit; every element in
    arrays is converted to that unit using chempy.units.to_unitless before the
    numeric concatenation is performed with numpy.concatenate. The numeric result
    is then re-attached to the chosen unit and returned. The original input objects
    are not modified.
    
    Args:
        arrays (list): A list of array-like objects to concatenate. Each element is
            expected to be an array (for example a numpy.ndarray or a quantities.Quantity
            array provided by the chempy.units-backed package) that carries a unit or
            is convertible to the unit of the first element. In the chemical domain,
            typical uses include concatenating time arrays (e.g., seconds) or concentration
            vectors (e.g., molar). The unit of the first element (arrays[0]) determines
            the unit of the returned array and is used as the target for conversion of
            all other elements prior to concatenation.
        kwargs (dict): Additional keyword arguments forwarded directly to
            numpy.concatenate (for example, axis). These control how the underlying
            numeric concatenation is performed and follow the same semantics and
            defaults as numpy.concatenate.
    
    Returns:
        quantities.Quantity: A numeric array (numpy.ndarray-like) whose underlying
        numeric contents equal numpy.concatenate([to_unitless(arr, unit_of(arrays[0])) for arr in arrays], **kwargs),
        and whose unit is the unit of arrays[0]. In practice this is a quantities
        object (an array with an attached unit) suitable for further chempy operations
        that expect unit-carrying arrays.
    
    Raises:
        IndexError: If arrays is empty, accessing arrays[0] to determine the unit will fail.
        TypeError or ValueError: If an element in arrays cannot be converted to the unit
            of arrays[0] using chempy.units.to_unitless, or if an element is not an
            array-like object acceptable to to_unitless/numpy.concatenate.
        ValueError: If the numeric shapes or axes of the provided arrays are incompatible
            for concatenation, the underlying numpy.concatenate will raise a ValueError.
    
    Notes:
        - The unit selection strategy (use unit of the first element) is deliberate:
          it avoids ambiguity when concatenating heterogeneous unit-carrying arrays and
          mirrors typical chemical workflows where time or concentration units are fixed.
        - No in-place modification of the input arrays occurs; the function returns a
          new array with the chosen unit attached.
        - Because this function delegates numeric concatenation to numpy.concatenate,
          it inherits numpy's performance characteristics, broadcasting rules for the
          concatenated axis, and limitations on memory usage.
    """
    from chempy.units import concatenate
    return concatenate(arrays, **kwargs)


################################################################################
# Source: chempy.units.format_string
# File: chempy/units.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def chempy_units_format_string(value: float, precision: str = "%.5g", tex: bool = False):
    """Format a scalar with an associated unit into two strings: a formatted numeric value and a unit representation suitable for display or LaTeX rendering.
    
    Args:
        value (float): A scalar "float with unit" as used in chempy.units (typically a quantities.Quantity-like object whose numeric part is available as value.magnitude and whose unit/dimensionality is available as value.dimensionality). In the ChemPy domain this function is used when printing concentrations, rates, equilibrium constants, and other quantified chemical properties so that the numeric magnitude and the unit label can be shown separately (for example in Reaction/Equilibrium printing or examples in the README). The function will extract the numeric magnitude via float(value.magnitude) and the unit representation via value.dimensionality (or via latex_of_unit when tex is True). If the supplied object does not provide the expected attributes (magnitude or dimensionality) an AttributeError will be raised.
        precision (str): A printf-style format string used to format the numeric magnitude. Default is "%.5g". This string is applied as precision % float(value.magnitude). The caller is responsible for providing a format compatible with converting the magnitude to float; incompatible format strings or non-numeric magnitudes may raise TypeError or ValueError.
        tex (bool): If True, return a LaTeX-compatible unit representation produced by latex_of_unit(value). The LaTeX string does not include surrounding dollar signs and is suitable for embedding inside larger LaTeX expressions (for example " \\mathrm{\\frac{1}{s}}" as shown in README examples). If False, the unit string is taken from value.dimensionality using quantities.markup.config to decide between a unicode representation or a plain ASCII string (config.use_unicode controls this). Default is False.
    
    Returns:
        tuple(str, str): A 2-tuple where the first element is the formatted numeric value (string) produced by applying the precision format to float(value.magnitude) and the second element is the unit string. The unit string is either a LaTeX fragment (when tex is True) or a unicode/plain string derived from value.dimensionality (when tex is False). No additional side effects occur beyond importing quantities.markup.config when tex is False and calling latex_of_unit when tex is True.
    
    Behavior, defaults, and failure modes:
        The function does not add currency or math delimiters around LaTeX output; callers must add surrounding $ if desired. When tex is False, the output unit depends on quantities.markup.config.use_unicode: if True a unicode unit representation is returned, otherwise an ASCII string is returned. This function performs no localization of numeric formatting beyond the provided printf-style precision. If value lacks .magnitude or .dimensionality attributes an AttributeError will be raised. If latex_of_unit cannot produce a LaTeX representation for the unit, it may raise an exception from that helper; such exceptions are propagated to the caller. If the precision string is invalid for the numeric magnitude, a TypeError or ValueError may be raised. The function performs no mutation of global state.
    """
    from chempy.units import format_string
    return format_string(value, precision, tex)


################################################################################
# Source: chempy.units.get_derived_unit
# File: chempy/units.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def chempy_units_get_derived_unit(registry: dict, key: str):
    """Get the unit for a derived physical quantity from a provided unit registry.
    
    This function is part of chempy.units, the small units layer used in ChemPy to check and manipulate units for chemical calculations (kinetics, equilibria, concentration/density handling, radiolytic dose calculations, etc.). It accepts a registry that maps base physical dimensions to unit objects (for example, mapping 'length' to a meter unit, 'mass' to a kilogram unit, and so on) and returns a unit expression for a requested derived quantity (for example, diffusivity -> length**2/time, concentration -> amount/length**3). If registry is None the function returns 1.0 to indicate a unitless context (useful when units are intentionally disabled or unavailable).
    
    Args:
        registry (dict): Mapping of base dimension names to unit objects used by ChemPy. The registry is expected to contain keys for the base dimensions used by ChemPy: 'length', 'mass', 'time', 'current', 'temperature', 'luminous_intensity', and 'amount'. The values are unit objects provided by the underlying units implementation wrapped by chempy.units (for example, objects from the quantities package or a compatible unit representation). If registry is None the function treats the environment as unitless and returns 1.0. The registry is not modified by this function (no side effects).
        key (str): The requested quantity name. This may be either one of the base-dimension keys described above (in which case the function returns registry[key]) or one of the supported derived-quantity names computed from the base units. Supported derived keys computed by this function include: "diffusivity" (diffusion coefficient, length**2/time), "electrical_mobility" (electrical mobility, current*time**2/mass), "permittivity" (electric permittivity, current**2 * time**4 / (length**3 * mass)), "charge" (electric charge, current*time), "energy" (energy, mass * length**2 / time**2), "concentration" (amount per volume, amount / length**3), and "density" (mass per volume, mass / length**3). Additional supported derived aliases and quantities provided by the implementation are "diffusion" (deprecated alias for "diffusivity"), "radiolytic_yield" (amount per energy), "doserate" (energy per mass per time), and "linear_energy_transfer" (energy per length). If the provided key matches a derived quantity the corresponding derived unit expression is returned; otherwise the function falls back to returning registry[key]. If the key is not present in either the derived mapping or the registry a KeyError will be raised by the registry lookup (this is the function's failure mode for unknown keys).
    
    Returns:
        unit or float: The unit object representing the requested quantity, constructed from the units in registry. When a derived quantity name is given, the returned unit is the composite unit computed from the registry base units (for example, diffusivity -> registry['length']**2/registry['time']). When a base-dimension key is given, the function returns registry[key] directly. If registry is None the function returns 1.0 to indicate a unitless value. On failure to find the key in the registry (and it is not a known derived key) a KeyError is raised.
    """
    from chempy.units import get_derived_unit
    return get_derived_unit(registry, key)


################################################################################
# Source: chempy.units.linspace
# File: chempy/units.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def chempy_units_linspace(start: float, stop: float, num: int = 50):
    """chempy.units.linspace generates an array of evenly spaced numeric values between two scalar endpoints while preserving and returning the unit associated with the start value. It is analogous to numpy.linspace but integrated with ChemPy's units subsystem (which wraps the quantities package) so the returned array carries the same physical unit as the start argument. This function is useful in chemical modelling workflows (for example creating time grids, temperature ramps, concentration sequences, or other evenly spaced parameter sweeps where units must be tracked consistently).
    
    Args:
        start (float): The left endpoint of the interval. In practice this is the numeric value or a quantities-aware scalar carrying a physical unit (for example a concentration, temperature or time). The function queries the unit of start (via unit_of) and uses that unit for the returned array. If start is a plain Python float it is treated as dimensionless.
        stop (float): The right endpoint of the interval. This is converted to the same unit as start (via to_unitless with the unit derived from start) before generating the sequence. If stop has an incompatible unit (for example meters vs seconds) the underlying unit conversion routine will raise an error (see failure modes below).
        num (int): Number of samples to generate. This integer controls the length of the returned sequence and defaults to 50. The value is passed directly to numpy.linspace, so its detailed semantics (including behavior for zero or negative values) follow numpy.linspace.
    
    Returns:
        numpy.ndarray: A one-dimensional NumPy array of length num whose entries are the evenly spaced values between start and stop, with the unit of start attached. In practice the return is a quantities-aware array (NumPy array multiplied by the unit object returned by unit_of(start)), so the results can be used directly in ChemPy workflows that expect unit-bearing quantities (for plotting, ODE integration time grids, concentration sweeps, etc.).
    
    Behavior, side effects, defaults, and failure modes:
        The function first determines the unit associated with start using unit_of(start). Both start and stop are converted to unitless numeric values with respect to this unit using to_unitless, and numpy.linspace is called on those unitless numbers. The numeric linspace output is multiplied by the unit and returned. There are no external side effects; the function is pure with respect to global state. The default for num is 50. If stop cannot be converted to the unit of start, the conversion function (from the quantities wrapper) will raise an exception (for example a ValueError indicating incompatible units), and that exception is propagated. Floating-point rounding and finite precision follow NumPy semantics; equality checks should allow for floating-point tolerances (the original example checks closeness within 1e-15). Invalid values for num or other arguments will produce the same errors as numpy.linspace or the underlying unit conversion routines.
    """
    from chempy.units import linspace
    return linspace(start, stop, num)


################################################################################
# Source: chempy.units.logspace_from_lin
# File: chempy/units.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def chempy_units_logspace_from_lin(start: float, stop: float, num: int = 50):
    """Logarithmically spaced data points with units preserved.
    
    This function is part of chempy.units, which wraps the quantities package for
    unit-aware numerical work in chemistry (e.g. concentrations, amounts, rates).
    logspace_from_lin produces an array of values that are evenly spaced in
    base-2 logarithmic space between two endpoint values. The unit of the
    returned array is taken from start; stop is converted to that unit before
    computing the logarithmic spacing. This is useful when sampling a physical
    quantity (for example, concentration, activity, or pressure) across orders
    of magnitude while keeping the values' units consistent with the rest of a
    chempy workflow.
    
    Args:
        start (float): Left endpoint of the interval expressed in a unit-aware
            numeric value. The unit of start determines the unit of the returned
            array. In the context of chempy.units this is typically a quantities
            object or a numeric value associated with a unit; unit_of(start) is
            used internally to extract the unit. The returned values are spaced
            so that log2(value) is linearly spaced between log2(start) and
            log2(stop_in_start_unit). start must be finite.
        stop (float): Right endpoint of the interval expressed in a unit-aware
            numeric value. stop is converted to the unit of start using
            chempy.units.to_unitless(start_unit) semantics (i.e., the function
            computes log2(to_unitless(stop, unit_of(start)))). If stop has
            incompatible units such that conversion to the unit of start fails,
            a conversion error (for example ValueError) from the underlying units
            handling code will be propagated. stop must be finite after
            conversion.
        num (int): Number of samples to generate. Defaults to 50. num is passed
            directly to numpy.linspace to create num evenly spaced points in the
            exponent (log2) domain and therefore must be a positive integer
            accepted by numpy.linspace (typical use is an integer >= 1).
    
    Returns:
        numpy.ndarray: One-dimensional array containing num values with the same
        unit as start. The numeric values are generated by taking num points
        linearly spaced between log2(start) and log2(stop_converted), applying
        2**x (exp2) to those points, and multiplying by the unit of start. The
        returned array is suitable for unit-aware numerical computations in
        chempy (for example, creating logarithmically spaced concentration
        vectors). Exceptions raised during unit extraction or conversion (e.g.
        due to incompatible units) are not caught and will propagate to the
        caller.
    """
    from chempy.units import logspace_from_lin
    return logspace_from_lin(start, stop, num)


################################################################################
# Source: chempy.units.tile
# File: chempy/units.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def chempy_units_tile(array: numpy.ndarray, *args, **kwargs):
    """chempy.units.tile: Tile a numpy.ndarray while preserving and propagating physical units.
    
    This function is a patched replacement for numpy.tile that is aware of the units used in chempy.units (the units/wrapping provided around the quantities package referenced in the README). It is used in chemical computations where arrays of values carry an associated physical unit (for example concentration arrays, rate constants, or other property arrays) and a repeated/tiled copy of the array is required while keeping the original unit attached. The function determines the unit from the first element of the input array, converts the entire array to unitless numbers with respect to that unit, performs numpy.tile using the supplied positional and keyword arguments, and finally re-attaches the inferred unit to the tiled numeric result.
    
    Args:
        array (numpy.ndarray): Input array to be tiled. In the context of chempy.units this is expected to contain elements that carry a physical unit (as handled by chempy.units utilities such as unit_of and to_unitless) or plain numeric values. The unit used for the entire operation is inferred from the first element of this array (array[0, ...] is attempted first, with a fallback to array[0] if indexing with slicing raises TypeError). Practical significance: this allows callers to tile arrays of concentrations, property values, or other chemistry-relevant quantities without manually stripping and re-applying units.
        args (tuple): Positional arguments forwarded directly to numpy.tile. These arguments control the tiling pattern (for example the repeats argument or sequence of repeats) and thus determine how the input array is repeated to form the output. The semantics and valid values for these arguments are those of numpy.tile; this function does not reinterpret them but ensures they are applied to the unitless numeric data before reattaching units.
        kwargs (dict): Keyword arguments forwarded directly to numpy.tile. These are passed unchanged to numpy.tile and follow numpy.tile's documented behavior. They exist to allow the same flexibility as numpy.tile when controlling how the tiling is performed.
    
    Returns:
        numpy.ndarray: A new numpy.ndarray containing the tiled numeric values with the same physical unit as inferred from the first element of the input array. The returned array is unit-aware in the same manner as inputs handled by chempy.units: numeric values are the result of np.tile applied to the unitless representation of the input array, and the physical unit (unit_of(array[0])) is multiplied back onto the result. Practical significance: callers receive a tiled array they can use directly in downstream chemistry calculations with units preserved.
    
    Notes and behavior details:
        - The function infers the physical unit from the first element of the input array. If elements in the array use different or incompatible units, conversion via to_unitless may fail or produce incorrect results; callers should ensure consistent units in the input array before calling tile.
        - The implementation converts the full input array to unitless values with respect to the inferred unit using to_unitless(array, unit), calls numpy.tile on those unitless values, and multiplies the tiled numeric result by the same unit. This sequence guarantees that numeric tiling semantics are identical to numpy.tile while preserving units at the end.
        - The function does not mutate the input array; it returns a new array.
        - All errors raised by unit_of, to_unitless, or numpy.tile are propagated to the caller. Common failure modes include IndexError if the input array is empty (no element from which to infer a unit), unit conversion errors if elements are incompatible with the inferred unit, or numpy.tile errors if the provided args/kwargs are invalid.
        - This function is intended for use in chemical computations (e.g., repeating concentration vectors, property arrays, or rate arrays) where preserving the physical unit through array operations reduces the risk of unit-inconsistency bugs.
    """
    from chempy.units import tile
    return tile(array, *args, **kwargs)


################################################################################
# Source: chempy.units.uniform
# File: chempy/units.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def chempy_units_uniform(container: dict):
    """Convert a container of quantities with mixed units into a container whose values all share the same unit.
    
    This function is part of chempy.units, the units-handling utilities used throughout ChemPy (for example when preparing concentration vectors for kinetics solvers, comparing equilibrium constants, or checking unit consistency for reactions). uniform chooses the unit of the first element in the provided container and converts every element to that unit. The conversion uses the local helpers unit_of(...) to determine the unit of an item and to_unitless(..., unit) to extract a unitless magnitude relative to that unit before reattaching the chosen unit. The function intentionally preserves the mapping/sequence type for dict inputs (it constructs and returns an instance of the original dict-like class) and does not modify the input in place.
    
    Args:
        container (tuple, list or dict): A sequence or mapping whose values are quantities (objects with attached units, typically provided by the quantities package wrapped by chempy.units) or numeric values associated with units. When container is a tuple or list, the function inspects container[0] to determine the target unit and converts the entire sequence to that unit. When container is a dict, the function inspects the first value returned by list(container.values())[0] to determine the target unit and converts every value in the mapping to that unit. The element chosen to determine the unit is therefore significant: all returned values will be expressed in that element's unit. The function accepts the concrete container types listed above; other input types are returned unchanged.
    
    Returns:
        dict, numpy.ndarray or original type: If container is a dict (or a dict subclass), returns a new instance of the same mapping class constructed from (key, converted_value) pairs where each converted_value is the original value expressed in the unit of the first value. The example in the original code shows dict(a=3*km, b=200*m) returning {'b': array(200.0) * m, 'a': array(3000.0) * m}. If container is a tuple or list, returns the converted sequence as an array-like quantity result of to_unitless(container, unit) * unit (i.e., numeric magnitudes in the chosen unit multiplied by that unit). If container is neither a tuple/list nor a dict, the original object is returned unchanged.
    
    Behavior, side effects, defaults and failure modes:
        - The chosen target unit is always the unit of the first element inspected: container[0] for sequences, list(container.values())[0] for mappings. Users must therefore ensure the first element carries the intended unit.
        - The function does not mutate the input container; it constructs and returns a new object (a new mapping instance for dict inputs; a new array-like quantity for list/tuple inputs).
        - If the container is empty, attempting to access the first element will raise an IndexError (for tuple/list) or will raise an IndexError when list(container.values())[0] is attempted (for dict). Callers should handle or avoid empty containers.
        - If elements cannot be converted to the chosen unit (for example, because they have incompatible dimensions or because values are not valid quantity objects), the underlying unit conversion helpers (unit_of or to_unitless) will raise an error (ValueError or an error from the underlying quantities implementation). These errors propagate to the caller.
        - The function relies on the chempy.units wrappers (unit_of and to_unitless) and therefore inherits their semantics and limitations (for example, how dimensionless quantities are treated, or how array-like inputs are handled).
        - The function preserves the mapping class for dict-like inputs by calling container.__class__(...), so subclasses of dict will be reconstructed using their constructor with the converted (key, value) pairs.
    """
    from chempy.units import uniform
    return uniform(container)


################################################################################
# Source: chempy.units.unit_registry_from_human_readable
# File: chempy/units.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def chempy_units_unit_registry_from_human_readable(unit_registry: dict):
    """chempy.units.unit_registry_from_human_readable: Deserialize a human-readable unit registry into the internal unit-quantity mapping used by chempy.units. This function is used by ChemPy's units subsystem (which wraps the quantities package) to reconstruct runtime unit objects from a serialized representation suitable for human inspection or storage (for example JSON-like dicts produced when saving configuration or exchanging unit settings). The resulting registry maps SI base keys (the keys iterated from SI_base_registry) to unit quantities used for unit consistency checks and conversions in chemical computations (kinetics, equilibria, properties) as described in the project README.
    
    Args:
        unit_registry (dict): A serialized, human-readable registry. The dict is expected to contain an entry for every key present in SI_base_registry. For each key k the value must be a two-item sequence (factor, u_symbol) where:
            factor is a numeric multiplicative factor (applied to the unit quantity) and
            u_symbol is either the integer 1 to indicate a dimensionless unit, or a unit symbol/string accepted by pq.Quantity (the quantities package) that can be used as the second argument to pq.Quantity(0, u_symbol).
        The practical role of this argument is to provide a compact, persistable description of units that this function converts into the actual quantity objects used internally by chempy.units. If unit_registry is None the function returns None (convention: no registry to deserialize).
    
    Returns:
        dict or None: If unit_registry is None the function returns None (no side effects). Otherwise returns a new dict mapping each SI base key (the same keys iterated from SI_base_registry) to a unit quantity value computed as factor * base_unit_quantity where base_unit_quantity is the single dimensionality key returned by pq.Quantity(0, u_symbol).dimensionality. These returned values are the runtime unit objects used by chempy for unit consistency and conversion (e.g., in Reaction parsing and to_unitless checks described in the README).
    
    Behavior, side effects, defaults, and failure modes:
        The function iterates over the predefined SI_base_registry and for each key looks up unit_registry[k]. A missing key in unit_registry will raise a KeyError. For each entry, if u_symbol == 1 the function treats the entry as dimensionless and uses a unit quantity placeholder corresponding to 1. Otherwise it calls pq.Quantity(0, u_symbol).dimensionality and expects exactly one dimensionality key; that single key is taken as the base unit quantity and multiplied by factor to form the stored value. If pq.Quantity cannot parse u_symbol the underlying quantities package will raise its own exception (for example ValueError); this will propagate unless caught by the caller. If pq.Quantity(0, u_symbol).dimensionality yields zero or multiple keys (i.e., the unit symbol does not correspond to a single base unit), the function raises TypeError("Unknown UnitQuantity: {}".format(unit_registry[k])) to signal an unrecognized or ambiguous unit specification. The function has no other side effects beyond allocating and returning the new dictionary.
    """
    from chempy.units import unit_registry_from_human_readable
    return unit_registry_from_human_readable(unit_registry)


################################################################################
# Source: chempy.units.unit_registry_to_human_readable
# File: chempy/units.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def chempy_units_unit_registry_to_human_readable(unit_registry: dict):
    """chempy.units.unit_registry_to_human_readable converts an internal unit registry used by ChemPy's units subsystem into a human-readable, serializable mapping suitable for JSON-style storage or display. It is used when persisting or inspecting the set of base unit definitions (the SI base keys listed in the module-level SI_base_registry) so that numeric multipliers and unit symbols can be represented as simple Python primitives.
    
    Args:
        unit_registry (dict): A mapping representing the unit registry used by ChemPy's units module. In the domain of ChemPy this registry is expected to contain entries for each key in the module-level SI_base_registry; each value is either the integer 1 (the literal integer object used here to denote a pure dimensionless base entry) or an object from the underlying units/quantities package with a .dimensionality attribute. The .dimensionality attribute is iterated to produce a list of unit descriptor objects; each unit descriptor must expose a u_symbol attribute containing the unit symbol string. This function does not accept inputs other than a dict or None, and it will attempt to read unit_registry[k] for every k in SI_base_registry, so missing keys will raise a KeyError.
    
    Returns:
        dict or None: If unit_registry is None, returns None (no side effects). Otherwise returns a new dict mapping the same SI base keys (the keys iterated from SI_base_registry) to two-element tuples suitable for human-readable serialization: for entries equal to the literal integer 1 the tuple is (1, 1) which denotes a dimensionless base entry; for other entries the tuple is (multiplier, u_symbol) where multiplier is float(unit_registry[k]) and u_symbol is the unit symbol (the u_symbol attribute of the single element in the dimensionality list). The returned dict contains only Python built-in types (floats and strings or the integers 1), making it safe to encode as JSON or present to users.
    
    Behavior, defaults, and failure modes:
        This function is pure and has no side effects on the provided unit_registry; it constructs and returns a new dict. If unit_registry is None the function short-circuits and returns None. For each key k in SI_base_registry the function reads unit_registry[k]; if that lookup fails a KeyError is raised. If unit_registry[k] is the literal integer 1 (compared by identity in the source), the output for that key is the tuple (1, 1) meaning a dimensionless base entry. Otherwise the function obtains list(unit_registry[k].dimensionality) and requires that the resulting list has exactly one element; if len(dim_list) != 1 a TypeError("Compound units not allowed: ...") is raised because compound (multi-dimensional) units are not supported by this serialization routine. If the dimensionality list contains one element, the element's u_symbol attribute is used as the human-readable unit symbol and float(unit_registry[k]) is used as the numeric multiplier; failures to call float(...) or to access .dimensionality/.u_symbol will raise the underlying TypeError or ValueError propagated from those operations. This function does not attempt to validate or normalize unit symbols beyond reading u_symbol, nor does it introduce or accept unit representations other than those described above.
    """
    from chempy.units import unit_registry_to_human_readable
    return unit_registry_to_human_readable(unit_registry)


################################################################################
# Source: chempy.util._aqueous.ions_from_formula
# File: chempy/util/_aqueous.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def chempy_util__aqueous_ions_from_formula(formula: str):
    """ions_from_formula
    Parse a chemical formula string to (eventually) produce the constituent ionic species and their stoichiometric counts for use in aqueous/ionic chemistry workflows.
    
    This utility is intended for use in ChemPy code paths that need to convert a chemical formula (as typically found in the README examples and in Substance.from_formula) into the ions that the formula represents so that functions working with equilibria, ionic strength, or reaction balancing can operate on ionic species (for example, mapping 'NaCl' -> {'Na+': 1, 'Cl-': 1} or 'Fe(NO3)3' -> {'Fe+3': 1, 'NO3-': 3}). The current source implementation is a placeholder (the function body contains only pass) and therefore does not perform parsing; the docstring documents both the intended behavior and the current behavior so callers understand practical significance and limitations.
    
    Args:
        formula (str): A chemical formula expressed as a string. In the intended/target usage this string represents an ionic (or ion-forming) compound and may use element symbols, parentheses for grouped subunits, integer stoichiometric multipliers, and the dot notation for hydrates (examples from the original docstring: 'NaCl', 'Fe(NO3)3', 'FeSO4', '(NH4)3PO4', 'KAl(SO4)2.11H2O'). The parameter role is to supply the formula to be analyzed and converted into ionic components for downstream tasks such as computing ionic strength, constructing equilibria, or balancing reactions. Because the function is presently a stub, passing any string for formula will not raise an exception but will not produce parsed ions either.
    
    Returns:
        None: The current implementation is a no-op (function body is pass) and therefore returns None and has no side effects. Intended/future behavior: return a dict mapping ionic species (str) to integer stoichiometric counts, e.g. {'Na+': 1, 'Cl-': 1} for 'NaCl'. The returned mapping is conceptually significant in aqueous chemistry workflows because it lets higher-level code treat substances as collections of ions when setting up equilibrium systems, calculating ionic strength, or converting between molecular and ionic representations.
    
    Behavioral notes, limitations, and failure modes:
        - Current behavior: the function is a placeholder and returns None. Callers must not rely on it in the present codebase.
        - Intended behavior: recognize common inorganic ions and polyatomic groups, apply stoichiometric multipliers (including those given inside parentheses), and ignore hydrate water of crystallization for the purpose of listing primary ionic constituents (as illustrated by the example 'KAl(SO4)2.11H2O' which intends {'K+': 1, 'Al+3': 1, 'SO4-2': 2}). The docstring examples in the source illustrate the practical goal but are not an exhaustive specification.
        - Ambiguities and limits: deducing oxidation states or splitting a neutral molecule into ionic constituents is not always unambiguous. The intended parser will be conservative and target typical inorganic ionic salts and common polyatomic ions; organic molecules or covalent species that do not form discrete ions are outside the intended reliable scope.
        - Error handling (intended): when implemented, unparsable or clearly invalid formula strings should raise a ValueError (or a documented parser-specific exception). Since the current implementation is a no-op, no such errors are raised now.
        - Side effects: none in the current implementation; the intended function also has no side effects (pure parsing utility).
    
    Examples (intended usage shown in original source docstring):
        'NaCl' -> {'Na+': 1, 'Cl-': 1}
        'Fe(NO3)3' -> {'Fe+3': 1, 'NO3-': 3}
        'FeSO4' -> {'Fe+2': 1, 'SO4-2': 1}
        '(NH4)3PO4' -> {'NH4+': 3, 'PO4-3': 1}
        'KAl(SO4)2.11H2O' -> {'K+': 1, 'Al+3': 1, 'SO4-2': 2}
    """
    from chempy.util._aqueous import ions_from_formula
    return ions_from_formula(formula)


################################################################################
# Source: chempy.util._expr.create_Piecewise
# File: chempy/util/_expr.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def chempy_util__expr_create_Piecewise(parameter_name: str, nan_fallback: bool = False):
    """create_Piecewise creates a parameterized piecewise expression factory for use in ChemPy expression trees and symbolic/numeric backends.
    
    This function returns an Expr factory (produced by Expr.from_callback) that constructs a piecewise expression which selects one of several sub-expressions based on the value of a single runtime parameter. It is intended for use in ChemPy contexts where expressions depend on a single named parameter (for example 'x' in concentration- or position-dependent expressions used in kinetics, equilibria, or property functions). The factory expects a specific sequence encoding alternating interval bounds and expressions (see behavior below). The implementation supports two execution modes: if the provided backend exposes a Piecewise constructor (e.g. a symbolic backend such as SymPy) a symbolic backend.Piecewise expression is returned (with evaluate=False to avoid eager simplification); otherwise a simple numeric selection is performed by iterating the supplied bounds and returning the matching expression. If the runtime parameter is a quantity, bounds are converted to unitless values using the parameter's unit so comparisons are meaningful in unit-aware contexts.
    
    Args:
        parameter_name (str): Name of the single runtime parameter used to select which branch of the piecewise expression to evaluate. This exact string is passed as the parameter key when creating the Expr via Expr.from_callback, so callers and downstream code must use the same name to provide the parameter value (for example, 'x' to indicate the independent variable in a spatial or concentration-dependent expression). The returned Expr factory therefore produces expressions that expect a mapping or value for this parameter at evaluation time.
        nan_fallback (bool): When True, the constructed symbolic Piecewise will include a final fallback branch that returns a backend Symbol called 'NAN' with an unconditional (True) predicate; this ensures the Piecewise is always defined in symbolic backends even when the parameter lies outside all provided intervals. When False (the default), no unconditional fallback is added for the symbolic Piecewise; in numeric/backends-without-Piecewise mode, omission of a matching interval causes a ValueError to be raised. Use nan_fallback=True when you need a defined symbolic placeholder for out-of-range values and are prepared to handle the backend-specific 'NAN' symbol.
    
    Behavior and usage details, side effects, and failure modes:
        The Expr factory returned by this function expects a single positional argument bounds_exprs, a sequence with an odd number of items and at least three items. The required ordering of bounds_exprs is strict and must be:
            lower0, expr0, lower1, expr1, ..., lowerN, upperN
        where each loweri and upperi are numeric values or quantities defining inclusive interval endpoints and each expri is the expression (numeric value, Expr, or object understood by the backend) to select when the runtime parameter lies between loweri and upperi (inclusive). There are n_exprs = (len(bounds_exprs) - 1) // 2 branches, and the final element of bounds_exprs is the upper bound for the last branch.
        If the runtime parameter value is a quantity (detected via is_quantity), the bounds are converted to unitless values using unit_of(parameter_value) and to_unitless before any comparisons; this permits unit-aware selection without requiring callers to manually normalize units.
        Execution modes:
        - Symbolic/backend mode: If the provided backend exposes a Piecewise attribute, the function builds and returns backend.Piecewise(..., evaluate=False) composed of tuples (expr, backend.And(backend.Le(lower, ux), backend.Le(ux, upper))) for each branch, where ux is the unitless form of the runtime parameter. If nan_fallback is True a final branch (backend.Symbol('NAN'), True) is appended. evaluate=False is used to avoid automatic simplification by the backend.
        - Numeric/python mode: If the backend does not provide a Piecewise constructor, the function performs a simple numeric search: it iterates over branches and returns the first expr for which lower <= x <= upper holds. If no branch matches, a ValueError("not within any bounds: %s" % x) is raised.
        Preconditions and failures:
        - A ValueError is raised when the provided bounds_exprs sequence has fewer than three elements.
        - A ValueError is raised when bounds_exprs contains an even number of elements (the function requires an odd number: alternating lower/expr pairs ending with the final upper bound).
        - If the backend lacks Piecewise and the runtime parameter does not fall into any interval, a ValueError is raised (see message above). If a symbolic backend is used without nan_fallback and the parameter lies outside all intervals, behavior depends on the backend's handling of Piecewise with no unconditional branch.
        - The function assumes parameter_name is a string and uses it as the single parameter key in the produced Expr; supplying a non-string may lead to downstream errors from Expr.from_callback or from consumers expecting a string parameter key.
        Side effects:
        - No global state is modified. The returned object is an Expr factory; creating and evaluating expressions may invoke backend constructors or comparisons at runtime.
    
    Returns:
        Expr: An Expr factory (callable) created via Expr.from_callback that constructs a piecewise expression parameterized by the given parameter_name. The returned factory should be called with a single positional argument bounds_exprs (the ordered sequence described above). At evaluation time, the produced Expr uses the runtime value associated with parameter_name to select and return the appropriate branch expression according to the inclusive interval tests.
    """
    from chempy.util._expr import create_Piecewise
    return create_Piecewise(parameter_name, nan_fallback)


################################################################################
# Source: chempy.util._expr.create_Poly
# File: chempy/util/_expr.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def chempy_util__expr_create_Poly(
    parameter_name: str,
    reciprocal: bool = False,
    shift: str = None,
    name: str = None
):
    """Create a polynomial Expression factory for a single scalar parameter.
    
    This function is used in ChemPy to build Expr objects (chempy.util._expr.Expr) that evaluate polynomials in a single named scalar parameter — for example to represent temperature-dependent empirical fits or other scalar-parameter-dependent property approximations that occur throughout the ChemPy codebase (see README examples for temperature-dependent properties and kinetics). The returned Expr implements a callback that, when called with a sequence of arguments and a mapping of parameter values, evaluates the polynomial
    sum_{n=0}^{N-1} coeff_n * x0^{n},
    where x0 is either the parameter value or a shifted / reciprocal transform of it as specified by the arguments below. The implementation uses successive multiplication (or division for reciprocal polynomials) to build powers efficiently.
    
    Args:
        parameter_name (str): The name of the scalar parameter the polynomial depends on. This string is used as the single parameter key in the Expr returned by create_Poly. In practice this is a physical variable name such as 'T' for temperature or 'x' for a generic scalar; when evaluating the Expr you provide a mapping (e.g. {'T': 298.15}) that supplies the parameter value.
        reciprocal (bool): If False (default), the polynomial is in nonnegative integer powers of the parameter (x0^0, x0^1, x0^2, ...). If True, the polynomial uses successive powers of the reciprocal of x0 (1/x0, 1/x0^2, ...), producing terms coeff_n * (1/x0)^n with the first term still being coeff_0 (multiplied by 1). This option is useful for fits that are naturally expressed as polynomials in 1/T or other reciprocals (examples in code show reciprocal=True for temperature reciprocals). Beware that setting reciprocal=True will raise a division-by-zero error if the evaluated x0 equals zero.
        shift (str or bool or None): If None (the default), the polynomial is evaluated directly at the parameter value x (so x0 = x). If a string is provided, that string is used as the name of the first argument that supplies the shift/reference value and the polynomial is evaluated in powers of (x - x_shift) where x_shift is the first argument in the argument list for the Expr. If shift is True, this is a shorthand that is converted to the string 'shift' internally (so the Expr will expect a first argument named 'shift'). Concretely, when shift is not None the callback expects the first element of the provided args to be the shift/reference value and the remaining elements to be the polynomial coefficients. This is useful for representing polynomials expanded about a reference point (for example, expansion about a reference temperature). Note: the function also supports passing the shift value at call-time via the returned Expr's argument mechanisms (see examples in source).
        name (str or None): Optional Python function name to assign to the internal callback created for the Expr. If provided, the callback's __name__ will be set to this value; otherwise the default function name is left unchanged. This has no effect on numerical behavior but can make debugging and introspection outputs more readable.
    
    Returns:
        Expr: An Expr object (created via Expr.from_callback) configured to evaluate the described polynomial. The Expr is constructed with parameter_keys=(parameter_name,) and with argument_names set to None when shift is None, or to (shift, Ellipsis) when shift is provided (so the Expr's expected argument layout reflects whether a shift value must be supplied). The returned Expr is callable: typically you invoke it by passing a sequence/tuple of numeric arguments (coefficients, and if shift is used then first the shift value) and a mapping of parameter values, e.g. p = create_Poly('T'); p([a0, a1, a2], {'T': 300}) or, for shifted polynomials, p([Tref, a0, a1], {'T': 310}). The Expr callback uses Python's math-like numeric operations (backend=math by default) and will raise standard Python exceptions when given non-numeric inputs or when a division by zero occurs for reciprocal polynomials.
    
    Behavior, defaults, and failure modes:
        - Coefficient ordering: coefficients are interpreted in increasing power order: the first coefficient multiplies x0^0 (the constant term), the next multiplies x0^1, then x0^2, and so on.
        - Shift behavior: when shift is provided (string or True), the first element of the arguments passed to the Expr is interpreted as the shift/reference value (x_shift) and x0 is computed as x - x_shift. The remaining elements are treated as coefficients. Passing shift=True is shorthand for shift='shift'.
        - Reciprocal behavior: when reciprocal=True, powers are formed by successive division by x0, so the terms correspond to x0^0, x0^{-1}, x0^{-2}, ...; this will raise a ZeroDivisionError if x0 == 0 at evaluation.
        - Naming: supplying the name parameter only changes the __name__ of the internal callback for readability and diagnostics; it does not affect evaluation semantics.
        - Input validation and errors: the function does not perform extensive input validation; errors will typically be raised at evaluation time if non-iterable or non-numeric arguments are provided, or if required arguments (e.g. the shift value when shift is set) are missing. Division-by-zero exceptions can occur when reciprocal=True and x or (x - x_shift) equals zero.
        - Numerical backend: the internal callback accepts a backend keyword (defaulting to Python's math module) for numeric operations; this is an implementation detail of the returned Expr's callback and not part of create_Poly's signature.
    
    Practical significance:
        - This factory is intended for creating concise, reusable polynomial evaluators for single scalar parameters used across ChemPy (for example for temperature-dependent empirical fits, property approximations, or other scalar-parameter models). Using create_Poly keeps the polynomial-evaluation logic consistent and efficient (via iterative power accumulation) and integrates the polynomial as an Expr so it can be combined with other Expr-based machinery in ChemPy (fitting, expression composition, and evaluation with named parameter mappings).
    """
    from chempy.util._expr import create_Poly
    return create_Poly(parameter_name, reciprocal, shift, name)


################################################################################
# Source: chempy.util._quantities.format_units_html
# File: chempy/util/_quantities.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def chempy_util__quantities_format_units_html(
    udict: dict,
    font: str = "%s",
    mult: str = "&sdot;",
    paren: bool = False
):
    """format_units_html returns an HTML-formatted string representation of the units described by the udict mapping.
    
    This function is used in ChemPy's units/markup code paths (for example when rendering Substance.html_name or other human-readable HTML output) to convert a units description produced/accepted by the quantities.markup.format_units helper into a small piece of HTML suitable for display in notebooks, web pages, or other HTML-rendering contexts. The conversion preserves semantics of the units (multiplication, exponentiation, compound grouping) while replacing plain-text markers with HTML constructs: exponentiation like m**2 becomes m<sup>2</sup>, the ASCII multiplication operator '*' is replaced by the HTML symbol provided via mult, and an optional wrapper format (font) is applied last. The function depends on quantities.markup.format_units to produce the initial unit string and then post-processes that string with regular-expression replacements.
    
    Args:
        udict (dict): A units dictionary as accepted by quantities.markup.format_units. In ChemPy this dict encodes the units to be displayed (unit names/identifiers and their exponents as used by the quantities/chempy units machinery). This argument is passed unchanged to quantities.markup.format_units to produce an initial textual unit representation; if udict is not in the accepted form for quantities.markup.format_units that underlying function may raise ValueError/TypeError which will propagate out of format_units_html.
        font (str): A Python format string containing a single "%s" placeholder that will be used to wrap the final HTML output. Default "%s" applies no extra wrapper. Practical examples: '<span style="color: #0000a0">%s</span>' to color the unit text blue, or '<em>%s</em>' to emphasize. If font does not contain a "%s" placeholder, Python string-formatting will raise a TypeError.
        mult (str): HTML snippet or plain string used to replace ASCII multiplication signs ('*') in the unit string. Default is the HTML entity "&sdot;" (a centered dot). Other common choices are the empty string "" to omit visible multiplication, or "*" to keep the original character. This parameter controls only the visual symbol inserted for multiplication; it does not change numeric semantics.
        paren (bool): If True, request that the returned string be enclosed in parentheses when the unit is not already rendered as a compound unit by quantities.markup.format_units. The function first asks quantities.markup.format_units for a representation and treats a result that already starts with "(" and ends with ")" as a compound unit; in that case no extra parentheses are added even if paren is True. Default is False.
    
    Returns:
        str: An HTML string representing the units in udict, with integer exponentiation converted to <sup>...</sup>, '*' replaced by the specified mult string, optional surrounding parentheses added per the paren rule, and finally wrapped using the font format string. The returned string is intended for embedding into HTML output (not escaped); it may therefore contain HTML entities and tags like <sup> and any markup supplied via font or mult.
    
    Behavior, defaults, and failure modes:
        - The function delegates initial formatting to quantities.markup.format_units(udict). Any exceptions raised by that call (for example due to an invalid udict) will propagate unchanged.
        - After obtaining the initial unit string, the function replaces occurrences of exponentiation of the form '**<digits>' (one or more ASCII digits) with an HTML superscript tag using the pattern <sup>digits</sup>. Note the limitation: only positive integer exponents made of digits are matched by the regular expression; non-integer or signed exponents will not be transformed by this regex.
        - All ASCII '*' characters in the units string are replaced with the mult argument value. This replacement is global.
        - Compound-unit detection is performed by checking whether the string returned by quantities.markup.format_units starts with '(' and ends with ')'. If so, the string is considered a compound unit and the paren flag will not induce additional wrapping.
        - The font argument is applied last using Python's old-style ("%") string formatting. The font string must contain exactly one "%s" placeholder; otherwise a TypeError will be raised by Python's string formatting. The default "%s" leaves the transformed unit string unwrapped.
        - No I/O or global state is modified; the function is pure in the sense that it returns a new string and has no side effects. However, because the returned string may contain HTML tags and is not escaped, embedding untrusted unit names into udict could result in unexpected HTML injection—take care when rendering units that originate from untrusted input.
        - The function relies on regular-expression patterns to perform replacements; if the initial string from quantities.markup.format_units already contains HTML tags or exotic characters, the regex-based substitutions may interact with those tags in ways that are not further sanitized or validated.
    """
    from chempy.util._quantities import format_units_html
    return format_units_html(udict, font, mult, paren)


################################################################################
# Source: chempy.util.parsing.formula_to_composition
# File: chempy/util/parsing.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def chempy_util_parsing_formula_to_composition(
    formula: str,
    prefixes: list = None,
    suffixes: tuple = ('(s)', '(l)', '(g)', '(aq)')
):
    """Parse a chemical formula string and return its composition as a mapping from atomic number to multiplicity.
    
    This function is used in ChemPy to convert human-readable chemical formulae (as found in the README examples and reaction/solution specifications) into a machine-friendly composition dictionary. The composition keys are integer atomic numbers (0 reserved for net charge) and the values are numeric multiplicities aggregated across possible dot-separated parts (hydrates, adducts). The parser ignores specified textual prefixes and physical-state suffixes, supports both the Unicode middle dot (·, U+00B7) and the legacy double-dot ("..") hydrate notation, and handles leading integer multipliers on subsequent parts (e.g., "Na2CO3..7H2O" treats "7" as a multiplier for the water part).
    
    Args:
        formula (str): Chemical formula to parse. Typical inputs are molecular formulae or species names that encode charge and stoichiometry, for example 'H2O', 'Fe+3', 'Cl-', '.NHO-(aq)', 'Na2CO3..7H2O', or 'UO2.3'. The string may include a leading prefix (to be ignored), a trailing suffix indicating phase (to be ignored), charge tokens (e.g. "+3", "-"), and dot-separated hydrate/adduct parts. This argument is required and must be a text string; a non-string will result in a TypeError in downstream parsing.
        prefixes (list): Iterable of prefix strings to ignore before parsing the core formula. Examples are '.' or 'alpha-'. If prefixes is None (the default), the function uses the internal LaTeX-to-text mapping keys (i.e., a predefined set of LaTeX prefixes known to ChemPy) as the prefixes to strip. Passing an explicit iterable restricts which leading substrings are stripped from the input formula prior to parsing.
        suffixes (tuple): Tuple of suffix strings to ignore after parsing the core formula. The default is ('(s)', '(l)', '(g)', '(aq)') to remove common physical-state annotations like solid, liquid, gas and aqueous. Any matching suffix found at the end of the input is removed before the chemical composition is parsed.
    
    Returns:
        dict: A dictionary mapping integer atomic numbers to numeric multiplicities describing the composition. Keys are ints representing atomic numbers as used throughout ChemPy (0 is reserved for net charge). Values are numeric multiplicities (typically integers for atomic counts, but the parser can also yield non-integer numeric stoichiometries as allowed by the input, e.g., 2.3 for 'UO2.3'). Multiplicities from dot-separated parts are multiplied by any leading integer multiplier for that part and aggregated into a single composition dictionary. If a charge token is present in the input, the net charge is placed under key 0 (with sign according to the token).
    
    Behavior, defaults, and failure modes:
        - The function first strips any matching prefixes and suffixes (controlled by the prefixes and suffixes arguments) before parsing the core formula.
        - Hydrate/adduct notation is recognized either with the Unicode middle dot (·, U+00B7) or with the legacy double-dot (".."). Parts after the first may have a leading integer multiplier which multiplies all multiplicities in that part (e.g., "A..3B2" treats B as appearing 6 times).
        - Charge tokens parsed from the input (e.g., '+2', '-') are converted to a signed integer and stored under key 0.
        - If prefixes is None, the function uses ChemPy's internal LaTeX mapping keys as the set of prefixes to ignore; providing an explicit iterable overrides this behavior.
        - The function has no side effects; it returns a new dict and does not modify global state.
        - On malformed input (non-string formula, unparseable token, invalid numeric formats, or other lexical errors encountered by internal helpers), the function will propagate an exception (for example, ValueError or TypeError) raised by the underlying parsing helpers. Callers should validate or catch exceptions when parsing user-provided formulae.
    """
    from chempy.util.parsing import formula_to_composition
    return formula_to_composition(formula, prefixes, suffixes)


################################################################################
# Source: chempy.util.parsing.formula_to_html
# File: chempy/util/parsing.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def chempy_util_parsing_formula_to_html(
    formula: str,
    prefixes: dict = None,
    infixes: dict = None,
    **kwargs
):
    """Convert a chemical formula string into an HTML string representation suitable for display in HTML contexts (for example, in Jupyter notebooks or web pages showing parsed Substance names as in the README).
    
    This function is used in ChemPy to render parsed chemical formulas (such as those produced by Substance.from_formula) into readable HTML where stoichiometric numbers are wrapped in <sub> tags and charges in <sup> tags, and where certain textual prefixes/infixes (for example greek letter names or a leading dot) are replaced by HTML entities. Typical use in the project is to produce Substance.html_name or to present formulas in documentation and user interfaces.
    
    Args:
        formula (str): Chemical formula to convert, for example 'H2O', 'Fe+3', 'Cl-'. The value must be a Python str; non-str inputs are not supported and may raise a TypeError. The function interprets digits as subscripts and leading/trailing charges or sign tokens as superscripts; parenthesized group counts are converted to subscripted numbers. Practical significance: this argument is the canonical textual chemical representation that users or other library code supply when they want an HTML-safe visual representation of a substance.
        prefixes (dict): Mapping used to transform prefix tokens in the formula into HTML fragments. If None (the default) the function uses the module default mapping (internal variable _html_mapping) that includes common transformations such as mapping 'alpha' to '&alpha;' and a leading '.' to '&sdot;'. Role: allows callers to override or extend how leading textual prefixes (for example 'alpha-' or a dot indicating a radical or surface site) are converted into HTML entities or strings for presentation.
        infixes (dict): Mapping used to transform infix tokens inside the formula into HTML fragments. If None (the default) the function uses the module default mapping (internal variable _html_infix_mapping). Role: allows callers to control replacements of tokens that appear within a formula (for example converting specific textual separators or annotations to HTML), enabling consistent rendering across different naming conventions.
        kwargs (dict): Additional keyword arguments forwarded to the underlying formatter _formula_to_format. Known and supported keys (passed through for compatibility) include:
            suffixes (tuple of str): A tuple of suffix strings to preserve and append unchanged (examples: ('(g)', '(s)', '(aq)')). When provided, recognized suffixes are left as-is after the sub/sup transformations so state annotations like phase or solvation are retained. Other keys are accepted by the internal formatter but are implementation details; callers should only rely on documented keys. Practical significance: kwargs let callers control minor formatting behaviors (such as keeping phase annotations) without modifying the global mappings.
    
    Behavior, defaults, and side effects:
        The function applies two formatting callbacks to the parsed formula: digits and parenthesized counts are wrapped in <sub>...</sub>, and charges or other superscript-like tokens are wrapped in <sup>...</sup>. Prefix and infix textual substitutions are applied according to the provided mappings. By default, prefixes and infixes are set to the module defaults that include common chemistry-related transformations (e.g., greek letters and a centered dot). The function returns a new string and has no side effects on inputs or global state. It does not perform HTML escaping beyond the specific replacements and tag insertions it makes; callers embedding the result into larger HTML contexts should ensure content safety as appropriate.
    
    Failure modes:
        If formula is not a str the function may raise a TypeError. If mappings contain unexpected values (non-string replacements) the resulting output may be invalid HTML or may raise runtime errors in the underlying formatter. Unrecognized tokens in the formula are generally left unchanged (other than sub/sup handling) so the function is tolerant of unusual names but will not validate chemical correctness.
    
    Examples of practical outputs (illustrative and matching typical project usage):
        formula_to_html('NH4+') -> 'NH<sub>4</sub><sup>+</sup>'
        formula_to_html('Fe(CN)6+2') -> 'Fe(CN)<sub>6</sub><sup>2+</sup>'
        formula_to_html('Fe(CN)6+2(aq)') -> 'Fe(CN)<sub>6</sub><sup>2+</sup>(aq)'
        formula_to_html('.NHO-(aq)') -> '&sdot;NHO<sup>-</sup>(aq)'
        formula_to_html('alpha-FeOOH(s)') -> '&alpha;-FeOOH(s)'
    
    Returns:
        str: An HTML-formatted string representing the input chemical formula. The returned string contains <sub>...</sub> fragments for stoichiometric counts, <sup>...</sup> fragments for charges and superscripted annotations, and any prefix/infix substitutions applied as HTML entities or fragments according to the provided mappings. No in-place modification of the input occurs.
    """
    from chempy.util.parsing import formula_to_html
    return formula_to_html(formula, prefixes, infixes, **kwargs)


################################################################################
# Source: chempy.util.parsing.formula_to_latex
# File: chempy/util/parsing.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def chempy_util_parsing_formula_to_latex(
    formula: str,
    prefixes: dict = None,
    infixes: dict = None,
    **kwargs
):
    """Convert a chemical formula string into a LaTeX-ready string fragment suitable for use in reports, notebooks, or publication-quality renderings. This function is used by higher-level ChemPy utilities that present chemical substances (for example Substance.latex_name) and by examples in the README to render molecular formulas with numeric subscripts, charge superscripts, and common textual prefixes/infixes (greek letters, bullets, etc.) while preserving common suffixes such as phase indicators.
    
    Args:
        formula (str): Chemical formula to convert, for example 'H2O', 'Fe+3', 'Cl-',
            'Fe(CN)6+2(aq)'. The function interprets integer counts that follow element symbols
            or closing parentheses as subscripts and charge annotations (sequences containing
            '+' or '-' optionally prefixed by a magnitude) as superscripts. Curly braces '{' or
            '}' in the input are escaped so they do not break LaTeX. This argument must be a
            Python string; passing a non-string will typically raise a TypeError when the
            function attempts string operations.
        prefixes (dict): Optional mapping of textual prefixes to LaTeX replacements. When
            provided, keys are substrings that appear at the start of the formula (prefixes)
            and values are the LaTeX text that should replace them (for example mapping
            'alpha' to '\\alpha'). If None (the default) the function uses the module's
            default mapping of common chemical prefixes to LaTeX (_latex_mapping). The mapping
            is applied during conversion; entries not present in the mapping are left
            unchanged.
        infixes (dict): Optional mapping of textual infixes to LaTeX replacements. When
            provided, keys are substrings that appear inside the formula (infixes) and values
            are the LaTeX replacements (for example mapping '.' to '^\\bullet' for radical
            notation). If None (the default) the function uses the module's default infix
            mapping (_latex_infix_mapping). The mapping is applied during conversion; entries
            not present in the mapping are left unchanged.
        kwargs (dict): Additional keyword arguments forwarded to the internal formatter
            function. Recognized keyword: 'suffixes' (iterable of str) which lists suffix
            substrings that should be preserved verbatim and not interpreted or transformed
            (common phase/physical state suffixes such as '(s)', '(l)', '(g)', '(aq)' are
            treated this way). If 'suffixes' is not provided, the default is the tuple
            ('(s)', '(l)', '(g)', '(aq)') as in the original implementation. Any other
            keyword arguments are passed unchanged to the underlying helper; unknown keys
            may be ignored or handled by that helper.
    
    Returns:
        str: A new string containing the LaTeX-formatted representation of the input formula.
        Numeric counts are rendered as subscripts using the pattern _{...} and charges are
        rendered as superscripts using the pattern ^{...}. Prefix and infix mappings (either
        defaults or those supplied via the prefixes/infixes arguments) are applied so that
        common textual conventions (greek letters, bullets, etc.) are converted to their
        LaTeX equivalents. The returned fragment is suitable for inclusion in LaTeX source
        (it does not automatically add math delimiters like $...$). The function has no
        external side effects; it only returns the converted string.
    
    Notes on behavior, defaults, and failure modes:
        The function delegates formatting work to an internal helper and, prior to that,
        escapes any literal curly braces in the input formula so they do not interfere with
        LaTeX. If prefixes or infixes are provided they must be mapping-like (dict) objects
        whose keys and values are strings; providing non-dict objects may lead to runtime
        errors. Suffixes specified via kwargs are treated as literal tails that are not
        parsed into subscripts/superscripts (useful for phase labels like '(aq)'). If the
        input contains tokens or patterns not recognized by the formatter or the supplied
        mappings, those parts are left unchanged in the returned string rather than causing
        conversion to fail.
    """
    from chempy.util.parsing import formula_to_latex
    return formula_to_latex(formula, prefixes, infixes, **kwargs)


################################################################################
# Source: chempy.util.parsing.formula_to_unicode
# File: chempy/util/parsing.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def chempy_util_parsing_formula_to_unicode(
    formula: str,
    prefixes: dict = None,
    infixes: dict = None,
    **kwargs
):
    """Convert a chemical formula string into a Unicode-rendered formula suitable for human-readable display.
    
    This function takes a chemical formula (for example, 'H2O', 'Fe+3', 'Cl-') and returns a Unicode string where numeric counts are rendered as subscripts, charge signs and magnitudes as superscripts, and known textual prefixes/infixes (such as "alpha" or "." for a middle dot) are replaced by their Unicode equivalents. In the ChemPy codebase this utility is used when generating human-friendly names for parsed substances (for example Substance.unicode_name) and when printing or rendering formulas in logs, UIs, or reports so that chemical notation appears correctly (e.g., 'NH4+' -> 'NH₄⁺', 'Fe(CN)6+2(aq)' -> 'Fe(CN)₆²⁺(aq)').
    
    Args:
        formula (str): Chemical formula to convert. This is the textual representation of a substance using common ASCII conventions for elements, integer counts, and charges (examples: 'H2O', 'NH4+', 'Fe(CN)6+2(aq)'). The function expects a Python str and will process element symbols, parentheses, digits, plus/minus charge signs, and any textual prefixes/infixes. Providing a non-str value is a misuse and may raise a TypeError from downstream processing.
        prefixes (dict): Mapping of textual prefix tokens to Unicode strings used to transform leading tokens in the formula. Typical default mapping (_unicode_mapping) contains entries used in chemistry notation, for example mapping the word 'alpha' to the Greek letter 'α'. The prefixes argument allows callers to override or extend these translations; if None, the function uses the module's default _unicode_mapping. The mapping must be a dict; missing or incompatible mappings required during conversion may result in KeyError.
        infixes (dict): Mapping of textual infix tokens to Unicode strings used to transform in-between tokens in the formula. Typical default mapping (_unicode_infix_mapping) contains entries used for characters like '.' which are commonly rendered as a centered dot '⋅' in hydrates and adducts. If None, the module's default _unicode_infix_mapping is used. The mapping must be a dict; missing or incompatible mappings required during conversion may result in KeyError.
        kwargs (dict): Additional keyword arguments forwarded to the underlying formatter (_formula_to_format). In practice the most common supported keyword is suffixes (a tuple of str) which lists suffix substrings to preserve verbatim (for example ('(g)', '(s)', '(aq)') so that phase labels remain unchanged after conversion). Any kwargs must be accepted by the underlying _formula_to_format implementation; unsupported or incorrectly typed kwargs may raise exceptions from that function.
    
    Returns:
        str: A Unicode string representing the input chemical formula with subscripts, superscripts, and prefix/infix Unicode replacements applied. This return value is a new string (no in-place modification of the input occurs). The result is intended for display and printing; it preserves specified suffixes such as phase annotations when provided via kwargs.
    
    Behavior and failure modes:
        - The function is pure and has no observable side effects on module state aside from using the provided mapping objects.
        - Defaults: when prefixes or infixes is None the function uses the module-private defaults _unicode_mapping and _unicode_infix_mapping respectively.
        - The conversion algorithm maps digits to Unicode subscript characters and maps charge signs/magnitudes to Unicode superscript characters by looking up characters in module mappings. If a required mapping entry is absent (for example a digit or sign not present in the mapping dictionaries), a KeyError may be raised during conversion.
        - If formula is not a str, downstream operations expecting str may raise TypeError; callers should pass a string.
        - kwargs are passed directly to the internal formatter; incorrect kwargs (wrong names or types) will cause errors raised by that formatter. The documented and commonly used kwarg is suffixes (tuple of str) to keep suffixes like '(aq)' unchanged.
        - This function does not perform chemical validation (it does not check element validity, atomic weights, or stoichiometric consistency); it only formats textual formulas for display.
    """
    from chempy.util.parsing import formula_to_unicode
    return formula_to_unicode(formula, prefixes, infixes, **kwargs)


################################################################################
# Source: chempy.util.periodic.atomic_number
# File: chempy/util/periodic.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def chempy_util_periodic_atomic_number(name: str):
    """chempy.util.periodic.atomic_number: Return the atomic number (Z) for a given element specified by name or chemical symbol.
    
    Args:
        name (str): Full element name or chemical symbol to look up. This function accepts a Python string representing either a chemical symbol (e.g. "Fe", "Cl", case-insensitive variants such as "fe" or "FE") or a full element name (e.g. "iron", "chlorine", case-insensitive). The implementation first capitalizes the input and searches the module-level symbols list (a sequence of canonical element symbols indexed so that index + 1 == atomic number). If that lookup fails, it lower-cases the input and searches the module-level lower_names list (a sequence of full element names indexed so that index + 1 == atomic number). The parameter must be a str; passing a non-str value will likely raise an AttributeError or TypeError because the code calls string methods on the argument.
    
    Returns:
        int: The atomic number (positive integer) corresponding to the supplied element. Atomic numbers follow the conventional sequence where hydrogen is 1, helium is 2, etc. The returned integer is intended for use in ChemPy contexts that use atomic numbers as identifiers (for example, Substance.composition maps element atomic numbers to counts, as shown in the project README).
    
    Raises and failure modes:
        ValueError: Raised if the provided name is not found in either the symbols list or the lower_names list. This indicates the element string is unrecognized by the module's internal lists.
        AttributeError or TypeError: May be raised if name is not a string (because the code calls str methods like capitalize and lower).
    
    Behavior and side effects:
        The function performs no side effects; it only performs lookups in module-level sequences (symbols and lower_names) defined in chempy.util.periodic. There is no default value for name; it must be provided. Lookup is case-normalized as described above, but ambiguous or nonstandard names (e.g., isotopic labels, malformed strings, or alternate vernacular names not present in lower_names) will not be resolved and will cause a ValueError. The function is used throughout ChemPy to convert human-readable element identifiers into the integer atomic-number keys required by stoichiometry, compositions, and other chemistry routines described in the README.
    """
    from chempy.util.periodic import atomic_number
    return atomic_number(name)


################################################################################
# Source: chempy.util.periodic.mass_from_composition
# File: chempy/util/periodic.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def chempy_util_periodic_mass_from_composition(composition: dict):
    """chempy.util.periodic.mass_from_composition: Calculate the molecular mass (molecular weight) from a composition mapping that relates atomic numbers to integer stoichiometric coefficients. This utility is used in ChemPy for tasks that require a molecular weight computed from a parsed or programmatic composition (for example, computing mass fractions, reporting Substance.mass or balancing stoichiometry in reaction systems as shown in the project README).
    
    Args:
        composition (dict): Dictionary mapping atomic number (int) to coefficient (int). Keys are integer atomic numbers where 1 denotes hydrogen, 2 helium, etc.; by convention 0 denotes net electron count / charge (see Notes). Values are integer stoichiometric coefficients for each atomic number in the species or aggregate being described. The function iterates over composition.items() and for each key k:
            - if k == 0, the contribution to the returned mass is computed as -v * 5.489e-4 (the electron mass expressed in atomic mass units) so the sign of v changes whether mass is added or removed (negative v is an electron deficiency / net positive charge).
            - otherwise the contribution is v * relative_atomic_masses[k - 1], where relative_atomic_masses is the module-level sequence of standard atomic masses indexed by (atomic number - 1).
            The mapping must therefore use integers in the valid atomic-number range for which relative_atomic_masses contains entries; coefficients are expected to be integers as in typical stoichiometric descriptions. If other numeric types are provided for values they will be multiplied as in normal Python arithmetic, but the documented expectation is int coefficients.
    
    Returns:
        float: Molecular weight in atomic mass units (u). The returned value is the sum of contributions from nuclei (using relative_atomic_masses[k - 1] for k > 0) adjusted by the electron mass contribution computed for key 0. The result is a Python float.
    
    Notes:
        Atomic number 0 denotes charge or "net electron deficiency" and is handled specially as described above. The constant 5.489e-4 in the implementation is the electron mass expressed in atomic mass units and is applied with the sign convention implemented in the code (mass is decremented by v * 5.489e-4).
        The function is pure (no side effects): it does not modify the input mapping or module-level data. It relies on the module-level sequence relative_atomic_masses for nuclear mass values.
    
    Failure modes:
        TypeError: May be raised if composition is not a mapping or if keys cannot be used as integer indices (e.g., non-integer keys that cannot be used to compute k - 1 for list indexing).
        IndexError: May be raised when an atomic number key k > 0 is outside the range supported by relative_atomic_masses (i.e., k - 1 is out of bounds).
        ValueError or other arithmetic errors may propagate if invalid numeric values are supplied as coefficients.
        Callers should validate that keys are integer atomic numbers within the supported range and that coefficients reflect the intended stoichiometry to avoid these errors.
    """
    from chempy.util.periodic import mass_from_composition
    return mass_from_composition(composition)


################################################################################
# Source: chempy.util.pyutil.defaultnamedtuple
# File: chempy/util/pyutil.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def chempy_util_pyutil_defaultnamedtuple(
    typename: str,
    field_names: str,
    defaults: tuple = ()
):
    """Generates and returns a new named tuple subclass (a lightweight immutable record type) with configurable default values for its trailing fields. This helper wraps collections.namedtuple and then adjusts the generated class's __new__.__defaults__ so that instances can be created with omitted trailing fields supplied from defaults. In the ChemPy codebase this is useful for defining compact, tuple-backed data containers used in chemical modelling (for example, simple spatial/property records such as the Body example in the original docstring: Body(x, y, z, density)), where some attributes commonly have sensible defaults (e.g., density).
    
    Args:
        typename (string): The name to assign to the generated class. This becomes the class __name__ and appears in the class __doc__ (for example 'Body'). The name must be a valid Python identifier acceptable to collections.namedtuple; if it is not, namedtuple will raise ValueError.
        field_names (str or iterable): The field names for the record, given either as a space/comma-separated string or as an iterable of strings. These determine the order of fields in the tuple, the Tuple._fields sequence, and the order used when converting instances to dictionaries via _asdict(). Invalid field names or malformed input will cause namedtuple to raise ValueError.
        defaults (iterable): Default values to assign to the last N fields, where N is len(defaults). The function accepts either: (a) a mapping (e.g., dict) where keys are field names and values are the defaults to apply to those fields (the mapping is used to construct a temporary instance Tuple(**defaults) and its values are used), or (b) a sequence/iterable of default values which are applied to the final fields in order. If fewer defaults than fields are provided, the earlier (leading) fields receive None as their default. If defaults is omitted or empty (the default ()), all fields will default to None. The defaults are applied by setting Tuple.__new__.__defaults__, which affects how the class can be instantiated with positional and keyword arguments.
    
    Behavior and side effects:
        The function first calls collections.namedtuple(typename, field_names) to create a new tuple subclass (referred to here as Tuple). It then sets Tuple.__new__.__defaults__ so that creating instances without supplying trailing fields will use the supplied defaults. For a mapping defaults, Tuple(**defaults) is used to obtain values in field order and those values become the __new__.__defaults__. For a sequence defaults, the sequence is right-aligned to the full field list by prefixing with None values so that default values correspond to the last fields. The function mutates the generated Tuple class by assigning to its __new__.__defaults__ and returns that class. No other global state is modified.
    
    Failure modes and exceptions:
        Invalid typename or field_names as required by collections.namedtuple will raise ValueError. If defaults is a mapping that contains keys not matching the field names, attempting Tuple(**defaults) will raise TypeError. If a sequence defaults is longer than the number of fields, the subsequent attempt to construct Tuple(*defaults) will raise TypeError due to too many positional arguments. Providing defaults with types incompatible with intended usage will not be checked by this function and may cause errors later when constructing instances or using their values.
    
    Returns:
        type: A new tuple subclass named according to typename. The returned class is a subclass of tuple produced by collections.namedtuple and has its __new__.__defaults__ adjusted so that omitted trailing fields are filled from the provided defaults. Instances are immutable tuple objects that support the typical namedtuple attributes and methods such as _fields and _asdict(), and are suitable as compact records for small chemical modelling data structures (e.g., positions, simple property containers, or parameters).
    """
    from chempy.util.pyutil import defaultnamedtuple
    return defaultnamedtuple(typename, field_names, defaults)


################################################################################
# Source: chempy.util.regression.avg_params
# File: chempy/util/regression.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def chempy_util_regression_avg_params(opt_params: numpy.ndarray, cov_params: numpy.ndarray):
    """chempy.util.regression.avg_params computes an inverse-variance weighted average of parameter vectors obtained from multiple regression fits and returns both the weighted mean parameters and an estimate of the variance of those averaged parameters. This function is intended for post-processing results from repeated fits (for example, fitting integrated rate expressions or kinetic rate constants across replicate experiments or bootstrap samples) where each fit yields a parameter vector and an associated parameter covariance matrix.
    
    Args:
        opt_params (numpy.ndarray): Array of shape (nfits, nparams) containing the estimated parameter vectors from each independent fit. Each row corresponds to one fit and each column corresponds to a specific model parameter (for example, a rate constant or coefficient used in ChemPy's kinetics or integrated rate-expression fitting routines). The practical role of this argument is to supply the set of point estimates that are to be combined into a single representative parameter vector.
        cov_params (numpy.ndarray): Array of shape (nfits, nparams, nparams) containing the variance-covariance matrices for the parameter estimates from each fit. Each cov_params[i] is the covariance matrix corresponding to opt_params[i]. The implementation uses the diagonal variances of the first two parameters (cov_params[:, 0, 0] and cov_params[:, 1, 1]) as the per-parameter variances used for inverse-variance weighting; therefore cov_params must contain those entries and be consistent with opt_params. In practice these covariance matrices are produced by regression/fit routines and encode the estimated uncertainty for each fitted parameter.
    
    Returns:
        avg_beta (numpy.ndarray): Weighted average of the parameters with shape (nparams,). This array contains the inverse-variance weighted mean across the nfits input parameter vectors. In the context of ChemPy, avg_beta can be interpreted as the consensus estimate for model parameters (e.g., rate constants or fitted coefficients) derived from multiple independent fits.
        var_avg_beta (numpy.ndarray): Estimated variance of the averaged parameters with shape (nparams,). This is not a full variance-covariance matrix but a per-parameter variance estimate computed from the scatter of opt_params around avg_beta and the per-fit variances extracted from cov_params. The value gives an indication of the uncertainty of each averaged parameter under the assumptions of the implemented weighting and variance-estimation formula.
    
    Behavior and important implementation notes:
        The function uses inverse-variance weighting: weights = 1 / var_beta, where var_beta is constructed from the diagonal elements cov_params[:, 0, 0] and cov_params[:, 1, 1]. As implemented, the code extracts only the variances for parameter indices 0 and 1 and stacks them, which effectively assumes nparams == 2. If your fits have a different number of parameters, this implementation will raise an IndexError or produce incorrect results.
        The weighted mean is computed with numpy.average(..., weights=1/var_beta). The variance estimate var_avg_beta is computed from the weighted squared deviations of opt_params from avg_beta divided by ((avg_beta.shape[0] - 1) * sum_of_weights) according to the formula in the source. Note that avg_beta.shape[0] refers to the number of parameters in the returned mean and not the number of fits; this implementation detail follows the source code but may be unintuitive.
        Side effects: None. The function does not modify its inputs.
        Failure modes and cautions: If any extracted variance in var_beta is zero or negative, the reciprocal 1/var_beta will produce infinities or NaNs and may lead to invalid outputs or runtime warnings. If cov_params does not have the expected shape or does not contain valid diagonal elements for indices 0 and 1, an IndexError or unexpected results will occur. The function assumes independent fits and uses only diagonal variances for weighting; off-diagonal covariances are ignored by this implementation.
    """
    from chempy.util.regression import avg_params
    return avg_params(opt_params, cov_params)


################################################################################
# Source: chempy.util.regression.irls_units
# File: chempy/util/regression.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def chempy_util_regression_irls_units(x: numpy.ndarray, y: numpy.ndarray, **kwargs):
    """chempy.util.regression.irls_units is a units-aware wrapper around chempy.util.regression.irls. It accepts numeric data arrays that may carry units (quantities as used in ChemPy's units support), converts them to unitless numbers for the core iterative reweighted least squares fit, and then re-attaches appropriate units to the returned regression coefficients so the results are directly meaningful in chemical/physical applications (for example when x and y represent concentrations, pressures, temperatures, or rates).
    
    Args:
        x (numpy.ndarray): Independent variable data supplied to the regression. In the ChemPy domain this typically represents measured or controlled quantities (e.g., concentrations, temperatures, times) and may be a plain numpy array or a quantities/units-aware array supported by chempy.units. The function reads the units with unit_of(x), converts the numeric values to unitless form with to_unitless, and uses the resulting unitless array as the predictor input to irls. If x does not carry units the function treats it as already unitless.
        y (numpy.ndarray): Dependent (response) variable data corresponding to x. In chemical modelling this commonly represents observed responses such as reaction rates, measured concentrations, or other state variables. As with x, y may be a plain numpy array or a units-aware quantity; its units are read and removed prior to fitting so irls operates on pure numbers. If units are incompatible with the expected numeric conversion, an error from unit_of/to_unitless may be raised.
        kwargs (dict): Keyword arguments forwarded verbatim to chempy.util.regression.irls. Common keywords expected by irls (e.g., options controlling tolerances, maximum iterations, weighting schemes) should be passed here. The behavior and defaults of those options are governed by irls; irls_units does not alter or document those defaults beyond forwarding the arguments. Any unexpected or unsupported kwargs will be handled (or rejected) by irls itself and may cause exceptions.
    
    Detailed behavior:
        The function extracts units from x and y using unit_of, converts both arrays to unitless numpy arrays with to_unitless, and then calls irls(x_ul, y_ul, **kwargs) where x_ul and y_ul are the unitless numeric data. The returned beta (numeric coefficients), vcv (variance-covariance matrix), and info (fit metadata) come from irls. irls_units then calls _beta_tup(beta, x_unit, y_unit) to produce beta_tup, a coefficients object/tuple where the numeric coefficients have been converted back to quantities with units consistent with the original x and y. This makes the coefficients directly interpretable in chemical contexts (for example, a slope in units of concentration per time will carry those units).
    
    Side effects and defaults:
        No external side effects (files, global state) are produced; the function only computes and returns values. Default behaviors for the fitting algorithm (tolerances, iteration limits, weighting defaults, etc.) follow the implementation of irls and are not changed here. If x or y are already unitless, the conversion steps are effectively no-ops and behavior reduces to calling irls on the provided arrays.
    
    Failure modes and error conditions:
        Errors may be raised if unit_of or to_unitless cannot determine or convert units for x or y (for instance, if the input type is not supported by chempy.units). The underlying irls call may raise exceptions for ill-conditioned or singular design matrices, non-finite inputs, or if convergence is not achieved; in such cases the returned info object from irls (when available) will contain diagnostic information. The variance-covariance matrix vcv may be None or ill-conditioned if the fit is not identifiable; callers should inspect info and vcv to assess fit quality.
    
    Returns:
        tuple: A 3-tuple (beta_tup, vcv, info) where beta_tup is the coefficients tuple/object produced by _beta_tup(beta, x_unit, y_unit) with units attached to the numeric coefficients so they are meaningful in chemical applications; vcv (typically a numpy.ndarray or None) is the variance-covariance matrix corresponding to the fitted numeric coefficients (in a form consistent with how irls reports vcv); and info is the diagnostics/metadata object returned by irls (containing convergence status, iteration counts, residuals, weights, or other solver-specific information).
    """
    from chempy.util.regression import irls_units
    return irls_units(x, y, **kwargs)


################################################################################
# Source: chempy.util.regression.least_squares
# File: chempy/util/regression.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def chempy_util_regression_least_squares(
    x: numpy.ndarray,
    y: numpy.ndarray,
    w: numpy.ndarray = 1
):
    """chempy.util.regression.least_squares performs a linear least-squares regression (ordinary or weighted) on paired data (x, y) and returns the parameter estimates (intercept and slope), their estimated 2x2 variance-covariance matrix, and the coefficient of determination R2. In the context of ChemPy (see README) this function is a convenience fitting routine that can be used for tasks such as fitting integrated rate expressions or calibrations where a linear relation is appropriate; when measurement variances are known or suspected to be heteroscedastic, the weighting argument w enables weighted least squares (WLS) for more reliable parameter estimates.
    
    Args:
        x (numpy.ndarray): One-dimensional array of independent-variable observations (predictor). The function constructs a design matrix with a column of ones for the intercept and a column with these x values for the slope. Practical significance: x typically represents e.g. time or a transformed variable in kinetic/integrated rate fits; its length determines the sample size n. x must be length-matched to y and (if provided as an array) to w; if x contains identical values (no variation) the normal equations matrix X^T X can be singular and estimation will fail.
        y (numpy.ndarray): One-dimensional array of dependent-variable observations (response) with the same length as x. The code converts y to float64 and, when weights are used, multiplies by sqrt(w) before fitting. Practical significance: y typically represents measured concentrations, transformed concentrations, or other observables in chemical modelling; constant y (zero total sum of squares) will make R2 undefined (division by zero).
        w (numpy.ndarray): Optional weights for weighted least squares (WLS). The default value 1 (scalar) yields ordinary least squares (OLS). When w is not the scalar 1, the implementation uses sqrt(w) to weight both X and y (pre-multiplying each row), equivalent to standard WLS where larger weights give more influence to corresponding observations. If provided as a one-dimensional numpy.ndarray it must have the same length as x and y; broadcasting of a scalar weight to all observations is supported. Practical significance: use w to reflect known inverse-variance weights from measurement uncertainty or to de-emphasize outliers. Note that degenerate or non-positive weights, mismatched lengths, or improper shapes may lead to incorrect results or exceptions.
    
    Behavior, defaults, side effects, and failure modes:
        The function forms the weighted design matrix X with two columns [1, x] and multiplies rows by sqrt(w). It solves for beta = [intercept, slope] using numpy.linalg.lstsq with rcond set proportionally to the sample size. Residuals eps = X beta - Y are used to compute SSR (sum of squared residuals). The variance-covariance matrix is estimated as SSR / (n - 2) * inv(X^T X), where n is the number of observations (length of x). R2 is computed as 1 - SSR / TSS where TSS is the weighted total sum of squares of Y around its weighted mean.
        Defaults: w defaults to the scalar 1 to select OLS; y is converted to numpy.float64 internally.
        Side effects: None persistent; the function performs in-memory numpy operations and returns new numpy arrays and a Python float. It does not modify its input arguments.
        Failure modes and warnings: If n <= 2, the denominator (n - 2) in the variance estimate is zero or negative leading to division by zero or nonsensical variance estimates. If X^T X is singular (for example, when all x values are identical) numpy.linalg.inv will raise a numpy.linalg.LinAlgError. If TSS == 0 (for example, constant weighted Y), R2 will be undefined (division by zero) and may be numpy.nan or numpy.inf. For WLS, interpret R2 with caution (see Willett & Singer, 1988); the weighted R2 may not have the same interpretation as OLS R2. The function relies on numpy linear algebra routines and thus may raise numpy.linalg.LinAlgError or propagate floating-point warnings from numpy.
    
    Returns:
        tuple: A 3-tuple with the following entries returned in order and their practical roles in chemical data analysis:
            beta (numpy.ndarray): Length-2 array [intercept, slope] containing the estimated regression parameters. In kinetics applications the slope may correspond to a rate constant or linearized rate parameter and the intercept to an offset or initial value in the transformed domain.
            vcv (numpy.ndarray): 2x2 variance-covariance matrix for the parameter estimates computed as SSR / (n - 2) * inv(X^T X) using the (possibly weighted) design matrix. Diagonal entries give estimated variances of intercept and slope; off-diagonals give their covariance. These estimates depend on the residual variance estimate and are unreliable if n is small or model assumptions are violated.
            R2 (float): Coefficient of determination computed as 1 - SSR / TSS using the weighted Y values. R2 indicates the proportion of weighted variance in Y explained by the linear model; for WLS its interpretation is subtler and should be used in conjunction with other diagnostics.
    
    References:
        Standard texts on least squares and the cautionary note about R2 in WLS: Willett, John B., and Judith D. Singer. "Another cautionary note about R2: Its use in weighted least-squares regression analysis." The American Statistician 42.3 (1988): 236-238.
    """
    from chempy.util.regression import least_squares
    return least_squares(x, y, w)


################################################################################
# Source: chempy.util.regression.least_squares_units
# File: chempy/util/regression.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def chempy_util_regression_least_squares_units(
    x: numpy.ndarray,
    y: numpy.ndarray,
    w: numpy.ndarray = 1
):
    """Units-aware least-squares fit for a data series, returning unit-aware parameter estimates, the variance-covariance matrix, and the coefficient of determination. This function is intended for chemistry-related numeric fitting tasks (for example fitting concentration vs time, integrated rate laws, or other measured quantities from the README examples) where the input arrays may carry physical units. The implementation first extracts units from x, y, and optional weights w, converts the numeric data to unitless values for the underlying numerical least-squares solver, and then re-attaches appropriate units to the fitted parameters before returning them. The function validates compatibility between the units of y and the units of w when explicit weights are provided.
    
    Args:
        x (numpy.ndarray): Independent-variable data (array-like) for the fit. In chemical applications this typically represents measured quantities such as time, temperature, or concentration. The function reads the unit of x via unit_of(x) and converts x to unitless numbers with to_unitless(x, x_unit) before performing the numerical fit. The unit of x is used when producing unit-aware fitted parameters so that slopes, intercepts, or other coefficients have consistent units.
        y (numpy.ndarray): Dependent-variable data (array-like) corresponding to x, e.g. concentration or signal. The function reads the unit of y via unit_of(y) and converts y to unitless numbers with to_unitless(y, y_unit) for the numerical fit. The y unit is required to validate and interpret any provided weights and to construct the units of the returned fit parameters.
        w (numpy.ndarray): Optional weights or explicit error specification for each data point. Default is 1 (the integer 1), which signals that no explicit weights are provided and an unweighted least-squares fit is performed. If an explicit weights array is supplied, its units must be compatible with the data: either unit_of(w) == y_unit ** -2 (weights proportional to 1/variance with units matching y^-2), in which case the weights are converted to unitless values via to_unitless(w, y_unit ** -2), or unit_of(w) == unit_of(1) (unitless weights), in which case the array is used as-is. If w is provided with incompatible units, the function raises a ValueError indicating incompatible units between y and w. Note that the code distinguishes the default unweighted case by comparing w to the integer 1 (the literal default).
    
    Returns:
        tuple: A 3-tuple (beta_tup, vcv, r2) where:
            beta_tup: tuple containing the fitted parameter values with units restored according to the original x and y units. These are the parameters returned by the internal helper _beta_tup(beta, x_unit, y_unit) and represent the practical fit results (for example intercept and slope for a linear fit) expressed with appropriate physical units for use in chemical calculations and reporting.
            vcv: numpy.ndarray containing the variance–covariance matrix for the fitted parameters. The matrix has shape (n_params, n_params) where n_params is the number of fitted parameters in beta_tup; it describes the estimated covariances between fitted parameters produced by the underlying least_squares routine on the unitless data.
            r2: float giving the coefficient of determination computed by the underlying least_squares routine, cast to a Python float for convenience. This value is returned after unitless fitting and is intended as a numeric goodness-of-fit indicator for the converted data.
    
    Behavior, defaults, and failure modes:
        - The function is unit-aware: it uses unit_of(...) to inspect units and to_unitless(...) to convert data to unitless numbers before calling the numerical least-squares solver least_squares.
        - Default behavior with w equal to the integer 1 performs an unweighted least-squares fit.
        - If explicit weights are provided, they must either be unitless or have units equal to y_unit**-2. If weights have incompatible units a ValueError("Incompatible units in y and w") is raised.
        - No input arrays are mutated; all unit conversions create unitless views or copies for the numerical solver, and the returned beta_tup contains new unit-aware parameter objects.
        - The returned variance–covariance matrix corresponds to the parameters obtained from fitting the unitless-converted data; users should interpret vcv in the context of the parameter units given in beta_tup.
        - The function relies on the presence and semantics of unit handling utilities (unit_of and to_unitless) and the underlying least_squares implementation; missing or incompatible unit backends or a failing least_squares call may propagate exceptions from those utilities.
    """
    from chempy.util.regression import least_squares_units
    return least_squares_units(x, y, w)


################################################################################
# Source: chempy.util.stoich.decompose_yields
# File: chempy/util/stoich.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def chempy_util_stoich_decompose_yields(yields: dict, rxns: list, atol: float = 1e-10):
    """Decomposes a target vector of product yields into a linear combination of provided mass-action reactions.
    
    This function is used in stoichiometric/kinetic modelling (as in ChemPy) to express a desired production pattern (yields) for a set of chemical species as a linear combination of the net stoichiometric vectors of given Reaction instances. Mathematically it formulates A k = y, where A is the net-stoichiometry matrix (n_species x n_reactions), k is the vector of effective rate coefficients (unknowns), and y is the target yields vector. The computation is performed in a least-squares sense using numpy.linalg.lstsq and returns the effective coefficients that, when combined with the provided reactions, reproduce the yields within a specified absolute tolerance. This is useful when converting non-integer or composite product distributions into contributions from discrete production reactions for downstream kinetic or mass-balance modelling.
    
    Args:
        yields (OrderedDict): Mapping of species names to target yields (values). The keys are the species identifiers that should appear among the substances handled by the provided reactions; the order of yields.keys() determines the row ordering used to build the stoichiometry matrix A (so use an OrderedDict when deterministic ordering is required). Values may be plain numeric scalars or quantities with units; the function will take the unit of the first yield value and convert the right-hand side to unitless for the linear algebra solve, and the returned coefficients will be multiplied by that same unit.
        rxns (iterable of Reaction): Iterable of Reaction instances that provide the available production/consumption stoichiometries. The union of substance keys from these reactions defines the set of substances known to the decomposition routine. Each Reaction contributes a net stoichiometric column to the matrix A used to solve for effective coefficients. All species named in yields must be present in at least one of these reactions, otherwise a ValueError is raised.
        atol (float): Absolute tolerance for residuals (default 1e-10). After solving the least-squares problem, the residual sum(s) squared reported by numpy.linalg.lstsq are compared against this tolerance; if any residual exceeds atol a ValueError is raised. Choose a tolerance appropriate for the numeric scale and units of your yields.
    
    Returns:
        numpy.ndarray: 1-dimensional array of effective rate coefficients k (length equal to number of provided reactions). If the input yield values carried a unit, the returned numpy array is multiplied by that same unit (the unit of the first yield value), so elements become quantities with that unit. The coefficients are the least-squares solution that reconstructs the yields from the reactions' net stoichiometries within atol.
    
    Behavior, side effects, defaults, and failure modes:
        - The function first checks that every key in yields is present among the substances appearing in rxns; missing species cause an immediate ValueError with a message identifying the missing key.
        - A ReactionSystem is constructed from rxns and the union of reaction substance keys; its net_stoichs(yields.keys()) method is used to assemble the matrix A with rows ordered according to yields.keys().
        - The right-hand side vector b is constructed from yields.values() in the same order. If values carry units, the unit of the first element is used to convert b to unitless numbers for the solve; this implies that mixing incompatible units in yields may produce incorrect results or require prior unit normalization by the caller.
        - The least-squares solve is performed on a float64 copy of A.T (via numpy.linalg.lstsq) with an rcond chosen relative to matrix size. Numerical rounding/conditioning may affect results for ill-conditioned stoichiometry matrices; users should inspect the returned residuals when needed.
        - If numpy.linalg.lstsq fails to converge or raises an error, numpy.linalg.LinAlgError may be propagated.
        - After the solve, if any reported residual (sum of squared residuals per solution) exceeds atol a ValueError("atol not satisfied") is raised.
        - The default atol is 1e-10; adjust this if your yields are large in magnitude or if you expect larger numerical residuals.
        - No other external side effects occur (no file I/O or global state modification).
    """
    from chempy.util.stoich import decompose_yields
    return decompose_yields(yields, rxns, atol)


################################################################################
# Source: chempy.util.stoich.get_coeff_mtx
# File: chempy/util/stoich.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def chempy_util_stoich_get_coeff_mtx(substances: list, stoichs: list):
    """chempy.util.stoich.get_coeff_mtx computes the net stoichiometry coefficient matrix for a set of chemical reactions. It converts a sequence of reaction stoichiometries (each given as a pair of reactant and product dictionaries mapping substance keys to stoichiometric coefficients) into a 2-D integer NumPy array where each row corresponds to a substance and each column corresponds to a reaction. This matrix is the standard coefficient matrix used in balancing stoichiometric equations and in forming linear systems for chemical-equilibrium and reaction-network calculations (for example, as an intermediate in balance_stoichiometry and related routines in ChemPy).
    
    Args:
        substances (list): Ordered list of substance keys. Each entry identifies a substance (for example a chemical formula string or any hashable key used consistently in the stoichiometry dictionaries). The order of this list defines the row order of the returned matrix: row i corresponds to substances[i]. If a substance does not appear in a particular reaction pair, it is treated as having coefficient zero for that reaction.
        stoichs (list): List of reaction stoichiometries, one element per reaction, in the same column order as the desired output. Each element must be a pair (reactant_dict, product_dict), where each dict maps the same kind of substance keys (as used in substances) to stoichiometric coefficients (integers are expected). The function iterates over this sequence and uses each pair to compute the net production minus consumption for each substance.
    
    Returns:
        numpy.ndarray: Integer 2-D array of shape (len(substances), len(stoichs)). Entry (i, j) equals prod.get(substances[i], 0) - reac.get(substances[i], 0) where (reac, prod) is the j-th pair from stoichs. Positive values indicate net production of the substance in that reaction, negative values indicate net consumption, and zero indicates no net change. The array is created with NumPy integer dtype so returned values are integers.
    
    Behavior, side effects, and failure modes:
        The function allocates and returns a freshly created NumPy array; it has no other side effects. It performs a nested loop over substances (rows) and reactions (columns) and thus runs in O(len(substances) * len(stoichs)) time and uses O(len(substances) * len(stoichs)) memory for the result.
        The function relies on each element of stoichs being a two-element sequence that can be unpacked into (reac, prod). If an element of stoichs cannot be unpacked into exactly two items, a ValueError (unpacking error) will be raised by the interpreter. Each reac and prod is expected to implement the mapping interface used by dict.get; if they do not provide a .get method, an AttributeError will be raised at runtime.
        Stoichiometric coefficients in the mapping values are expected to be integers. Because the returned array is created with integer dtype, non-integer numeric values will be cast to integers by NumPy (with the usual truncation behavior), potentially losing fractional information. Values or objects that cannot be coerced to integers by NumPy (for example symbolic objects or arbitrary Python objects) will raise an exception during array assignment. To preserve non-integer or symbolic coefficients, call sites should use a different representation or modify this function accordingly.
        Keys in substances are compared to keys in the reactant/product dictionaries using standard Python equality and hashing semantics; mismatched key representations (for example differing string capitalization or different key types) will be treated as distinct and may result in rows of zeros.
    """
    from chempy.util.stoich import get_coeff_mtx
    return get_coeff_mtx(substances, stoichs)


################################################################################
# Source: chempy.util.table.render_tex_to_pdf
# File: chempy/util/table.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def chempy_util_table_render_tex_to_pdf(
    contents: str,
    texfname: str,
    pdffname: str,
    output_dir: str,
    save: bool
):
    """Generates a PDF file by writing LaTeX source to a .tex file and invoking the external pdflatex program twice. This utility is used in ChemPy (for example in table rendering, documentation/examples and generating figures for notebooks or reports) to convert LaTeX-formatted contents into a PDF file using the system pdflatex executable. The function writes the provided LaTeX source to a file named texfname inside output_dir (or a temporary directory if output_dir is None), runs pdflatex two times in batch mode while capturing stdout/stderr to a logfile, and returns the filesystem path to the generated PDF. It also implements configurable post-processing behavior via save to either remove temporary files, keep them, or copy the PDF to another location.
    
    Args:
        contents (str): LaTeX source text to render. This is the literal contents written to the .tex file named by texfname and represents the document (for example a table or figure) that ChemPy users want to convert to PDF for inclusion in documentation, examples, or reports.
        texfname (path): Filename (not full path) for the .tex file to be created inside output_dir. Typical usage is a basename such as "table.tex"; the function will join output_dir and texfname to form the full path where contents is written.
        pdffname (path): Filename (not full path) for the resulting .pdf file inside output_dir. Typical usage is a basename such as "table.pdf"; the function will join output_dir and pdffname to form the full path returned on success.
        output_dir (path): Directory in which to write the .tex file, run pdflatex, and place the generated .pdf and logfile. If output_dir is None, the function creates a temporary directory using tempfile.mkdtemp() and uses that directory for all files; a flag created_tempdir is set internally so the function can decide whether to remove the directory later depending on save.
        save (path or bool or str(bool)): Controls post-processing and persistence of generated files.
            If save is True or the string "True", no cleanup is performed and all files (tex, pdf, logfile) are left in output_dir.
            If save is False or the string "False", and output_dir was created as a temporary directory by this call, that temporary directory (and all files it contains) is removed before the function returns.
            Otherwise save is interpreted as a filesystem path (string) to which the generated PDF should be copied; in that case the PDF is copied to that path if it is not the same file as the generated pdfpath. When save is a path the original output_dir (including the generated files) is retained (no automatic deletion).
    
    Behavior and side effects:
        The function writes contents to the file os.path.join(output_dir, texfname) using text mode and UTF-8 default encoding behavior of open(). It constructs the PDF output path as os.path.join(output_dir, pdffname) and opens a logfile at pdfpath + ".out" in binary write mode; both pdflatex stdout and stderr are redirected to this logfile.
        The function invokes the external command ["pdflatex", "-halt-on-error", "-interaction", "batchmode", texfname] twice using subprocess.Popen with cwd=output_dir. The two return codes are summed; if the aggregate return code is non-zero the function raises RuntimeError with a message that includes the full command string and the aggregated exit status. Successful execution returns the full path to the generated PDF (the pdfpath inside output_dir).
        If output_dir is None the function creates a temporary directory and sets created_tempdir; whether that temporary directory is removed is controlled by save as described above.
        The logfile at pdfpath + ".out" contains combined stdout and stderr from both pdflatex invocations and is useful for diagnosing LaTeX errors when RuntimeError is raised.
    
    Failure modes and exceptions:
        If pdflatex is not available in the system PATH or cannot be executed, the subprocess invocation will raise an OSError/FileNotFoundError (propagated to the caller).
        If pdflatex returns a non-zero exit status (summed over the two runs), the function raises RuntimeError with a message describing the command and exit status; in that case the logfile (pdfpath + ".out") and other files may still be present in output_dir for debugging unless save was explicitly False and output_dir was a created temporary directory (in which case the temporary directory is removed).
        Errors while copying the PDF to the save path (when save is a path) or while removing the temporary directory (when filesystem permissions prevent deletion) will propagate the corresponding exceptions (for example shutil.Error, OSError) to the caller.
    
    Returns:
        str: The full filesystem path to the generated PDF file (os.path.join(output_dir, pdffname)). On success the caller can use this path to access the PDF. If the function raises an exception no path is returned.
    """
    from chempy.util.table import render_tex_to_pdf
    return render_tex_to_pdf(contents, texfname, pdffname, output_dir, save)


################################################################################
# Source: chempy.util.terminal.limit_logging
# File: ../../../../../opt/conda/lib/python3.10/contextlib.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def chempy_util_terminal_limit_logging(max_lvl: int = 50):
    """chempy.util.terminal.limit_logging provides a context manager that temporarily raises the global logging threshold to suppress logging messages up to and including a specified numeric level. It is intended for use in ChemPy workflows where noisy log output from numerical integrators, equilibrium solvers, or optional backend libraries (see README: ODE solver front-end, equilibria solvers, and optional backends) should be silenced for clarity during tests, example runs, or user scripts.
    
    Args:
        max_lvl (int): Numeric logging threshold to disable. This integer is forwarded to logging.disable and therefore follows the semantics of the standard Python logging levels (for example, logging.CRITICAL == 50). The default value in the function signature is 50, which corresponds to logging.CRITICAL and will suppress all messages at that level and below while the context is active. The parameter controls the global logging state for the process; passing a lower value will allow fewer messages through and a higher value will suppress more.
    
    Returns:
        None: This function is a context manager and does not return a value to the caller. Instead, it yields control to the with-block while having the side effect of setting logging.disable(max_lvl). On exit from the context (including when an exception is raised within the with-block) the original global disable value is restored by writing back the saved integer retrieved from logging.root.manager.disable, ensuring the previous logging behavior is reestablished.
    
    Behavior and side effects:
        - On entering the context, the current global logging.disable value is saved (read from logging.root.manager.disable) and replaced by the supplied max_lvl via logging.disable(max_lvl).
        - On exit, the saved original value is restored with logging.disable(_ori) even if the with-block raised an exception.
        - The change affects the global logging configuration for the entire process and therefore can suppress log messages from other modules and threads; callers should be aware that this is a process-wide change and may impact concurrent code.
        - Typical usage is to silence verbose or non-critical logging produced during numerical integrations, root-finding, or when running examples/documentation snippets as described in the ChemPy README.
    
    Failure modes and notes:
        - The function expects an integer compatible with logging.disable semantics; supplying values that are not integers or are otherwise incompatible will result in behavior dictated by the standard logging module (e.g., a TypeError or unexpected behavior from logging.disable).
        - Because the function modifies global logging state, nested uses will overwrite the global disable level; each context restores the value it saw on entry, so correct nesting order ensures values are restored as intended.
    """
    from chempy.util.terminal import limit_logging
    return limit_logging(max_lvl)


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
