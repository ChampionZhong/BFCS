"""
Regenerated Google-style docstrings for module 'molmass'.
README source: others/readme/molmass/README.rst
Generated at: 2025-12-02T00:51:59.343282Z

Total functions: 14
"""


################################################################################
# Source: molmass.analyze
# File: molmass/molmass.py
# Category: valid
################################################################################

def molmass_analyze(
    formula: str,
    maxatoms: int = 512,
    min_intensity: float = 0.0001,
    debug: bool = False
):
    """molmass.analyze
    Return a human-readable analysis of a chemical formula suitable for console display or logging.
    
    This function is used in the molmass library (a molecular mass calculation toolkit and web/console application) to produce a multi-line textual summary of a molecule described by a chemical formula. The analysis includes the formula in Hill notation when different, the empirical formula, nominal mass, average mass, monoisotopic mass and its isotopic abundance, the most abundant mass and the mean of the mass distribution when available, the number of atoms, an elemental composition table, and a mass distribution table (spectrum) when computed. Calculations are based on elemental isotopic compositions provided by the molmass element database; chemical bonding mass deficiency is not taken into account. The returned string is intended for human inspection (console output or logs) and not guaranteed to follow a stable machine-parsable format.
    
    Args:
        formula (str): Chemical formula to analyze. This is the input string parsed by molmass.Formula and may be given in conventional chemical notation (including isotopic and charged-species forms supported by molmass). The function constructs a Formula object from this string to compute masses, composition, and optionally the mass distribution. If the provided string cannot be parsed as a valid formula, behavior depends on debug: when debug is True the underlying parse exception is re-raised; when debug is False the function catches the exception and returns a single-line error message in the result string.
        maxatoms (int): Threshold for spectrum calculation. If the total atom count of the parsed formula (Formula.atoms) is strictly less than this integer, the function computes the mass distribution (spectrum) and includes it in the returned text; otherwise the spectrum is omitted to avoid expensive computations for very large molecules. Default value is 512.
        min_intensity (float): Minimum relative intensity cutoff used when computing the mass distribution (spectrum). Peaks with intensity below this fraction (for example 0.0001 for 0.01%) are excluded from the displayed spectrum. This parameter only affects output when the spectrum is computed (see maxatoms). Default value is 0.0001.
        debug (bool): Error-handling mode. If True, exceptions raised during parsing or calculation (for example invalid formula syntax or unexpected internal errors) are propagated to the caller so they can be handled or cause program termination. If False (the default), exceptions are caught and converted into a human-readable "Error: <message>" line appended to the returned string; no exception is raised.
    
    Returns:
        str: A multi-line string containing the analysis. Typical content includes (when available) the original and Hill notation formulas, empirical formula, nominal mass, average mass formatted to a precision appropriate for the magnitude, monoisotopic mass with isotopic abundance percentage, most abundant mass and mean of distribution (when spectrum computed), total atom count, a tabular elemental composition (counts, relative mass contribution, fraction percentage), and a tabular mass distribution (mass number, relative mass, fraction percentage, intensity percentage) filtered by min_intensity. If the spectrum is not computed because the atom count is >= maxatoms, the mass distribution section is omitted. If an error occurred and debug is False the returned string contains a single "Error: ..." line describing the failure. There are no other side effects (the function does not print to stdout); the string can be printed by the caller to obtain console output equivalent to the examples in the molmass README.
    """
    from molmass import analyze
    return analyze(formula, maxatoms, min_intensity, debug)


################################################################################
# Source: molmass.format_charge
# File: molmass/molmass.py
# Category: valid
################################################################################

def molmass_format_charge(charge: int, prefix: str = ""):
    """Return a standardized string representation of an ionic charge for use in
    chemical formula display, mass/charge (m/z) annotations, and related output in
    the molmass library. This function is used by higher-level objects (for
    example Formula and spectrum formatting) to append or display the net ionic
    charge of a molecule or particle in a compact, human- and machine-readable
    form.
    
    The formatting rules implemented here follow the conventions used in the
    molmass codebase and README examples: a zero charge is represented by the
    single character '0'; a charge of magnitude one is represented only by its
    sign ('+' or '-'); charges with absolute value greater than one are rendered
    as the absolute count immediately followed by the sign, optionally preceded
    by a single-character prefix when provided (for example, formatting -2 with
    prefix '_' yields '_2-'). The function has no side effects and returns a new
    string; it does not modify external state.
    
    Args:
        charge (int): Net ionic charge of the molecule or particle being
            formatted. In the chemical domain this is the integer net charge (for
            example +1 for ammonium NH4+, -2 for sulfate SO4(2-)). The function
            treats 0 specially and returns the literal '0'. Callers should pass
            an integer; passing a non-integer may raise TypeError or lead to
            undefined formatting behavior.
        prefix (str): Single-character prefix placed before the numeric magnitude
            when the absolute value of charge is greater than 1. This allows
            use-cases in the molmass ecosystem where a separator or marker is
            desired before the magnitude (for example '_' to produce '_2-'). The
            default is the empty string, which results in no prefix. If prefix is
            non-empty but abs(charge) is 0 or 1, the prefix is ignored and not
            included in the result (e.g., prefix '_' with charge 1 returns '+').
    
    Returns:
        str: The formatted charge string suitable for inclusion in chemical
        notation and display. The returned string is one of:
        '0' when charge == 0;
        '+' or '-' when abs(charge) == 1;
        '<prefix><N><sign>' when abs(charge) > 1, where <prefix> is the provided
        prefix (possibly empty), <N> is the decimal absolute value of the charge,
        and <sign> is '+' for positive and '-' for negative charges. This string
        is intended for direct concatenation with formula strings or display in
        reports and does not include whitespace.
    """
    from molmass import format_charge
    return format_charge(charge, prefix)


################################################################################
# Source: molmass.from_fractions
# File: molmass/molmass.py
# Category: valid
################################################################################

def molmass_from_fractions(
    fractions: dict[str, float],
    maxcount: int = 10,
    precision: float = 0.0001
):
    """molmass.from_fractions returns a chemical formula string derived from elemental mass fractions by converting mass-based abundances into the smallest consistent integer atom counts, using element atomic masses and isotopic masses from the package elemental database.
    
    Args:
        fractions (dict[str, float]): Mapping of element symbols or isotope identifiers to mass abundances (mass fractions or relative weights). Keys may be element symbols like 'C', 'O' or isotope forms like '30Si' or '[30Si]'. The special symbol 'D' (deuterium) is accepted and treated as the isotope '2H'. Values are numeric mass proportions; they need not be normalized (the function normalizes by the sum of provided values). This argument is used to derive relative mole counts by dividing each provided mass fraction by the corresponding atomic or isotopic mass from the internal ELEMENTS database, and thus is the primary input from which the chemical formula is inferred (for example from elemental analysis or mass-spectrometry-derived composition).
        maxcount (int): Upper bound used when searching for a small integer multiplier that converts the computed relative atom counts into integers. The routine tests integer multipliers i in range(1, maxcount) (note: the loop excludes maxcount itself) to find the factor that minimizes the summed rounding error. Larger values allow finding larger integer stoichiometries but increase computation; default is 10. This parameter therefore limits the maximum integer scaling factor explored when converting fractional atom ratios into integer atom counts for the returned formula.
        precision (float): Threshold that controls when the search for an integer multiplier may stop early. The algorithm multiplies this threshold by the number of distinct symbols present and compares it to the cumulative rounding error for a candidate multiplier; if the error is below multiplier * (precision * number_of_symbols) the multiplier is accepted. The default is 1e-4. This value tunes the tolerance for accepting near-integer ratios and affects whether small deviations (due to measurement noise or rounding) are tolerated in the inferred integer formula.
    
    Returns:
        str: A chemical formula string constructed from the inferred integer atom counts. The returned string is deterministic: element/isotope tokens are sorted lexicographically by their symbol (isotopes are represented in bracketed form like '[30Si]'). If the input mapping is empty the function returns the empty string ''. Counts are omitted when equal to 1 (symbol alone implies one atom), and integer counts greater than 1 are appended directly after the symbol (e.g., 'H2O', 'O[2H]2'). Note that if rounding yields zero for a symbol, the implementation will still append the symbol without a numeric suffix, which is interpreted as a count of one in the output.
    
    Behavior, defaults, and failure modes:
        The function first normalizes the provided mass fractions by their sum, then converts each normalized mass fraction into a relative mole count by dividing by the element's or isotope's atomic mass as obtained from the package ELEMENTS database. For element symbols that begin with a lowercase character or that are provided in bracketed form (e.g., '[30Si]' or '30Si') the leading digits are parsed as an isotope mass number and the corresponding isotopic mass is used. The special key 'D' is mapped to '2H' (deuterium). After computing relative counts, all counts are divided by the smallest count to produce ratios, and the routine searches for a small integer multiplier (bounded by maxcount) that makes these ratios near-integers within the precision tolerance scaled by the number of symbols.
        The function has no side effects beyond reading the internal ELEMENTS database. If an element symbol or isotope identifier is not found in the ELEMENTS database, the function raises FormulaError describing the unknown element or isotope. If no multiplier within the tested range produces exact integers within the specified precision, the function returns the formula based on the best multiplier found (the one minimizing total rounding error). The function does not account for mass deficiency due to chemical bonding; it treats input values purely as mass fractions and converts to atom counts using tabulated atomic/isotopic masses.
    """
    from molmass import from_fractions
    return from_fractions(fractions, maxcount, precision)


################################################################################
# Source: molmass.from_oligo
# File: molmass/molmass.py
# Category: valid
################################################################################

def molmass_from_oligo(sequence: str, dtype: str = "ssdna"):
    """Return the chemical formula for a polymer composed of unmodified (deoxy)nucleotides derived from a DNA or RNA sequence. This function constructs a Hill-notation style chemical formula string that represents the polymer used in molmass molecular-mass and composition calculations. Each strand produced by this function includes a 5' monophosphate; single-stranded polymers have one appended water unit (H2O) and double-stranded polymers have two appended water units ((H2O)2) as part of the returned grouped formula. The resulting string is suitable for feeding into molmass.Formula or other molmass parsers to compute average/monoisotopic mass, elemental composition, and isotopic spectra. The function also accepts and preserves an optional ionic charge suffix in the input sequence (for example "G_2+"), which is parsed and appended to the returned formula.
    
    Args:
        sequence (str): DNA or RNA sequence for the polymer. Whitespace in the sequence is ignored before parsing. A trailing ionic charge suffix may be included in the sequence (examples in the package use forms such as "_2+"); the function uses the package's charge-parsing helper to extract and reapply that charge to the returned formula. The sequence is interpreted using the nucleotide sets appropriate for the selected dtype; invalid or unrecognized characters will cause the underlying parsing routines to raise an exception.
        dtype (str): Nucleic acid sequence type, case-insensitive. One of 'ssdna', 'dsdna', 'ssrna', or 'dsrna'. This parameter determines two orthogonal choices: whether to use deoxyribonucleotide monomer formulas (for 'ssdna' and 'dsdna') or ribonucleotide monomer formulas (for 'ssrna' and 'dsrna'), and whether to produce a single-stranded polymer (prefix 'ss') or a double-stranded polymer (prefix 'ds'). For double-stranded types the complementary strand is generated and the final formula groups both strands together and appends two water units.
    
    Returns:
        str: A chemical formula string representing the unmodified nucleotide polymer, including 5' monophosphates and water groups appropriate for single- or double-stranded polymers, and with any parsed ionic charge appended in the package's formula charge notation. The format is compatible with molmass.Formula and other molmass parsers for downstream mass, composition, and spectrum calculations.
    
    Raises:
        ValueError: If the sequence contains characters or symbols that are not valid nucleotides for the selected dtype or if the package's sequence/charge parsing helpers detect malformed input. The concrete exception may be ValueError or a more specific subclass provided by the package's parsing utilities.
    """
    from molmass import from_oligo
    return from_oligo(sequence, dtype)


################################################################################
# Source: molmass.from_peptide
# File: molmass/molmass.py
# Category: valid
################################################################################

def molmass_from_peptide(sequence: str):
    """Return chemical formula for a peptide polymer built from unmodified amino acids.
    
    This function converts an amino acid sequence into a chemical formula string that represents the corresponding peptide polymer. It is intended for use with the molmass library to compute molecular mass, elemental composition, and mass distribution by passing its output to molmass.Formula. The function accepts typical single-letter amino acid sequences, tolerates embedded spaces for readability, and recognizes trailing ionic charge descriptors as accepted by the package (for example, rdkit-style descriptors such as "_2+"). Internally the function removes whitespace, extracts any trailing charge with split_charge, maps residues to their residue formulas using the AMINOACIDS mapping via from_sequence, appends "H2O" to represent the polymer termini/hydration stoichiometry, and then reattaches any parsed charge with join_charge. The returned string uses the same formula syntax expected by molmass.Formula (parentheses, optional bracketed charge notation).
    
    Args:
        sequence (str): Amino acid sequence for which to build a peptide chemical formula. The sequence may contain embedded spaces (they are removed) and may include a trailing charge descriptor (for example "_2+" or "2+"); such a descriptor will be parsed and encoded in the returned formula. The sequence must use standard single-letter amino acid symbols that are present in the AMINOACIDS residue mapping used by molmass; sequences containing unknown symbols will cause the underlying from_sequence routine to raise an error (typically FormulaError or ValueError) indicating invalid residue symbols. The function does not modify the original sequence object; it only returns a new formula string.
    
    Returns:
        str: A chemical formula string representing the peptide polymer built from the provided sequence. The formula string encloses the concatenated residue formulas and the appended H2O in parentheses and, if a charge was parsed from the input, includes the corresponding charge annotation (for example, '((C2H3NO)2H2O)' for two glycines or '[C107H159N29O30S2]2+' for a charged sequence). This string is suitable as input to molmass.Formula to compute mass, composition, and spectrum. The function has no side effects beyond returning this string.
    """
    from molmass import from_peptide
    return from_peptide(sequence)


################################################################################
# Source: molmass.from_sequence
# File: molmass/molmass.py
# Category: valid
################################################################################

def molmass_from_sequence(sequence: str, groups: dict[str, str]):
    """Convert a biological sequence (DNA, RNA, or peptide) to a chemical formula string in Hill notation.
    
    Args:
        sequence (str): A sequence of single-character item codes representing monomers in a biological polymer (for example, single-letter amino acid codes for peptides or single-letter nucleotide codes for DNA/RNA). The function iterates the string one character at a time; each character must be a key in the groups mapping. The sequence may include an ionic charge using rdkit-style or other supported charge notation; any such charge is parsed and preserved (see notes on charge handling below). Examples of valid domain usage are converting a peptide like 'ACDE' or a nucleotide sequence like 'ATG' into a concatenated chemical formula for downstream molecular mass, elemental composition, or mass spectrum calculations performed by the molmass library.
        groups (dict[str, str]): Mapping from sequence item (single-character string) to its chemical formula given in Hill notation (string). Typical values for this argument are the module mappings DEOXYNUCLEOTIDES, NUCLEOTIDES, or AMINOACIDS which map item codes (e.g., 'A', 'C', 'G', 'T', amino-acid single letters) to the corresponding group chemical formula strings. The function requires that every character in sequence exists as a key in this mapping; keys and values must be strings.
    
    Behavior and important details:
    - The function counts occurrences of each sequence item (single-character key) and constructs the output by concatenating group formulas according to those counts.
    - For each key in groups, processed in deterministic sorted order of the mapping keys (lexicographic order of the keys), the corresponding group formula is emitted in parentheses. If the count for that key is 1, the number is omitted (for example, '(B)'); if the count is greater than 1 the count is appended directly after the parenthesized group formula (for example, '(B)2'). Group ordering in the output follows sorted(groups) and therefore is deterministic but is not the same as chemical-element Hill ordering unless the group keys are chosen to produce that effect.
    - The function treats the input string as a sequence of single-character items. It does not perform multi-character tokenization. If a sequence item is not present as a key in groups the function will raise a KeyError.
    - The input sequence is first passed to split_charge which extracts and returns any ionic charge notation present in the sequence; the computed formula string is then passed to join_charge to append the same charge notation to the returned formula. This preserves ionic charge notation supported by the molmass package (including rdkit-style ionic charges) so downstream mass/charge (m/z) and composition calculations remain consistent.
    - Empty sequence: if sequence contains no item characters, the function produces an empty concatenated formula (''), except that any parsed ionic charge is still appended by join_charge so the return may contain only the charge notation. This behavior enables constructing charged formulas even from an empty monomer list.
    - Types and errors: the function expects sequence to be a str and groups to be a dict[str, str]. If types differ, a TypeError or other type-related exception may be raised by Python. KeyError is raised when sequence contains characters not present in groups. The function has no side effects (it does not modify the input mapping) and performs only string construction.
    
    Practical significance:
    - This helper converts biological sequences into a concise chemical formula representation suitable for use with the molmass Formula class and other molmass routines that compute average mass, monoisotopic mass, elemental composition, and mass spectra. By supplying group formulas in Hill notation (e.g., via DEOXYNUCLEOTIDES, NUCLEOTIDES, or AMINOACIDS), users can translate sequence-level information into the chemico-physical inputs required for molecular-mass calculations performed by the library.
    
    Returns:
        str: A chemical formula string in Hill notation formed by concatenating the mapped group formulas with counts appended (parenthesized group formulas, count omitted for single occurrences). Any ionic charge parsed from the input sequence is appended using the library's join_charge semantics. The returned string can be passed directly to other molmass APIs (for example, Formula) to compute mass, composition, or spectra.
    """
    from molmass import from_sequence
    return from_sequence(sequence, groups)


################################################################################
# Source: molmass.join_charge
# File: molmass/molmass.py
# Category: valid
################################################################################

def molmass_join_charge(formula: str, charge: int, separator: str = ""):
    """Return a chemical formula string with an ionic charge suffix suitable for use
    in molmass calculations, web output, and textual representations of charged
    species.
    
    Args:
        formula (str): Chemical formula without charge. This is the core formula
            string describing the molecular composition (for example "C8H10N4O2")
            that will be augmented with an ionic charge notation. The string is
            inserted either inside square brackets (when separator is the empty
            string) or left as-is (when a non-empty separator such as '_' is
            provided). The function does not modify element ordering or validate
            chemical correctness of the formula; it only modifies the textual
            representation to include the charge.
        charge (int): Charge number of the species. Zero indicates a neutral
            species and leaves the input formula unchanged. Non-zero integers are
            formatted as a charge suffix where the sign character is '+' for
            positive charges and '-' for negative charges. The magnitude '1' is
            omitted when formatting a single charge (e.g. 1 -> '+', -1 -> '-'),
            while magnitudes greater than 1 are shown before the sign
            (e.g. 2 -> '2+', -2 -> '2-'). This integer value is used directly to
            produce the textual suffix appended to the formula and is the standard
            ionic notation used elsewhere in the molmass library for charged ions.
        separator (str): Character separating the formula from the formatted charge.
            Allowed values are the empty string '' (the default) or '_' as used in
            molmass code and outputs. If separator is the empty string, the input
            formula is wrapped in square brackets and the formatted charge is
            appended immediately after the closing bracket (for example
            '[Formula]2+'); if separator is '_' the separator is inserted between
            the formula and the formatted charge without adding brackets
            (for example 'Formula_2-'). The default empty string produces the
            bracketed representation commonly used for ionic formulas.
    
    Returns:
        str: A new formula string with the ionic charge appended when charge != 0.
        If charge == 0 the original input string is returned unchanged. The return
        value is intended for display, parsing by other molmass functions, and
        export in reports or spectra where a standardized charge notation is
        required.
    
    Behavior and failure modes:
        The function is pure and has no side effects. It performs simple string
        operations and relies on Python's f-string formatting to construct the
        result. If arguments are not of the expected types (formula: str,
        charge: int, separator: str) or if a separator other than '' or '_' is
        supplied, the function will still attempt to format the result but may
        produce an unexpected representation; supplying non-string types for
        `formula` or `separator` or a non-integer `charge` can raise standard
        Python runtime errors (for example TypeError) from string operations. The
        function does not perform chemical validation of the formula and does not
        account for mass deficiency due to bonding; it only creates the textual
        representation used elsewhere in the molmass library.
    """
    from molmass import join_charge
    return join_charge(formula, charge, separator)


################################################################################
# Source: molmass.mass_charge_ratio
# File: molmass/molmass.py
# Category: valid
################################################################################

def molmass_mass_charge_ratio(mass: float, charge: int):
    """molmass.mass_charge_ratio returns the mass-to-charge ratio (m/z) for an ion or neutral species given its mass and integer charge. This helper is used in the molmass library and examples to convert masses (for example Formula.mass or Formula.monoisotopic_mass, expressed in unified atomic mass units / daltons) and integer charge numbers (from Formula.charge or other sources) into the m/z value used in mass spectrometry and spectrum calculations.
    
    Args:
        mass (float): Mass value to be divided by the charge magnitude. In the molmass domain this is typically a molecular or isotopic mass in unified atomic mass units (u, also called daltons) returned by functions or properties such as Formula.mass, Formula.monoisotopic_mass, or Element/Isotope mass values. The function performs no unit conversion; the returned ratio has the same mass units per unit charge. The function does not validate that mass is non-negative or finite; passing NaN or infinite values will propagate those values.
        charge (int): Integer charge number in units of the elementary charge (e). Positive integers represent cations, negative integers represent anions, and zero represents a neutral species. In practice this value is typically obtained from Formula.charge or parsed ion notation. The function expects an int per its signature; passing non-integer types is not supported by the documented API and may result in a TypeError or undefined behavior.
    
    Returns:
        float: The computed mass-to-charge ratio (m/z). If charge == 0 (neutral species) the function returns mass unchanged to avoid division by zero; otherwise it returns mass divided by the absolute value of charge (mass / abs(charge)). The returned float represents the mass in the same units as the input mass per unit charge magnitude and is suitable for use as an m/z value in mass spectrum peak position calculations.
    
    Behavior, side effects, and failure modes:
        This function is pure and has no side effects. It implements a design choice to treat neutral species (charge == 0) by returning the input mass unchanged rather than raising an exception or returning infinity; this avoids division-by-zero errors when computing m/z for neutral molecules in contexts where an m/z value is still meaningful. The sign of a nonzero charge is ignored because mass spectrometers measure the magnitude m/z; the absolute value of charge is used for division. The function does not coerce or validate units; callers must supply mass in the appropriate units used throughout molmass. Passing non-numeric or incompatible types (for example, a non-float mass or non-int charge) is outside the documented contract and may raise TypeError or produce unexpected results.
    """
    from molmass import mass_charge_ratio
    return mass_charge_ratio(mass, charge)


################################################################################
# Source: molmass.split_charge
# File: molmass/molmass.py
# Category: valid
################################################################################

def molmass_split_charge(formula: str):
    """molmass.split_charge: Extract the chemical formula with any trailing ionic charge notation removed and return the cleaned formula plus the integer net charge. This utility is used throughout the molmass library to accept user-supplied chemical formulas that include appended ionic charge annotations (for example "H2O+", "[Fe(CN)6]3-", "M_2+" or "[[M]]2-") so the core formula parsing and mass calculations operate on the neutral/formula part while the ionic charge is handled separately by mass/charge computations.
    
    Args:
        formula (str): Chemical formula string that may include an appended ionic charge notation. The string may contain bracketed formulas (leading and trailing square brackets), an underscore '_' delimiter separating a numeric count from a trailing sign, explicit numeric charge counts, or repeated '+' and '-' characters. This function expects the common molmass and RDKit-style charge annotations appended to the formula (examples: 'Formula', 'Formula+', 'Formula+2', '[Formula]2-', 'Formula_2-'). The function does not validate chemical correctness of the formula content (element symbols, counts, or parentheses); it only analyzes and removes trailing charge notation.
    
    Behavior:
        The function inspects the end of the input string using regular-expression patterns to detect three trailing patterns in order of precedence: (1) a delimiter character ']' or '_' followed by a decimal count and one-or-more sign characters; (2) a leading sign character followed by a decimal count; (3) an optional delimiter and one-or-more sign characters. When a numeric count is present it is combined with the sign to produce the integer charge (for example '2+' -> +2, '3-' -> -3). When only sign characters are present (for example '++--+'), the function computes the net charge by summing +1 for each '+' and -1 for each '-' in the matched trailing sign sequence. If the matched delimiter is '_' the trailing "_<count><signs>" is removed; if the matched delimiter is ']' the right-most ']' and preceding text are adjusted so that nested brackets are preserved (for example '[[Formula]]2-' yields '[Formula]' as the cleaned formula); if the match has no delimiter the function strips the matched sign characters from both ends of the string using str.strip, which may remove sign characters at the front as well as the back in malformed inputs. If no charge annotation is found, the original input string is returned unchanged and the charge is 0.
    
    Returns:
        tuple[str, int]: A 2-tuple where the first element is the chemical formula string with any recognized trailing charge notation removed (brackets are unwrapped if appropriate) and the second element is an integer representing the net ionic charge (positive for cations, negative for anions, zero if no charge found). For example, ('Formula', 0), ('Formula', 1), ('Formula', -2), or ('[Formula]', -2) depending on the input.
    
    Failure modes and caveats:
        - The function uses heuristic regular-expression parsing only and does not validate element symbols, stoichiometry, or bonding; callers must validate formula content separately if required.
        - In inputs where sign characters appear elsewhere in the formula (not only at the end), the behavior when only sign characters are matched can remove those characters from both ends because str.strip is used for the delimiter-less case.
        - Ambiguous or malformed charge annotations (for example 'Formula+-') are interpreted by algebraic summation of sign characters (in this example net 0) rather than raising an error.
        - No exceptions are raised for unrecognized formats; the function returns the original string and a charge of 0 when no trailing charge notation matches.
    """
    from molmass import split_charge
    return split_charge(formula)


################################################################################
# Source: molmass.elements.word_wrap
# File: molmass/elements.py
# Category: fix_docstring
# Reason: Schema parsing failed: Cannot generate JSON schema for word_wrap because the docstring has no description for the argument 'text'
################################################################################

def molmass_elements_word_wrap(
    text: str,
    linelen: int = 79,
    indent: int = 0,
    joinstr: str = ""
):
    """molmass.elements.word_wrap: Return a word-wrapped copy of a text string for use in molmass element descriptions, console output, and web display. This helper formats long description strings (for example Element.description entries shown in the README) into lines that are easier to read in terminals, logs, and simple text UIs by breaking on whitespace between words without performing hyphenation.
    
    Args:
        text (str): The input string to wrap. The function splits this string on whitespace (using text.split()), so sequences of spaces and newlines are collapsed into single separators and words are preserved. This parameter is the raw descriptive text used in molmass (for example element or compound descriptions) and is returned wrapped for improved readability. If the full text length is less than linelen, the original text is returned unchanged.
        linelen (int): Target maximum line length in characters. Default 79. The routine attempts to keep each output line shorter than this limit by adding words until adding the next word would meet or exceed linelen, at which point it starts a new line. Note that a single word longer than linelen will be placed on its own line and may therefore exceed linelen. linelen is treated as an integer character budget; passing a non-integer or otherwise inappropriate value may raise a TypeError elsewhere.
        indent (int): Number of characters to reserve for an external left indentation when computing line lengths. Default 0. This value reduces the available space per line by adjusting an internal length counter (initial llen = -indent) so callers that will prepend indentation externally can account for that space when wrapping. The function does not insert indentation characters into the returned lines; callers must add indentation themselves after wrapping.
        joinstr (str): String used to join the wrapped lines into the single returned string. Default '\n'. Use the default newline for multiline output (e.g., console or text files) or supply another separator (for example a single space to rejoin lines with spaces).
    
    Returns:
        str: A new string with words wrapped into lines according to linelen and joined using joinstr. The returned string has no leading/trailing added indentation (indent only affects internal line-length calculation). There are no side effects; the input text is not modified in place. If text is not a str, a TypeError will occur when the function attempts to call text.split(). Behavior for non-positive or otherwise out-of-range linelen or indent values is not specially handled by this function and may produce unexpected wrapping results.
    """
    from molmass.elements import word_wrap
    return word_wrap(text, linelen, indent, joinstr)


################################################################################
# Source: molmass.molmass.precision_digits
# File: molmass/molmass.py
# Category: valid
################################################################################

def molmass_molmass_precision_digits(f: float, width: int):
    """molmass.molmass.precision_digits returns the recommended number of digits after the decimal point to print a floating point value f so that the formatted number fits within a field of width characters. This helper is used by the molmass library when formatting numeric outputs (for example average mass, monoisotopic_mass, nominal_mass, m/z values in spectra, and intensity percentages shown in the console, web UI, and dataframes) so that numbers are presented compactly and consistently within fixed-width displays.
    
    Args:
        f (float): Floating point number to be displayed. The function uses the base-10 logarithm of the absolute value of f to estimate the length of the integer portion of the number and to determine how many characters remain for the fractional part. Negative values are supported (the sign consumes one character). Passing f == 0 will raise a ValueError because log10(0) is undefined. Behavior for non-finite floats (inf, -inf, nan) is not specified and may raise a math domain error or produce unexpected results.
        width (int): Maximum allowed total length in characters for the printed number, including any sign, digits before the decimal point, the decimal point itself, and digits after the decimal point. This is an integer number of characters as used by fixed-width formatting routines. If width is too small to accommodate the integer part plus sign and decimal point, the function still returns at least 1 (it guarantees at least one fractional digit), but the formatted string may exceed the requested width.
    
    Behavior and implementation notes:
        The function computes the base-10 logarithm of abs(f) to estimate the magnitude and integer part length, clamps that estimate to zero for numbers with absolute value less than 1, and subtracts the estimated integer-part length from width. It then subtracts one character for the decimal point and one additional character for the sign when f is negative (implemented as subtracting 3 for negative f, 2 for non-negative f). Finally, it ensures the returned precision is at least 1. There are no side effects; the function is pure and does not modify global state or perform I/O.
    
    Failure modes and limitations:
        - f == 0 raises ValueError because math.log(abs(f), 10) is undefined.
        - Non-finite values (math.inf, -math.inf, math.nan) are not handled explicitly and may raise errors or return meaningless values.
        - If width is too small to display the integer portion plus sign and decimal point, the function still returns 1, but a formatted representation using that precision may exceed the requested width.
        - The function assumes fixed-point decimal formatting; it does not account for scientific/exponential notation and therefore may not be appropriate if the caller intends to use exponential formatting.
    
    Returns:
        int: Number of digits after the decimal point recommended for printing f within a field of width characters. This value is >= 1. The returned integer can be passed to standard formatting functions (for example, format(value, f'.{precision}f')) to produce a fixed-point string representation.
    """
    from molmass.molmass import precision_digits
    return precision_digits(f, width)


################################################################################
# Source: molmass.web.analyze
# File: molmass/web.py
# Category: valid
################################################################################

def molmass_web_analyze(formula: str, maxatoms: int = 512, min_intensity: float = 0.0001):
    """Return an HTML fragment that contains a human-readable analysis of a chemical formula suitable for inclusion in the molmass web application.
    
    This function is part of the molmass web interface. It uses the molmass library to parse a chemical formula, compute elemental composition, and — when feasible — compute the mass distribution spectrum based on isotopic compositions (the molmass library computes average, nominal, and isotopic masses; mass deficiency due to chemical bonding is not taken into account). The returned string embeds results as HTML tables and formatted Hill/empirical formulas using HTML superscripts/subscripts so the output can be rendered directly in a web page.
    
    Args:
        formula (str): Chemical formula to analyze as a plain Python string. This is the primary input and must be a formula accepted by molmass.Formula (examples: 'C8H10N4O2', 'H2O', isotopic notation and ionic syntax supported by molmass). The function passes this string to molmass.Formula to parse atom counts, isotope specifications, and optional ionic charge. The practical role is to identify elements, atom counts, and isotopic labels for mass and spectrum calculations used by the web UI.
        maxatoms (int): Threshold for the number of atoms below which the isotopic mass distribution (spectrum) is calculated. Default is 512. If the parsed formula has f.atoms < maxatoms the function will attempt to generate a Spectrum via f.spectrum(min_intensity=min_intensity); otherwise the potentially expensive spectrum calculation is skipped and the returned HTML omits mass-distribution details. This parameter controls a practical performance trade-off in the web application to avoid excessive computation for very large molecules.
        min_intensity (float): Minimum relative intensity threshold for including peaks in the computed spectrum. Default is 0.0001 (1e-4). When a spectrum is calculated, peaks with intensity below this value are discarded; if the resulting spectrum has fewer than two peaks the function treats the spectrum as not available and omits spectrum-related sections from the HTML. This value is interpreted as a fractional intensity (not a percentage) and is forwarded to molmass.Formula.spectrum.
    
    Returns:
        str: An HTML fragment (string) containing the analysis. The returned HTML includes one or more of the following, depending on the input and computed results:
            - A results table with Hill notation, empirical formula (if different), nominal mass, average mass (when applicable), monoisotopic mass with abundance shown as a percent, number of atoms, and, when a spectrum is computed, most abundant mass and distribution mean. Numerical formatting uses molmass.precision_digits to select display precision based on computed masses.
            - An elemental composition table when the formula contains more than one element; each row lists element symbol (with numeric isotopic prefixes rendered as superscripts), atom count, relative mass contribution, and fraction percentage.
            - A mass distribution table when a spectrum was computed and contains more than one peak; each row lists mass number, relative mass, fraction percent, intensity percent, and, when applicable, m/z computed from the spectrum charge. The presence of an m/z column depends on the absolute value of the spectrum charge.
        The function never raises parsing or calculation exceptions to the caller; instead, if molmass.Formula parsing or any calculation raises an exception, analyze returns an HTML fragment that contains an error heading and escaped details suitable for display in the web UI. Thus callers always receive a string and may embed it directly into a web page.
    
    Behavior and side effects:
        - The function is pure with respect to program state (no global mutation); its observable side effect is producing an HTML string. It calls into the molmass library (molmass.Formula, composition(), spectrum(), precision_digits()) and uses regular expressions to convert chemical-formula notation to HTML with <sup> and <sub>.
        - Spectrum computation is conditional to limit CPU and memory usage: it is only attempted when f.atoms < maxatoms and only retains peaks with intensity >= min_intensity. If the computed spectrum has fewer than two peaks, it is discarded (treated as None) and no mass-distribution table is added.
        - Error handling is defensive: any Exception raised during parsing or computation is caught, its message lines are escaped for safe HTML display, and an error fragment is included in the returned string. Because exceptions are caught internally, callers should inspect the returned HTML for an error heading rather than rely on a raised exception.
        - Numeric values are formatted to a precision derived from molmass.precision_digits for mass values; percentage fields are presented with fixed decimal formatting where the code specifies it.
    
    Failure modes and limitations:
        - If the input formula is not valid according to molmass.Formula parsing rules (for example malformed element symbols, counts, or isotope/charge syntax), the function returns an HTML error fragment describing the parsing failure. Recent molmass revisions derive FormulaError from ValueError; such parsing errors are handled and included in the returned HTML.
        - Very large formulas or extreme parameter values may cause spectrum computation to be skipped (by design via maxatoms) or to be computationally expensive if maxatoms is increased; callers should use maxatoms and min_intensity to balance detail vs. performance.
        - The function does not model mass deficiency due to chemical bonding; masses and spectra are computed from element isotopic compositions provided by the molmass element database, consistent with the library behavior documented in the project README.
    """
    from molmass.web import analyze
    return analyze(formula, maxatoms, min_intensity)


################################################################################
# Source: molmass.web.cgi
# File: molmass/web.py
# Category: valid
################################################################################

def molmass_web_cgi(url: str, open_browser: bool = True, debug: bool = True):
    """Run the molmass web application as a local CGI-capable server process.
    
    This function is part of the molmass web interface and is used to serve the molmass web application (the component that renders HTML pages and handles requests to calculate molecular mass, elemental composition, and mass distribution spectra from chemical formulas as described in the project README). It either handles a single CGI request when executed inside a CGI-capable environment, or starts a local HTTP server that recognizes this module as a CGI script and serves it at the provided URL. Behavior includes normalizing the URL, changing the current working directory to the module directory, optionally opening a web browser to the URL, and enabling detailed CGI tracebacks in debug mode.
    
    Args:
        url (str): URL at which the web application is served. The function requires a fully qualified URL with a hostname and port when starting the built-in local HTTP server. The implementation will ensure the URL ends with a slash and will append the module filename (without .pyc or .pyo) so the final target points at the CGI script. Example practical use: "http://localhost:8000/molmass.cgi" (or a base URL that will be normalized).
        open_browser (bool): Open url in a web browser when starting the local HTTP server. Default True. When True the function calls the package's webbrowser helper to launch the user's default browser pointing at the normalized URL. This is a convenience for interactive use when testing or running the molmass web application locally. Note that opening a browser is a side effect and may fail for headless environments; the server will still be started even if the browser cannot be opened.
        debug (bool): Enable debug mode. Default True. When True this function enables cgitb (detailed CGI tracebacks) which causes detailed error pages and tracebacks to be written to standard output on exceptions during CGI request handling; this aids debugging of response generation and request parsing for the molmass web application. In non-debug mode, tracebacks are not enabled and error output is less verbose.
    
    Detailed behavior and side effects:
        - If the environment variable SERVER_NAME is set (the typical sign the process is running as a CGI script under a web server), the function treats the invocation as a single CGI request: it writes the HTTP content-type header to stdout, parses form/request fields using cgi.FieldStorage(), adapts request.get to return the first value (request.getfirst), calls the package response(request, url) function to obtain HTML output for the molmass web application, prints that HTML to stdout, and then returns the integer exit code described below. This branch is used when the module is invoked by an external web server to compute and return molecular mass results for a single request.
        - Otherwise, the function starts a local HTTP server using http.server.CGIHTTPRequestHandler and HTTPServer. It monkey-patches CGIHTTPRequestHandler.is_cgi so that requests whose path contains the module filename are treated as CGI requests served by this module. It prints a diagnostic message ("Running CGI script at", url), optionally opens the normalized URL in the user's web browser (if open_browser is True), parses the URL with urllib.parse.urlparse and validates that a hostname and port are present, and then calls HTTPServer(...).serve_forever() to run a blocking server loop that will handle incoming HTTP requests and dispatch CGI invocations of this module. This branch is intended for local development or demonstration of the molmass web interface.
        - The function changes the current working directory (os.chdir) to the directory that contains this module so that relative imports and module filename resolution for CGI execution work correctly. This is a permanent side effect for the running process.
        - The function normalizes the module filename to avoid using compiled file extensions (.pyc, .pyo) when constructing the CGI script URL.
        - The function prints status and HTTP headers to standard output in the CGI branch and prints status messages to standard output in the local server branch. The local server block is blocking: serve_forever() will run until interrupted (KeyboardInterrupt) or until an error occurs.
    
    Failure modes and exceptions:
        - ValueError is raised if the provided url does not contain both a hostname and a port when attempting to start the built-in HTTP server (the code parses the URL and requires urlparse(url).hostname and .port to be non-None).
        - OSError or other exceptions may be raised when attempting to change the current working directory (os.chdir) or when binding the server socket (e.g., if the port is already in use).
        - Exceptions raised by response(request, url) (the function that generates HTML for the molmass web application) will propagate; when debug is True cgitb will cause detailed tracebacks to be printed to standard output for CGI invocations.
        - Opening the browser is a best-effort side effect and may fail silently or raise exceptions depending on the platform and environment (for example in headless servers).
        - The function will block while serving requests in the local server mode; normal return only occurs if the server stops (due to an error or explicit shutdown).
    
    Returns:
        int: Exit code value. The function returns 0 on successful setup/completion. Note that when running the built-in HTTP server the function normally does not return because serve_forever() blocks; in the CGI invocation branch the function will complete the request/response cycle and return 0. The return value is intended as a conventional process exit code for callers that invoke this function in scripting contexts.
    """
    from molmass.web import cgi
    return cgi(url, open_browser, debug)


################################################################################
# Source: molmass.web.webbrowser
# File: molmass/web.py
# Category: valid
################################################################################

def molmass_web_webbrowser(url: str, delay: float = 1.0):
    """molmass.web.webbrowser: Open a URL in the system default web browser after a short delay.
    
    This helper function is used by the molmass web application to launch the web-based user interface (for example when running the package with the web server option). It schedules a background timer that calls the standard library webbrowser.open() on the provided URL after the specified delay. The short delay is intended to give a locally started web server time to bind to its port before the browser attempts to connect.
    
    Args:
        url (str): URL to open in the web browser. In the molmass context this is typically the local address served by the molmass web application (for example "http://127.0.0.1:5000/"), but any valid URL string accepted by the Python standard library webbrowser.open() may be supplied. The function will pass this string verbatim to webbrowser.open(); it does not validate or canonicalize the URL.
        delay (float): Delay in seconds before opening the web browser. A non-negative float; the default is 1.0 which is used to allow the molmass Flask server (if used) to start. The function schedules the open call with threading.Timer and returns immediately; the delay only affects when the background thread calls webbrowser.open().
    
    Behavior and side effects:
        The function creates and starts a threading.Timer object that will invoke webbrowser.open(url) after delay seconds. It does not block the caller and returns immediately. The actual act of opening the browser is performed by the standard library webbrowser module and therefore depends on the host environment, the presence of a graphical desktop, and the system default browser configuration. If the environment does not support opening a browser (for example a headless server without a GUI), webbrowser.open() may fail or be unable to display any page; such failures are governed by webbrowser and the underlying OS and are not explicitly handled here. Exceptions raised by webbrowser.open() (rare) occur in the background thread and are not propagated to the caller. The function does not provide mechanisms to cancel the timer once started.
    
    Returns:
        None: This function returns None and its purpose is the side effect of scheduling a browser-open operation on a background thread. The caller should not expect a return value or synchronous confirmation that the browser was opened.
    """
    from molmass.web import webbrowser
    return webbrowser(url, delay)


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
