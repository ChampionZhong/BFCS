"""
Regenerated Google-style docstrings for module 'cirpy'.
README source: others/readme/cirpy/README.rst
Generated at: 2025-12-02T00:49:30.642798Z

Total functions: 6
"""


################################################################################
# Source: cirpy.construct_api_url
# File: cirpy.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def cirpy_construct_api_url(
    input: str,
    representation: str,
    resolvers: list = None,
    get3d: bool = False,
    tautomers: bool = False,
    xml: bool = True,
    **kwargs
):
    """Construct and return the CIR (Chemical Identifier Resolver) API URL for a desired resolution request.
    
    This function is used by CIRpy to build the exact HTTP request URL that will be sent to the CIR web service (the NCI/NIH Chemical Identifier Resolver). Given a chemical identifier (for example, a chemical name or registry number) and the desired output representation (for example, a SMILES string, InChI, or a file format), this function encodes the identifier, selects the appropriate path and query parameters accepted by the CIR API, and returns a single percent-encoded URL string. The function does not perform any network I/O; it only composes the URL. It respects module-level constants such as FILE_FORMATS (which cause a representation to be sent as representation=file with a format=... query parameter) and API_BASE (the CIR endpoint base path).
    
    Args:
        input (str): Chemical identifier to resolve. This is the raw identifier provided by the caller (for example "Aspirin" or "50-78-2"). The value will be percent-encoded (quoted) for safe inclusion in a URL path; if non-string values are passed, the underlying quote call may raise a TypeError. If tautomers is True, the function will prepend the literal prefix "tautomers:" to this string before encoding to instruct CIR to return all tautomers for the identifier.
        representation (str): Desired output representation requested from CIR. This specifies the CIR resource path segment (for example "smiles", "inchi", or other representation names). If this value is a member of the module FILE_FORMATS set, the function will instead set the path representation to "file" and add a query parameter format=<representation> so CIR returns the requested file format. The returned URL path will include this representation (or "file" when a file format is requested).
        resolvers (list(str)): Optional ordered list of resolver names to pass to the CIR service. When provided, the list elements are joined with commas and added as the resolver query parameter (resolver=name1,name2,...). The order in this list indicates the order in which CIR should attempt resolution. Elements must be strings; non-string elements will cause a TypeError during the join operation.
        get3d (bool): Optional flag indicating whether to request 3D coordinates from CIR when applicable. When True, the function adds get3d=True to the query string. Default is False. This flag is meaningful for representations and resolver combinations that can return 3D coordinate data.
        tautomers (bool): Optional flag indicating whether to request all tautomers of the given identifier. When True, the function prepends the literal prefix "tautomers:" to the input identifier (before percent-encoding) which tells the CIR service to return alternate tautomeric forms. Default is False.
        xml (bool): Optional flag indicating whether to request the CIR XML wrapper for the response. When True (the default), the function appends the literal path segment "/xml" to the constructed path so the CIR service returns its full XML response wrapper. When False, the "/xml" suffix is omitted and the raw resource endpoint is returned.
        kwargs (dict): Additional optional query parameters to include in the URL as a query string. Keys and values are encoded with urllib.parse.urlencode. Common uses include specifying format (for file outputs), page or detail options supported by CIR, or any other query parameters accepted by the CIR API. Values should be types accepted by urlencode (strings or sequences as appropriate); unsupported types may cause an exception during encoding.
    
    Returns:
        str: The fully constructed CIR API URL as a percent-encoded string. The returned URL combines the module-level API_BASE, the quoted input identifier (with any tautomers: prefix if requested), the chosen representation (or "file" plus format=... for file formats), an optional "/xml" suffix when xml is True, and a query string containing resolver, get3d, format, and any additional kwargs. No network request is made; the caller must use this URL with an HTTP client to contact the CIR web service.
    
    Behavior notes and failure modes:
        - If representation is found in the module FILE_FORMATS set, the function places the original representation into the format query parameter and uses "file" as the path representation. This behavior is required by the CIR API to request downloadable file formats.
        - The input identifier is percent-encoded using urllib.parse.quote; passing non-string input is likely to raise a TypeError.
        - resolvers must be an iterable of strings; otherwise the string join operation will raise a TypeError.
        - kwargs keys and values are passed to urllib.parse.urlencode; values that are not strings or sequences acceptable to urlencode may raise an exception.
        - The function has no side effects beyond returning the URL string and does not modify external state or perform HTTP requests.
    """
    from cirpy import construct_api_url
    return construct_api_url(
        input,
        representation,
        resolvers,
        get3d,
        tautomers,
        xml,
        **kwargs
    )


################################################################################
# Source: cirpy.download
# File: cirpy.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def cirpy_download(
    input: str,
    filename: str,
    representation: str,
    overwrite: bool = False,
    resolvers: list = None,
    get3d: bool = False,
    **kwargs
):
    """Convenience function to save a Chemical Identifier Resolver (CIR) response to a file.
    
    This function is part of CIRpy, a Python interface for the CIR web service (Chemical Identifier Resolver) described in the project README. It wraps cirpy.resolve to request a chemical representation (for example, "smiles" for SMILES strings) from CIR and then writes the textual response to a local file. Typical use is to persist resolved chemical representations (SMILES, InChI, SDF blocks, etc.) for later use in cheminformatics workflows without manually constructing HTTP requests or parsing responses.
    
    Args:
        input (str): Chemical identifier to resolve. This is the query passed to the CIR service via cirpy.resolve; examples include common chemical names (e.g., "Aspirin"), registry identifiers, or other identifiers that CIR accepts. The meaning and role in the domain is to identify the chemical whose representation is requested.
        filename (str): File path to save the resolved representation to. This is a filesystem path (relative or absolute) where the textual CIR response will be written. The function opens this path in text write mode ('w') and overwrites only when allowed by the overwrite parameter.
        representation (str): Desired output representation requested from CIR. This string specifies the chemical representation to retrieve (for example, "smiles" as shown in the README). It is passed directly to cirpy.resolve and determines the format of the text that will be saved to filename.
        overwrite (bool): Whether to allow overwriting of an existing file at filename. Defaults to False. If False and a file already exists at filename (detected using os.path.isfile), the function will raise an IOError to avoid accidental loss of data. If True, any existing file at filename will be replaced.
        resolvers (list): Ordered list of resolvers to use when resolving the input. This optional list (or None) is forwarded to cirpy.resolve and influences which CIR resolver backends are tried and in what order. In the domain context, different resolvers may provide different mapping strategies or sources for resolving identifiers.
        get3d (bool): Whether to request 3D coordinates where applicable. Defaults to False. When True and the requested representation supports 3D coordinate output (for example, certain molecular file formats), cirpy.resolve is asked to obtain 3D coordinates; the returned text (if any) may therefore include 3D coordinate data which will be written to file.
        kwargs (dict): Additional keyword arguments forwarded to cirpy.resolve. These are passed through unchanged to the underlying resolve function and ultimately to the CIR service or resolver-specific handlers; they allow caller-specified options supported by cirpy.resolve. The practical significance is to enable resolver-specific or service-specific flags without modifying this convenience wrapper.
    
    Returns:
        None: The function does not return the retrieved representation. Instead, it has the side effect of writing the resolved textual representation to filename. If cirpy.resolve returns a falsy value (no result), the function logs a debug message ("No file to download.") and returns without creating or modifying any file.
    
    Raises:
        HTTPError: If the CIR web service returns an HTTP error code during resolution via cirpy.resolve. This indicates a transport or server-side error when contacting CIR.
        ParseError: If the CIR response cannot be interpreted by cirpy.resolve (unparseable or malformed response). This indicates the service returned data that could not be converted to a usable string representation.
        IOError: If overwrite is False and a file already exists at filename, an IOError is raised to prevent overwriting. The function may also raise an IOError for filesystem write errors when opening or writing to filename.
    
    Behavior and side effects:
        The function calls cirpy.resolve(input, representation, resolvers, get3d, **kwargs) to obtain a textual representation. If the resolved result is falsy (None or empty string), the function logs a debug message and returns None without touching the filesystem. If a non-empty string is returned, the function ensures the string ends with a newline character; if it does not, a single newline is appended. The file at filename is then opened in text write mode ('w') and the (possibly newline-terminated) result is written to disk. Existing files are preserved unless overwrite is True. Exceptions raised by cirpy.resolve are propagated to the caller as described above.
    """
    from cirpy import download
    return download(
        input,
        filename,
        representation,
        overwrite,
        resolvers,
        get3d,
        **kwargs
    )


################################################################################
# Source: cirpy.query
# File: cirpy.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def cirpy_query(
    input: str,
    representation: str,
    resolvers: list = None,
    get3d: bool = False,
    tautomers: bool = False,
    **kwargs
):
    """Resolve a chemical identifier using the CIR (Chemical Identifier Resolver) web service and return all parsed results for the requested output representation.
    
    This function is part of CIRpy, a Python interface to the CIR web service maintained by the CADD Group at NCI/NIH (see README). It issues a network request (via request()) to CIR, parses the XML response, and constructs a list of Result objects representing each resolution returned by CIR. Each Result contains the original input string, the representation requested, the resolver that produced the result, the input format class, the notation, and the resolved value (a single string when a single value is returned or a list of strings when multiple items are returned, e.g., multiple tautomers). The function logs the number of results at debug level and may perform HTTP network I/O.
    
    Args:
        input (str): Chemical identifier to resolve. This is the query string sent to the CIR web service and can be a common chemical name, registry number, InChI, SMILES, or any identifier that CIR accepts. The exact meaning and interpretation depend on CIR; the field is used to populate the 'string' attribute in each returned Result.
        representation (str): Desired output representation requested from CIR (for example, 'smiles' as in the README example). This selects the representation CIR should attempt to return and becomes the 'representation' attribute on each Result. If CIR cannot produce the requested representation for a particular resolver, that resolver may not contribute results.
        resolvers (list(str)): (Optional) Ordered list of resolver names to use when querying CIR. When provided, the function instructs CIR to try resolvers in this order; when None (the default) CIR's own default resolver order is used. Each result's 'resolver' attribute records which resolver produced that result.
        get3d (bool): (Optional) Whether to request 3D coordinates where applicable. Default is False. When True and when the requested representation supports 3D coordinates, CIR may return coordinate-containing formats; otherwise this flag has no effect. Results that include coordinates will still be returned as Result objects.
        tautomers (bool): (Optional) Whether to request all tautomers from CIR. Default is False. When True the function may receive and return multiple resolved values (e.g., a list of tautomers) for a single resolver; when False CIR typically returns a single canonical form per resolver.
        kwargs (dict): Additional keyword arguments forwarded to the internal request() function and ultimately used to build or control the HTTP request to CIR (for example, timeout or other request-specific parameters supported by request()). These are passed through unchanged and may influence network behavior and error handling.
    
    Returns:
        list(Result): A list of Result objects, one per resolved entry parsed from the CIR XML response. Each Result is constructed from XML attributes and items as follows:
            input: value of the top-level 'string' attribute (the original query string).
            representation: value of the top-level 'representation' attribute (the requested representation).
            resolver: value of the 'resolver' attribute on the individual <data> element (which resolver produced the entry).
            input_format: value of the 'string_class' attribute on the <data> element (how CIR classified the input).
            notation: value of the 'notation' attribute on the <data> element (notation metadata from CIR).
            value: the resolved data taken from <item> elements; a single string when the <data> element contains one <item>, or a list of strings when multiple <item> elements are present (e.g., multiple tautomers).
        The returned list may be empty if CIR returns no matches. The function logs the number of results at debug level before returning.
    
    Raises:
        HTTPError: If the CIR web service returns an HTTP error code (propagated from the request() call). This indicates a network-level or server error returned by CIR.
        ParseError: If the CIR XML response cannot be parsed or does not conform to the expected XML structure (for example, missing expected attributes or elements). This indicates that the response from CIR was uninterpretable by the parser.
    """
    from cirpy import query
    return query(input, representation, resolvers, get3d, tautomers, **kwargs)


################################################################################
# Source: cirpy.request
# File: cirpy.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def cirpy_request(
    input: str,
    representation: str,
    resolvers: list = None,
    get3d: bool = False,
    tautomers: bool = False,
    **kwargs
):
    """Make a request to the Chemical Identifier Resolver (CIR) web service and return the parsed XML root element of the CIR response.
    
    This function is the low-level network operation used by CIRpy to resolve chemical identifiers (for example, a common name like "Aspirin", a CAS, InChI, or SMILES) into another chemical representation (for example, "smiles", "inchi", "mol"). It constructs a CIR API URL via construct_api_url, performs an HTTP request (using urlopen), and parses the returned XML into an Element tree root (using etree.parse). The function performs blocking network I/O and will propagate network and parsing errors to the caller. Typical use in the CIRpy domain is to call higher-level helpers (such as cirpy.resolve) which in turn call this function to obtain the CIR XML response.
    
    Args:
        input (string): Chemical identifier to resolve. This is the source identifier you want CIR to translate (for example, a common name "Aspirin", a SMILES string, or another identifier supported by the CIR web service). The function does not validate that the identifier is syntactically correct for any particular identifier system; it forwards the value to the CIR API.
        representation (string): Desired output representation to request from CIR (for example, 'smiles', 'inchi', 'mol'). This tells CIR which chemical representation you want returned. The representation must be one supported by the CIR web service; unsupported representations will cause CIR to return an error which is propagated as an HTTPError or visible in the returned XML.
        resolvers (list(string)): (Optional) Ordered list of resolver names to use within CIR. When provided, this list directs CIR to try the specified resolver implementations in the given order. If None (the default), CIR's default resolver ordering is used. This parameter is passed to construct_api_url and affects which back-end resolver(s) CIR queries.
        get3d (bool): (Optional) Whether to request 3D coordinates when the requested representation supports 3D output. Defaults to False. When True and when the chosen representation can include 3D coordinates, the constructed CIR request will ask for 3D coordinate output; whether 3D data is returned depends on the resolver and the input.
        tautomers (bool): (Optional) Whether to request all tautomeric forms from CIR. Defaults to False. When True, CIR may return multiple tautomeric variants in the XML response; callers should expect the response XML root to contain multiple structure entries in that case.
        kwargs (dict): Additional keyword arguments forwarded to construct_api_url and ultimately included in the CIR request URL or request options. These implementation-specific options are appended to the request and can influence CIR behavior; they are passed through without local validation by this function.
    
    Returns:
        Element: The root XML element parsed from the CIR HTTP response (an XML Element as produced by etree.parse(...).getroot()). The returned element contains the CIR response data (the requested representation, any resolver metadata, or error information) and must be inspected by the caller to extract the desired fields. This function has the side effect of performing network I/O (an HTTP request) and logging a debug message about the request URL.
    
    Raises:
        HTTPError: If the CIR service returns an HTTP error status code (for example, 4xx or 5xx). The underlying urlopen call will raise this and it is propagated to the caller.
        ParseError: If the CIR response cannot be parsed as XML (for example, if the response is malformed or not XML), the XML parser (etree.parse) will raise a ParseError which is propagated to the caller.
    
    Behavior and failure modes:
        This function always constructs the CIR API URL by calling construct_api_url with the same arguments and then performs an HTTP GET via urlopen on that URL. It blocks until the HTTP response is received and then attempts to parse the response as XML. Network failures, DNS errors, or other lower-level URL errors raised by urlopen (such as URLError) may also propagate to the caller. The function logs the constructed URL at debug level prior to issuing the request. Callers should be prepared to handle network and XML parsing exceptions and to validate that the returned XML contains the expected representation data.
    """
    from cirpy import request
    return request(input, representation, resolvers, get3d, tautomers, **kwargs)


################################################################################
# Source: cirpy.resolve
# File: cirpy.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def cirpy_resolve(
    input: str,
    representation: str,
    resolvers: list = None,
    get3d: bool = False,
    **kwargs
):
    """Resolve a chemical identifier to a specified output representation using the Chemical Identifier Resolver (CIR) web service.
    
    This function is the high-level convenience wrapper used in CIRpy to convert one form of chemical identifier (for example a common name such as "Aspirin") into another representation (for example a SMILES string). It issues a network query to the CIR service via the internal cirpy.query function, takes the first result from the CIR XML response, and returns that result's value as a string. If CIR returns no results the function returns None. This function therefore encapsulates the practical workflow described in the project README: no manual URL construction or XML parsing is required — cirpy.resolve performs the request, parsing, and selection of the first returned representation.
    
    Args:
        input (string): Chemical identifier to resolve. This is the input token you want translated (for example a trivial/common name, InChI, CAS number, etc.) and is passed directly to the CIR web service. Its practical role is to identify the chemical entity to be looked up.
        representation (string): Desired output representation. This string names the CIR output type you require (for example "smiles", "inchi", etc.). The function requests this representation from CIR and returns the first matching value when available.
        resolvers (list(string)): (Optional) Ordered list of resolvers to use. When provided, this list constrains and orders which backend resolvers CIR should attempt. If None (the default), resolver selection is left to CIR's default behavior. The list items are resolver identifiers accepted by the CIR service.
        get3d (bool): (Optional) Whether to request 3D coordinates where applicable. If True, the request will ask CIR for three-dimensional coordinate output when the chosen representation and CIR backend support it. Default is False. Note that not all representations or resolver backends can provide 3D coordinates; support is dependent on the CIR service.
        kwargs (dict): Additional keyword arguments forwarded to cirpy.query and ultimately to the CIR web service client. These keyword arguments are not interpreted by cirpy.resolve itself but are passed through to the underlying query/HTTP layer (for example to influence network behavior or parser options supported by cirpy.query). Users should consult cirpy.query documentation for the set of accepted extra options.
    
    Returns:
        string or None: The first output representation value returned by CIR as a string, or None if CIR returned no results for the given input and representation. A successful return means the caller can use the string directly (for example as a SMILES or InChI). If no match is found, None indicates absence of a resolved representation.
    
    Raises:
        HTTPError: If the CIR web service responds with an HTTP error code (network-level or server-side error). This indicates the request failed and no valid CIR response could be obtained.
        ParseError: If the CIR response cannot be parsed or interpreted as expected (for example malformed XML). This indicates a failure during response interpretation by the internal query/parsing routines.
    
    Side effects and behavior notes:
        - This function performs a network call to the external CIR web service; callers should be prepared for network latency, timeouts, and related exceptions.
        - Only the first result from the CIR XML response is returned; any additional candidate results are ignored by this wrapper.
        - Default parameter values: resolvers defaults to None (use CIR defaults), get3d defaults to False.
        - The function does not modify persistent state; its primary observable effect is the network request and the returned string (or None).
    """
    from cirpy import resolve
    return resolve(input, representation, resolvers, get3d, **kwargs)


################################################################################
# Source: cirpy.resolve_image
# File: cirpy.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def cirpy_resolve_image(
    input: str,
    resolvers: list = None,
    fmt: str = "png",
    width: int = 300,
    height: int = 300,
    frame: bool = False,
    crop: int = None,
    bgcolor: str = None,
    atomcolor: str = None,
    hcolor: str = None,
    bondcolor: str = None,
    framecolor: str = None,
    symbolfontsize: int = 11,
    linewidth: int = 2,
    hsymbol: str = "special",
    csymbol: str = "special",
    stereolabels: bool = False,
    stereowedges: bool = True,
    header: str = None,
    footer: str = None,
    **kwargs
):
    """Resolve a chemical identifier to a 2D image depiction using the CIR (Chemical Identifier Resolver) web service.
    
    This function is part of CIRpy, a Python interface to the NCI/NIH Chemical Identifier Resolver (CIR). Given a chemical identifier (name, InChI, SMILES, registry number, etc.) it builds a CIR API request that asks the service to produce a 2D image representation of the structure and returns the raw image bytes. The function aggregates its explicit parameters into the API parameter map, applies a small set of deterministic transformations required by the CIR API (for example, renaming fmt to format and stereolabels to showstereo, mapping stereowedges to both wedges and dashes), enforces a non-antialiased image when bgcolor is "transparent", and always requests a representation of type "image" and xml=False. The constructed URL is produced by construct_api_url(...) and the image is fetched with urlopen(...). The returned bytes are the exact response body from the CIR service and must be interpreted by the caller according to the requested fmt (png or gif).
    
    Args:
        input (str): Chemical identifier to resolve. This is the primary piece of information sent to the CIR web service (for example a chemical name like "Aspirin", a SMILES string, an InChI, CAS registry number, etc.). The identifier determines which structure CIR will attempt to depict.
        resolvers (list): (Optional) Ordered list of resolvers to use. Each item is a resolver name understood by the CIR service; the order controls which resolver is tried first when CIR attempts to interpret the input identifier.
        fmt (str): (Optional) Image format requested from CIR; supported values are "png" or "gif" (default "png"). Internally this parameter is renamed to the API parameter "format" before constructing the request.
        width (int): (Optional) Image width in pixels (default 300). Controls the horizontal pixel dimension requested from CIR.
        height (int): (Optional) Image height in pixels (default 300). Controls the vertical pixel dimension requested from CIR.
        frame (bool): (Optional) Whether to show a border frame around the structure (default False). When True, a frame parameter is included in the API request to instruct CIR to render a border.
        crop (int): (Optional) Crop image with specified padding. If provided, this integer value is forwarded to CIR to request cropping with the specified padding in pixels.
        bgcolor (str): (Optional) Background color for the image. This string is forwarded to CIR as the background color parameter; use values accepted by the CIR API (for example color names or hex codes as supported by the service). If set to the literal "transparent" the function will also disable antialiasing by setting the "antialiasing" API parameter to False to preserve transparency.
        atomcolor (str): (Optional) Atom label color forwarded to CIR. The string is passed unchanged to the API as the atom label color specification.
        hcolor (str): (Optional) Hydrogen atom label color forwarded to CIR. The string is passed unchanged to the API as the hydrogen atom label color specification.
        bondcolor (str): (Optional) Bond color forwarded to CIR. The string is passed unchanged to the API as the bond color specification.
        framecolor (str): (Optional) Border frame color forwarded to CIR. The string is passed unchanged to the API as the frame/border color specification.
        symbolfontsize (int): (Optional) Atom label font size in points (default 11). Controls the font size used for element symbols/labels in the depiction and is forwarded to CIR.
        linewidth (int): (Optional) Bond line width in pixels (default 2). Controls how thick rendered bonds appear and is forwarded to CIR.
        hsymbol (str): (Optional) Hydrogens display mode: "all", "special" or "none" (default "special"). This string controls whether hydrogen symbols are drawn and is forwarded to the CIR API as provided.
        csymbol (str): (Optional) Carbons display mode: "all", "special" or "none" (default "special"). This string controls whether carbon symbols are drawn and is forwarded to the CIR API as provided.
        stereolabels (bool): (Optional) Whether to show stereochemistry labels (default False). This boolean is renamed to the API parameter "showstereo" before constructing the request.
        stereowedges (bool): (Optional) Whether to show wedge/dash bonds (default True). This boolean is mapped to two API parameters, "wedges" and "dashes", both set to the same boolean value to enable or disable wedge/dash rendering consistently.
        header (str): (Optional) Header text to render above the structure. The string is forwarded to CIR and will appear on the returned depiction if the service supports it.
        footer (str): (Optional) Footer text to render below the structure. The string is forwarded to CIR and will appear on the returned depiction if the service supports it.
        kwargs (dict): Additional named parameters forwarded directly to the CIR API. The function first aggregates all explicit parameters into this mapping (only non-None values are included), then applies the specific transformations described above (fmt -> format, stereolabels -> showstereo, stereowedges -> wedges+dashes, antialiasing=False when bgcolor is "transparent"), then adds constant keys representation="image" and xml=False before calling construct_api_url(**kwargs). Any key/value pairs you include here will be sent to construct_api_url and ultimately to the CIR web service; use this to access CIR options not covered by the explicit parameters.
    
    Returns:
        bytes: Raw response body returned by the CIR web service for the constructed image request. For successful image requests this will be the binary image data in the requested format (PNG or GIF) and can be written directly to a file or passed to an image decoder. This function makes a network request (via construct_api_url and urlopen) and has no other side effects on local state. Network errors, HTTP errors, or CIR service errors will propagate as exceptions from urlopen (for example URLError or HTTPError) and, in some failure cases, the returned bytes may contain an error message or non-image payload from the CIR service rather than image data.
    """
    from cirpy import resolve_image
    return resolve_image(
        input,
        resolvers,
        fmt,
        width,
        height,
        frame,
        crop,
        bgcolor,
        atomcolor,
        hcolor,
        bondcolor,
        framecolor,
        symbolfontsize,
        linewidth,
        hsymbol,
        csymbol,
        stereolabels,
        stereowedges,
        header,
        footer,
        **kwargs
    )


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
