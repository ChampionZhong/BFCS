"""
Regenerated Google-style docstrings for module 'pybel'.
README source: others/readme/pybel/README.rst
Generated at: 2025-12-02T01:41:36.185839Z

Total functions: 46
"""


from typing import List

################################################################################
# Source: pybel.canonicalize.postpend_location
# File: pybel/canonicalize.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def pybel_canonicalize_postpend_location(bel_string: str, location_model: dict):
    """Append a canonicalized BEL location clause to an existing BEL node string.
    
    Args:
        bel_string (str): A BEL string representing a node, expected to be the full BEL term
            for an entity including a trailing closing parenthesis. In the PyBEL domain this
            is a fragment produced by the parser/serializer for a node (for example "p(hgnc:1234)")
            and this function appends a loc(...) clause to represent a subcellular or biological
            location. The function removes the final character of this string (normally the
            closing ')') and inserts the canonicalized location clause before re-closing the
            term. If the input does not end with a closing parenthesis the result may be a
            syntactically incorrect BEL fragment; callers should supply a well-formed BEL
            node string produced by the parser or serializer.
        location_model (dict): A dictionary that encodes the location to append and must
            contain the keys NAMESPACE and NAME (these constants are defined in the PyBEL
            canonicalization constants used by the parser and compiler). In practice this
            dictionary is one of the location entries produced by the parsing/normalization
            pipeline (for example the values associated with pybel.constants.TO_LOC or
            pybel.constants.FROM_LOC in parsed statement metadata). NAMESPACE should be a
            string namespace (e.g., a controlled vocabulary like "go" or a textual namespace)
            and NAME should be the location identifier or label. The NAME value will be
            passed through ensure_quotes to produce a valid BEL string literal for the
            loc(...) clause.
    
    Behavior and side effects:
        This function performs a pure string transformation and has no side effects on
        external state or on the supplied location_model. It constructs and returns a new
        BEL fragment by removing the final character of bel_string and concatenating
        ", loc(<NAMESPACE>:<NAME>))" where <NAMESPACE> is taken directly from
        location_model[NAMESPACE] and <NAME> is wrapped by ensure_quotes(location_model[NAME]).
        The produced string is intended for use in BEL graph compilation, serialization,
        or downstream exporters (e.g., Node-Link JSON, CX, or BEL script output) to
        represent entities annotated with cellular or tissue locations.
    
    Failure modes and validation:
        The function validates that location_model contains both NAMESPACE and NAME keys and
        will raise ValueError if either is missing. It does not validate that bel_string
        itself is a well-formed BEL term beyond slicing off its last character; supplying a
        malformed bel_string (for example lacking a trailing ')') will not raise a specific
        error but may produce an incorrect BEL fragment. ensure_quotes is used to mitigate
        quoting/escaping issues for the NAME value, but callers should ensure that the
        namespace and name values come from controlled vocabularies or the grounding step
        in the PyBEL pipeline when canonicalization correctness matters.
    
    Returns:
        str: A new BEL string fragment representing the original node with a canonicalized
        loc(...) clause appended. Example output form: "p(hgnc:1234, loc(GO:'cytoplasm'))".
    """
    from pybel.canonicalize import postpend_location
    return postpend_location(bel_string, location_model)


################################################################################
# Source: pybel.io.biodati_client.from_biodati
# File: pybel/io/biodati_client.py
# Category: valid
################################################################################

def pybel_io_biodati_client_from_biodati(
    network_id: str,
    username: str = "demo@biodati.com",
    password: str = "demo",
    base_url: str = "https://networkstore.demo.biodati.com"
):
    """pybel.io.biodati_client.from_biodati: Download and return a BELGraph from a BioDati network store given its network identifier.
    
    This function connects to a BioDati "network store" web service, authenticates with the provided credentials, requests the network identified by network_id, and returns it as a pybel.struct.graph.BELGraph. In the PyBEL ecosystem, this is used to import networks that were stored or published via BioDati Studio/Network Store so they can be analyzed, summarized, serialized, or exported using PyBEL tooling (for example, graph.summarize(), pybel.dump(...), or exports to CX/GraphDati/NDEx). The function performs network I/O and authentication as a side effect and relies on the BiodatiClient implementation to transform the remote network representation into a BELGraph instance.
    
    Args:
        network_id (str): The internal identifier of the network to download from the BioDati network store. This is the unique string assigned to a stored network in BioDati (for example, '01E46GDFQAGK5W8EFS9S9WMH12' in the original example). In the domain of BEL and PyBEL, this identifier points to a stored BEL network that will be converted into a pybel.struct.graph.BELGraph for downstream analysis.
        username (str): The email address or username used to authenticate to the BioDati network store. Defaults to "demo@biodati.com", which is the public demo account on the demo server. In practice, replace this with the account that has permission to access the target network on your BioDati instance.
        password (str): The password used to authenticate to the BioDati network store. Defaults to "demo" for the demo server. Keep in mind that using the default demo credentials only works for the public demo server; for private or production BioDati instances, provide the appropriate password for your account.
        base_url (str): The base URL of the BioDati network store API endpoint. Defaults to "https://networkstore.demo.biodati.com", the public demo server used in examples. In production use, substitute the URL of your BioDati network store (commonly of the form "https://networkstore.<YOUR NAME>.biodati.com"). This URL is used by the underlying BiodatiClient to construct HTTP requests to fetch the network data.
    
    Returns:
        pybel.struct.graph.BELGraph: A PyBEL BELGraph object representing the requested BioDati network. The returned BELGraph is a fully constructed in-memory BEL graph that can be summarized (graph.summarize()), serialized to files (pybel.dump), or converted to other interchange formats supported by PyBEL. If the remote content cannot be converted to a BELGraph, an exception will be raised by the BiodatiClient or its underlying HTTP client.
    
    Behavior, side effects, defaults, and failure modes:
        - Authentication and network I/O: The function creates a BiodatiClient with the provided username, password, and base_url and performs one or more HTTP requests to the remote BioDati server. These actions are side effects that may incur network latency and require network connectivity.
        - Defaults are set for convenience to point at the public BioDati demo server (username "demo@biodati.com", password "demo", base_url "https://networkstore.demo.biodati.com"). These defaults are appropriate only for public demo usage; you should change them to your credentials and server for real data.
        - Errors and exceptions: Network errors (connection timeouts, DNS failures), HTTP errors (4xx/5xx status codes), and authentication failures will be raised by the underlying BiodatiClient or HTTP library. If network_id does not correspond to an accessible or convertible network, the client will raise an informative exception (for example, not found or conversion failure). Callers should handle these exceptions as appropriate for their application (retry logic, user feedback, or logging).
        - Security: Credentials are sent to the server to authenticate. Ensure you use TLS/HTTPS endpoints (the default uses https) and manage credentials securely (avoid hard-coding production credentials in source code).
        - Result usage: The returned BELGraph is a standard PyBEL graph and participates in all PyBEL workflows, such as summarization, grounding, exporting to GraphDati/CX/NDEx, and further programmatic analysis.
    """
    from pybel.io.biodati_client import from_biodati
    return from_biodati(network_id, username, password, base_url)


################################################################################
# Source: pybel.io.cx.from_cx
# File: pybel/io/cx.py
# Category: valid
################################################################################

def pybel_io_cx_from_cx(cx: List[Dict]):
    """pybel.io.cx.from_cx: Rebuild a BELGraph from CX JSON output produced by PyBEL.
    
    Reconstructs a pybel.struct.graph.BELGraph from a CX-format JSON object (the CX
    export produced by PyBEL/NDEx-compatible pipelines). This function is used in
    I/O and interchange workflows described in the project README to import a
    previously exported BEL graph (nodes, edges, annotations, citations, network
    metadata, namespaces, and annotation lists) back into an in-memory BELGraph
    data structure for further analysis, validation, modification, or re-export to
    other formats (for example: NetworkX, Node-Link JSON, GraphDati, INDRA, or
    NDEx).
    
    The function iterates the CX list-of-dictionaries and reconstructs internal
    aspects (context_legend, annotation_lists, @context entries, networkAttributes,
    nodes, nodeAttributes, edges, edgeAttributes, and meta entries). It restores
    node-level constructs (including fusion and variant encodings, products/reactants
    and members lists), maps CX node name attributes to the BEL node NAME field,
    parses node data into PyBEL DSL node representations via parse_result_to_dsl,
    adds nodes to the BELGraph with add_node_from_data, reconstructs edge relations
    and per-edge qualified data (evidence, citation, source/target modifiers,
    annotations), converts annotation lists to sets, expands serialized dicts where
    necessary, and finally adds edges to the graph using add_qualified_edge or
    add_unqualified_edge.
    
    Args:
        cx (List[Dict]): A CX JSON object represented as a list of dictionaries,
            exactly as produced by PyBEL's CX exporter or a compatible NDEx/CX
            pipeline. Each list element corresponds to an aspect (for example,
            "nodes", "edges", "nodeAttributes", "edgeAttributes", "networkAttributes",
            "@context", "annotation_lists", "context_legend", or other CX aspects)
            and the dictionaries inside those aspects must follow the CX conventions
            expected by PyBEL. This argument is the primary serialized representation
            of a BEL graph produced by PyBEL and is required for rebuilding the
            in-memory BELGraph. Passing a value that is not a list of dictionaries
            will raise standard Python type errors or lead to downstream parsing
            failures. The function expects that CX node/edge attribute keys used by
            PyBEL appear (for example CX_NODE_NAME, FUSION, VARIANTS, PRODUCTS,
            REACTANTS, MEMBERS, CITATION-prefixed keys, EVIDENCE, and NDEx source/target
            modifier keys) so that they can be restored to their BEL equivalents.
    
    Returns:
        pybel.struct.graph.BELGraph: A newly constructed BELGraph instance containing
        the nodes, edges, graph metadata, namespaces, annotation definitions, and
        per-edge qualifications (citations, evidence, source/target modifiers,
        annotations) restored from the provided CX object. The returned BELGraph is
        suitable for all standard PyBEL operations (summarization, grounding,
        serialization to other formats, analysis pipelines) described in the README.
        Side effects: the function instantiates and populates a BELGraph entirely in
        memory; it does not write files or modify external state. Failure modes: if
        the CX input omits required structural elements or contains malformed JSON
        strings where lists/dicts are expected, the function may raise ValueError,
        TypeError, KeyError, or JSON decoding errors. In particular, edges with
        relations that are neither present in the reconstructed edge_relation mapping
        nor in the UNQUALIFIED_EDGES set will trigger a ValueError("problem adding
        edge: <eid>"). Network attributes named NDEX_SOURCE_FORMAT are ignored per
        implementation. The function may also raise errors coming from helper
        routines called during reconstruction (for example parse_result_to_dsl,
        expand_dict, _restore_fusion_dict, or the Entity constructor) when node or
        edge attribute values are invalid or inconsistent with expected CX encodings.
    """
    from pybel.io.cx import from_cx
    return from_cx(cx)


################################################################################
# Source: pybel.io.cx.from_cx_gz
# File: pybel/io/cx.py
# Category: fix_docstring
# Reason: Schema parsing failed: Cannot generate JSON schema for from_cx_gz because the docstring has no description for the argument 'path'
################################################################################

def pybel_io_cx_from_cx_gz(path: str):
    """pybel.io.cx.from_cx_gz reads a BELGraph from a CX JSON file compressed with gzip and returns the in-memory PyBEL BELGraph representation.
    
    This function is part of PyBEL's I/O utilities for interoperability between CX (the Cytoscape/NDEx JSON data model) and PyBEL's BELGraph. It opens the given filesystem path in gzip text mode, decodes the CX JSON content using the standard json loader, and delegates parsing and conversion to pybel.io.cx.from_cx to produce a BELGraph. Practically, use this to import networks exported in CX/NDEx formats into PyBEL for downstream analysis, visualization, or serialization to other formats supported by PyBEL.
    
    Args:
        path (str): Filesystem path to a gzip-compressed file that contains CX JSON. The file should be a CX (NDEx) JSON document exported by NDEx or other CX writers and accessible from the local filesystem. This parameter is the sole input and its role is to identify the source file to read; the function does not modify the file.
    
    Behavior and side effects:
        The function opens the file with gzip.open(path, "rt") to read text from the gzip stream, then calls json.load(file) to parse the entire JSON document into a Python object, and finally calls pybel.io.cx.from_cx with that object to obtain a pybel.struct.graph.BELGraph. There are no persistent side effects: the input file is only read and is not altered or written to. Because the entire JSON is loaded into memory, very large CX files may use substantial memory and could raise MemoryError on resource-constrained systems.
    
    Failure modes and notes:
        If the file does not exist, a FileNotFoundError will be raised. If the file is not a valid gzip-compressed file or cannot be opened, an OSError may be raised. If the file contents are not valid JSON, json.JSONDecodeError will be raised. If the CX JSON structure is malformed or contains content that pybel.io.cx.from_cx cannot convert to a BELGraph, that function may raise ValueError or other parsing/validation exceptions. The function relies on pybel.io.cx.from_cx for semantic validation and conversion; it does not itself perform additional BEL validation beyond what from_cx implements.
    
    Returns:
        pybel.struct.graph.BELGraph: An in-memory PyBEL BELGraph constructed from the CX JSON input. The BELGraph represents the biological network encoded by the CX document and can be used with the rest of PyBEL's API for summarization, analysis, serialization to other formats (for example BEL, Node-Link JSON, GraphDati, INDRA JSON), grounding, or visualization workflows.
    """
    from pybel.io.cx import from_cx_gz
    return from_cx_gz(path)


################################################################################
# Source: pybel.io.cx.from_cx_jsons
# File: pybel/io/cx.py
# Category: fix_docstring
# Reason: Schema parsing failed: Cannot generate JSON schema for from_cx_jsons because the docstring has no description for the argument 'graph_json_str'
################################################################################

def pybel_io_cx_from_cx_jsons(graph_json_str: str):
    """pybel.io.cx.from_cx_jsons reads a BELGraph from a CX-format JSON string and returns a populated BELGraph suitable for PyBEL workflows.
    
    Args:
        graph_json_str (str): A JSON-encoded string containing a CX representation of a network following the CX/NDEx data model. This string must be valid JSON text. The function first parses this string with json.loads to produce the native Python JSON object (dict/list) and then delegates to pybel.io.cx.from_cx to convert that CX object into a BELGraph. In the PyBEL domain, this is used to import networks produced by CX-capable tools and services (for example, NDEx) so they can be manipulated, analyzed, summarized, and exported with PyBEL's BELGraph APIs.
    
    Behavior and side effects:
        This function performs two steps: (1) decoding the input JSON text into a Python data structure using json.loads, and (2) converting the decoded CX object into a pybel.struct.graph.BELGraph by calling pybel.io.cx.from_cx. It does not mutate the input string. The primary side effect is the allocation and return of a new BELGraph object populated with nodes, edges, annotations, and citations extracted from the CX structure. No files are written and no global state is modified by this function.
    
    Failure modes:
        If graph_json_str is not valid JSON, json.loads will raise a json.JSONDecodeError (or a TypeError if an inappropriate non-string type is passed), which will propagate to the caller. If the decoded JSON does not conform to the expected CX structure required by pybel.io.cx.from_cx, that function may raise ValueError, KeyError, or other exceptions describing the CX validation/conversion failure; those exceptions are propagated. Callers should validate or catch these exceptions when loading untrusted or uncertain input.
    
    Returns:
        pybel.struct.graph.BELGraph: A newly constructed BELGraph representing the BEL biological network encoded in the provided CX JSON string. This BELGraph is the primary in-memory graph representation used throughout PyBEL for downstream tasks such as summarization, grounding, analysis, and export to other formats (Node-Link JSON, GraphML, CX, INDRA, etc.).
    """
    from pybel.io.cx import from_cx_jsons
    return from_cx_jsons(graph_json_str)


################################################################################
# Source: pybel.io.gpickle.from_pickle_gz
# File: pybel/io/gpickle.py
# Category: fix_docstring
# Reason: Schema parsing failed: Cannot generate JSON schema for from_pickle_gz because the docstring has no description for the argument 'path'
################################################################################

def pybel_io_gpickle_from_pickle_gz(path: str):
    """pybel.io.gpickle.from_pickle_gz reads a BELGraph from a gzipped pickle file and returns the in-memory BELGraph object for downstream biological-network analysis and format interchange.
    
    Args:
        path (str): Filesystem path to the gzipped pickle file to read. This path should point to a file that was created by serializing a pybel.struct.graph.BELGraph instance with Python's pickle protocol and then compressed with gzip. The string is interpreted by gzip.open as a path-like filename; it must be accessible to the running process and refer to a regular file (not a directory). In practical PyBEL workflows, such files are used to persist compiled BEL graphs so they can be quickly reloaded for downstream tasks such as exporting to Node-Link JSON, CX, GraphML, or running analyses and summaries described in the PyBEL README.
    
    Returns:
        pybel.struct.graph.BELGraph: The deserialized BELGraph instance reconstructed from the gzipped pickle. This returned object is the same BELGraph data structure used throughout PyBEL to represent biological expression networks (nodes with namespaces/identifiers, edges with relations, annotations, citations, etc.) and can be used immediately for summarization, grounding, exporting, or other analyses.
    
    Behavior and side effects:
        The function opens the file at path using gzip.open(path, "rb") and delegates reading and unpickling to the internal from_pickle routine, returning its result. The file is opened in binary read mode and is closed automatically when the function exits because a context manager is used. The function loads the entire graph into memory; for very large graphs this may consume substantial memory and may impact performance or fail if system memory is insufficient.
    
    Failure modes and errors:
        If the path does not exist or is not readable, a FileNotFoundError or OSError will be raised by gzip.open. If the file is not a valid gzip archive, gzip.BadGzipFile or OSError may be raised. If the gzip stream is valid but the contained data is not a valid pickle for a BELGraph (for example, the pickle is corrupted, uses an incompatible protocol, or references unknown classes), pickle.UnpicklingError, EOFError, AttributeError, or TypeError may be raised by the unpickling machinery invoked by from_pickle. Because pickle deserialization can execute arbitrary code embedded in the pickle payload, only load gzipped pickles from trusted sources to avoid security risks.
    
    Notes:
        This function is intended to be used in PyBEL I/O workflows to quickly restore a previously serialized BELGraph for reuse, analysis, or conversion to other network formats as described in the project README. It does not perform validation of BEL semantics beyond whatever validation the underlying unpickling and from_pickle routines perform.
    """
    from pybel.io.gpickle import from_pickle_gz
    return from_pickle_gz(path)


################################################################################
# Source: pybel.io.graphdati.from_graphdati
# File: pybel/io/graphdati.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def pybel_io_graphdati_from_graphdati(j: dict, use_tqdm: bool = True):
    """Convert a GraphDati / BioDati "normal" network JSON dictionary into a compiled PyBEL BELGraph.
    
    This function ingests a GraphDati-style JSON object (as produced by BioDati / GraphDati exports in the repository described by the README) and constructs a PyBEL BELGraph by parsing BEL statements embedded in the edges and attaching the per-statement metadata (citations, nanopub ids, annotations) to the resulting graph. It expects the input to follow the "normal" network format produced by BioDati/GraphDati; requesting the BioDati "full" network format is known to crash BioDati and therefore is not supported by this converter. The function populates top-level graph metadata (name, version, authors, description), records the originating BioDati network id on the graph (graph.graph["biodati_network_id"]), and parses each edge label as a BEL statement with the BELParser configured with NAMESPACE_TO_PATTERN. For every metadata entry attached to an edge (each nanopub_data entry) the function clears and reuses parser.control_parser, sets citation fields parsed from the GraphDati CURIE, attaches a default evidence string ("No evidence available from BioDai"), stores the BioDati nanopub id under the annotation key "biodati_nanopub_id", and merges any other parsed annotations before parsing the BEL statement. Because each BEL statement is parsed once per metadata entry, the same BEL statement may be added multiple times with different metadata sets (intentional to preserve nanopub-level provenance).
    
    Args:
        j (dict): A GraphDati JSON object representing the network. The function reads j["graph"] as the root and expects keys in specific places: root.get("label") becomes the BELGraph name; root["metadata"].get("gd_rev") becomes the BELGraph version; root["metadata"].get("gd_creator") becomes the BELGraph authors; root.get("gd_description") becomes the BELGraph description; root["metadata"]["id"] is saved to graph.graph["biodati_network_id"]; and the edge list is expected at root["edges"]. If these keys are missing the function may raise a KeyError because the code dereferences these locations directly. The dict is assumed to conform to GraphDati/BioDati "normal" export structure; other JSON shapes are not documented or supported by this function.
        use_tqdm (bool): When True (default), the edge iterator is wrapped with tqdm to display a progress bar while iterating over edges. When False, no progress bar is shown. This only affects user-facing progress reporting and does not change parsing logic or graph contents.
    
    Returns:
        BELGraph: A PyBEL BELGraph object containing nodes and edges parsed from the GraphDati JSON. The returned BELGraph has its name, version, authors, and description set from the GraphDati metadata when available, and contains an entry graph.graph["biodati_network_id"] set to the original GraphDati network id. Edge provenance and citation metadata are attached via the parser during parsing; repeated parsing of the same BEL statement for different nanopub metadata entries produces separate metadata-bearing instances in the graph to preserve provenance.
    
    Behavior and side effects:
        The function iterates over root["edges"]. For each edge it first checks for a "relation" and logs a warning if missing. Edges with relation values "actsIn" or "translocates" are skipped because they correspond to legacy formats not needed for BEL conversion. If an edge lacks a BEL statement in edge["label"], it is skipped and a debug message is emitted. For each metadata entry in edge["metadata"]["nanopub_data"], parser.control_parser is cleared and populated: citation_id is parsed via _parse_biodati_citation (if this returns None the specific metadata entry is skipped), parser.control_parser.citation_db and citation_db_id are set, parser.control_parser.evidence is set to the literal string "No evidence available from BioDai", parser.control_parser.annotations["biodati_nanopub_id"] is assigned a single-item list containing the nanopub id, and parser.control_parser.annotations is updated with the return value of _parse_biodati_annotations. After setting these control fields the BEL statement (edge["label"]) is parsed with parser.parseString(bel_statement, line_number=i). Parse errors raised by pyparsing.ParseException are caught and logged as warnings; they do not stop the overall conversion. The function reuses the same BELParser instance (initialized with namespace_to_pattern=NAMESPACE_TO_PATTERN) across edges; NAMESPACE_TO_PATTERN is a module-level mapping and may need updating depending on external namespace conventions used by the BioDati export.
    
    Failure modes and logging:
        Missing expected keys in the input JSON (for example, if "graph", "metadata", or "edges" are absent or malformed) can lead to KeyError or TypeError exceptions propagating from this function. Parser pyparsing.ParseException errors for individual BEL statements are handled locally: the specific BEL statement is skipped and a warning is logged, and processing continues. If _parse_biodati_citation returns None for a metadata entry the entry is skipped (logged indirectly by logic), preserving robustness to incomplete metadata. Because each BEL statement is parsed once per metadata entry, this can result in duplicate structural entries in the BELGraph with different provenance annotations; this behavior preserves nanopub-level provenance but has performance implications (multiple parses per edge) which are noted in the source as an area for potential improvement.
    
    Notes:
        This function is intended to be used as part of PyBEL's I/O machinery to import GraphDati/BioDati networks into the BELGraph data model described in the README, enabling downstream analysis, serialization, and export to other formats supported by PyBEL.
    """
    from pybel.io.graphdati import from_graphdati
    return from_graphdati(j, use_tqdm)


################################################################################
# Source: pybel.io.graphdati.from_graphdati_gz
# File: pybel/io/graphdati.py
# Category: fix_docstring
# Reason: Schema parsing failed: Cannot generate JSON schema for from_graphdati_gz because the docstring has no description for the argument 'path'
################################################################################

def pybel_io_graphdati_from_graphdati_gz(path: str):
    """pybel.io.graphdati.from_graphdati_gz reads a GraphDati JSON graph from a gzip-compressed file and converts it into a pybel.struct.graph.BELGraph for use with PyBEL's analysis, export, and visualization tools.
    
    This function is used in the PyBEL I/O subsystem to import networks produced or exported in the GraphDati (BioDati) JSON format into PyBEL's native BELGraph representation. It opens the given gzip-compressed file in text mode, decodes the contained JSON document with the standard json loader, and then delegates construction of the BELGraph to the from_graphdati parser, which expects the JSON to follow the GraphDati schema referenced by the GraphDati project.
    
    Args:
        path (str): The filesystem path to a gzip-compressed file containing a single GraphDati JSON document. This is typically a filename ending in ".gz" that was produced by tooling which exports GraphDati/GraphDati-compatible network JSON (for example, BioDati Studio exports). The path must be accessible to the running process; the function does not modify the file on disk and uses a context manager to ensure the file is closed after reading.
    
    Returns:
        pybel.struct.graph.BELGraph: A BELGraph constructed from the GraphDati JSON content. The returned BELGraph is the PyBEL in-memory graph object representing nodes, edges, annotations, and metadata as translated from the GraphDati schema; it can be used with PyBEL functions for summarization, grounding, exporting to other formats (Node-Link JSON, CX, BEL), and graph analysis.
    
    Behavior, side effects, and failure modes:
        - The function opens the file using gzip.open(path, "rt"), reading text using the interpreter's default text encoding unless the environment overrides it.
        - It parses the file contents with json.load and then calls from_graphdati(parsed_json) to produce the BELGraph.
        - The only side effect is reading the provided file; no files are written or global state mutated.
        - Common exceptions that can be raised by this function include: FileNotFoundError or PermissionError if the given path is not available or not readable; OSError or gzip.BadGzipFile if the file is not a valid gzip archive; json.JSONDecodeError if the file contents are not valid JSON; and errors (e.g., ValueError, KeyError) propagated from from_graphdati if the JSON does not conform to the expected GraphDati schema or required fields are missing. Callers should handle or propagate these exceptions as appropriate for their application.
    """
    from pybel.io.graphdati import from_graphdati_gz
    return from_graphdati_gz(path)


################################################################################
# Source: pybel.io.graphdati.from_graphdati_jsons
# File: pybel/io/graphdati.py
# Category: fix_docstring
# Reason: Schema parsing failed: Cannot generate JSON schema for from_graphdati_jsons because the docstring has no description for the argument 's'
################################################################################

def pybel_io_graphdati_from_graphdati_jsons(s: str):
    """pybel.io.graphdati.from_graphdati_jsons: Load a BELGraph from a GraphDati JSON string and convert it into PyBEL's BELGraph data structure used for biological networks.
    
    Args:
        s (str): A GraphDati-formatted JSON document provided as a Python string (typically UTF-8 text). This string must be a complete JSON serialization of a graph conforming to the GraphDati schemas used by BioDati Studio and the GraphDati export format that PyBEL can import (for example, output produced by pybel.dump(..., '...graphdati.json')). The function will call json.loads(s) to parse this string into Python objects and then delegate to pybel.io.graphdati.from_graphdati to convert that parsed mapping into a BELGraph. Provide a valid JSON string; passing non-JSON text will raise json.JSONDecodeError. Passing a value that is not a str (for example bytes) is not supported by this wrapper and may raise a TypeError before parsing.
    
    Returns:
        BELGraph: A newly constructed pybel.struct.graph.BELGraph representing the biological network encoded in the GraphDati JSON. The returned BELGraph contains nodes, edges, namespaces, annotations, citations, and other metadata reconstructed from the GraphDati representation and is suitable for further PyBEL operations such as analysis (graph.summarize.*), serialization to other formats (pybel.dump), grounding, or display in Jupyter. No file I/O is performed by this function; it does not modify the input string. Errors raised during conversion may include json.JSONDecodeError for invalid JSON and any exceptions propagated from pybel.io.graphdati.from_graphdati if required GraphDati fields are missing or malformed. For very large GraphDati JSON strings, parsing and conversion may be memory- and CPU-intensive.
    """
    from pybel.io.graphdati import from_graphdati_jsons
    return from_graphdati_jsons(s)


################################################################################
# Source: pybel.io.hetionet.hetionet.from_hetionet_gz
# File: pybel/io/hetionet/hetionet.py
# Category: fix_docstring
# Reason: Schema parsing failed: Cannot generate JSON schema for from_hetionet_gz because the docstring has no description for the argument 'path'
################################################################################

def pybel_io_hetionet_hetionet_from_hetionet_gz(path: str):
    """pybel.io.hetionet.hetionet.from_hetionet_gz retrieves a Hetionet-formatted JSON file from disk, decompresses it, parses it into PyBEL's BELGraph data structure, and returns that graph for downstream biological-network analysis and format interchange. In the PyBEL domain, this function is used to import Hetionet (a consolidated biomedical knowledge network) into a BELGraph so it can be validated, summarized, serialized to other formats (NetworkX, CX, GraphML, INDRA, etc.), grounded against controlled vocabularies, and used by analytical tools and pipelines described in the PyBEL README.
    
    Args:
        path (str): Filesystem path to a compressed Hetionet JSON file. The function opens the path with bz2.open (i.e., it expects a bzip2-compressed file-like object containing Hetionet JSON). This parameter should be a string accessible to the running process and refer to a readable file. The path is logged at INFO level before opening. Practical significance: supplying the correct path is required to read the compressed Hetionet JSON that from_hetionet_file will parse into a BELGraph; passing an incorrect path, a non-readable file, or a file compressed in an unsupported format (for example gzip .gz instead of bzip2 .bz2) will cause I/O or decompression errors.
    
    Returns:
        pybel.struct.graph.BELGraph: A BELGraph instance representing the Hetionet content parsed from the provided compressed JSON. The returned BELGraph is the primary in-memory PyBEL representation of a biological network and can be used directly with PyBEL APIs for summarization, grounding, exporting to other formats, programmatic analysis, and persistence. The BELGraph is produced by delegating parsing to from_hetionet_file using the decompressed file-like object.
    
    Behavior, side effects, and failure modes:
        This function logs an informational message identifying the path, opens the file at the given path using bz2.open, and passes the resulting file-like object to from_hetionet_file to perform JSON parsing and BELGraph construction. Side effects include reading the file from disk and allocating memory for the resulting BELGraph; large Hetionet files may consume significant memory and take substantial time to parse. Possible failures include FileNotFoundError or PermissionError if the path is invalid or not readable, OSError/EOFError if decompression fails because the file is not a valid bzip2 archive, JSON decoding errors if the decompressed content is not valid Hetionet JSON, and parsing/validation exceptions raised by from_hetionet_file if the content does not match expected Hetionet structure. The function does not perform retries or automatic format detection; callers should ensure the input is a bzip2-compressed JSON file compatible with from_hetionet_file.
    """
    from pybel.io.hetionet.hetionet import from_hetionet_gz
    return from_hetionet_gz(path)


################################################################################
# Source: pybel.io.hipathia.from_hipathia_paths
# File: pybel/io/hipathia.py
# Category: fix_docstring
# Reason: Schema parsing failed: Cannot generate JSON schema for from_hipathia_paths because the docstring has no description for the argument 'name'
################################################################################

def pybel_io_hipathia_from_hipathia_paths(name: str, att_path: str, sif_path: str):
    """Get a BEL graph from HiPathia-formatted files.
    
    This function is a convenience loader that reads two HiPathia files from disk and converts them
    into a PyBEL BELGraph for downstream network analysis and export. In the PyBEL ecosystem,
    HiPathia files (a .att attribute table and a .sif simple interaction file) are commonly used
    to represent pathway topology and accompanying node/edge attributes produced by the HiPathia
    tool for pathway-based analysis. This function uses pandas to parse those files and then
    delegates construction and validation of the BELGraph to from_hipathia_dfs, allowing users
    to interoperate HiPathia outputs with PyBEL workflows (for example, to summarize, serialize,
    or export graphs to other formats supported by PyBEL).
    
    Args:
        name (str): The name to assign to the resulting BELGraph. This becomes the graph-level
            identifier (BELGraph.name) used in summaries, serialization, and exports so that the
            produced graph can be referenced in downstream analyses and provenance records.
        att_path (str): Path to the HiPathia attribute file on disk. The function reads this file
            with pandas.read_csv(..., sep="\t") and expects a tab-separated values (TSV) file
            containing the attributes required by from_hipathia_dfs. The file is loaded into a
            pandas.DataFrame (att_df) and passed through without modification; therefore the
            caller must ensure the ATT file uses a compatible TSV layout and encoding.
        sif_path (str): Path to the HiPathia SIF (simple interaction format) file on disk. The
            function reads this file with pandas.read_csv(..., sep="\t", header=None,
            names=["source", "relation", "target"]) and thus expects a tab-separated file with
            three columns per row representing an interaction edge: source node, relation
            (edge type), and target node. The header is assumed absent; the loader assigns the
            column names "source", "relation", and "target" to the resulting DataFrame (sif_df).
            The produced sif_df is passed to from_hipathia_dfs for graph construction.
    
    Returns:
        pybel.struct.graph.BELGraph: A BELGraph constructed from the provided HiPathia files. The
        returned object encodes nodes and edges derived from the SIF topology and attributes
        supplied in the ATT file. This BELGraph can be inspected with BELGraph.summarize,
        serialized using pybel.dump, exported to other formats supported by PyBEL, or used in
        further programmatic analyses.
    
    Behavior, side effects, defaults, and failure modes:
        - Side effects: The function reads the two input files into memory using pandas, which
          may consume significant memory for very large HiPathia files. It does not write to disk
          or modify the input files. All data transformation and graph construction is performed
          in-memory and delegated to from_hipathia_dfs.
        - Defaults and parsing details: att_path is parsed with pandas.read_csv using sep="\\t".
          sif_path is parsed with pandas.read_csv using sep="\\t", header=None, and the explicit
          column names ["source", "relation", "target"] so callers must provide files compatible
          with these conventions (no header row for SIF, tab-separated columns).
        - Common failure modes: If att_path or sif_path do not exist or are inaccessible, a
          FileNotFoundError or an OSError will be raised by pandas. If the files are not valid TSV
          or contain malformed rows, pandas may raise pandas.errors.ParserError or similar. If the
          loaded DataFrames are missing expected columns or contain unexpected values, the
          downstream from_hipathia_dfs call may raise ValueError or other validation errors.
          Unicode/encoding issues in the files may raise UnicodeDecodeError. Callers should
          validate file existence, permissions, and encoding before invoking this function.
        - Validation and provenance: Any semantic validation, namespace grounding, or BEL-specific
          normalization is handled by from_hipathia_dfs; this loader does not perform additional
          BEL validation beyond reading the input TSVs and passing DataFrames forward.
    """
    from pybel.io.hipathia import from_hipathia_paths
    return from_hipathia_paths(name, att_path, sif_path)


################################################################################
# Source: pybel.io.hipathia.group_delimited_list
# File: pybel/io/hipathia.py
# Category: fix_docstring
# Reason: Schema parsing failed: Cannot generate JSON schema for group_delimited_list because the docstring has no description for the argument 'entries'
################################################################################

def pybel_io_hipathia_group_delimited_list(entries: List[str], sep: str = "/"):
    """pybel.io.hipathia.group_delimited_list groups a flat sequence of string entries into ordered sublists by treating a particular string token as a delimiter. This function is used in the HiPathia I/O codepath of PyBEL (pybel.io.hipathia) to convert flat, serialized lists (for example, path components, segment lists, or other exported string sequences) into the grouped list-of-lists structure expected by downstream HiPathia exporters and analytic consumers.
    
    The implementation partitions the input by contiguous runs of elements that are not equal to the delimiter token sep using itertools.groupby. Delimiter elements (entries exactly equal to sep) are treated as boundaries and are omitted from the output groups; only the non-delimiter runs are returned, in the original order.
    
    Args:
        entries (List[str]): The input flat list of string tokens to be grouped. Each element is compared for equality against sep; elements exactly equal to sep act as delimiters and are not included in any output sublist. In the PyBEL/HiPathia context, entries typically represent serialized components (e.g., path or expression tokens) produced during export. Passing an empty list returns an empty result. Although the annotation is List[str], passing non-string elements may lead to unexpected comparison behavior since grouping is based on equality with sep.
        sep (str): The delimiter token string that marks boundaries between groups. Its default value is "/" (as in the function signature). The function treats only list elements that are exactly equal to this string as separators; it does not split individual strings containing this character nor perform substring matching. If no element equals sep, the entire entries list is returned as a single group.
    
    Returns:
        List[List[str]]: A new list where each element is a sublist of contiguous entries from the input that were not equal to sep. Each sublist preserves the relative order of those elements from the original entries list. No separator tokens appear in the returned sublists. Examples of behavior: an empty input yields [], an input with no separators yields [entries], and consecutive separators do not produce empty sublists between them.
    
    Behavior, side effects, and failure modes:
        This function is pure and has no side effects; it constructs and returns new list objects. It runs in linear time relative to the length of entries (one pass using itertools.groupby) and uses additional memory proportional to the size of the returned groups. Because delimiting is performed by equality comparison (element == sep), elements that are not strings or a sep value with a different type may result in comparisons that are always False or otherwise unexpected; callers should ensure entries are strings and sep is a string as annotated.
    """
    from pybel.io.hipathia import group_delimited_list
    return group_delimited_list(entries, sep)


################################################################################
# Source: pybel.io.jgif.from_cbn_jgif
# File: pybel/io/jgif.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def pybel_io_jgif_from_cbn_jgif(graph_jgif_dict: dict):
    """pybel.io.jgif.from_cbn_jgif builds a PyBEL BELGraph from a JGIF JSON object produced by the Causal Biological Network (CBN) Database by mapping CBN-specific JGIF fields to the standard JGIF namespace and annotation URLs and then delegating graph construction to pybel.from_jgif.
    
    This function is used when integrating networks distributed by the CBN Database into PyBEL workflows described in the project README: it converts the CBN JGIF representation (a JSON graph format used by the Causal Biological Network Database) into a BELGraph suitable for downstream analysis, visualization, export, and interchange with tools that PyBEL supports (for example, NetworkX, Node-Link JSON, CX/NDEx, and others). The implementation first calls map_cbn to translate CBN-specific naming/structure, then inserts the constants GRAPH_NAMESPACE_URL and GRAPH_ANNOTATION_URL with NAMESPACE_URLS and ANNOTATION_URLS into the graph's top-level "graph" object, updates graph["metadata"] with METADATA_AUTHORS, METADATA_LICENSES, and METADATA_CONTACT, and finally calls pybel.from_jgif to produce the BELGraph.
    
    Args:
        graph_jgif_dict (dict): The JSON object representing the graph in JGIF format as provided by the Causal Biological Network Database API or a saved JGIF file. This parameter is expected to be a Python dict parsed from the CBN JGIF JSON; it should contain at minimum the top-level "graph" mapping and a nested "metadata" mapping as required by the JGIF convention. The dict is passed through map_cbn() to normalize CBN-specific fields to the standard JGIF keys, and then the function updates graph_jgif_dict["graph"] by setting GRAPH_NAMESPACE_URL to NAMESPACE_URLS and GRAPH_ANNOTATION_URL to ANNOTATION_URLS and by updating graph_jgif_dict["graph"]["metadata"] with METADATA_AUTHORS, METADATA_LICENSES, and METADATA_CONTACT. Callers should therefore be aware that the object they pass may be mutated (modified in place) by this function or that the function may replace it with a mapped version returned by map_cbn; subsequent code that holds references to the same dict may observe those changes.
    
    Returns:
        BELGraph: A PyBEL BELGraph constructed from the (mapped) JGIF dictionary by calling pybel.from_jgif. The returned BELGraph is a PyBEL in-memory representation of the biological network encoded by the input JGIF and is suitable for all PyBEL operations (summarization, serialization to BEL, Node-Link JSON, CX/NDEx, grounding, analysis pipelines, etc.) as described in the project README.
    
    Behavior and side effects:
        The function normalizes CBN-specific JGIF structure via map_cbn(graph_jgif_dict), sets the graph-level namespace and annotation URL mappings using the module constants GRAPH_NAMESPACE_URL and GRAPH_ANNOTATION_URL to NAMESPACE_URLS and ANNOTATION_URLS, and augments the "metadata" section with authorship, license text, and contact information (METADATA_AUTHORS, METADATA_LICENSES, METADATA_CONTACT). These modifications prepare the JGIF for pybel.from_jgif and may mutate the provided dict or the object returned by map_cbn. The function then invokes pybel.from_jgif to construct the BELGraph from the adjusted JGIF representation and returns that BELGraph.
    
    Failure modes and errors:
        If graph_jgif_dict is not a dict or lacks the expected "graph" or "metadata" mappings, the function may raise TypeError or KeyError. map_cbn() or pybel.from_jgif() may raise ValueError, TypeError, or other exceptions if the input JGIF is malformed or contains unsupported constructs; callers should handle these exceptions as appropriate for their workflow. The function does not perform network I/O; it only processes the provided dict.
    
    Notes and limitations:
        The CBN JGIF documents contain annotations whose provenance (the resources used to create them) is not encoded in the documents. Handling of those annotations is not yet supported by this function; annotations may need to be stripped before uploading graphs to the CBN network store using pybel.struct.mutation.strip_annotations. The function also inserts CBN-specific metadata (a license notice and contact) into the graph metadata field to preserve provenance of the source.
    """
    from pybel.io.jgif import from_cbn_jgif
    return from_cbn_jgif(graph_jgif_dict)


################################################################################
# Source: pybel.io.jgif.from_jgif_gz
# File: pybel/io/jgif.py
# Category: fix_docstring
# Reason: Schema parsing failed: Cannot generate JSON schema for from_jgif_gz because the docstring has no description for the argument 'path'
################################################################################

def pybel_io_jgif_from_jgif_gz(path: str):
    """pybel.io.jgif.from_jgif_gz reads a JGIF (JSON Graph) representation of a BEL graph from a gzip-compressed file and reconstructs it as a PyBEL BELGraph.
    
    This is a convenience wrapper used in the PyBEL I/O suite to load graphs exported in the JGIF/JSON Graph specification (for example, exports intended for network viewers or interchange) that have been compressed with gzip. The function opens the file in text mode, parses the contained JSON using the standard json library, and delegates to pybel.io.jgif.from_jgif to convert the JSON object into a pybel.struct.graph.BELGraph. The resulting BELGraph is PyBEL's in-memory representation of a Biological Expression Language (BEL) network and contains nodes (biological entities such as Proteins, Complexes, Abundances), edges (BEL relations), namespaces, annotations, citations, and other metadata that downstream PyBEL tools and exports expect.
    
    Args:
        path (str): The filesystem path to the gzip-compressed file containing JGIF JSON. This must be a path string pointing to a .gz (or other gzip-compressed) file whose uncompressed content is a JSON document conforming to the JGIF/JSON Graph structure that pybel.io.jgif.from_jgif can consume (for example, an output of pybel.io.jgif.to_jgif or other JGIF-compliant exporters). The function opens the file with gzip.open(path, "rt") (text mode) and thus decodes the compressed bytes into text using the interpreter's default text encoding unless an environment or wrapper changes that behavior.
    
    Returns:
        pybel.struct.graph.BELGraph: A BELGraph reconstructed from the JSON content. This object is PyBEL's primary in-memory graph representation for BEL networks and is ready for PyBEL analyses, serialization to other formats (Node-Link JSON, CX, GraphML, BEL script), validation, grounding, and visualization. The returned graph encapsulates nodes, edges, namespaces, annotations, citations, and other BEL metadata parsed from the JGIF JSON.
    
    Raises:
        FileNotFoundError: If the given path does not exist.
        OSError: If the file cannot be opened as a gzip file or there is an I/O error reading it.
        json.JSONDecodeError: If the uncompressed file contents are not valid JSON.
        Exception: Any exception raised by pybel.io.jgif.from_jgif while converting the parsed JSON to a BELGraph (for example, validation errors, missing required fields, or internal conversion errors) will propagate to the caller.
    
    Behavior and side effects:
        The function has no side effects other than reading from disk. It does not modify the input file or write any output. It performs three main steps: open gzip file in text mode, parse JSON from the file, and convert the JSON into a BELGraph by calling pybel.io.jgif.from_jgif. Consumers should ensure the JSON structure matches what from_jgif expects (JGIF representation of a BEL graph).
    """
    from pybel.io.jgif import from_jgif_gz
    return from_jgif_gz(path)


################################################################################
# Source: pybel.io.jgif.from_jgif_jsons
# File: pybel/io/jgif.py
# Category: fix_docstring
# Reason: Schema parsing failed: Cannot generate JSON schema for from_jgif_jsons because the docstring has no description for the argument 'graph_json_str'
################################################################################

def pybel_io_jgif_from_jgif_jsons(graph_json_str: str):
    """pybel.io.jgif.from_jgif_jsons reads a BELGraph from a JGIF-format JSON string.
    
    This function is part of PyBEL's I/O utilities for interoperability between BEL graphs and JSON-based network formats. It accepts a string containing a JSON document that conforms to the JGIF (JSON Graph Interchange Format) representation of a BEL graph and returns a pybel.struct.graph.BELGraph, the in-memory PyBEL representation used throughout PyBEL for storing biological networks encoded in the Biological Expression Language (BEL). Internally, the function decodes the JSON string with the standard library json.loads and delegates the construction of the BELGraph to pybel.io.jgif.from_jgif, enabling downstream analysis, serialization to other formats (for example Node-Link JSON, GraphML, CX), visualization in tools like Cytoscape, or further PyBEL processing such as grounding and summarization.
    
    Args:
        graph_json_str (str): A JSON document encoded as a Python string. This string must contain a JGIF-conformant JSON representation of a BEL graph (as used by PyBEL for exchanging BEL graphs with tools and viewers). The practical role of this parameter is to provide the serialized network data that will be parsed and converted into a PyBEL BELGraph. This function does not perform any file I/O; the caller is responsible for reading JSON text from files, network responses, or other sources and supplying it as this string.
    
    Returns:
        BELGraph: A pybel.struct.graph.BELGraph instance representing the parsed BEL network. This return value is the central in-memory graph object used by PyBEL for analyses (summaries, grounding, exports, and algorithmic processing). The constructed BELGraph contains the nodes, edges, annotations, and citations encoded in the input JGIF JSON and can be passed to other PyBEL functions for further manipulation or export.
    
    Raises:
        json.JSONDecodeError: If graph_json_str is not valid JSON, the underlying json.loads call will raise this exception; it is propagated to the caller.
        Exception: Any exceptions raised by pybel.io.jgif.from_jgif when the decoded JSON does not conform to the expected JGIF/BEL graph structure (for example malformed node or edge fields) are propagated. Callers should handle or report these errors as appropriate for their data ingestion pipeline.
    
    Behavior and side effects:
        - No file or network I/O is performed by this function; it only parses the provided string and constructs a BELGraph in memory.
        - The function relies on the JGIF structure and semantics expected by pybel.io.jgif.from_jgif; providing JSON that deviates from that expectation may result in exceptions.
        - Successful execution returns a ready-to-use BELGraph for PyBEL workflows such as serialization to other formats, visualization, grounding with identifiers, and graph analyses described in the PyBEL documentation.
    """
    from pybel.io.jgif import from_jgif_jsons
    return from_jgif_jsons(graph_json_str)


################################################################################
# Source: pybel.io.jgif.map_cbn
# File: pybel/io/jgif.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def pybel_io_jgif_map_cbn(d: dict):
    """pybel.io.jgif.map_cbn pre-processes a JGIF (JSON Graph Interchange Format) document produced by the CBN to normalize experimental-context evidence attached to edges so the data can be more reliably consumed by PyBEL I/O pipelines. In the biological-network domain (PyBEL), this function standardizes per-evidence annotation keys (for example mapping "species_common_name" to a canonical "Species" value via a species_map and remapping other keys via annotation_map), strips surrounding whitespace from keys and values, drops empty/placeholder values, and replaces each evidence's EXPERIMENT_CONTEXT mapping with the cleaned, standardized mapping.
    
    Args:
        d (dict): Raw JGIF dictionary output from the CBN. This function expects d to follow the JGIF structure used by the CBN: a top-level "graph" key containing an "edges" list. Each edge may contain a "metadata" mapping with an "evidences" list; each evidence may contain an EXPERIMENT_CONTEXT mapping of annotation keys to string values. The parameter d is both the input and the object that will be modified in place: the function updates d["graph"]["edges"][i]["metadata"]["evidences"][j][EXPERIMENT_CONTEXT] with a new normalized dictionary for each evidence that contains EXPERIMENT_CONTEXT.
    
    Returns:
        dict: The same JGIF dictionary object passed in as d, after in-place preprocessing. The returned dict has had each evidence's EXPERIMENT_CONTEXT replaced by a cleaned mapping where keys are stripped and lowercased (then mapped via annotation_map when applicable), empty or whitespace-only values are removed, and "species_common_name" values are converted to a canonical Species entry using species_map. This function mutates the input and returns it for convenience; callers that must preserve the original should pass a deep copy.
    
    Behavior, side effects, and failure modes:
        This function iterates over d["graph"]["edges"] and inspects each edge's "metadata" and "evidences". If an edge lacks "metadata" or "evidences", it is left unchanged and skipped. For each evidence that contains EXPERIMENT_CONTEXT, the function constructs a new mapping new_context by:
        - ignoring any key/value pairs whose value is empty or only whitespace (these pairs are omitted and a debug log entry is emitted),
        - stripping surrounding whitespace from values,
        - lowercasing and stripping keys before mapping,
        - mapping "species_common_name" values to a canonical Species value via the module-level species_map (using value.lower() as the lookup key),
        - mapping other known context keys via the module-level annotation_map to standardized annotation names,
        - preserving unknown keys (after lowercasing and stripping) under their cleaned name.
        After processing, the evidence's EXPERIMENT_CONTEXT is replaced with new_context.
    
        Side effects: the input dict d is modified in place. The function relies on module-level names EXPERIMENT_CONTEXT, annotation_map, and species_map to be defined and populated appropriately; they determine which key is treated as the experimental context container and how keys/values are remapped.
    
        Failure modes and exceptions:
        - A KeyError will be raised if the expected top-level keys ("graph" or "edges") are missing from d because the function directly indexes d["graph"]["edges"].
        - A TypeError may be raised if d is not a dict or if d["graph"]["edges"] is not an iterable of mappings with the expected structure.
        - A KeyError may occur when using species_map if a species value is not present in species_map.
        - A NameError will occur if EXPERIMENT_CONTEXT, annotation_map, or species_map are not defined at the module level.
        - No edges are removed by this function; edges that lack evidence or an EXPERIMENT_CONTEXT are simply not modified. If callers require removal of edges without evidence, that must be performed separately.
    
        Practical significance: in PyBEL workflows that import causal-network outputs (CBN) via JGIF, consistent and cleaned experimental-context annotations are necessary for later steps such as grounding, annotation-aware filtering, and conversion to BELGraph nodes/edges. This function centralizes that normalization so subsequent I/O functions can rely on a standardized context structure.
    """
    from pybel.io.jgif import map_cbn
    return map_cbn(d)


################################################################################
# Source: pybel.io.jinja_utils.build_template_environment
# File: pybel/io/jinja_utils.py
# Category: valid
################################################################################

def pybel_io_jinja_utils_build_template_environment(here: str):
    """Build and return a preconfigured jinja2.Environment for use by Flask apps and Jupyter displays in PyBEL.
    
    This function constructs a Jinja2 templating environment that is configured to load templates from a "templates" subdirectory located under the provided base directory and to expose a STATIC_PREFIX global that points to a "static" subpath. In the PyBEL codebase this environment is used when rendering BEL graph views in Jupyter and when serving templates from Flask-based applications so that templates and static assets (CSS/JS/images) can be located relative to a module's file system location. The returned Environment is created with autoescape=True and trim_blocks=False and uses a FileSystemLoader rooted at os.path.join(here, "templates").
    
    Args:
        here (str): Filesystem path to the directory that should serve as the base for template and static asset lookup. In PyBEL this is intended to be the directory of the calling module and is commonly provided as os.path.dirname(os.path.abspath(__file__)). The function constructs the templates search path as os.path.join(here, "templates") and sets a global STATIC_PREFIX equal to here + "/static/". The function does not itself create or validate the existence of the "templates" or "static" directories; if they are missing, template resolution or static asset references will fail later when rendering or serving.
    
    Returns:
        jinja2.Environment: A configured Jinja2 Environment object with the following concrete properties relevant to PyBEL:
            - loader: a FileSystemLoader pointed at the "templates" subdirectory under the provided here path.
            - autoescape: True, enabling automatic escaping for templates (suitable for HTML output).
            - trim_blocks: False, preserving block trailing newlines.
            - globals["STATIC_PREFIX"]: set to the string here + "/static/", providing a template-accessible prefix for static assets.
        This Environment is returned to the caller for use in rendering templates; the function performs no rendering itself and has no disk-write side effects.
    
    Notes on behavior and failure modes:
        - The function imports jinja2's Environment and FileSystemLoader at call time; if the jinja2 package is not installed, calling this function will raise ImportError.
        - The function expects here to be a string. Passing a non-string value may lead to TypeError or incorrect path construction when the loader is created or when templates are looked up.
        - Template lookup failures occur at render time if the "templates" directory does not exist or lacks the requested templates; this function does not raise on creation in that case.
        - The STATIC_PREFIX value is constructed by simple string concatenation using a forward slash (here + "/static/"), which mirrors the original code behavior; on some platforms or deployment setups you may need to normalize the path before using it as a filesystem path (the value is primarily intended for template references/URLs).
    """
    from pybel.io.jinja_utils import build_template_environment
    return build_template_environment(here)


################################################################################
# Source: pybel.io.jinja_utils.build_template_renderer
# File: pybel/io/jinja_utils.py
# Category: valid
################################################################################

def pybel_io_jinja_utils_build_template_renderer(path: str):
    """Build a render-template function that locates Jinja2 templates relative to a module file and returns a closure for rendering them.
    
    Args:
        path (str): The filesystem path of the current file used to locate the template directory. In practice call this with the module __file__ value (for example, render_template = build_template_renderer(__file__)). The function uses os.path.abspath and os.path.dirname on this path to determine the template search directory; the path must be a string path to a file within the package that contains the templates.
    
    This function constructs a Jinja2 template environment rooted at the directory containing the provided path by calling build_template_environment(here) where here = os.path.dirname(os.path.abspath(path)). It returns a closure that, when invoked, will load a template by filename from that environment and render it to a Python str using the provided rendering context. The returned closure is intended for use in I/O and display code in PyBEL (for example, rendering HTML fragments for Jupyter display of BEL graphs), and therefore requires Jinja2 to be available in the runtime. The closure attaches the underlying Jinja2 Environment to its environment attribute so callers can inspect or modify loaders, filters, globals, and other environment-level settings before or after rendering.
    
    Behavior and side effects:
    - The call to build_template_renderer does not itself read or render templates; it constructs a Jinja2 Environment configured to load templates from the directory computed from path.
    - The returned closure, when called, will synchronously load the named template from the filesystem (via the environment's loader) and render it with the provided context into a Python str. The closure signature is (template_filename: str, **context) -> str.
    - The returned closure has an attribute environment pointing to the created Jinja2 Environment object; modifying that object (for example, adding filters) affects subsequent renders.
    - No defaults are provided for path; it must be supplied.
    
    Failure modes and exceptions:
    - If Jinja2 is not installed or build_template_environment fails to construct an environment, an ImportError or the originating exception will be raised when build_template_renderer is called.
    - When rendering, template loading or rendering errors from Jinja2 (for example, jinja2.exceptions.TemplateNotFound, TemplateSyntaxError, or runtime errors raised during template evaluation) will propagate to the caller.
    - Providing a non-string path will result in a TypeError when os.path.abspath is called or otherwise fail; callers should pass a str.
    
    Returns:
        function: A callable render_template_enclosure(template_filename: str, **context) -> str that loads template_filename from the environment rooted at the directory of path and returns the rendered template as a Python str. The returned callable has an attribute environment referencing the Jinja2 Environment used for loading and rendering templates.
    """
    from pybel.io.jinja_utils import build_template_renderer
    return build_template_renderer(path)


################################################################################
# Source: pybel.io.lines.from_bel_script_gz
# File: pybel/io/lines.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def pybel_io_lines_from_bel_script_gz(path: str, **kwargs):
    """pybel.io.lines.from_bel_script_gz parses a gzipped BEL Script file and returns a compiled BELGraph.
    
    This function is used in the PyBEL I/O pipeline to read BEL documents that have been compressed with gzip (commonly carrying the .bel.gz extension) and to convert the textual BEL statements into a BELGraph, PyBEL's in-memory representation of a biological network encoded in the Biological Expression Language (BEL). It opens the file at the given filesystem path in text mode (gzip.open(..., "rt")), decodes the text, and delegates parsing and graph construction to pybel.io.lines.from_bel_script, forwarding any keyword arguments. The resulting BELGraph can then be used with PyBEL features (serialization, analysis, grounding, exports to formats like CX/GraphML/Node-Link JSON, and downstream tools described in the README).
    
    Args:
        path (str): Filesystem path to a gzipped BEL Script file. This should be a path-like string pointing to a file whose contents are a BEL Script compressed with gzip (for example, a file named "my_graph.bel.gz"). The function opens this path for reading in text mode using gzip.open(path, "rt"), so the caller must ensure the file exists and is a valid gzip file containing textual BEL statements encoded in a decodable text encoding. This parameter is the primary input used to locate and read the BEL Script to be parsed into a BELGraph.
        kwargs (dict): Additional keyword arguments forwarded directly to pybel.io.lines.from_bel_script. These control parsing behavior and options of the BEL-to-BELGraph compilation step (for example, parser strictness, BEL version handling, annotation handling, citation/namespace resolution, or other parser-specific flags supported by from_bel_script). The exact accepted keys, defaults, and semantics are defined by from_bel_script; callers should consult that function's documentation. Using kwargs allows callers to customize parsing to support different BEL versions (BEL 1.0 vs BEL 2.0+), error handling, and other domain-specific parsing behaviors documented in PyBEL.
    
    Returns:
        BELGraph: A BELGraph instance containing the nodes, edges, annotations, citations, and metadata extracted from the BEL Script. The BELGraph is PyBEL's primary in-memory biological network representation and can be used for further processing such as grounding, summarization, serialization to other formats (Node-Link JSON, CX, GraphML), and analysis with PyBEL tools. The returned graph reflects the compiled semantics of the BEL statements in the gzipped input file.
    
    Raises:
        FileNotFoundError: If the path does not exist.
        OSError: If the file cannot be opened due to I/O errors.
        gzip.BadGzipFile: If the file at path is not a valid gzip file or is corrupted.
        UnicodeDecodeError: If the gzipped file's bytes cannot be decoded as text in the environment's default encoding when opened in text mode.
        Exception: Any exceptions raised by pybel.io.lines.from_bel_script (parsing errors, validation errors, or other parser-specific exceptions) are propagated unchanged. These indicate problems in parsing the BEL statements or in constructing the BELGraph.
    
    Behavior and side effects:
        The function performs a blocking read of the specified file and returns only after parsing and graph construction complete. It does not modify the input file. It opens the gzipped file in text mode ("rt"), so newline translation and text decoding are applied according to the runtime environment; callers concerned about encoding should ensure the file uses a compatible encoding. All keyword arguments are forwarded to the underlying from_bel_script parser, so their defaults and effects are determined by that function. This function is intended for local filesystem paths; it does not fetch remote URLs.
    """
    from pybel.io.lines import from_bel_script_gz
    return from_bel_script_gz(path, **kwargs)


################################################################################
# Source: pybel.io.nodelink.from_nodelink_gz
# File: pybel/io/nodelink.py
# Category: fix_docstring
# Reason: Schema parsing failed: Cannot generate JSON schema for from_nodelink_gz because the docstring has no description for the argument 'path'
################################################################################

def pybel_io_nodelink_from_nodelink_gz(path: str):
    """Read a BEL graph encoded as Node-Link JSON from a gzip-compressed file and convert it to a BELGraph.
    
    This function is used in the PyBEL I/O pipeline to load graphs that were exported in the Node-Link JSON representation (a JSON format compatible with NetworkX/JGIF and used by PyBEL for interchange with viewers and other tools) and compressed with gzip. It opens the given filesystem path in text mode, decodes and parses the JSON payload, and delegates to pybel.io.nodelink.from_nodelink to construct and return a pybel.struct.graph.BELGraph. The resulting BELGraph represents a biological network in BEL and can be used with PyBEL functionality such as grounding, summarization, serialization to other formats (GraphML, CX, INDRA JSON), and analysis.
    
    Args:
        path (str): Filesystem path to the gzip-compressed Node-Link JSON file to read. This must be a path to an existing file accessible by the running process. The function opens the file with gzip.open(..., "rt") (text mode) and passes the decoded text to json.load; therefore the file should contain valid Node-Link JSON text (typically UTF-8 encoded). The path parameter is the primary input identifying which compressed Node-Link JSON artifact to load into a BELGraph.
    
    Returns:
        pybel.struct.graph.BELGraph: A BELGraph constructed from the parsed Node-Link JSON. The returned BELGraph encodes the nodes, edges, namespaces, annotations, and citations present in the Node-Link data and is ready for downstream PyBEL operations (grounding, exporting, summarizing, analysis). The conversion is performed by pybel.io.nodelink.from_nodelink after parsing the JSON.
    
    Behavior and side effects:
        The function reads from the filesystem and has the side effect of allocating in-memory Python objects for the parsed JSON and the resulting BELGraph. It does not write to disk or modify the input file. The gzip file is opened in text mode and closed before returning.
    
    Failure modes and exceptions:
        If the path does not exist or is not readable, a FileNotFoundError or OSError will be raised by gzip.open. If the file is not a valid gzip archive, gzip.BadGzipFile (or an OSError) will be raised. If the file contents are not valid JSON text, json.JSONDecodeError will be raised. If the parsed JSON does not conform to the expected Node-Link structure required by pybel.io.nodelink.from_nodelink, that function may raise ValueError or other exceptions indicating an invalid graph representation. Callers should handle these exceptions as appropriate for their application.
    """
    from pybel.io.nodelink import from_nodelink_gz
    return from_nodelink_gz(path)


################################################################################
# Source: pybel.io.nodelink.from_nodelink_jsons
# File: pybel/io/nodelink.py
# Category: fix_docstring
# Reason: Schema parsing failed: Cannot generate JSON schema for from_nodelink_jsons because the docstring has no description for the argument 'graph_json_str'
################################################################################

def pybel_io_nodelink_from_nodelink_jsons(graph_json_str: str, check_version: bool = True):
    """pybel.io.nodelink.from_nodelink_jsons reads a BEL graph encoded as a Node-Link JSON string and returns a compiled BELGraph usable by PyBEL for analysis, summarization, grounding, and format interconversion.
    
    Args:
        graph_json_str (str): A JSON-formatted string containing a Node-Link representation of a BEL graph. This string is expected to be valid UTF-8 text parseable by json.loads and to follow the Node-Link JSON structure used by PyBEL exports (for example files named '*.bel.nodelink.json' produced by pybel.dump). The Node-Link JSON is the interchange format used for network viewers and tools (e.g., D3 or Cytoscape) and for serializing NetworkX-style graphs; this function will parse that serialized representation into an in-memory BELGraph.
        check_version (bool): Whether to perform format version validation on the parsed Node-Link JSON before constructing the BELGraph. The default is True. When True, the function delegates to pybel.io.nodelink.from_nodelink to validate that the JSON contains the expected version/format metadata and will raise an exception (for example ValueError) if the version or required structural fields are incompatible with the current PyBEL node-link reader. When False, the function skips or relaxes that validation, which can allow loading older or slightly different node-link payloads but may produce a BELGraph with missing or unexpected attributes.
    
    Returns:
        pybel.struct.graph.BELGraph: A BELGraph instance representing the parsed biological network encoded in the Node-Link JSON. The returned BELGraph is a PyBEL graph object that can be inspected, summarized (graph.summarize.*), grounded (pybel.grounding.ground), exported to other formats (GraphML, CX, INDRA JSON, etc.), and used with downstream analysis tools. No files are written by this function; it performs in-memory parsing and graph construction.
    
    Behavior and failure modes:
        This function decodes graph_json_str using json.loads and then delegates to pybel.io.nodelink.from_nodelink with the parsed JSON object and the check_version flag. It requires a string input (not bytes); passing non-string types will raise a TypeError. If graph_json_str is not valid JSON, json.loads will raise json.JSONDecodeError. If the parsed JSON does not conform to the expected node-link structure or fails version validation (when check_version is True), the delegated from_nodelink call will raise a ValueError or other descriptive exception. The function has no other side effects beyond creating and returning the BELGraph.
    """
    from pybel.io.nodelink import from_nodelink_jsons
    return from_nodelink_jsons(graph_json_str, check_version)


################################################################################
# Source: pybel.io.pykeen.get_triples_from_bel
# File: pybel/io/pykeen.py
# Category: valid
################################################################################

def pybel_io_pykeen_get_triples_from_bel(path: str):
    """Get triples from a BEL Script file and return them as a NumPy array suitable for downstream knowledge-graph workflows (for example, PyKEEN training). This function is a thin wrapper that parses a BEL Script using pybel.from_bel_script and then extracts subject-relation-object triples by delegating to the internal _from_bel helper which calls pybel.io.tsv.api.get_triples. It is intended to convert BEL statements (BEL 1.0 and BEL 2.0+) into a standardized three-column representation (head, relation, tail) for interchange and machine-learning pipelines.
    
    Args:
        path (str): The file path to a BEL Script to be read from disk. This should be a path to a local BEL document containing BEL statements in BEL 1.0 or BEL 2.0+ syntax. The function will open and parse the file using pybel.from_bel_script; common failure modes are FileNotFoundError if the path does not exist, and parser errors raised by pybel if the file contains malformed BEL. The parameter is required and must be a string file path.
    
    Returns:
        numpy.ndarray: A two-dimensional numpy.ndarray with shape (n, 3) where n is the number of extracted triples. Each row is a triple in the exact order [head, relation, tail]. The elements are the string serializations produced by pybel.io.tsv.api.get_triples (i.e., the textual representation of the BEL subject and object and the BEL relation). If no triples are found, an array of shape (0, 3) is returned. Exceptions raised by file I/O or the underlying pybel parser are propagated to the caller; callers should handle FileNotFoundError and parsing exceptions as appropriate.
    """
    from pybel.io.pykeen import get_triples_from_bel
    return get_triples_from_bel(path)


################################################################################
# Source: pybel.io.pykeen.get_triples_from_bel_commons
# File: pybel/io/pykeen.py
# Category: valid
################################################################################

def pybel_io_pykeen_get_triples_from_bel_commons(network_id: str):
    """pybel.io.pykeen.get_triples_from_bel_commons loads a BEL document from BEL Commons and returns its statement triples as a 2-D numpy array suitable for downstream conversion or analysis.
    
    This function is a convenience wrapper that converts the given network identifier to a string, invokes the internal BEL Commons loader, and returns the extracted triples in a compact tabular form. In the PyBEL ecosystem (see README), BEL Commons is a web service that hosts BEL networks; this function fetches a BEL document for the specified network and extracts head-relation-tail triples that can be used to compile a BELGraph, export to other formats, or feed downstream tools such as machine-learning pipelines (for example PyKEEN) or graph exporters.
    
    Args:
        network_id (str): The network identifier for a graph in BEL Commons. This is the identifier used by BEL Commons to locate a BEL document; the function will coerce the provided value to str (via str(network_id)) before requesting the resource. Provide a valid BEL Commons network id (as documented by the BEL Commons service) to retrieve that network's BEL content.
    
    Returns:
        numpy.ndarray: A two-dimensional numpy array with shape (n, 3) where each row is a triple and the columns are, in order, head, relation, and tail. Each element in the array contains the textual representation of the node or relation as extracted from the BEL document. The array is suitable for programmatic processing (iteration, conversion to BELGraph, or serialization to TSV/TSV-derived formats).
    
    Behavior, side effects, and failure modes:
        The function performs network I/O to retrieve the BEL document from the BEL Commons service and then parses that document to extract triples. Side effects include network requests and associated latency; it does not mutate caller-provided data structures. If the requested network_id does not correspond to an available BEL document, or if there are network connectivity issues, authentication/permission problems, or parsing failures, the underlying errors raised by the HTTP client, loader, or parser will propagate to the caller. If the BEL document contains no triples, the function will return an empty numpy.ndarray with shape (0, 3).
    """
    from pybel.io.pykeen import get_triples_from_bel_commons
    return get_triples_from_bel_commons(network_id)


################################################################################
# Source: pybel.io.pykeen.get_triples_from_bel_nodelink
# File: pybel/io/pykeen.py
# Category: valid
################################################################################

def pybel_io_pykeen_get_triples_from_bel_nodelink(path: str):
    """Get triples from a BEL Node-Link JSON file and return them as a NumPy array suitable for downstream consumption (for example, by machine-learning tools such as PyKEEN). This function is a thin wrapper that delegates parsing to the internal BEL loader and to the TSV triple extraction implementation used by PyBEL. It is intended to be used when a BEL graph has been serialized in the Node-Link JSON format and you need a flat list of subject–predicate–object triples for export, analysis, or input to embedding/training pipelines.
    
    Args:
        path (str): The filesystem path to a BEL Node-Link JSON file. This must be a path accessible to the running process and point to a file that was written according to PyBEL's BEL node-link JSON conventions (the format used by pybel.dump(..., '...bel.nodelink.json')). The value of this parameter determines which file is opened and parsed; the function reads the file contents and does not modify or overwrite the file.
    
    Behavior and side effects:
        The function opens and reads the file at the given path, parses it as BEL Node-Link JSON, converts the graph representation into triples using PyBEL's TSV triple extraction logic (it wraps pybel.io.tsv.api.get_triples by delegating to the internal helper _from_bel with from_nodelink_file), and returns the triples as a two-dimensional NumPy array. No files are written or persisted by this function. The returned array preserves the order in which triples are produced by the underlying PyBEL parser and extractor. Common failure modes are file-related I/O errors (for example, the path does not exist or is not readable), JSON parsing errors when the file is not valid JSON, or parsing/validation errors when the file contents do not conform to the expected BEL node-link structure; such exceptions raised by the underlying I/O and parsing routines propagate to the caller.
    
    Returns:
        numpy.ndarray: A two-dimensional NumPy array of shape (n, 3) where n is the number of extracted triples. Each row contains three Python strings in the order [head, relation, tail], representing the subject (head) node identifier, the BEL relation label (predicate), and the object (tail) node identifier, respectively. The ndarray is ready for use by downstream consumers that expect a flat triple table (for example, conversion to TSV, input to PyKEEN, or other graph-embedding workflows).
    """
    from pybel.io.pykeen import get_triples_from_bel_nodelink
    return get_triples_from_bel_nodelink(path)


################################################################################
# Source: pybel.io.pykeen.get_triples_from_bel_pickle
# File: pybel/io/pykeen.py
# Category: valid
################################################################################

def pybel_io_pykeen_get_triples_from_bel_pickle(path: str):
    """pybel.io.pykeen.get_triples_from_bel_pickle: Load triples from a BEL pickle file and return them as a NumPy array of (head, relation, tail) rows suitable for downstream consumers such as PyKEEN and other knowledge-graph tooling.
    
    Args:
        path (str): The filesystem path to a BEL pickle file. This should point to a file produced by PyBEL serialization (for example via pybel.dump or other PyBEL save routines) that contains either a serialized BELGraph or an exported triples structure. The function will open and read this file, unpickle its contents, and extract triples. Supplying a path that does not exist, is not readable, or does not contain a PyBEL-compatible pickle will raise an I/O or unpickling-related exception.
    
    Returns:
        numpy.ndarray: A two-dimensional NumPy array with shape (n, 3), where each row is a triple in the order (head, relation, tail). Each element in the array is the string representation of the corresponding node or relation as produced by PyBEL's triple exporter. This array is intended for use by machine-learning and embedding pipelines (for example, PyKEEN or OpenBioLink) that expect triples in row-wise (subject, predicate, object) form. The function returns an empty array with shape (0, 3) if no triples are present in the pickle.
    
    Behavior and side effects:
        This function is a thin wrapper around the internal _from_bel(path, from_pickle) helper which, in turn, delegates to pybel.io.tsv.api.get_triples to perform extraction. The function reads the entire pickle into memory and therefore may consume significant memory for very large BEL graphs or large triple sets. It performs no mutation of the source file. The exact string formatting of heads, relations, and tails follows PyBEL's triple export conventions and will reflect BEL functions, namespaces, and identifiers as represented in the original graph.
    
    Failure modes and errors:
        If the file at path does not exist, a FileNotFoundError or OSError will be raised. If the file is not a valid pickle or is corrupted, the underlying unpickling will raise pickle.UnpicklingError or another exception from the pickle module. If the unpickled object does not contain a structure that _from_bel/from_pickle recognize as extractable triples, a ValueError or a custom extraction error may be raised. Consumers should catch these exceptions and validate that the input pickle was produced by a compatible PyBEL serialization process.
    """
    from pybel.io.pykeen import get_triples_from_bel_pickle
    return get_triples_from_bel_pickle(path)


################################################################################
# Source: pybel.io.sbel.from_sbel_gz
# File: pybel/io/sbel.py
# Category: fix_docstring
# Reason: Schema parsing failed: Cannot generate JSON schema for from_sbel_gz because the docstring has no description for the argument 'path'
################################################################################

def pybel_io_sbel_from_sbel_gz(path: str):
    """pybel.io.sbel.from_sbel_gz reads a BEL JSONL graph from a gzip-compressed file and returns a pybel.struct.graph.BELGraph.
    
    This function opens the file at the given filesystem path using Python's gzip.open in text mode ("rt") and delegates parsing to from_sbel_file(file). The input file is expected to be a gzip-compressed newline-delimited JSON (JSONL) representation of a BEL graph (SBEL/JSONL), where each line encodes a BEL statement or SBEL record. The returned object is a PyBEL BELGraph, the primary in-memory representation of a Biological Expression Language (BEL) network used throughout the PyBEL ecosystem for analysis, serialization, and export to formats and services described in the project README (for example NetworkX/Node-Link JSON, CX/NDEx, GraphML/Cytoscape, INDRA, and others).
    
    Behavior and side effects:
    This function performs I/O by opening and reading the gzip file and will close the file before returning. It relies on from_sbel_file to parse the text stream into a BELGraph; therefore parsing behavior, validation, and any warnings or errors follow from that parser's implementation. The gzip file is opened in text mode using Python's default text encoding and newline handling unless the environment or Python defaults are changed. Memory usage and performance depend on the size of the file and the implementation details of from_sbel_file (e.g., whether it streams or accumulates content).
    
    Failure modes and exceptions:
    The call may raise file-related exceptions (for example FileNotFoundError, OSError) if the path does not exist or is not accessible, gzip.BadGzipFile (or other gzip-related errors) if the file is not a valid gzip archive, and parsing errors propagated from from_sbel_file if the JSONL content is malformed or violates expected SBEL structure. Callers should handle these exceptions as appropriate for their application.
    
    Args:
        path (str): Filesystem path to a gzip-compressed file containing a BEL JSONL (SBEL/JSONL) graph. This should be an absolute or relative path to a .gz file; the file must be a valid gzip archive whose text contents consist of newline-delimited JSON records representing BEL statements or SBEL records. The path is used as the input source for constructing the returned BELGraph.
    
    Returns:
        BELGraph: A pybel.struct.graph.BELGraph instance built by parsing the provided gzip-compressed BEL JSONL file. The BELGraph represents the parsed biological network (nodes, edges, annotations, citations, and metadata) and can be used with PyBEL I/O and analysis functions (for example pybel.dump for serialization and the BELGraph.summarize dispatch for summaries).
    """
    from pybel.io.sbel import from_sbel_gz
    return from_sbel_gz(path)



################################################################################
# Source: pybel.manager.cache_manager.not_resource_cachable
# File: pybel/manager/cache_manager.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def pybel_manager_cache_manager_not_resource_cachable(bel_resource: dict):
    """Check if a BEL resource entry should not be cached.
    
    This function inspects the dictionary metadata for a BEL resource (as returned by
    get_bel_resource) and determines whether the resource is considered non-cacheable.
    Within the PyBEL project this is used by the cache manager to decide whether to
    persist a downloaded or generated resource to the local cache or other storage.
    The function expects the resource dictionary to contain a "Processing" mapping
    with an optional "CacheableFlag" string value. The function treats the resource
    as cacheable only when the "CacheableFlag" exactly equals one of the
    case-sensitive strings: "yes", "Yes", "True", or "true". Any other value,
    including a missing "CacheableFlag" or None, is interpreted as non-cacheable.
    
    Args:
        bel_resource (dict): A dictionary representing a BEL resource metadata
            returned by get_bel_resource. In practice this dict must contain a
            "Processing" key whose value is a mapping (dict-like) that may include
            the "CacheableFlag" key. The "CacheableFlag" value is expected to be a
            string indicating whether the resource may be cached; accepted
            cacheable string values are "yes", "Yes", "True", and "true".
    
    Returns:
        bool: True if the resource should NOT be cached (i.e., it is non-cacheable);
        False if the resource is explicitly marked cacheable by having
        "CacheableFlag" equal to one of "yes", "Yes", "True", or "true". There are
        no side effects. Failure modes: if bel_resource is not a dict or if the
        "Processing" key is missing, a KeyError may be raised; if
        bel_resource["Processing"] is present but not a mapping with a .get
        method, an AttributeError may be raised. The function performs no mutation
        of its input.
    """
    from pybel.manager.cache_manager import not_resource_cachable
    return not_resource_cachable(bel_resource)


################################################################################
# Source: pybel.manager.citation_utils.sanitize_date
# File: pybel/manager/citation_utils.py
# Category: fix_docstring
# Reason: Schema parsing failed: Cannot generate JSON schema for sanitize_date because the docstring has no description for the argument 'publication_date'
################################################################################

def pybel_manager_citation_utils_sanitize_date(publication_date: str):
    """Sanitize a variety of free-form publication date strings into an ISO-8601 date string (YYYY-MM-DD) for use in citation metadata and BEL graph versioning within PyBEL.
    
    This function is used by PyBEL's citation utilities (pybel.manager.citation_utils) to convert heterogeneous date representations found in BEL documents, external data sources, or citation records into a consistent ISO-8601 format required for downstream storage, comparison, and display (for example, in graph summaries, exports, and database fields).
    
    Args:
        publication_date (str): The raw publication date string to normalize. This is typically a date extracted from a citation record or BEL document (for example, the "date" field when compiling or loading BEL content). The function expects a Python str and will apply a sequence of compiled regular-expression checks and datetime parsing to interpret common forms of publication dates. Recognized forms derived from the implementation include an exact year-month-day with abbreviated month (parsed with "%Y %b %d"), year and abbreviated month ("%Y %b" -> day defaults to "01"), a bare four-digit year (interpreted as YYYY-01-01), seasonal expressions (year plus a season token mapped via the internal season_map to a month), and variants that include hyphenated qualifiers after the day (these are matched and incorporated into the parsing format). The input is not modified in place; it is read and converted. If the string does not match any of the handled patterns or if parsing fails, datetime.strptime can raise ValueError; if a non-str is passed, a TypeError may be raised when regular-expression functions are applied.
    
    Returns:
        str: An ISO-8601 formatted date string ("YYYY-MM-DD") suitable for use in PyBEL citation metadata and graph versioning. When only a year or year+month are provided, the function uses sensible defaults consistent with the implementation: missing month or day values become "01" so that the returned string is a complete YYYY-MM-DD date. The mapping from seasonal tokens to month numbers is performed by the module-level season_map used by this function. The function has no external side effects (it does not modify files, global state, or the input object).
    """
    from pybel.manager.citation_utils import sanitize_date
    return sanitize_date(publication_date)


################################################################################
# Source: pybel.manager.utils.extract_shared_optional
# File: pybel/manager/utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def pybel_manager_utils_extract_shared_optional(
    bel_resource: dict,
    definition_header: str = "Namespace"
):
    """Extract the optional annotations that are shared between BEL namespace documents and BEL annotation resource documents.
    
    This function inspects a BEL resource configuration dictionary (for example, a parsed Namespace or AnnotationDefinition document used by PyBEL to describe controlled vocabularies and annotation metadata) and returns a new dictionary containing a canonical set of optional metadata fields that are common to both namespace and annotation resources. It applies a fixed mapping (shared_mapping) from canonical keys used by PyBEL (such as "description", "version", "author", "license", "contact", "citation", etc.) to the locations and element names expected inside a BEL resource document. The function writes values into a fresh result dictionary using update_insert_values and, if the resource contains a Citation.PublishedDate, converts that string into a datetime via parse_datetime and stores it under the key "citation_published". The returned mapping is intended for use by PyBEL components that assemble resource metadata (for example, when loading namespaces or annotations to ground BEL graphs or to include descriptive metadata in BELGraph exports).
    
    Args:
        bel_resource (dict): A configuration dictionary representing a BEL resource document. In PyBEL this is typically the parsed content of a Namespace or AnnotationDefinition resource and is expected to contain sections like the definition header (e.g., "Namespace" or "AnnotationDefinition"), "Author", and "Citation" with nested string fields such as "NameString", "DescriptionString", "VersionString", "ContactInfoString", "CopyrightString", "ReferenceURL", and "PublishedDate". This parameter is read by the function; the function does not mutate bel_resource but will read its nested keys to populate the returned mapping. Supplying a non-dict value will cause a TypeError when the function attempts dict operations.
        definition_header (str): The top-level definition header name to use when mapping fields in bel_resource. By default this is "Namespace", matching typical BEL namespace documents; it can also be "AnnotationDefinition" for annotation resource documents. This string determines which section name is looked up in the resource when extracting fields that belong to the main resource definition (for example, mapping ("Namespace", "DescriptionString") when definition_header is "Namespace"). Changing this value alters which top-level section the shared_mapping will target.
    
    Returns:
        dict: A dictionary of the optional shared annotation fields extracted from bel_resource. The returned dictionary contains zero or more of the following canonical keys when present in the input resource:
            description: The definition-level description string (copied from the resource element at (definition_header, "DescriptionString")), type str when present.
            version: The definition-level version string (copied from (definition_header, "VersionString")), type str when present.
            author: The author name string (copied from ("Author", "NameString")), type str when present.
            license: The author copyright/license string (copied from ("Author", "CopyrightString")), type str when present.
            contact: The author contact information string (copied from ("Author", "ContactInfoString")), type str when present.
            citation: The citation name string (copied from ("Citation", "NameString")), type str when present.
            citation_description: The citation description string (copied from ("Citation", "DescriptionString")), type str when present.
            citation_version: The citation published version string (copied from ("Citation", "PublishedVersionString")), type str when present.
            citation_published: If the resource contains Citation.PublishedDate, this key will be present and its value will be the result of parse_datetime applied to that date string (a datetime.datetime object). If PublishedDate is absent, this key will not be present.
        The function returns a new dictionary and does not modify bel_resource. Missing fields in bel_resource are omitted from the result rather than included with None values.
    
    Raises and failure modes:
        - If bel_resource is not a dict, dictionary access will fail (TypeError) when the function attempts to read sections.
        - If nested expected keys are absent, those canonical keys are simply not included in the returned dict (no exception).
        - If a Citation.PublishedDate is present but cannot be parsed by parse_datetime, parse_datetime may raise a ValueError (or another parsing-related exception) which will propagate to the caller; callers that cannot tolerate parsing errors should validate or catch exceptions accordingly.
    
    Practical significance:
        This utility centralizes extraction of commonly useful metadata from BEL namespace and annotation resource documents so that downstream PyBEL code (for example, resource loaders, graph metadata builders, and exporters) can rely on a consistent set of optional fields when annotating BELGraphs, performing grounding, or exporting resource metadata to external formats and services (NDEx, CX, GraphDati, etc.).
    """
    from pybel.manager.utils import extract_shared_optional
    return extract_shared_optional(bel_resource, definition_header)


################################################################################
# Source: pybel.manager.utils.extract_shared_required
# File: pybel/manager/utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def pybel_manager_utils_extract_shared_required(
    config: dict,
    definition_header: str = "Namespace"
):
    """Extract shared required annotation metadata from a BEL resource configuration.
    
    This function is used by PyBEL manager utilities when loading or validating BEL namespace and annotation resource documents. Given the parsed configuration dictionary for a BEL resource (for example, a BEL namespace file or an annotation definition file), it extracts the two required metadata fields that are common to both kinds of resource documents: the resource Keyword (the short name used in BEL statements to reference terms from that resource) and the resource creation timestamp. The creation timestamp value is converted by calling parse_datetime on the raw CreatedDateTime value found in the configuration so the caller receives a parsed timestamp object appropriate for downstream metadata comparisons, caching, or versioning logic.
    
    Args:
        config (dict): The configuration dictionary representing a BEL resource document. This dictionary is expected to have a top-level mapping keyed by the document type name (for example, "Namespace" or "AnnotationDefinition") whose value is another mapping containing the keys "Keyword" and "CreatedDateTime". This argument is the in-memory representation of a BEL namespace or annotation definition resource produced by PyBEL I/O utilities.
        definition_header (str): The top-level section name in config that contains the required fields. By default this is "Namespace" for BEL namespace documents. It may be set to "AnnotationDefinition" when extracting the same required metadata from an annotation definition document. The function will index config[definition_header] to find the "Keyword" and "CreatedDateTime" values.
    
    Returns:
        dict: A dictionary with two entries describing the shared required metadata for the specified BEL resource. The mapping always contains the keys "keyword" and "created". The "keyword" entry is the value taken directly from config[definition_header]["Keyword"] and represents the resource keyword used in BEL statements to identify that namespace or annotation resource. The "created" entry is the result of parse_datetime(config[definition_header]["CreatedDateTime"]) and represents the parsed creation timestamp from the resource document. Note: the function does not modify the input config in place; it only reads values and returns the new dictionary.
    
    Behavioral notes and failure modes:
        This function performs straightforward dictionary lookups and a timestamp parse. It will raise a KeyError if config does not contain the expected definition_header key or if the required "Keyword" or "CreatedDateTime" keys are missing under that header. If config is not a mapping type, a TypeError may be raised during indexing. parse_datetime may raise its own exceptions (for example, if the CreatedDateTime string is not a valid timestamp); callers should handle or propagate those exceptions as appropriate. There are no other side effects.
    """
    from pybel.manager.utils import extract_shared_required
    return extract_shared_required(config, definition_header)


################################################################################
# Source: pybel.struct.filters.node_predicate_builders.data_missing_key_builder
# File: pybel/struct/filters/node_predicate_builders.py
# Category: fix_docstring
# Reason: Schema parsing failed: ("Couldn't parse this type hint, likely due to a custom class or object: ", typing.Callable[[pybel.struct.graph.BELGraph, pybel.dsl.node_classes.BaseEntity], bool])
################################################################################

def pybel_struct_filters_node_predicate_builders_data_missing_key_builder(key: str):
    """pybel.struct.filters.node_predicate_builders.data_missing_key_builder: Build and return a node predicate that tests whether a BELGraph node's data dictionary lacks a given key.
    
    Args:
        key (str): The dictionary key to look for in each node's data mapping. In the context of PyBEL BELGraph nodes, this is the literal key that would appear in the node attribute dict (for example, metadata keys such as a grounding namespace, identifier, or other annotation keys). The function captures this string in a closure and does not validate its semantics; it is used verbatim as a lookup key in graph.nodes[node].
    
    Behavior:
        This function constructs and returns a predicate callable that, when invoked with a BELGraph and a BaseEntity node, inspects the node's attribute dictionary via graph.nodes[node] and returns True exactly when the specified key is not present in that dictionary. If the key exists in the node's data mapping, the predicate returns False regardless of the associated value (that is, the predicate treats a present key with value None as present and therefore returns False). The predicate performs a single membership test ("key in graph.nodes[node]") and does not inspect nested structures or interpret values; it only checks for the presence or absence of the top-level key.
    
    Side effects and defaults:
        The builder itself has no side effects: it only creates and returns a pure function. The returned predicate has no side effects either; it only reads node attributes from the provided BELGraph. There are no defaults beyond the literal key provided. The builder does not modify the graph or nodes.
    
    Failure modes:
        If the provided node is not present in graph.nodes, the attempt to access graph.nodes[node] will raise a KeyError from the underlying mapping (as with typical NetworkX-style graph node access). If the node object is not a valid node identifier for the graph (for example, unhashable or otherwise incompatible with the graph's node indexing), the predicate may raise the corresponding exception (for example, TypeError). The builder does not perform defensive checks for these conditions; callers should ensure the graph and node are valid before calling the predicate.
    
    Returns:
        Callable[[pybel.struct.graph.BELGraph, pybel.dsl.node_classes.BaseEntity], bool]: A predicate function that accepts a BELGraph and a BaseEntity node and returns True if and only if the captured key is not present in the node's data dictionary. The returned callable is intended for use in node-filtering pipelines within PyBEL to select nodes that are missing specific metadata keys (for example, to find nodes that still need grounding or annotation).
    """
    from pybel.struct.filters.node_predicate_builders import data_missing_key_builder
    return data_missing_key_builder(key)


################################################################################
# Source: pybel.struct.operations.node_intersection
# File: pybel/struct/operations.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def pybel_struct_operations_node_intersection(graphs: list):
    """Take the node intersection over a collection of BEL graphs and return a new BELGraph that contains only the nodes present in every input graph.
    
    This function is part of pybel.struct.operations and is used in the PyBEL ecosystem to compute the common node set across multiple BELGraph instances (biological networks encoded in the Biological Expression Language). The intersection semantics follow the same definition as left_node_intersection_join: the set of nodes present in all provided graphs is computed, each original graph is restricted to that node set (an induced subgraph), and the final result is the union of those induced subgraphs. This is useful in comparative analyses of biological networks where downstream operations should operate only on entities (nodes) shared by all networks.
    
    Args:
        graphs (iter[BELGraph]): An iterable of BELGraph objects to intersect. The iterable is converted to a tuple internally because the code iterates over it more than once; therefore this function is not safe for infinite iterables. Each BELGraph is expected to implement the .nodes() method (returning an iterable/collection of nodes) and to be accepted by the subgraph(...) and union(...) helpers used internally. Typical usage in PyBEL passes a list of BELGraph instances.
    
    Behavior and side effects:
        The function first materializes the provided iterable into a tuple and computes its length. If no graphs are provided, the function raises ValueError("no graphs given"). If exactly one graph is provided, the function returns that same BELGraph instance unchanged (no copy is made). For two or more graphs, the function computes the intersection of node sets across all graphs, then constructs the induced subgraph of each input graph restricted to that intersection and returns the union of those induced subgraphs. The returned BELGraph is a newly composed graph representing all edges and node data present in the induced subgraphs; original input graphs are not modified by this operation. Because the implementation delegates to subgraph(...) and union(...), any errors raised by those helpers (for example, due to malformed graph objects) will propagate out of this function.
    
    Failure modes:
        ValueError is raised if the input iterable contains zero graphs. TypeErrors or AttributeErrors may occur if elements of the iterable are not BELGraph-like objects exposing .nodes() or are otherwise incompatible with subgraph/union. Passing an infinite iterable will result in non-termination or memory exhaustion because the iterable is converted to a tuple.
    
    Returns:
        BELGraph: A BELGraph containing only nodes present in every input graph. For multiple input graphs, this graph is constructed by taking the induced subgraph of each input graph restricted to the intersection node set and returning the union of those subgraphs. If a single graph is provided, that same BELGraph instance is returned unchanged.
    """
    from pybel.struct.operations import node_intersection
    return node_intersection(graphs)


################################################################################
# Source: pybel.struct.operations.union
# File: pybel/struct/operations.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def pybel_struct_operations_union(graphs: list, use_tqdm: bool = False):
    """Take the union of multiple BELGraph objects into a single BELGraph.
    
    This function is used within PyBEL to merge separate BEL graphs (BELGraph instances) that represent biological networks encoded in the Biological Expression Language (BEL), for example when combining compiled BEL scripts, network fragments, or outputs from different data sources into a single network for downstream analysis, export, or visualization. The function iterates over the provided collection of graphs, using the first graph as the base (target) and merging subsequent graphs into that base via left_full_join. The merge operation is performed sequentially and is suitable for finite collections of graphs; it is not safe to pass an infinite iterator.
    
    Args:
        graphs (list): A finite list (or other iterable that can be converted to an iterator) of BELGraph objects to be merged. Each element is expected to be a BELGraph as defined by PyBEL (nodes, edges, annotations, citations, metadata). The function will consume the iterable; if it is empty a ValueError("no graphs given") is raised. If the iterable contains a single BELGraph, that object is returned directly (no copy is made). If it contains two or more BELGraph objects, the first BELGraph is copied and then subsequent graphs are merged into that copy in order. The caller should therefore pass a concrete finite collection (e.g., a list of BELGraph) and be aware that elements that are not BELGraph instances may cause downstream errors from left_full_join.
        use_tqdm (bool): Whether to display a progress bar while iterating over the graphs. Default is False. If True, the iterator over graphs will be wrapped with tqdm to show progress (this requires tqdm to be importable in the execution environment). This flag only affects user-facing progress reporting and does not change merge semantics.
    
    Returns:
        BELGraph: The merged BELGraph containing the union of nodes, edges, and associated metadata from the provided graphs. Practical behavior by input size:
        - If zero graphs are provided, the function raises ValueError("no graphs given") and nothing is returned.
        - If exactly one BELGraph is provided, that same BELGraph object is returned unchanged (no copy), which means no new object is allocated and the caller retains the original reference.
        - If two or more BELGraph objects are provided, the function returns a new BELGraph (a copy of the first graph) that has had subsequent graphs merged into it via left_full_join; the returned BELGraph is a distinct object and the original first graph is not modified.
    
    Failure modes and side effects:
        - Raises ValueError("no graphs given") when the provided iterable contains no elements.
        - If use_tqdm is True but tqdm is not available in the environment, attempting to use this flag may raise an ImportError or NameError depending on the import context.
        - The exact rules for merging nodes, edges, attributes, annotations, and citations are determined by left_full_join; errors raised by left_full_join (for example when elements are not valid BELGraph instances or when graph contents are incompatible) will propagate to the caller.
        - For collections with two or more graphs, the merge is performed in-place on the internal target copy (i.e., the returned BELGraph is mutated during construction). For a single-input collection, there are no side effects because the original BELGraph is returned without modification.
    """
    from pybel.struct.operations import union
    return union(graphs, use_tqdm)


################################################################################
# Source: pybel.struct.pipeline.decorators.get_transformation
# File: pybel/struct/pipeline/decorators.py
# Category: valid
################################################################################

def pybel_struct_pipeline_decorators_get_transformation(name: str):
    """Get a registered pipeline transformation by name.
    
    This function looks up and returns a previously registered transformation function from the internal registry used by PyBEL's pipeline decorators. In the context of PyBEL (a framework for parsing and manipulating Biological Expression Language graphs), pipeline transformations are callables that perform a discrete graph-processing step (for example, grounding, validation, or format conversion) and are referenced by name when composing processing pipelines. get_transformation(name) performs a dictionary-style lookup (via mapped.get(name)) into that registry and returns the transformation so the caller can invoke it as part of a pipeline. This function has no side effects on the registry; it only reads from it.
    
    Args:
        name (str): The string name of the transformation to retrieve from the pipeline registry. This name is the identifier under which a transformation function was registered (for example, commonly used transformation step names in PyBEL pipelines). The caller is responsible for supplying the exact registered name; misspellings or different casing will cause a lookup failure.
    
    Returns:
        callable: The transformation function associated with the given name. The returned value is a callable that implements a pipeline step (a function that accepts the expected pipeline arguments, such as a BELGraph or related context, and performs a transformation). The function is returned directly so the caller can execute or compose it into pipeline workflows.
    
    Raises:
        MissingPipelineFunctionError: If the registry does not contain an entry for the provided name (i.e., mapped.get(name) is None), this exception is raised with a message indicating that the given name is not registered as a pipeline function. This is the primary failure mode; callers should catch this exception when dynamically resolving transformation names or validate names against known registrations before calling get_transformation.
    """
    from pybel.struct.pipeline.decorators import get_transformation
    return get_transformation(name)


################################################################################
# Source: pybel.testing.generate.generate_random_graph
# File: pybel/testing/generate.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def pybel_testing_generate_generate_random_graph(
    n_nodes: int,
    n_edges: int,
    namespace: str = "NS"
):
    """Generate a small random BEL subgraph containing protein nodes and sampled "increases" edges for use in tests and examples within the PyBEL ecosystem. The function constructs protein nodes labeled with consecutive integer strings and a given BEL namespace, enumerates all unordered node pairs, samples a requested number of pairs at random, and adds an "increases" relation for each sampled pair with placeholder citation and evidence.
    
    Args:
        n_nodes (int): Integer parameter used to generate node identifiers. The function calls protein(namespace=namespace, name=str(i)) for i in range(1, n_nodes), so the created protein nodes will have names "1", "2", ..., str(n_nodes - 1). Note that, because of the implementation using range(1, n_nodes), the number of node objects actually created is n_nodes - 1. In the context of BEL and PyBEL, each created node represents a Protein function with its concept namespace set to the provided namespace; these nodes are intended for synthetic/testing BEL graphs rather than representing real biological entities.
        n_edges (int): Number of edges to add to the graph. The function first forms all unordered pairs of the created nodes via itertools.combinations(nodes, r=2) and then selects n_edges distinct pairs using random.sample. Therefore n_edges must be less than or equal to the number of possible unordered node pairs C(m, 2), where m == max(0, n_nodes - 1). If n_edges is larger than the available combinations, random.sample will raise a ValueError. Practically, this parameter controls how many sampled "increases" relations are present in the returned BELGraph for testing connectivity and algorithm behavior.
        namespace (str): BEL namespace string assigned to every generated protein node (default "NS"). This maps to the node concept namespace in BEL nodes created by protein(...). Use this to simulate nodes grounded to a particular namespace for testing grounding, namespace-specific analyses, or serialization. The default "NS" is a short placeholder namespace commonly used in tests.
    
    Behavior, side effects, defaults, and failure modes:
        - Node creation: protein(namespace=namespace, name=str(i)) is invoked for each integer i in range(1, n_nodes). This produces Protein-function nodes with the given namespace and name equal to the decimal string of i. Because of the exclusive upper bound in range, passing n_nodes==1 or n_nodes==0 will produce zero nodes; passing n_nodes<2 means there are no unordered pairs to sample, so requesting n_edges>0 will cause a ValueError from random.sample.
        - Edge sampling: all unordered pairs of the created nodes are enumerated with itertools.combinations; random.sample performs uniform sampling without replacement. The sampling is non-deterministic unless the caller seeds the random module (for example via random.seed()) before calling this function.
        - Edge creation: for each sampled pair (u, v), graph.add_increases(u, v, citation=n(), evidence=n()) is called. This adds an "increases" relation from u to v in the BELGraph and attaches placeholder citation and evidence objects produced by n(). These placeholders are intended for testing and are not real bibliographic citations or textual evidence.
        - Exceptions: ValueError is raised by random.sample when n_edges is greater than the number of available unordered node pairs. Other exceptions may propagate from the helper functions (protein, BELGraph.add_increases, n()) if they are provided invalid inputs; those are not handled by this function.
        - Defaults: namespace defaults to the string "NS". No other global state is modified; the function returns a newly constructed BELGraph and has no persistent side effects beyond that returned object.
    
    Returns:
        pybel.BELGraph: A newly created BELGraph containing the generated Protein nodes (with namespace set to the provided namespace and names "1".."str(n_nodes-1)") and the requested number of sampled "increases" edges. The returned graph is suitable for unit tests, examples, or synthetic-data pipelines within the PyBEL framework.
    """
    from pybel.testing.generate import generate_random_graph
    return generate_random_graph(n_nodes, n_edges, namespace)


################################################################################
# Source: pybel.testing.utils.get_uri_name
# File: pybel/testing/utils.py
# Category: fix_docstring
# Reason: Schema parsing failed: Cannot generate JSON schema for get_uri_name because the docstring has no description for the argument 'url'
################################################################################

def pybel_testing_utils_get_uri_name(url: str):
    """pybel.testing.utils.get_uri_name: Extract the file name or resource name from the given URL string for use when saving, caching, or naming downloaded BEL-related resources in PyBEL testing utilities.
    
    This function parses the input URL with urllib.parse.urlparse and returns a short, human-usable name describing the remote resource. In practice within PyBEL (for example when calling urllib.request.urlretrieve as shown in the README), this name is used to generate local filenames for downloaded BEL documents, JSON serializations, or other test resources without performing any network I/O.
    
    Args:
        url (str): The URL to parse and extract a terminal resource name from. This is expected to be a Python string representing a web resource location (for example, a raw GitHub URL or other HTTP(S) link to a BEL document). The function treats the input purely as text and does not attempt to download or validate the resource. If a non-str value is passed, the caller will encounter a TypeError due to the function signature and the expectation that url is a string.
    
    Returns:
        str: The extracted file name or resource name. For typical URLs, this is the last path segment (the substring after the final '/' in the URL path). For URLs that begin with the module-level _FRAUNHOFER_RESOURCES prefix (a special-case prefix used in this codebase for Fraunhofer-hosted resources), the function instead returns the last value after an '=' in the query string (urlparse(url).query.split('=')[-1]) because those resources encode the desired name in the query. If the URL path ends with a slash, the returned string will be empty. If the query contains no '=', the entire query component is returned. The function performs no network requests, does not validate that the returned name corresponds to an existing remote file, and is deterministic and side-effect free.
    """
    from pybel.testing.utils import get_uri_name
    return get_uri_name(url)


################################################################################
# Source: pybel.tokens.parse_result_to_dsl
# File: pybel/tokens.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def pybel_tokens_parse_result_to_dsl(tokens: dict):
    """Convert a ParseResult (the output of the BEL parser) into a PyBEL DSL BaseEntity.
    
    This function is part of the PyBEL parsing pipeline described in the README: after the pyparsing-based BEL grammar produces a ParseResults/dictionary for an entity or post-translational operation (PO), parse_result_to_dsl translates that raw parse representation into the corresponding PyBEL DSL object (BaseEntity). The resulting DSL object is the canonical in-memory representation used by higher-level APIs such as pybel.parse (which can emit JSON for BEL statements), graph construction routines that populate a BELGraph, and export/conversion utilities that interoperate with formats and tools described in the README (NetworkX, CX/NDEx, INDRA, GraphDati, Cytoscape, etc.).
    
    Args:
        tokens (dict or pyparsing.ParseResults): The parse result representing a BEL term, function, or post-translational operation.
            This argument is expected to be the direct output from the BEL parser (pyparsing.ParseResults) or an equivalent dict
            with the same keys. The parse result contains keys such as FUNCTION, VARIANTS, MEMBERS, CONCEPT, and FUSION (as used
            by the parser implementation). Each key maps to the parsed substructure for that part of the BEL expression:
            FUNCTION indicates the top-level BEL function for the term (for example a REACTION), VARIANTS encodes variant/modifier
            information, MEMBERS encodes list/complex membership, CONCEPT carries the namespace/identifier/name tuple for an entity,
            and FUSION denotes fusion constructs. The function inspects these keys in a fixed order and dispatches to specialized
            helpers to produce the appropriate DSL object. Provide the exact parse output produced by the parser; supplying
            a different structure or missing expected keys may raise exceptions (see Failure modes).
    
    Behavior:
        The conversion follows a deterministic dispatch order implemented in the source:
        1) If tokens[FUNCTION] equals the REACTION sentinel, the function returns the result of _reaction_po_to_dict(tokens).
        2) Elif the VARIANTS key is present in tokens, it returns _variant_po_to_dict(tokens).
        3) Elif the MEMBERS key is present, it checks for CONCEPT and returns _list_po_with_concept_to_dict(tokens) when CONCEPT exists,
           otherwise _list_po_to_dict(tokens).
        4) Elif the FUSION key is present, it returns _fusion_to_dsl(tokens).
        5) If none of the above conditions match, it returns _simple_po_to_dict(tokens) as the fallback for simple entities.
        This order and the helper function names are the canonical mapping from parsed BEL constructs to PyBEL DSL entities,
        ensuring correct representation of reactions, variants, lists/complexes, fusions, and simple protein/abundance terms.
    
    Side effects and defaults:
        This function does not perform I/O and does not modify external state; it constructs and returns a new BaseEntity instance
        (or equivalent DSL structure produced by the helper functions). It does not mutate the provided tokens mapping. There are no
        implicit defaults beyond the dispatch order above: when no special keys are present the function falls back to the simple
        entity conversion helper.
    
    Failure modes:
        If tokens is not a dict-like object or a pyparsing.ParseResults with the expected keys and structure, the function may raise
        TypeError, KeyError, or ValueError propagated from the key access or from the specialized helper functions. In particular,
        the expression tokens[FUNCTION] is read without a guard in the REACTION check; if FUNCTION is absent this will raise KeyError.
        The helper functions (_reaction_po_to_dict, _variant_po_to_dict, _list_po_with_concept_to_dict, _list_po_to_dict,
        _fusion_to_dsl, _simple_po_to_dict) may themselves raise parsing/validation errors if substructures are malformed. Callers
        should validate or catch these exceptions when converting arbitrary or user-provided parse results.
    
    Returns:
        BaseEntity: A PyBEL DSL object representing the parsed BEL entity or post-translational operation. The returned BaseEntity
        is the in-memory representation used to build BELGraph nodes and edges, to serialize statements (for example via pybel.parse
        to JSON), and to interoperate with downstream export and analysis tools described in the README. The concrete subtype and
        structure of the returned BaseEntity depend on the keys present in tokens and which helper was invoked (reaction, variant,
        list/complex with concept, list/complex without concept, fusion, or simple entity).
    """
    from pybel.tokens import parse_result_to_dsl
    return parse_result_to_dsl(tokens)


################################################################################
# Source: pybel.utils.ensure_quotes
# File: pybel/utils.py
# Category: fix_docstring
# Reason: Schema parsing failed: Cannot generate JSON schema for ensure_quotes because the docstring has no description for the argument 's'
################################################################################

def pybel_utils_ensure_quotes(s: str):
    """Ensure that a BEL token or label is wrapped in double quotes when it contains non-alphanumeric characters.
    
    Args:
        s (str): A text token, typically a node name, label, or identifier string used in BEL statements and PyBEL serialization routines (for example, a gene/protein name, a namespace entry, or an annotation value). This function examines the supplied string to decide whether it is already an acceptable unquoted token (consisting only of characters matched by the module-level regular expression used for unquoted BEL tokens) or whether it must be represented as a quoted string in serialized BEL. The argument must be a Python str as required by the function signature; other types are not accepted by this function.
    
    Returns:
        str: The input string if it is considered an acceptable unquoted token by the module-level regular expression (_re.match(s) is truthy). If the regular expression does not match, a new string is returned that wraps the original input in double quotes (f'"{s}"'). This returned value is intended for use in BEL serialization or other PyBEL output formats where literal labels with spaces or punctuation must be quoted. Note that the function does not perform any escaping of characters inside s (for example, embedded double-quote characters are not escaped) and does not modify the input in any other way.
    
    Behavior and failure modes:
        This function is pure (no side effects) and deterministic: given the same string s it will always return the same output. Whether a string is returned unchanged or quoted depends solely on the module-level regular expression _re and therefore may vary if that regex is changed elsewhere in the module. An empty string will be quoted (resulting in '""') if the regex does not match it. If s contains internal double-quote characters, those characters are preserved and not escaped, which may result in output that requires additional processing before safe use in contexts that require escaped quotes. The function does not validate full BEL syntax beyond the regex test; callers that need strict BEL-compliant escaping or validation should perform additional checks or escaping as appropriate.
    """
    from pybel.utils import ensure_quotes
    return ensure_quotes(s)


################################################################################
# Source: pybel.utils.expand_dict
# File: pybel/utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def pybel_utils_expand_dict(flat_dict: dict, sep: str = "_"):
    """pybel.utils.expand_dict expands a flattened dictionary whose keys are composite strings (concatenated nested keys) into a nested dictionary structure by splitting keys on a separator and recursing. In the PyBEL codebase this is useful for rebuilding nested attribute or annotation structures (for example, attributes serialized as concatenated strings in export formats) back into hierarchical Python dicts suitable for BEL graph processing and downstream I/O.
    
    Args:
        flat_dict (dict): A flattened dictionary to expand. Keys must be strings that represent one or more nested key components concatenated with the separator string sep (for example, "a_b_c"). Values are treated as opaque Python objects and are preserved at the leaf positions of the reconstructed structure. This function does not mutate the passed-in flat_dict; it builds and returns a new dict.
        sep (str): The literal string used to split composite keys into their first component and the remainder. This function uses str.split(sep, 1) so only the first occurrence of sep is split on each recursion step; deeper occurrences are handled by recursive calls. sep must be a non-empty string (an empty sep will cause str.split to raise ValueError). The default separator is "_" which is commonly used in PyBEL-generated flattened keys.
    
    Returns:
        dict: A new nested dictionary reconstructed from flat_dict. Top-level keys are the first components from composite keys. If a flat key contains multiple separator occurrences (for example "a_b_c"), the function produces nested dictionaries {"a": {"b": {"c": value}}} by recurring on the remainder. Leaf values from flat_dict are preserved. Note important behaviors and failure modes: if flat_dict contains both a simple key "a" and composite keys beginning with "a" (for example "a_b"), the simple key's value will be replaced by the nested dict built from the composite keys (the nested dict assignment overwrites the earlier scalar). If a key in flat_dict is not a string, calling split will raise an AttributeError. Extremely deep nesting can lead to RecursionError due to the recursive implementation. There are no other side effects; the original flat_dict is unchanged and the function returns the expanded structure.
    """
    from pybel.utils import expand_dict
    return expand_dict(flat_dict, sep)


################################################################################
# Source: pybel.utils.get_corresponding_pickle_path
# File: pybel/utils.py
# Category: valid
################################################################################

def pybel_utils_get_corresponding_pickle_path(path: str):
    """pybel.utils.get_corresponding_pickle_path: Compute and return the filesystem path that corresponds to a pickled representation of a BEL file by appending the literal extension ".pickle" to the provided BEL file path.
    
    This utility is used in the PyBEL codebase to standardize the filename for a pickled (serialized) form of a BEL document or BELGraph object so calling code can consistently name cache files or serialized outputs. It performs a purely string-based transformation and does not read from or write to the filesystem, perform any validation of the input path, or attempt to detect or replace existing file extensions.
    
    Args:
        path (str): A filesystem path string referring to a BEL file or BELGraph output. In the PyBEL domain, this is typically the path used to save or load BEL content (for example, a file produced by pybel.dump). This function treats the argument as an opaque path string and appends the ".pickle" extension to it to produce the canonical pickled filename. The caller is responsible for providing a valid path string; the function itself does not check that the file exists or that the path is writable.
    
    Returns:
        str: A new path string equal to the input path with the literal suffix ".pickle" appended. For example, given "my_graph.bel" the function returns "my_graph.bel.pickle". Note that this function does not remove or replace existing extensions, so if the input already ends with ".pickle" the returned string will have ".pickle" appended again (for example, "x.pickle" becomes "x.pickle.pickle"). No filesystem side effects occur during this operation.
    """
    from pybel.utils import get_corresponding_pickle_path
    return get_corresponding_pickle_path(path)


################################################################################
# Source: pybel.utils.hash_dump
# File: pybel/utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def pybel_utils_hash_dump(data: dict):
    """Hash an arbitrary JSON dictionary by dumping it in sorted order, encoding it in UTF-8, then hashing the bytes with MD5.
    
    This function, pybel.utils.hash_dump, produces a deterministic short string fingerprint for a JSON-serializable Python object by performing three steps: (1) serializing the object to a JSON string with keys sorted (json.dumps(..., sort_keys=True)), (2) encoding that JSON string to UTF-8 bytes, and (3) computing the MD5 digest and returning its hexadecimal representation. In the context of PyBEL, this is useful for creating stable identifiers or cache keys for JSON representations produced by the library (for example, Node-Link JSON serializations of BEL graphs used for interchange and export). The returned string is intended to be used as an opaque identifier (e.g., filenames, cache keys, deduplication markers) and not as a cryptographic signature.
    
    Args:
        data (dict or list or tuple): The JSON-serializable Python object to hash. The original implementation accepts mappings (dict) and sequence types (list or tuple) that can be passed to json.dumps; this parameter should therefore be a structure composed of JSON-compatible types (strings, numbers, booleans, None, lists, tuples, and dicts with string keys). The function relies on json.dumps to serialize the object; therefore the determinism of the hash depends on json.dumps behavior (for example, non-deterministic ordering of custom objects or non-serializable types will cause errors). This parameter corresponds to data structures that PyBEL commonly writes or exchanges (for example, node-link dictionaries used when exporting BEL graphs).
    
    Returns:
        str: The hexadecimal MD5 digest of the UTF-8 encoding of the JSON serialization with keys sorted. This is a deterministic string identifier for the given input under json.dumps semantics and can be used as an opaque key. No other side effects occur.
    
    Behavior, defaults, and failure modes:
        The function forces JSON key ordering by calling json.dumps with sort_keys=True so that semantically equivalent mappings with different key order produce the same hash. It encodes the JSON string with UTF-8 before hashing. The function uses MD5 (hashlib.md5) and returns the hex digest string; MD5 is fast and produces short fixed-length identifiers but is not collision-resistant for cryptographic purposes. If the provided data is not JSON-serializable (for example, contains Python objects that json.dumps cannot encode), json.dumps will raise a TypeError (or other json-related exception), which propagates to the caller. The resulting hash is only as meaningful as the JSON serialization: changes in numerical formatting, floating point representation, or use of non-standard JSON encoders will change the hash.
    """
    from pybel.utils import hash_dump
    return hash_dump(data)


################################################################################
# Source: pybel.utils.parse_datetime
# File: pybel/utils.py
# Category: fix_docstring
# Reason: Schema parsing failed: Cannot generate JSON schema for parse_datetime because the docstring has no description for the argument 's'
################################################################################

def pybel_utils_parse_datetime(s: str):
    """pybel.utils.parse_datetime parses a date or datetime string used in BEL graph metadata (for example, version or publication timestamps) into a datetime.date value according to the standard PyBEL date/time formats.
    
    This helper is used by PyBEL when reading or constructing graph metadata fields such as creation date or published date (for example the from_emmaa example that accepts date='2020-05-29-17-31-58'). It tries the canonical PyBEL formats in a fixed order (CREATION_DATE_FMT, PUBLISHED_DATE_FMT, PUBLISHED_DATE_FMT_2) using datetime.strptime and returns a datetime.date-compatible object when one of the formats matches. It does not perform timezone handling, fuzzy parsing, or infer formats beyond the three supported format strings; if the input does not match any supported format it fails with a ValueError so callers can detect and handle malformed metadata.
    
    Args:
        s (str): A string containing a date or datetime to parse. In the PyBEL domain this is typically a metadata timestamp such as a BEL document creation or publication time (for example the version string '2020-05-29-17-31-58' shown in the README). The function expects the string to exactly match one of the three PyBEL formats referenced by the constants CREATION_DATE_FMT, PUBLISHED_DATE_FMT, and PUBLISHED_DATE_FMT_2.
    
    Returns:
        datetime.date: A date-compatible object representing the parsed calendar date/time for use in BEL graph metadata. The return value is produced by parsing with datetime.strptime using one of the supported format strings; callers can use it directly for storing or comparing graph version/published dates (it is compatible with datetime.date semantics).
    
    Raises:
        ValueError: If s does not match any of the supported PyBEL datetime/date formats, a ValueError is raised with the message "Incorrect datetime format for {s}". This signals malformed or unsupported metadata input and allows callers to validate or reject graph metadata.
    """
    from pybel.utils import parse_datetime
    return parse_datetime(s)


################################################################################
# Source: pybel.utils.tokenize_version
# File: pybel/utils.py
# Category: valid
################################################################################

def pybel_utils_tokenize_version(version_string: str):
    """pybel.utils.tokenize_version: Tokenize a version string into a three-integer tuple (major, minor, patch).
    
    Converts a textual version identifier commonly used in PyBEL metadata and BEL document exports into a canonical numeric form suitable for numeric comparison, sorting, and inclusion in graph summaries or exported manifests. The function first strips any qualifier starting at the first hyphen (for example "-dev" or "-rc1"), then splits the remaining portion on dots and returns the first three components as integers. This normalization is useful in the PyBEL codebase wherever semantic version-like strings (for example the BELGraph "Version" field shown in the README) must be compared, displayed in summaries, or used in I/O filenames.
    
    Args:
        version_string (str): A version string to parse. This should be a text string that contains at least three dot-separated numeric components in its core portion (for example "0.1.2" or "1.2.3-dev"). The function will ignore any trailing qualifier beginning with "-" (for example "-dev", "-rc1") and only parse the portion before the first hyphen. The parameter corresponds to version identifiers encountered in PyBEL graph metadata, package versions, or file naming conventions.
    
    Returns:
        Tuple[int, int, int]: A tuple of three integers (major, minor, patch). Each element is the integer conversion of the corresponding dot-separated component from the version_string before any hyphen qualifier. For example, tokenize_version("0.1.2-dev") returns (0, 1, 2). This returned tuple is intended for numeric comparison and deterministic ordering of versions within PyBEL workflows.
    
    Raises:
        ValueError: If the core portion of version_string (the substring before the first "-") does not contain at least three dot-separated components, or if any of the first three components cannot be converted to int. In such cases the function fails rather than attempting to guess missing fields.
        TypeError or AttributeError: If a non-string is passed that does not support the str.split method, an exception may be raised; callers should pass a str as required by the signature.
    
    Behavior and side effects:
        This function is a pure, deterministic transformation with no external side effects. It does not modify its input and does not perform I/O. There are no default values beyond the required string argument. The function intentionally truncates additional numeric components beyond the first three (for example "1.2.3.4" becomes (1, 2, 3)) and drops any hyphenated qualifiers (for example "1.2.3-dev.0" becomes (1, 2, 3)).
    """
    from pybel.utils import tokenize_version
    return tokenize_version(version_string)


################################################################################
# Source: pybel.utils.valid_date
# File: pybel/utils.py
# Category: fix_docstring
# Reason: Schema parsing failed: Cannot generate JSON schema for valid_date because the docstring has no description for the argument 's'
################################################################################

def pybel_utils_valid_date(s: str):
    """pybel.utils.valid_date: Return whether a string represents a valid ISO 8601 date in the YYYY-MM-DD form.
    
    Checks that its argument is a date string that conforms to the ISO 8601 calendar-date pattern "YYYY-MM-DD" and that the year, month, and day constitute a real calendar date (for example, 2020-02-29 is valid while 2019-02-29 is not). This validator is used in PyBEL to validate date-valued metadata commonly found in BEL graphs and workflows (for example published dates or simple version dates used when compiling or serializing BEL documents), ensuring consistency with the PUBLISHED_DATE_FMT applied throughout the codebase. The function delegates to the module-level _validate_date_fmt using the PUBLISHED_DATE_FMT constant, performs no I/O, and has no side effects.
    
    Args:
        s (str): The input string to validate as a date. This must be a text string representing a calendar date in the exact four-digit-year, two-digit-month, two-digit-day pattern separated by hyphens (for example "2021-07-15"). Callers should provide a str as required by the signature; passing a non-str value may result in exceptions from the underlying validator.
    
    Returns:
        bool: True if s exactly matches the YYYY-MM-DD ISO 8601 date format and represents a valid calendar date; False if s does not match the format or encodes a nonexistent date (invalid month, invalid day for the month, non-leap-year February 29, etc.). The function does not validate time, time zones, or datetime strings that include hours/minutes/seconds.
    """
    from pybel.utils import valid_date
    return valid_date(s)


################################################################################
# Source: pybel.utils.valid_date_version
# File: pybel/utils.py
# Category: fix_docstring
# Reason: Schema parsing failed: Cannot generate JSON schema for valid_date_version because the docstring has no description for the argument 's'
################################################################################

def pybel_utils_valid_date_version(s: str):
    """pybel.utils.valid_date_version validates whether a string is a valid PyBEL "date version" identifier.
    This function checks that the provided string conforms to the module's DATE_VERSION_FMT (the date-version
    format used across PyBEL to label graph versions and timestamps) by delegating to the internal
    _validator _validate_date_fmt(s, DATE_VERSION_FMT). It is intended for use anywhere PyBEL accepts or
    produces version/date strings for BEL graphs and I/O (for example, as the date argument to
    pybel.from_emmaa(...) and the Version field shown by BELGraph.summarize), ensuring consistent formatting
    for graph metadata, file naming, and interoperability with downstream tools.
    
    Args:
        s (str): The candidate date-version string to validate. This should be a Python str containing
            a date/time-like version label in the PyBEL convention (for example, the README shows
            "2020-05-29-17-31-58" as a typical date-version used for graph versions). The parameter's
            role is to provide the textual version identifier that will be checked for syntactic
            conformance to DATE_VERSION_FMT. Callers must supply a str; behavior for non-str inputs is
            not guaranteed by this function and depends on the underlying validator.
    
    Returns:
        bool: True if and only if the input string s matches the PyBEL DATE_VERSION_FMT and therefore
        represents a valid date-version label according to PyBEL conventions. False if s is a str but
        does not match the expected format. The function is deterministic and has no side effects;
        it performs only a format validation and does not modify global state or persist data. If a
        non-str value is passed, the underlying validator may raise an exception (behavior not defined
        beyond the function's signature).
    """
    from pybel.utils import valid_date_version
    return valid_date_version(s)


################################################################################
# Source: pybel.version.get_version
# File: pybel/version.py
# Category: fix_docstring
# Reason: Schema parsing failed: Cannot generate JSON schema for get_version because the docstring has no description for the argument 'with_git_hash'
################################################################################

def pybel_version_get_version(with_git_hash: bool = False):
    """Get the PyBEL package version string via pybel.version.get_version. Returns the package VERSION constant as a human-readable semantic version string and, optionally, appends the current git commit hash to the version string for more specific provenance tracking (for example in logs, CLI output, graph metadata, file exports, or reproducibility records used by PyBEL workflows that parse and serialize BEL graphs).
    
    Args:
        with_git_hash (bool): If True, include the repository git commit hash after the package VERSION. When True, this function calls get_git_hash() and returns the concatenation VERSION + "-" + get_git_hash() (exactly as implemented in the source). When False (the default), the function returns only the packaged VERSION constant without consulting git metadata. This parameter controls whether the returned string carries additional source-control provenance useful for debugging, reproducibility, or distinguishing installs from different commits.
    
    Returns:
        str: A version string. If with_git_hash is False, this is the package VERSION constant (e.g., "1.2.3"). If with_git_hash is True, this is the package VERSION followed by a hyphen and the git hash returned by get_git_hash() (e.g., "1.2.3-abcdef0"). The returned string is suitable for display in user interfaces, inclusion in BELGraph metadata (as shown by BELGraph.summarize output), serialization headers, and CLI --version output.
    
    Behavior and side effects:
        This function does not mutate global state. If with_git_hash is True, it will read git-related metadata by calling get_git_hash(); any I/O or lookup performed by get_git_hash() (for example, reading .git metadata or invoking git) is a side effect of that call. Any exceptions raised by get_git_hash() (for example if git metadata is unavailable in the installation) are propagated to the caller; callers that require robust behavior in environments without git metadata should call get_version(False) or handle exceptions accordingly.
    
    Failure modes:
        If with_git_hash is True and retrieving the git hash fails (for example because the installation is a packaged distribution without .git data or git is unavailable), get_git_hash() may raise an exception which will propagate from get_version. In such cases, callers can request the plain VERSION by passing with_git_hash=False to avoid depending on git metadata.
    """
    from pybel.version import get_version
    return get_version(with_git_hash)


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
