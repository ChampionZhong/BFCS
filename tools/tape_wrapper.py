"""
Regenerated Google-style docstrings for module 'tape'.
README source: others/readme/tape/README.md
Generated at: 2025-12-02T00:21:27.170369Z

Total functions: 7
"""


import torch

################################################################################
# Source: tape.models.file_utils.cached_path
# File: tape/models/file_utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def tape_models_file_utils_cached_path(
    url_or_filename: str,
    force_download: bool = False,
    cache_dir: str = None
):
    """tape.models.file_utils.cached_path: Determine whether an input is a URL or a local file path; if it is a URL, download and cache the remote file (using the repository's model/data cache) and return the local cached file path; if it is a local path, validate existence and return it.
    
    This function is used throughout TAPE to ensure pretrained model weights, tokenizer files, and dataset artifacts referenced by URL (for example when calling model.from_pretrained, tape-embed, or other helpers documented in the README) are available locally. For HTTP(S) and S3 URLs the file is downloaded and stored in a cache directory so subsequent calls reuse the cached copy. For local paths the function verifies the file exists and returns the same path. pathlib.Path objects are accepted on Python 3 and are converted to strings before processing.
    
    Args:
        url_or_filename (str): A string that is either (1) a URL using the http, https, or s3 schemes pointing to a remote resource (for example a pretrained model file hosted on a server), or (2) a local filesystem path to an existing file. In the TAPE domain this typically refers to pretrained model weights, tokenizer vocabularies, or data archives. If a pathlib.Path is passed on Python 3 it will be converted to str internally.
        force_download (bool): If True, ignore any existing cached copy and re-download the file from the URL. Defaults to False. When True this forces a fresh download even if a file with the same URL has already been fetched into the cache directory, which is useful when the remote file has been updated or a previous download is suspected to be corrupted.
        cache_dir (str): Optional path to a directory to use for caching downloaded files. If None, the function uses the module default cache directory (PROTEIN_MODELS_CACHE) used by TAPE to store pretrained models and other large artifacts. Supplying this argument overrides that default and directs where get_from_cache will store or look for cached copies.
    
    Returns:
        str: The local filesystem path to the file that can be used by downstream code. For HTTP(S)/S3 URLs this is the path to the cached file (after downloading if necessary). For local input paths this is the same path if the file exists. The returned value is a string path suitable for opening with standard file APIs.
    
    Behavior and side effects:
        - Recognizes URLs when the parsed scheme is 'http', 'https', or 's3'. In these cases it delegates to get_from_cache(url_or_filename, cache_dir, force_download) which may create the cache directory, perform network I/O to download the file, and write to disk. Network failures, permission errors when writing to cache_dir, or errors raised by get_from_cache will propagate to the caller.
        - If url_or_filename is an existing local file, the function returns that path immediately without copying or moving the file.
        - If url_or_filename parses to a URI with an empty scheme (parsed.scheme == '') and the local file does not exist, the function raises EnvironmentError indicating the file was not found.
        - If the parsed URI has a non-empty scheme that is not in ('http', 'https', 's3') the function raises ValueError indicating it cannot interpret the input as a URL or local path.
        - On Python 3, pathlib.Path instances passed for url_or_filename or cache_dir are converted to strings prior to processing.
    
    Failure modes and exceptions:
        - EnvironmentError is raised when a local path (no scheme) does not exist.
        - ValueError is raised when the input has an unrecognized scheme.
        - Errors from network operations, permission issues, or underlying filesystem operations (for example raised by get_from_cache) will propagate; callers should catch these as appropriate when invoking TAPE commands that rely on remote downloads (such as from_pretrained or tape-embed).
    """
    from tape.models.file_utils import cached_path
    return cached_path(url_or_filename, force_download, cache_dir)


################################################################################
# Source: tape.models.file_utils.filename_to_url
# File: tape/models/file_utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def tape_models_file_utils_filename_to_url(filename: str, cache_dir: str = None):
    """Return the remote URL and stored ETag associated with a cached pretrained-model
    file name. This function is used by TAPE's model-loading and caching logic to
    map a local cached filename (under the module's pretrained models cache) to the
    original download URL and the HTTP ETag value recorded when the file was
    cached. The URL is used to re-download or verify the origin of a pretrained
    protein model (for example, transformer or UniRep weights listed in the TAPE
    README) and the ETag is used to determine whether a cached copy is current;
    the ETag may be None if no ETag was recorded for that file.
    
    Args:
        filename (str): The file name (relative path segment) of a cached model
            artifact within the pretrained models cache. In the TAPE workflow this
            corresponds to a file that was previously downloaded and stored for a
            pretrained protein model (for example a weights file for 'bert-base'
            or 'babbler-1900'). The function constructs the cache path by joining
            cache_dir and filename; filename may include subdirectory components.
        cache_dir (str): Path to the cache directory that stores pretrained model
            files and their metadata (the default is the module-level
            PROTEIN_MODELS_CACHE). This should be a filesystem path given as a
            string; when running on Python 3 the implementation also accepts a
            pathlib.Path (it will be coerced to a string). If None, the function
            uses the module's PROTEIN_MODELS_CACHE default. The cache directory is
            expected to contain both the cached file named by filename and a
            metadata JSON file named filename + '.json'.
    
    Returns:
        tuple: A two-element tuple (url, etag).
            url (str): The original remote download URL stored in the metadata
                JSON. This is the address from which the pretrained model file
                was originally obtained and which can be used to re-download the
                file for TAPE tasks such as embedding, pretraining, or downstream
                evaluation.
            etag (str or None): The recorded HTTP ETag value from the time the
                file was cached, or None if no ETag was stored. The ETag can be
                used to perform conditional HTTP requests to avoid unnecessary
                downloads when the cached file is still current.
    
    Behavior and side effects:
        The function reads metadata but does not modify any files. It opens the
        metadata JSON with UTF-8 encoding and parses it with json.load to extract
        the 'url' and 'etag' fields. The function relies on the metadata file
        located at os.path.join(cache_dir, filename) + '.json'.
    
    Failure modes and exceptions:
        If the constructed cache file path (os.path.join(cache_dir, filename)) does
        not exist, the function raises EnvironmentError with the message
        "file {cache_path} not found". If the corresponding metadata JSON file
        (cache_path + '.json') does not exist, it raises EnvironmentError with the
        same form of message for the metadata path. If the metadata JSON is
        malformed, json.load will raise a JSONDecodeError; if the JSON does not
        contain the expected 'url' or 'etag' keys, a KeyError will be raised. Callers
        should handle these exceptions (EnvironmentError for missing files,
        JSONDecodeError/KeyError for malformed metadata) when integrating this
        function into automated model-download or verification workflows.
    """
    from tape.models.file_utils import filename_to_url
    return filename_to_url(filename, cache_dir)


################################################################################
# Source: tape.models.file_utils.get_from_cache
# File: tape/models/file_utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def tape_models_file_utils_get_from_cache(
    url: str,
    cache_dir: str = None,
    force_download: bool = False,
    resume_download: bool = False
):
    """tape.models.file_utils.get_from_cache retrieves a file referenced by a URL and ensures a stable local cached copy for use by the TAPE benchmark code (for example pretrained model weights, dataset archives, or other artifacts used when embedding, training, or evaluating protein models).
    
    This function locates a cached copy of the resource identified by url in cache_dir (or in the repository-wide PROTEIN_MODELS_CACHE when cache_dir is None). If a suitable cached copy is not present or force_download is True, it downloads the resource to a temporary file and atomically moves it into the cache. The cached filename incorporates the HTTP/S3 ETag when available so that different versions of the same URL are stored separately; when no ETag is available the function attempts to find previously downloaded candidates by filename pattern. The function also writes a small metadata JSON file next to the cached file containing the original URL and the ETag. A file-based lock prevents simultaneous parallel downloads of the same resource across processes. The function is used throughout TAPE to ensure pretrained models and data are automatically downloaded and reused (see the README section on loading pretrained models and embedding proteins).
    
    Args:
        url (str): The HTTP(S) or S3 URL identifying the resource to cache. This is the primary identifier used to download pretrained model weights, datasets, or other artifacts required by TAPE commands such as model.from_pretrained or tape-embed. The function queries the remote URL for an ETag (via a HEAD request for HTTP or s3_etag for s3:// URLs) and incorporates the ETag into the local filename when available to avoid staleness and allow multiple versions to coexist.
        cache_dir (str): Directory path where cached files and metadata will be stored. If None, the function uses the module-level PROTEIN_MODELS_CACHE default. The function will create this directory if it does not exist. Path-like objects are coerced to str for compatibility on Python 2/3. The cached file is written as cache_dir/<filename> with a sidecar metadata file <filename>.json.
        force_download (bool): If True, always download the resource even when a matching cached file exists. When False (the default), the function returns an existing cached file if it matches the resolved filename/ETag or if the function finds a suitable previously downloaded candidate when no ETag is provided. Use True to override the cache and refresh the local copy.
        resume_download (bool): If True, enable resumable downloads by writing into a persistent ".incomplete" file (opened in append-binary mode). When enabled, interrupted downloads can be resumed by appending to the incomplete file on subsequent calls. When False (the default), downloads use a temporary file (NamedTemporaryFile) in the cache directory and are renamed into place only after a successful complete download. Note that resume behavior requires server support and does not change the return type.
    
    Returns:
        str: Absolute or relative filesystem path to the cached file (cache_dir/<filename>). The returned path is the authoritative local copy that callers should open and read. Side effects: this function may create cache_dir, download data over the network, write the cached file, write a metadata JSON file containing {"url": url, "etag": etag}, and create or remove lock/temporary files. If resume_download was used an ".incomplete" file may persist between attempts.
    
    Behavior and failure modes:
        - Network operations: the function attempts to read the remote ETag via an HTTP HEAD request (or s3_etag for s3:// URLs). HEAD failures are caught and treated as "no ETag" (etag = None). Actual data transfer is performed by http_get or s3_get; network errors raised by those functions (timeouts, connection errors, HTTP errors surfaceable by http_get/s3_get) will propagate to the caller.
        - ETag handling: when an ETag is present it is incorporated into the generated filename (via url_to_filename) so that different versions of a URL do not clobber each other. When no ETag is present the function may return the first matching file it finds in cache_dir that resembles the expected filename (excluding .json files). If etag is None and multiple matching files exist, the implementation picks the last matching filename in the directory listing; callers should not rely on deterministic selection in this scenario.
        - Concurrency: a file lock (cache_path + ".lock") is used to prevent multiple processes from downloading the same URL in parallel. The lock is released when the function returns or raises. If another process completes the download while the lock is held, the function will detect the completed file and return it without re-downloading.
        - Atomicity and corruption: downloads are written to a temporary file (or an ".incomplete" file when resume_download is True) and only moved into the cache path after a successful download using os.replace to reduce the chance of corrupted cache entries from interrupted downloads. However, filesystem permission errors, insufficient disk space, or failures during os.replace may leave temporary or incomplete files; callers should handle such I/O errors.
        - Metadata: a JSON metadata file (<cached_filename>.json) is created next to the cached file containing the original URL and ETag. This file is used to record provenance but may be absent if creation failed due to I/O errors.
        - Python 2 compatibility: the function contains code paths that coerce types and decode ETag values for Python 2; these branches are present only to maintain behavior in legacy environments.
        - Errors: typical failure modes include network errors from http_get/s3_get, permission or filesystem errors when creating or writing files, and errors from the locking mechanism. These errors are not swallowed by this function (except for the HEAD request used to probe the ETag, which is treated as "no ETag" on error) and will raise exceptions to the caller.
    
    Practical significance in TAPE:
        - This function underpins the automatic download-and-cache behavior described in the README (for example, calling ProteinBertModel.from_pretrained or running tape-embed). By centralizing download, ETag-based versioning, atomic placement, metadata recording, and locking, callers can reliably obtain local copies of required model weights and datasets without manually managing files.
    """
    from tape.models.file_utils import get_from_cache
    return get_from_cache(url, cache_dir, force_download, resume_download)


################################################################################
# Source: tape.models.file_utils.s3_etag
# File: tape/models/file_utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def tape_models_file_utils_s3_etag(url: str):
    """tape.models.file_utils.s3_etag — Return the ETag of an S3 object identified by a URL.
    
    This function obtains the ETag metadata for an object stored on Amazon S3. In the TAPE codebase this is used when interacting with datasets and pretrained model artifacts hosted on AWS (see README references to data hosted on AWS and automatic downloading of pretrained models). The ETag is commonly used for simple cache validation or to detect whether a remote S3 object has changed, so callers typically use this function to decide whether to re-download or to validate a cached copy. The function performs a network request via boto3 to read object metadata; it does not download the object contents.
    
    Args:
        url (str): S3 object URL or path identifying the target object. This should be a string in an S3 form accepted by tape.models.file_utils.split_s3_path (for example, an "s3://bucket/key" style URL or another form supported by that helper). The string is interpreted as the S3 bucket and key to query. If the string is not a valid S3 path according to split_s3_path, that helper will raise an error and this function will propagate that error.
    
    Returns:
        str: The ETag value of the S3 object as returned by boto3's S3 Object resource (the value of s3_object.e_tag). This value is the object ETag metadata used for change detection and cache validation; callers should treat it as an opaque string returned by S3 and compare it against previously stored ETags to detect changes.
    
    Behavior and side effects:
        This function performs a network metadata request to AWS S3 using boto3.resource("s3") and will incur normal S3 request latency and any applicable service costs. It relies on boto3's default session and credential resolution: valid AWS credentials and network access to S3 are required. The function itself does not modify S3 objects or local state.
    
    Failure modes and errors:
        If AWS credentials are not configured, if the caller lacks permissions to read the object, if the object does not exist, or if there are network issues, the underlying boto3 call will raise an exception (for example botocore.exceptions.ClientError or network-related exceptions). These exceptions are propagated to the caller. If the provided url is malformed or unsupported by split_s3_path, that helper will raise (for example ValueError) and the error will propagate. Callers should catch and handle these exceptions when using this function.
    
    Notes:
        The function is decorated with @s3_request in this module, which centralizes S3 request handling behavior in the codebase (for example, consistent authentication, retry or logging behavior implemented by that decorator).
    """
    from tape.models.file_utils import s3_etag
    return s3_etag(url)


################################################################################
# Source: tape.models.file_utils.split_s3_path
# File: tape/models/file_utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def tape_models_file_utils_split_s3_path(url: str):
    """Split a full S3 URL into the bucket identifier and the object key path. This function is used throughout TAPE when parsing S3 locations for downloading data, pretrained model weights, or saving artifacts to AWS S3: it extracts the bucket component that identifies the S3 bucket and the relative object key that identifies the object within that bucket (the value typically passed to S3 API calls such as get_object or put_object). The function performs only string parsing (no network access) and normalizes the returned path by removing a leading '/' if present so that the result can be used directly as an S3 object key.
    
    Args:
        url (str): A full URL-like S3 location containing a network location (netloc) and a path component. In practice, TAPE expects URLs where the bucket name appears in the netloc portion of the URL, for example 's3://my-bucket/path/to/object'. The string is parsed with urlparse; if the bucket is encoded as part of a hostname (for example 'bucket.s3.amazonaws.com'), the returned bucket_name will be that hostname exactly (this function does not rewrite or validate hostnames vs bucket naming schemes).
    
    Returns:
        tuple: A 2-tuple (bucket_name, s3_path) where both elements are strings. bucket_name is the parsed network-location component (parsed.netloc) and represents the S3 bucket identifier to use with AWS APIs. s3_path is the parsed path component (parsed.path) with any leading '/' removed; it represents the object key relative to the bucket and can be passed directly as the key parameter in S3 operations. Trailing slashes in the original path are preserved. Note that query and fragment components of the input URL are not returned and are effectively ignored for the object key.
    
    Raises:
        ValueError: If the input url does not contain both a non-empty network-location (netloc) and a non-empty path according to urlparse (for example passing a bare 'bucket/key' string or an otherwise malformed URL). The raised error message will be "bad s3 path <url>" where <url> is the original input.
    """
    from tape.models.file_utils import split_s3_path
    return split_s3_path(url)


################################################################################
# Source: tape.models.file_utils.url_to_filename
# File: tape/models/file_utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def tape_models_file_utils_url_to_filename(url: str, etag: str = None):
    """tape.models.file_utils.url_to_filename converts a URL (and optional ETag) into a deterministic hashed filename used by TAPE to cache downloaded resources such as pretrained model weights. In the TAPE workflow (see README), pretrained models and other remote assets are automatically downloaded and stored in a local cache; this function maps the original remote identifier (the URL) and an optional HTTP ETag to a repeatable filename suitable for use as a cache key.
    
    Args:
        url (str): The resource identifier to convert into a filename. This is typically an HTTP(S) URL pointing to a pretrained model, weight file, or other remote asset used by TAPE (for example, a Hugging Face model URL). The function encodes this string using UTF-8 and computes its SHA-256 hash to produce a stable, deterministic, and opaque filename component. If `url` is not a str, calling this function will raise an exception when the code attempts to call `.encode('utf-8')`.
        etag (str, optional): An optional entity tag (ETag) string associated with the resource version on the server. If provided (not None), it is encoded with UTF-8 and hashed with SHA-256, and its hex digest is appended to the URL hash separated by a single period ('.'). This creates a filename that reflects both the resource identity (URL) and its version (ETag), allowing different versions of the same URL to map to different cache keys. If `etag` is provided but is not a str, an exception will be raised when `.encode('utf-8')` is called. The default is None, meaning no ETag component is appended.
    
    Returns:
        str: A deterministic filename string composed of the hexadecimal SHA-256 digest of the UTF-8 encoded `url`. If `etag` is provided, the returned string is the url-digest, a '.' character, and the etag-digest (i.e., "<url_sha256_hex>.<etag_sha256_hex>"). The returned value contains only ASCII hexadecimal characters and at most one period; it does not perform any filesystem operations (no file is created) and does not include file extensions or directory separators. Because the function uses SHA-256, collisions are extremely unlikely but not impossible; callers should treat the result as an opaque cache key rather than a human-readable filename.
    """
    from tape.models.file_utils import url_to_filename
    return url_to_filename(url, etag)


################################################################################
# Source: tape.models.modeling_utils.gelu
# File: tape/models/modeling_utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def tape_models_modeling_utils_gelu(x: torch.Tensor):
    """Implementation of the Gaussian Error Linear Unit (GELU) activation used in TAPE model implementations.
    
    This function computes the GELU nonlinearity as used in many deep learning models provided in the TAPE repository (for example Transformer, LSTM, ResNet, UniRep and trRosetta implementations described in the README). GELU is a smooth, non-linear activation that weights each element of the input tensor by the Gaussian cumulative distribution function; it is commonly used in modern transformer-style architectures and language models to provide improved optimization and representational properties compared to hard rectifiers. The implementation follows the exact mathematical form:
        x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
    which is equivalent to x * Phi(x) where Phi is the standard normal CDF expressed via the error function. See Hendrycks & Gimpel (2016) for the original GELU description: https://arxiv.org/abs/1606.08415
    
    Note: an alternative approximation used in some implementations (for example OpenAI GPT) is
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)))
    which is numerically different and can produce slightly different model outputs; the TAPE codebase uses the erf-based exact form above.
    
    Args:
        x (torch.Tensor): Input tensor of activations. This is the tensor on which the GELU nonlinearity is applied elementwise. In the TAPE codebase this will typically be the output of a linear or convolutional layer (float-valued activations) within models used for protein sequence embedding and downstream prediction tasks. The function expects a PyTorch tensor; non-floating dtypes will typically raise an error when calling torch.erf or will be implicitly promoted by PyTorch operations. The function does not modify x in-place.
    
    Returns:
        torch.Tensor: A new torch.Tensor containing the elementwise GELU-transformed values computed by x * 0.5 * (1.0 + torch.erf(x / sqrt(2.0)))). The returned tensor is produced by PyTorch operations, preserves autograd connectivity for gradient-based optimization used across TAPE training/evaluation pipelines, and will generally match the input tensor's device and floating dtype when supported by PyTorch. Failure modes include type or dtype errors if x is not a floating-point torch.Tensor, and propagation of NaN/Inf values if they are present in the input.
    """
    from tape.models.modeling_utils import gelu
    return gelu(x)


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
