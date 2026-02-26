"""
Regenerated Google-style docstrings for module 'mace'.
README source: others/readme/mace/README.md
Generated at: 2025-12-02T00:20:44.151377Z

Total functions: 7
"""


################################################################################
# Source: mace.calculators.lammps_mliap_mace.timer
# File: ../../../../../usr/local/lib/python3.10/contextlib.py
# Category: fix_docstring
# Reason: Schema parsing failed: Cannot generate JSON schema for timer because the docstring has no description for the argument 'name'
################################################################################

def mace_calculators_lammps_mliap_mace_timer(name: str, enabled: bool = True):
    """mace.calculators.lammps_mliap_mace.timer: Context manager that measures wall-clock elapsed time for a block of code and emits a logging.info entry with the measured duration in milliseconds. This utility is intended for use in the MACE codebase (for example in LAMMPS/MLIAP calculator integration, training, evaluation, preprocessing, or performance debugging) to label and record how long specific operations take (neighbor list construction, model forward passes, data preprocessing, etc.). It uses Python's high-resolution time.perf_counter() to measure elapsed real time and formats the log message as "Timer - {name}: {elapsed_ms:.3f} ms".
    
    Args:
        name (str): Human-readable label for the timed code block. This string is inserted verbatim into the log message and serves to identify which operation was measured (for example "forward_pass", "neighbor_list_build", or "data_preprocess"). Keep the label concise and descriptive so the log can be correlated with training/evaluation steps recorded when using MACE tools and scripts.
        enabled (bool): Flag that controls whether timing and logging are performed. If True (default) the context manager records the start time, yields control to the wrapped block, and upon exit computes elapsed time and logs it via logging.info. If False, the context manager yields immediately without measuring time or producing any log output; this allows conditional disabling of instrumentation without changing call sites.
    
    Behavior and side effects:
        When enabled is True, the context manager:
            - Captures a start timestamp using time.perf_counter() before yielding to the with-block.
            - On normal exit or if the with-block raises an exception, computes elapsed = time.perf_counter() - start, converts to milliseconds (elapsed*1000), and emits a logging.info message in the format "Timer - {name}: {elapsed_ms:.3f} ms".
            - Always performs the logging in a finally clause, so the elapsed time is logged even if the wrapped code raises an exception; the original exception is not suppressed and will propagate after logging.
            - Uses the standard Python logging framework; if the logging configuration does not include handlers or the logging level filters out INFO, the message may not appear. The function does not modify logging configuration.
        When enabled is False, the context manager yields control immediately and produces no timing measurement or logging side effects.
        The timing measured is wall-clock time from the host process perspective. For asynchronous device operations (for example CUDA kernels launched on the GPU), this context manager does not perform device synchronization; therefore measured durations may not reflect true device execution time unless the caller explicitly synchronizes (e.g., torch.cuda.synchronize()) before and/or after the timed block.
    
    Failure modes and exceptions:
        - If an exception occurs inside the with-block, the elapsed time is still computed and logged, and then the exception is re-raised to the caller.
        - Missing or misconfigured logging handlers may prevent the emitted message from appearing in logs; the function does not raise an error in this case.
        - The timer relies on time.perf_counter(); on platforms where perf_counter has low resolution, measurements for extremely short blocks may be imprecise.
    
    Returns:
        None: This function is a context manager and does not return a value to the with-statement. Its observable effect (when enabled) is the side effect of logging the elapsed time as described above.
    """
    from mace.calculators.lammps_mliap_mace import timer
    return timer(name, enabled)


################################################################################
# Source: mace.tools.run_train_utils.combine_datasets
# File: mace/tools/run_train_utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def mace_tools_run_train_utils_combine_datasets(datasets: list, head_name: str):
    """Combine multiple datasets which might be of different types and return a single object suitable for use by MACE training/evaluation code.
    
    This utility is used by the MACE training pipeline (for example in run_train.py and preprocessing scripts) to merge multiple data sources that represent configurations for a given model "head" into one combined dataset object or list. It supports inputs that are Python lists (e.g., lists of raw configuration dicts or ASE Atoms generated during preprocessing) and PyTorch Dataset objects (e.g., preprocessed HDF5-backed Dataset or TensorDataset). The function attempts safe fallbacks when inputs are mixed types, logs informative messages using head_name to aid debugging, and may perform conversions that materialize dataset items into memory.
    
    Args:
        datasets (list): Ordered list of dataset-like objects to combine for the given head. Each element may be a Python list (containing raw data items such as configuration dictionaries or Atoms) or a PyTorch Dataset-like object implementing __len__ and __getitem__. The function treats an empty list as "no data" and returns an empty list. In the MACE domain this parameter is typically provided by preprocessing scripts or the training CLI when multiple files or partitions (train/valid/test/config_type-specific) need to be merged for a particular loss head.
        head_name (str): Identifier for the current model head (for example a string naming a task or config_type). This value is used only in log messages to disambiguate which head the combination applies to, helping diagnose dataset composition issues when training multi-head or heterogeneous-label models.
    
    Returns:
        list or torch.utils.data.dataset.Dataset: The combined dataset for use by downstream training/evaluation code. The concrete return value depends on the input types and the success of internal conversion attempts:
            - If datasets is empty, returns an empty list.
            - If every element of datasets is a Python list, returns a single Python list containing all items in order (shallow concatenation, original item references preserved). This is used when data are represented as raw items and the training pipeline expects an in-memory list.
            - If every element of datasets is not a Python list (i.e., they are Dataset objects), returns a torch.utils.data.ConcatDataset containing the inputs when there is more than one Dataset, or returns the single Dataset object unchanged when only one element was provided. This path preserves Dataset behaviors such as lazy loading and HDF5-backed access preferred for large datasets.
            - If inputs are mixed (some lists, some Dataset objects), the function first attempts to build and return a single Python list by iterating over list elements and by indexing each Dataset via ds[i] for i in range(len(ds)). This requires that any Dataset objects implement __len__ and __getitem__ and will materialize all items into memory; it may be slow or memory intensive for large datasets.
            - If the above mixed-type list conversion fails (for example, indexing Datasets is unsupported or raises an exception), the function next attempts to convert Python lists to placeholder torch.utils.data.TensorDataset objects (creating small integer tensors to represent indices) and then returns a torch.utils.data.ConcatDataset of these converted Dataset objects and original Dataset inputs. This fallback preserves a Dataset return type but the placeholder conversion is a lossy transformation (it does not recreate original list item contents) and is primarily a compatibility attempt used by the training utilities.
            - If all conversion attempts fail, the function logs a warning and returns the first element of the provided datasets list as a last-resort fallback.
        The caller should be prepared to handle either an in-memory list of items or a PyTorch Dataset/ConcatDataset. No deep copies are made; list concatenation and Dataset wrapping preserve references to original items or datasets.
    
    Behavior, side effects, and failure modes:
        - Logging: The function logs informational messages that include head_name and the number/type of inputs; warnings are emitted when conversions fail. These logs are intended to help users of MACE understand how inputs were merged for a specific head.
        - Imports and tensor creation: When converting lists to Dataset placeholders, the function imports torch and torch.utils.data.TensorDataset and creates small torch.Tensor objects. This may add a runtime dependency on PyTorch if that fallback path is executed.
        - Memory and performance: Converting Dataset objects to an in-memory list requires iterating over range(len(ds)) and calling ds[i] for each index; this can be slow or memory-intensive for large datasets and may raise exceptions if __len__ or __getitem__ are not implemented. Prefer providing homogeneous Dataset inputs when possible to keep lazy loading and low memory usage.
        - Compatibility: The function assumes that non-list inputs behave like PyTorch Dataset objects. If provided custom objects that are neither lists nor Dataset-like, conversion attempts may fail and cause the function to return the first provided element.
        - No exception is raised by this function for conversion failures; instead it logs warnings and uses progressively weaker fallbacks. Callers that require strict failure behavior should validate inputs before calling this function.
    """
    from mace.tools.run_train_utils import combine_datasets
    return combine_datasets(datasets, head_name)


################################################################################
# Source: mace.tools.scripts_utils.get_config_type_weights
# File: mace/tools/scripts_utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def mace_tools_scripts_utils_get_config_type_weights(ct_weights: str):
    """Parse a command-line config_type_weights string into a mapping of configuration type names to numeric weights used to weight dataset entries (loss contributions) during MACE training and evaluation.
    
    Args:
        ct_weights (str): A string representation of a Python dictionary specifying per-config-type weights, typically provided via the command-line argument --config_type_weights or in a YAML config (see the README training examples). The string is parsed using ast.literal_eval and therefore must be a valid Python literal that evaluates to a dict (for example '{"Default": 1.0}' or "{'IsolatedAtom': 0.0, 'Default': 1.0}"). The keys are expected to be configuration type names (usually strings matching the dataset's config_type fields) and the values are the corresponding weights (typically floats). This function does not perform numeric-type coercion or detailed validation of individual values beyond ensuring the top-level parsed object is a dict.
    
    Returns:
        dict: A dictionary mapping configuration type names (str) to weights (float). On successful parse of ct_weights, the parsed dictionary is returned unchanged. If parsing fails or the parsed object is not a dict (for example due to malformed syntax, wrong literal type, or other errors), the function logs a warning via logging.warning explaining that the config type weights were not specified correctly and returns the default mapping {"Default": 1.0}. This return value is intended to be used by the training and evaluation pipelines to scale loss terms or to select behavior per config type (as illustrated in the README examples).
    """
    from mace.tools.scripts_utils import get_config_type_weights
    return get_config_type_weights(ct_weights)


################################################################################
# Source: mace.tools.tables_utils.custom_key
# File: mace/tools/tables_utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def mace_tools_tables_utils_custom_key(key: str):
    """mace.tools.tables_utils.custom_key returns a two-element sort key that prioritizes the "train" and "valid" entries when sorting the keys of a data-loader or results dictionary used during MACE training and evaluation. This ensures that the training set and validation set are evaluated (and therefore reported or plotted) before other datasets (for example test or per-config-type results) in scripts such as the training loop (run_train.py / mace_run_train) and evaluation utilities (mace_eval_configs), where deterministic ordering of dataset evaluation and logging is important.
    
    Args:
        key (str): The dictionary key/name for a dataset or configuration (for example "train", "valid", or "test"). In the MACE workflow this is typically the identifier used by the data loader or results collection to label different splits or config types. The function expects a string and uses its exact value to determine priority; passing a non-string may lead to type errors or undefined ordering when used as a sort key.
    
    Returns:
        tuple: A two-element tuple used as a sorting key. The first element is an integer priority (0 for "train", 1 for "valid", 2 for any other key) that enforces the evaluation order. The second element is the original key string, which preserves a deterministic alphabetical ordering among keys with the same priority (for example among all non-"train"/"valid" keys). No side effects; the function is pure and intended to be passed as the key argument to Python's sorted() or similar ordering utilities. Failure modes: supplying non-string inputs or keys that are not comparable to other keys in the sort will raise exceptions when used in sorting operations.
    """
    from mace.tools.tables_utils import custom_key
    return custom_key(key)


################################################################################
# Source: mace.tools.torch_geometric.seed.seed_everything
# File: mace/tools/torch_geometric/seed.py
# Category: fix_docstring
# Reason: Schema parsing failed: Cannot generate JSON schema for seed_everything because the docstring has no description for the argument 'seed'
################################################################################

def mace_tools_torch_geometric_seed_seed_everything(seed: int):
    """mace.tools.torch_geometric.seed.seed_everything sets the global random seed for Python's random module, NumPy, and PyTorch (including all CUDA devices accessible from the current process). This function is used throughout MACE to improve reproducibility of experiments such as model initialization, data shuffling and splitting, on-line preprocessing, and stochastic training procedures described in the README (training, evaluation, preprocessing, and distributed training workflows).
    
    Args:
        seed (int): The desired seed. This integer is forwarded verbatim to Python's random.seed, numpy.random.seed, torch.manual_seed, and torch.cuda.manual_seed_all. Its practical role is to initialize the pseudo-random number generators used by these libraries so that subsequent calls that depend on randomness (for example weight initialization, data augmentation, batching order, or random splits) produce repeatable sequences when the same seed and same execution environment are used.
    
    Behavior and side effects:
        Calling this function mutates the global RNG state for the Python random module, NumPy, and PyTorch in the current process. It sets the CPU RNG state via random.seed and numpy.random.seed, sets the PyTorch CPU RNG via torch.manual_seed, and sets the CUDA RNG state for all GPUs visible to the current process via torch.cuda.manual_seed_all. Because it only modifies these RNG states, reproducibility is limited by other factors: nondeterministic CUDA kernels, platform or hardware differences, PyTorch/CUDA/cuDNN versions, and certain PyTorch operations that are inherently nondeterministic. The function does not modify PyTorch backend flags (for example torch.backends.cudnn.deterministic or torch.backends.cudnn.benchmark); if full determinism is required, callers must set those flags and account for potential performance impacts. In multi-process or distributed training setups, this function sets RNGs for the current process only; users should ensure appropriate per-process seeding (for example using different seeds per rank or deriving per-rank seeds) as required by their distributed training strategy.
    
    Returns:
        None: This function does not return a value. Instead, its effect is to set global RNG states as described above so that subsequent library calls that use randomness are reproducible within the same execution environment and configuration.
    """
    from mace.tools.torch_geometric.seed import seed_everything
    return seed_everything(seed)


################################################################################
# Source: mace.tools.torch_geometric.utils.download_url
# File: mace/tools/torch_geometric/utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def mace_tools_torch_geometric_utils_download_url(url: str, folder: str, log: bool = True):
    """Downloads the content of a URL to a local folder and returns the local file path.
    
    Args:
        url (string): The URL to download. This should point to a single file resource (for example, a model weight file or a dataset archive hosted on GitHub releases or another HTTP/HTTPS server). The function extracts the filename by taking the substring after the last "/" in the URL and removing any query string (the portion after "?"). Supplying a URL that does not resolve to a file-like resource, or a malformed URL, will raise a URL/HTTP-related exception from urllib.
        folder (string): The destination folder on the local filesystem where the downloaded file will be stored. If the folder does not exist, the function will create it (using the repository's makedirs helper). If the process lacks permission to create or write to this folder, an OSError (or subclass) will be raised.
        log (bool, optional): If False, the function suppresses user-facing console messages. If True (default), the function prints progress messages such as "Downloading <url>" when starting a download and "Using exist file <filename>" if the target file already exists. This flag controls only printing; it does not affect exceptions or file I/O behavior.
    
    Returns:
        string: The absolute or relative filesystem path (as constructed by os.path.join) to the file that now exists in folder. If the file already existed at that path prior to the call, the existing path is returned without re-downloading. If the file was not present, the function downloads the URL content using urllib.request.urlopen with an SSL context created via ssl._create_unverified_context(), writes the bytes to the destination path in binary mode, and then returns the path.
    
    Behavior and side effects:
        The function determines the target filename from the URL, joins it to folder to form the destination path, and returns that path. If the destination file already exists on disk, the function returns immediately and does not perform a network request. If the file does not exist, the function:
        - Creates the destination folder if necessary (side effect: new directories may appear on disk).
        - Opens an HTTPS/HTTP connection using urllib.request.urlopen with an unverified SSL context (ssl._create_unverified_context()), which disables SSL certificate verification for the request.
        - Reads the response bytes and writes them to the destination path in binary mode.
        - Returns the destination path after successful write.
        Console output is controlled by the log parameter.
    
    Failure modes and warnings:
        - Network errors: urllib throws URLError, HTTPError, or related exceptions if the host is unreachable, the URL is invalid, or the server returns an error status. These propagate to the caller.
        - SSL and security: the function uses an unverified SSL context (intentionally bypasses certificate verification). This can expose the download to man-in-the-middle attacks; callers that require verified SSL should perform integrity checks (for example, checksum or signature verification) on the returned file.
        - File I/O errors: writing to disk can raise OSError/IOError if the filesystem is full, the process lacks permissions, or for other I/O failures.
        - Race conditions: if multiple processes invoke this function concurrently for the same destination path, a partial or corrupted file may be written; the function does not implement atomic download or locking semantics.
        - Filename extraction: query parameters in the URL are removed when forming the filename. If the URL does not contain a filename component, the resulting filename may be empty or unexpected and could lead to errors.
    
    Practical significance in MACE:
        This utility is used by MACE tooling to fetch external resources such as pretrained foundation models, example datasets, or release artifacts referenced in the README and tooling. It allows higher-level code (for example, model-loading helpers or CLI scripts) to obtain required files into a local cache directory and then load them from the returned path. Users of MACE should verify downloaded model files (e.g., via checksums) if provenance or integrity is required.
    """
    from mace.tools.torch_geometric.utils import download_url
    return download_url(url, folder, log)


################################################################################
# Source: mace.tools.torch_geometric.utils.extract_zip
# File: mace/tools/torch_geometric/utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def mace_tools_torch_geometric_utils_extract_zip(path: str, folder: str, log: bool = True):
    """mace.tools.torch_geometric.utils.extract_zip extracts a ZIP archive from the filesystem to a target folder. This utility is used in the MACE codebase (torch_geometric utilities) to unpack archived resources such as preprocessed datasets, example inputs, or model artifacts that may be distributed as .zip files for preprocessing, on-line data loading, training, or evaluation workflows.
    
    The function opens the ZIP file at the given path for reading and calls zipfile.ZipFile.extractall to write the archive contents under the destination folder. The primary practical significance is to make the files contained in an archive available on disk for subsequent preprocessing (for example, the preprocessing scripts and training workflows described in the README), model loading, or evaluation steps.
    
    Args:
        path (string): The filesystem path to the ZIP archive to extract. This should be a path to a readable .zip file on the local filesystem (for example a downloaded dataset or model archive). If the file does not exist or is not a valid ZIP file, the call will raise an exception (see Failure modes).
        folder (string): The destination folder where archive members will be written. Paths inside the archive are joined to this folder when creating files on disk; the destination must be writable by the current process. Existing files with the same names may be overwritten by the extraction. This parameter is typically used to point to a preprocessing or output directory (for example a processed_data/ directory used by MACE preprocessing scripts).
        log (bool, optional): If False, callers indicate they do not want informational console messages. (default: True) Note: in the current implementation the function does not emit logged messages, so this flag is accepted but not used to produce output; it exists for API compatibility with higher-level utilities that may pass logging preferences.
    
    Returns:
        None: The function does not return a value. Side effects: files and directories from the ZIP archive are created or overwritten under the filesystem path given by folder. After successful return the archive contents are available on disk for downstream steps (preprocessing, loading, training, evaluation).
    
    Behavior, side effects, defaults, and failure modes:
    - The function uses Python's zipfile.ZipFile(path, "r") and extractall(folder) to perform extraction.
    - If path does not point to an existing file, a FileNotFoundError (or an OSError subclass) will be raised by the underlying open operation.
    - If the file at path is not a valid ZIP archive, zipfile.BadZipFile will be raised.
    - If the destination folder is not writable or the process lacks permissions, a PermissionError or OSError may be raised when writing files.
    - Extraction will overwrite existing files in folder that share the same names as archive entries.
    - Extracting untrusted archives can be unsafe: archives may contain absolute paths or path traversal entries (e.g., "../") that can write outside folder. Callers should validate archives or sanitize entries before extraction if the archive content is not trusted.
    - The log parameter defaults to True for API compatibility, but the current implementation does not produce console output; callers should not rely on logging from this function.
    """
    from mace.tools.torch_geometric.utils import extract_zip
    return extract_zip(path, folder, log)


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
