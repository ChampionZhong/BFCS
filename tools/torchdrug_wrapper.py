"""
Regenerated Google-style docstrings for module 'torchdrug'.
README source: others/readme/torchdrug/README.md
Generated at: 2025-12-02T01:10:30.164484Z

Total functions: 16
"""


import torch

################################################################################
# Source: torchdrug.data.dataloader.graph_collate
# File: torchdrug/data/dataloader.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def torchdrug_data_dataloader_graph_collate(batch: list):
    """torchdrug.data.dataloader.graph_collate
    Collate a list of identically-structured samples into a single batched container suitable for model input and DataLoader output.
    
    This function is used in TorchDrug to prepare batches of graph-structured data (for example, molecules or interaction graphs used in drug discovery tasks) and other nested containers so they can be processed efficiently by downstream models and by PyTorch DataLoader workers. It supports torch.Tensor stacking with optional shared-memory allocation when running in a DataLoader worker process, conversion of numeric scalars into tensors, preservation of raw string/bytes lists, packing of Graph objects via data.Graph.pack for batched graph processing, and recursive collation of Mapping and Sequence containers. The caller must provide a non-empty list of samples where each sample has the same nested container structure and element types; mismatches or empty input are failure modes documented below.
    
    Args:
        batch (list): list of samples with the same nested container. Each sample is expected to be one of: torch.Tensor, float, int, str, bytes, data.Graph, Mapping, or Sequence (as used in the source). The role of this parameter is to supply per-sample data produced by a dataset so that graph_collate can merge them into a single batch for training or inference in TorchDrug. In practice, batch is the list passed by torch.utils.data.DataLoader when batch_size > 1.
    
    Returns:
        object: Collated batch. The concrete return type depends on the element type of batch[0] and is produced deterministically as follows:
        - If elements are torch.Tensor: returns a torch.Tensor created by stacking all tensors along dimension 0 (torch.stack(batch, 0)). If called inside a DataLoader worker process (torch.utils.data.get_worker_info() is not None), the function attempts to allocate a shared storage buffer and create an output tensor that uses that storage to enable inter-process shared memory for more efficient batching.
        - If elements are float: returns torch.tensor(batch, dtype=torch.float).
        - If elements are int: returns torch.tensor(batch) (integer tensor).
        - If elements are str or bytes: returns the original Python list batch unchanged (strings/bytes are not converted to tensors).
        - If elements are data.Graph: returns elem.pack(batch) where elem is batch[0]; this delegates to data.Graph.pack to produce a packed graph representation suitable for batched graph neural network processing in TorchDrug.
        - If elements are Mapping: returns a dict mapping each key to the result of recursively collating the list of corresponding values across samples ({key: graph_collate([d[key] for d in batch]) for key in elem}).
        - If elements are Sequence: first verifies that every sequence in batch has equal length; if so, returns a list by zipping corresponding positions and recursively collating each tuple (i.e., [graph_collate(samples) for samples in zip(*batch)]).
        The return value is intended for direct consumption by TorchDrug models, training loops, and PyTorch DataLoader consumers and preserves the nested container shape and types in a batched form.
    
    Behavior, side effects, defaults, and failure modes:
        - The function expects batch to be non-empty. If batch is empty, accessing batch[0] raises IndexError.
        - For torch.Tensor elements, when called inside a DataLoader worker (multiprocessing), the function creates a shared storage buffer sized to the total number of elements across tensors in batch and constructs an output tensor that uses that storage. This side effect enables shared-memory batching between worker processes and the parent process to reduce memory copying, mirroring the behavior in the source code lines that call elem.storage()._new_shared(numel) and elem.new(storage).
        - For Sequence elements, all sequences in the batch must have equal length; otherwise a RuntimeError is raised: "Each element in list of batch should be of equal size".
        - For Mapping elements, all samples are expected to contain the same keys; missing keys will raise a KeyError during list comprehension [d[key] for d in batch].
        - If an element type is encountered that is not one of the supported types above, the function raises TypeError with the message "Can't collate data with type `<type>`".
        - The function relies on data.Graph.pack for batching graph objects; any semantics, constraints, or side effects of packing are those of data.Graph.pack (see data.Graph.pack documentation). Packing is necessary in TorchDrug to combine multiple graph objects (e.g., molecular graphs) into a single representation that Graph-based models can process efficiently.
        - The function preserves Python-native types for strings/bytes and only converts numeric scalars and tensors to torch tensors; it does not perform device transfers or dtype coercion beyond the explicit float->torch.float behavior for floats.
        - This collate utility is intended for use in the TorchDrug domain of graph-structured drug discovery data to ensure that batched data fed into models have consistent nested structure and are memory-efficient when used with multi-worker DataLoader pipelines.
    """
    from torchdrug.data.dataloader import graph_collate
    return graph_collate(batch)


################################################################################
# Source: torchdrug.utils.comm.init_process_group
# File: torchdrug/utils/comm.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def torchdrug_utils_comm_init_process_group(backend: str, init_method: str = None, **kwargs):
    """Initialize CPU and/or GPU process groups used by TorchDrug to enable inter-process
    communication for distributed training and data-parallel execution.
    
    This function is a thin wrapper around torch.distributed.init_process_group and is used
    throughout TorchDrug to prepare the communication primitives that other modules (for
    example core.Engine and models) rely on when running on multiple CPUs or GPUs. It sets
    two module-level globals, cpu_group and gpu_group, which represent process groups for
    CPU-side and GPU-side collective operations respectively. After calling this function,
    collective operations in TorchDrug that depend on cpu_group or gpu_group can be used
    for synchronizing parameters, gradients, or other tensors across processes.
    
    Args:
        backend (str): Communication backend. Use "nccl" for GPUs and "gloo" for CPUs.
            This argument is passed directly to torch.distributed.init_process_group and
            determines the low-level transport used for collective operations. Choosing
            "nccl" is required for CUDA-based collective communication on NVIDIA GPUs;
            "gloo" is a CPU-capable backend and can also be used for GPU communication on
            platforms where NCCL is unavailable.
        init_method (str, optional): URL specifying how to initialize the process group.
            This argument is passed through to torch.distributed.init_process_group and
            controls how processes discover each other (for example via TCP or a shared file
            system). If None, the underlying torch.distributed implementation will use its
            default initialization behavior (for example, environment-variable-based
            initialization) as documented in torch.distributed.
        kwargs (dict): Additional keyword arguments forwarded unchanged to
            torch.distributed.init_process_group. These keyword arguments are used by the
            underlying PyTorch API to configure process group initialization and therefore
            must follow the torch.distributed.init_process_group specification. They are
            not interpreted by this function itself.
    
    Behavior and side effects:
        This function calls torch.distributed.init_process_group(backend, init_method,
        **kwargs) to initialize the global default process group. Immediately after that
        call, gpu_group is set to dist.group.WORLD (the global default group). If backend
        equals "nccl", the function creates a separate CPU-capable process group by calling
        dist.new_group(backend="gloo") and assigns it to the module-level cpu_group; this
        allows TorchDrug to perform CPU-based collective operations alongside GPU-based
        NCCL collectives. If backend is not "nccl", cpu_group is set to the same group as
        gpu_group (i.e., no separate CPU group is created). The function therefore
        mutates module-level state (cpu_group and gpu_group) but does not return a value.
    
    Defaults and usage notes:
        The practical effect in the TorchDrug domain is to prepare process groups so that
        training and evaluation code can run across multiple processes and devices as shown
        in the README examples (single CPU, multiple CPUs, single GPU, multiple GPUs,
        distributed GPUs). Call this function before performing any distributed collective
        operations. Typical callers are high-level Engine or training setup code that needs
        synchronized communication across replicas.
    
    Failure modes and errors:
        Errors raised by torch.distributed.init_process_group are propagated to the caller.
        Common failure modes include attempting to initialize an already-initialized
        distributed environment, providing an unsupported backend for the available devices,
        missing or incorrect initialization information (for multi-node setups the init
        method or rendezvous parameters must be correct), network connectivity failures, or
        mismatched world_size/rank configuration across processes. If backend == "nccl",
        ensure CUDA and the NCCL library are available on all participating processes.
    
    Returns:
        None: This function does not return a value. Its primary effect is to initialize the
        global torch.distributed process group and to set the module-level globals cpu_group
        and gpu_group so that the rest of the TorchDrug codebase can perform CPU- and
        GPU-based collective operations.
    """
    from torchdrug.utils.comm import init_process_group
    return init_process_group(backend, init_method, **kwargs)


################################################################################
# Source: torchdrug.utils.file.compute_md5
# File: torchdrug/utils/file.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def torchdrug_utils_file_compute_md5(file_name: str, chunk_size: int = 65536):
    """torchdrug.utils.file.compute_md5: Compute the MD5 checksum of a file.
    
    This function reads the file specified by file_name in binary mode and computes
    its MD5 digest using hashlib.md5. It processes the file in chunks of size
    chunk_size to limit peak memory usage when handling large files. In the
    TorchDrug project, this function is intended for practical tasks such as
    verifying downloaded dataset or model file integrity, generating stable cache
    keys for file-based caching, and detecting unintended file modifications to
    support reproducibility of experiments.
    
    Args:
        file_name (str): Path to the target file whose MD5 checksum will be
            computed. This should be a filesystem path string accessible to the
            running process. The function opens the file in binary mode ("rb")
            and reads its contents without modifying it; the file is closed
            automatically when reading completes or when an exception occurs.
        chunk_size (int): Chunk size in bytes used for incremental reading of the
            file. The default value is 65536 (64 KiB). A larger chunk_size may
            reduce total runtime by performing fewer read calls but increases
            transient memory per read; a smaller chunk_size reduces per-read
            memory at the cost of more read and update operations. For typical use
            in TorchDrug (dataset and model files), the default value is a
            reasonable trade-off. chunk_size is expected to be an integer; using
            non-positive or unusual values is not recommended because it can lead
            to unexpected behavior (for example, reading zero bytes will produce
            the MD5 of empty content).
    
    Returns:
        str: Lowercase hexadecimal MD5 digest of the file contents (32 hex
        characters). This string is a deterministic checksum for the current file
        contents and can be used to verify integrity, compare files, or as a
        stable identifier for caching.
    
    Notes on failure modes and side effects:
        - If file_name does not exist or is not accessible, the function will
          raise FileNotFoundError or PermissionError raised by the underlying
          open/read operations. Other I/O errors will raise OSError.
        - The function reads the entire file content (in chunks) and may take
          significant time for large files; choose chunk_size accordingly.
        - If the file is modified concurrently while being read, the resulting
          MD5 digest will reflect the bytes read and may not correspond to any
          atomic snapshot of the file.
        - The function does not write to or alter the file; it only reads and
          computes the checksum.
    """
    from torchdrug.utils.file import compute_md5
    return compute_md5(file_name, chunk_size)


################################################################################
# Source: torchdrug.utils.file.download
# File: torchdrug/utils/file.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def torchdrug_utils_file_download(
    url: str,
    path: str,
    save_file: str = None,
    md5: str = None
):
    """torchdrug.utils.file.download downloads a file from a URL and saves it to a local path, with an optional MD5-based skip mechanism commonly used by TorchDrug dataset and model asset installers to avoid re-downloading already-available files.
    
    This function is used throughout TorchDrug to fetch dataset files, pretrained model weights, and other remote assets required by the library. It writes a file to disk (side effect), logs the download action via the module logger, and returns the local pathname of the saved file for subsequent use by dataset loaders or model constructors. The function will skip downloading only when a file already exists at the target location and its MD5 matches the provided md5 parameter; otherwise, it will perform the network download using urlretrieve from six.moves.urllib.request.
    
    Args:
        url (str): URL to download. This is the remote HTTP/HTTPS location of the resource (for example, a dataset archive or pretrained model file) that TorchDrug components request when preparing data or models.
        path (str): path to store the downloaded file. This is a local directory path that must exist and be writable by the process; the function will save the file under this directory. The function does not create intermediate directories.
        save_file (str, optional): name of save file. If not specified (None), the function infers the filename from the URL by taking os.path.basename(url) and stripping any query parameters beginning with '?'. If provided, this exact filename is joined with the provided path to form the full destination pathname.
        md5 (str, optional): MD5 of the file. When provided, the function will compute the MD5 of any existing file at the destination and skip the download if the computed MD5 equals this value. If no file exists or the MD5 differs, the function proceeds to download. Note that the function checks MD5 only before downloading and does not re-check or validate the MD5 after the download completes.
    
    Returns:
        str: The full local filesystem path to the saved file (os.path.join(path, save_file)). Side effects include writing the downloaded file to disk and logging an informational message when a download occurs. The function may overwrite an existing file at the destination when the existing file is present but its MD5 does not match the provided md5 value.
    
    Failure modes and behavior notes:
        Network or HTTP errors raised by urlretrieve (for example, connection failures or HTTP errors) will propagate as exceptions from the underlying urllib implementation. File system errors (for example, missing destination directory, permission denied, or insufficient disk space) will raise the corresponding OS exceptions. The caller should ensure that the destination directory exists and is writable before calling this function. The function relies on compute_md5 and a module logger; if those utilities are unavailable or raise errors, those exceptions will also propagate.
    """
    from torchdrug.utils.file import download
    return download(url, path, save_file, md5)


################################################################################
# Source: torchdrug.utils.file.extract
# File: torchdrug/utils/file.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def torchdrug_utils_file_extract(zip_file: str, member: str = None):
    """Extract files from a compressed archive used by TorchDrug. This utility unpacks archives commonly used to distribute datasets, pretrained model weights, or other resources in the TorchDrug project and related workflows. Supported archive types are .zip, .gz, .tar.gz, .tgz and .tar. The function writes extracted files into the directory containing the archive (the save path) and returns either the single extracted file path or the save directory path.
    
    The function is typically used in the TorchDrug data-loading and setup pipelines to materialize files from downloaded compressed archives so subsequent dataset, model loading, or preprocessing code can read them from disk.
    
    Behavior summary and practical details:
    - The save directory is determined as os.path.dirname(zip_file), i.e., files are extracted next to the archive by default.
    - For .gz (single-file gzip) archives, the function derives the original member name from the gzip file name (the archive name without the .gz suffix or .tar.gz handling) and extracts that single file.
    - For .tar, .tar.gz and .tgz archives, the function iterates over archive members (tar.getnames()). For each member, directory members are created with os.makedirs(save_file, exist_ok=True). Non-directory members are extracted via tar.extractfile and written to disk.
    - For .zip archives, the function iterates over zip.namelist(), creates directories for directory members, and writes each file via zipped.open and a streaming copy to disk.
    - When member is provided, only that member is extracted. In that case the extracted file is written into the save path using the basename of the member (i.e., nested archive path information is not preserved when extracting a single member; the output file is save_path / os.path.basename(member)).
    - Before writing a member, the function compares the existing destination file size with the archived file size (for .gz the original file length is obtained via the gzip trailer and struct.unpack("<I", ...)). If sizes match, extraction of that member is skipped to avoid unnecessarily overwriting identical content.
    - The function logs extraction actions via logger.info calls present in the implementation.
    - Unknown or unsupported archive extensions will raise ValueError("Unknown file extension `%s`" % extension). Other errors from the underlying libraries (gzip, tarfile, zipfile) or I/O (e.g., permission errors) are propagated.
    
    Args:
        zip_file (str): Path to the compressed archive file to extract. In the TorchDrug context this is typically a downloaded dataset archive, pretrained model archive, or other resource bundle. The function determines the save directory as the directory containing this file (os.path.dirname(zip_file)) and uses the file name extension to select the appropriate extraction procedure (.zip, .gz, .tar.gz/.tgz, .tar).
        member (str, optional): If specified, the name of a single archive member to extract. For .zip and .tar archives this should match an entry in the archive (a member name returned by zip.namelist() or tar.getnames()). When provided, only this member is extracted and the output file is written to the save directory using os.path.basename(member). If omitted (the default), all members in the archive are extracted into the save directory, preserving member path structure when multiple members are written.
    
    Returns:
        str: If exactly one file was written (len(save_files) == 1), returns the filesystem path to that extracted file. Otherwise returns the path to the directory where files were extracted (the save directory, os.path.dirname(zip_file)). Side effects include creating directories, writing files to disk, and logging extraction actions. Exceptions from underlying archive libraries or I/O are not caught and will propagate; a ValueError is raised for unsupported file extensions.
    """
    from torchdrug.utils.file import extract
    return extract(zip_file, member)


################################################################################
# Source: torchdrug.utils.file.get_line_count
# File: torchdrug/utils/file.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def torchdrug_utils_file_get_line_count(file_name: str, chunk_size: int = 8388608):
    """Get the number of newline characters in a file by scanning it in binary chunks.
    
    Args:
        file_name (str): Path to the file whose lines are to be counted. In the TorchDrug project this is typically used when indexing or pre-processing large dataset files (for example dataset text files or CSVs) to obtain an approximate or exact line count for progress reporting, sharding, or dataset sizing. The function opens this path in binary mode ("rb") and will raise standard I/O exceptions (e.g. FileNotFoundError, PermissionError, OSError) if the file cannot be opened or read.
        chunk_size (int, optional): Size in bytes of each read operation used to iterate through the file. Defaults to 8388608 (8 * 1024 * 1024, as set in the function signature). Choosing a larger chunk_size may reduce the number of system calls and improve throughput at the cost of higher peak memory use; choosing a smaller chunk_size reduces memory use but may increase read overhead. The function expects a positive integer; non-positive values may produce incorrect results (for example, a value of 0 will cause the function to return 0) or cause fallback behavior of the underlying file.read call.
    
    Returns:
        int: The count of newline bytes (b"\n") found in the file. Because the file is read in binary mode and the function counts occurrences of the newline byte, this value equals the number of line terminators in the file. For typical Unix-style text files that end each line with '\n', this value equals the number of lines. If the final line in the file does not end with a newline byte, the returned count will be one less than the number of logical text lines. This return value is useful in TorchDrug workflows for estimating dataset sizes, driving progress bars, or partitioning large files without loading the entire file into memory.
    """
    from torchdrug.utils.file import get_line_count
    return get_line_count(file_name, chunk_size)


################################################################################
# Source: torchdrug.utils.file.smart_open
# File: torchdrug/utils/file.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def torchdrug_utils_file_smart_open(file_name: str, mode: str = "rb"):
    """Open a regular or compressed file and return a file-like object.
    
    This function is a drop-in replacement for the built-in open() that transparently handles files compressed with bzip2 (.bz2) and gzip (.gz). The implementation determines the compression type by examining the file extension using os.path.splitext(file_name)[1]; if the extension equals ".bz2" it opens the file with bz2.BZ2File, if the extension equals ".gz" it opens the file with gzip.GzipFile, otherwise it falls back to the builtin open(). In the TorchDrug project this is used to simplify code that loads datasets, model checkpoints, and other data files so callers do not need to handle compressed and uncompressed files separately. The default mode is "rb" (binary read) because many TorchDrug artifacts (for example PyTorch checkpoint files and binary dataset formats) are read in binary mode.
    
    Args:
        file_name (str): Path to the file to open. This should be a filesystem path string pointing to a regular file used by TorchDrug (for example a dataset file, a model checkpoint, or a molecule data file). The function only inspects the filename extension to decide whether to use a compressor-specific opener; it does not inspect file contents. If the path has no extension or an extension other than ".bz2" or ".gz", the function will call the builtin open(file_name, mode).
        mode (str): Open mode passed verbatim to the underlying opener. Defaults to "rb". Practically, this should follow the same semantics as Python's builtin open() (for example "r", "rb", "w", "wb", "rt", "wt"). The given mode is forwarded to bz2.BZ2File or gzip.GzipFile for compressed files, or to built-in open() for uncompressed files. The default "rb" is commonly used in TorchDrug for reading binary checkpoints and dataset files.
    
    Returns:
        file object: A file-like object opened in the requested mode. Concretely, this will be an instance of bz2.BZ2File when file_name ends with ".bz2", an instance of gzip.GzipFile when file_name ends with ".gz", or the object returned by the builtin open() for other extensions. The caller is responsible for closing the returned object (for example by using a with statement) to release system resources.
    
    Behavior, defaults, side effects, and failure modes:
        - Compression detection is based solely on the filename extension returned by os.path.splitext; a file with a mismatched extension (for example a gzipped file named "data" without ".gz") will not be automatically decompressed.
        - Only ".bz2" and ".gz" extensions are handled specially. Other compressed formats (for example ".zip") are not handled by this function.
        - The function does not support remote URLs, S3 paths, or other non-filesystem schemes; file_name must refer to a local filesystem path.
        - Underlying openers may raise standard file I/O exceptions (FileNotFoundError, OSError, IOError) or compression-specific exceptions from the gzip/bz2 modules; callers should handle these exceptions where appropriate.
        - Resource management is the caller's responsibility: always close the returned file object (use with smart_open(...) as f:) to avoid leaking file descriptors.
    """
    from torchdrug.utils.file import smart_open
    return smart_open(file_name, mode)


################################################################################
# Source: torchdrug.utils.io.input_choice
# File: torchdrug/utils/io.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def torchdrug_utils_io_input_choice(prompt: str, choice: tuple = ('y', 'n')):
    """torchdrug.utils.io.input_choice prints a formatted prompt to standard output and blocks until the user supplies a string that matches one of the allowed choices. This function is a small command-line I/O utility intended for interactive TorchDrug workflows (for example, confirming dataset downloads, overwriting files, or proceeding with potentially destructive operations during training or data preparation). It formats the displayed prompt by appending the allowed choices in parentheses and repeatedly calls the built-in input() until a valid choice is entered.
    
    Args:
        prompt (str): Prompt string to display to the user. The function constructs the final prompt text using the pattern "%s (%s)" % (prompt, "/".join(choice)) so the allowed choices appear after the prompt in parentheses (for example "Continue (y/n)"). This text is printed by the built-in input() function. The prompt must be provided as a Python str; non-str types may lead to formatting errors before calling input().
        choice (tuple): Candidate choices as a tuple of str. Default is ('y', 'n'), representing a conventional yes/no confirmation in interactive sessions. The function lowercases each element of this tuple and converts them into a set to perform case-insensitive membership testing; duplicate entries in the tuple are collapsed by the set conversion. During input checking, the user response is lowercased and compared against this set. Note that the comparison requires an exact match after lowercasing (leading or trailing whitespace in the user input is not stripped by this function), and if the tuple contains non-string elements calling lower() on them will raise an exception. If an empty tuple is supplied, no input can match and the function will continue to prompt until input() raises EOFError or the process is interrupted.
    
    Returns:
        str: The exact string entered by the user that matched one of the allowed choices. Matching is performed case-insensitively (by comparing result.lower() to the lowercased choice set), but the returned string preserves the original user-entered casing and spacing. The returned value is intended to be used by the caller to drive control flow (for example, proceeding when 'y' is returned and aborting when 'n' is returned). If the input stream is closed or the user interrupts the process, the built-in input() may raise EOFError or KeyboardInterrupt, which propagate to the caller.
    """
    from torchdrug.utils.io import input_choice
    return input_choice(prompt, choice)


################################################################################
# Source: torchdrug.utils.io.literal_eval
# File: torchdrug/utils/io.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def torchdrug_utils_io_literal_eval(string: str):
    """torchdrug.utils.io.literal_eval evaluates a string into a Python literal value using Python's ast.literal_eval and returns the original string unchanged if parsing fails. This utility is used in TorchDrug to convert textual representations (for example, attributes read from datasets, configuration files, or user-provided strings when registering custom node/edge/graph attributes) into native Python literal objects so downstream code (indexing, feature processing, model configuration) can operate on proper Python types rather than raw text.
    
    Args:
        string (str): The input text to parse. This should be a Python literal expression encoded as a string (for example, text representing numbers, strings, tuples, lists, dictionaries, booleans, or None). The function attempts a safe evaluation using ast.literal_eval, which does not execute arbitrary code and only parses Python literal structures. In the TorchDrug context this parameter typically comes from dataset metadata, attribute values attached to graph or molecule objects, or configuration entries that may contain literal values encoded as strings.
    
    Returns:
        object: The evaluated Python literal value produced by ast.literal_eval when the input string is a valid Python literal expression. If ast.literal_eval raises ValueError or SyntaxError because the input is not a valid literal expression, the function returns the original input string unchanged. There are no side effects (the function does not modify global state or its input), and failure modes are handled by returning the original string rather than raising an exception, providing a safe fallback for downstream code that may treat unparsed text as a string.
    """
    from torchdrug.utils.io import literal_eval
    return literal_eval(string)


################################################################################
# Source: torchdrug.utils.plot.reaction
# File: torchdrug/utils/plot.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def torchdrug_utils_plot_reaction(
    reactants: list,
    products: list,
    save_file: str = None,
    figure_size: tuple = (3, 3),
    atom_map: bool = False
):
    """torchdrug.utils.plot.reaction visualizes a chemical reaction by converting TorchDrug Molecule objects into RDKit reaction templates, rendering the combined reactant and product depiction, and either displaying the image in a viewer or saving it to a PNG file. This function is intended for use in drug discovery and molecular modeling workflows built with TorchDrug (a PyTorch-based toolbox) where quick visual inspection of reaction transformations, including optional atom mapping, is required.
    
    Args:
        reactants (list of Molecule): list of reactant molecules to include in the reaction. Each element is expected to be a TorchDrug Molecule (for example, torchdrug.data.Molecule) that implements a to_molecule() method returning an RDKit Mol. The function calls to_molecule() on each reactant and uses the returned RDKit Mol as a reactant template in an RDKit ChemicalReaction. If a reactant cannot be converted to an RDKit Mol, an exception from that conversion will propagate.
        products (list of Molecule): list of product molecules to include in the reaction. Each element is expected to be a TorchDrug Molecule implementing to_molecule() returning an RDKit Mol. The function calls to_molecule() on each product and uses the returned RDKit Mol as a product template in the RDKit ChemicalReaction. Conversion failures will raise the underlying exception.
        save_file (str, optional): ``png`` file path to save the rendered reaction image. If provided, the function will save the rendered image to this path using the image object's save method, which typically writes a PNG file; an IOError or OSError may be raised if the path is invalid or not writable. If save_file is None (default), the function will attempt to show the image using the image object's show method, which opens the system default image viewer as a side effect.
        figure_size (tuple of int, optional): width and height of the figure in abstract units (default: (3, 3)). The function multiplies each integer by 100 to compute pixel dimensions (size = [100 * s for s in figure_size]) and passes that pixel size to RDKit's Draw.ReactionToImage. Both elements should be integers; providing non-integer or non-positive values may produce unexpected image sizes or runtime errors from the drawing backend.
        atom_map (bool, optional): visualize atom mapping or not (default: False). When False, the function clears atom map numbers on every atom of the converted RDKit Mol by calling SetAtomMapNum(0) for each atom before adding templates to the reaction; when True, existing atom mapping annotations on the RDKit Mol are preserved and rendered if present.
    
    Returns:
        None: This function does not return a value. Its primary effects are side effects: constructing an RDKit ChemicalReaction from the provided reactants and products, rendering the reaction to an image via RDKit Draw.ReactionToImage with size computed from figure_size, and either displaying the image (img.show()) when save_file is None or saving the image to the provided PNG path (img.save(save_file)). Possible failure modes include missing RDKit imports (AllChem or Draw not available), conversion errors from Molecule.to_molecule(), invalid or non-writable save_file paths, and invalid figure_size values that cause the drawing backend to raise an exception.
    """
    from torchdrug.utils.plot import reaction
    return reaction(reactants, products, save_file, figure_size, atom_map)


################################################################################
# Source: torchdrug.utils.pretty.long_array
# File: torchdrug/utils/pretty.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def torchdrug_utils_pretty_long_array(array: list, truncation: int = 10, display: int = 3):
    """Format a list as a concise, human-readable string suitable for logging and display in TorchDrug utilities.
    
    This function is used throughout TorchDrug to produce compact textual summaries of potentially long Python lists (for example, lists of node indices for graphs, atom or bond index lists for molecules, dataset index lists, or other long enumerations) so that logs, visualizations, and printed representations remain readable. If the list length does not exceed the truncation threshold, the full list is returned as its standard Python string representation. If the list is longer than the truncation threshold, the function returns a summary that shows the first few and last few elements separated by the literal ", ..., " to indicate omitted middle elements. The function does not modify the input list.
    
    Args:
        array (list): The input list to format. In TorchDrug this commonly holds things like node or atom indices, attribute lists, or other ordered collections that benefit from compact display. The function expects a Python list (an object that supports len() and slicing); passing an object without these behaviors may raise a TypeError or produce unexpected output.
        truncation (int, optional): Threshold length above which the input list will be summarized by truncation. If len(array) is less than or equal to truncation, the function returns the full string representation of array. If len(array) is greater than truncation, the function returns a truncated summary showing the beginning and end of the list. Defaults to 10.
        display (int, optional): Number of elements to include from both the start and the end of the list when producing a truncated summary. For a truncated list, the output will contain the string form of array[:display] (with its closing bracket removed), then the literal ", ..., ", then the string form of array[-display:] (with its opening bracket removed). Defaults to 3. Note: if display is zero, negative, or larger than the list length, the produced string may be redundant or formatted in an unexpected way because slicing behavior and simple string slicing are used to assemble the result.
    
    Returns:
        str: A string representation of the input list. If len(array) <= truncation, this is the same as str(array). If len(array) > truncation, this is a concatenation of the stringified first display elements and the stringified last display elements separated by ", ..., " (for example, when display == 2 a result might look like "[a, b, ..., y, z]"). There are no side effects: the input list is not modified. Exceptions such as TypeError may be raised if the provided array does not support len() or slicing.
    """
    from torchdrug.utils.pretty import long_array
    return long_array(array, truncation, display)


################################################################################
# Source: torchdrug.utils.pretty.time
# File: torchdrug/utils/pretty.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def torchdrug_utils_pretty_time(seconds: float):
    """torchdrug.utils.pretty.time formats a duration given in seconds into a concise, human-readable string suitable for logging and display in TorchDrug training, evaluation, and data processing workflows (for example, epoch time, batch processing time, or dataset loading time).
    
    This function converts a numeric elapsed time in seconds to one of four unit strings ("secs", "mins", "hours", "days") using fixed thresholds. It is intended to be used in monitoring and reporting runtime information in the TorchDrug domain (PyTorch-based graph and molecular ML tasks) where readable elapsed-time strings simplify experiment logs and user-facing output. The conversion uses constant thresholds sec_per_min = 60, sec_per_hour = 3600, and sec_per_day = 86400 and selects units by strict greater-than comparisons: values strictly greater than a threshold are expressed in the next larger unit. Formatting is performed with two decimal places using "%.2f".
    
    Behavior details, side effects, defaults, and failure modes: The function has no side effects and does not modify input. There is no default for seconds; the caller must supply a float. Numeric values that exactly equal the thresholds (60, 3600, 86400) are formatted in the smaller unit because the comparisons are strict (for example, 60.0 -> "60.00 secs", 3600.0 -> "60.00 mins", 86400.0 -> "24.00 hours"). Negative float values are accepted by the implementation and will be formatted as negative durations in seconds (for example, -5.0 -> "-5.00 secs"). Non-numeric inputs or types incompatible with Python float formatting may raise a TypeError or produce unexpected output; callers should pass a float or a value coercible to float. Special float values such as NaN or infinity will be formatted according to Python's float-to-string behavior.
    
    Args:
        seconds (float): Elapsed time in seconds to format. In the TorchDrug context this typically represents measured durations such as per-batch, per-epoch, data-loading, or inference times during model training and evaluation. The value is compared against the constants sec_per_min (60), sec_per_hour (3600), and sec_per_day (86400) using strict greater-than checks to decide which unit to display. The function expects a numeric float; supply values already converted to seconds.
    
    Returns:
        str: A human-readable string representing the input duration formatted with two decimal places and a unit suffix. Possible outputs take the form "%.2f secs", "%.2f mins", "%.2f hours", or "%.2f days" (for example, "12.34 secs", "1.50 mins", "2.00 hours", "0.12 days"). The selection of unit follows the strict thresholds described above.
    """
    from torchdrug.utils.pretty import time
    return time(seconds)


################################################################################
# Source: torchdrug.utils.torch.cat
# File: torchdrug/utils/torch.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def torchdrug_utils_torch_cat(objs: list, *args, **kwargs):
    """Concatenate a list of nested containers with the same structure used in TorchDrug graph and molecular data processing.
    
    Args:
        objs (list): A non-empty list of objects that share an identical nested structure. Each element in the list must have the same container layout and corresponding element types (for example, the same dict keys, the same sequence lengths for lists/tuples, or the same tensor shapes except along the concatenation dimension when elements are torch.Tensor). In the TorchDrug codebase this function is used to combine outputs such as node/edge feature tensors, lists of per-graph attributes, or containerized graph objects produced during batching and model processing.
        args (tuple): Additional positional arguments forwarded to the underlying concatenation operation. When the elements at a given position are torch.Tensor, these args are passed to torch.cat(objs, *args, **kwargs) and therefore must follow torch.cat semantics (for example, specifying dim). During recursive concatenation, the same args are forwarded into recursive cat calls for nested containers. Note that for data.PackedGraph elements, this function delegates to data.cat(objs) and does not forward args to that call.
        kwargs (dict): Additional keyword arguments forwarded to the underlying concatenation operation. When the elements at a given position are torch.Tensor, these kwargs are passed to torch.cat(objs, *args, **kwargs) and therefore must follow torch.cat semantics (for example, any accepted keyword arguments of torch.cat). During recursive concatenation, the same kwargs are forwarded into recursive cat calls for nested containers. Note that for data.PackedGraph elements, this function delegates to data.cat(objs) and does not forward kwargs to that call.
    
    Behavior and practical significance:
        This function inspects the first element of objs to determine how to concatenate the list. If the first element is a torch.Tensor, it returns torch.cat(objs, *args, **kwargs), using the same semantics and constraints as torch.cat (for example, tensors must have the same dtype and device and matching shapes except along the concatenation dimension). If the first element is an instance of data.PackedGraph (a TorchDrug container for graph data), it delegates to data.cat(objs) to perform concatenation suitable for TorchDrug PackedGraph objects. If the first element is a dict, the function concatenates corresponding values for each key across the list and returns a dict with the same keys mapping to the concatenated results. If the first element is a list or tuple, it zips corresponding positions across objs, recursively concatenates each tuple of elements, and returns an object of the same sequence type (list or tuple) preserving the original sequence type. This recursive behavior enables concatenation of arbitrarily nested containers commonly used in TorchDrug for batching graph-structured data, molecular features, and model outputs.
    
    Side effects and defaults:
        The function itself performs no in-place modification of the input containers; it constructs and returns new container objects (or tensors) as the result of concatenation. The behavior for tensor concatenation, including defaults (for example, default dim) and constraints, follows torch.cat. For data.PackedGraph, behavior follows data.cat defined in TorchDrug. The function requires that objs is non-empty because it uses objs[0] to infer type and structure.
    
    Failure modes and errors:
        If objs is empty, an IndexError will be raised when accessing objs[0]. If the elements of objs do not share the same nested structure (for example, mismatched dict keys, differing sequence lengths, or incompatible tensor shapes for torch.cat), the function may raise KeyError, ValueError, or an error originating from torch.cat or data.cat during recursion. If the element type at the top level is not one of torch.Tensor, data.PackedGraph, dict, list, or tuple, the function raises TypeError("Can't perform concatenation over object type `%s`" % type(obj)), indicating that concatenation for that object type is unsupported.
    
    Returns:
        object: The concatenated result preserving the original nested container structure and container types. Concretely, this can be a torch.Tensor (when concatenating tensors), a data.PackedGraph (when concatenating PackedGraph objects via data.cat), a dict (with the same keys and concatenated values), or a list/tuple (of the same sequence type with concatenated elements). The exact return type and semantics follow the branch taken based on the type of objs[0].
    """
    from torchdrug.utils.torch import cat
    return cat(objs, *args, **kwargs)


################################################################################
# Source: torchdrug.utils.torch.load_extension
# File: torchdrug/utils/torch.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def torchdrug_utils_torch_load_extension(
    name: str,
    sources: list,
    extra_cflags: list = None,
    extra_cuda_cflags: list = None,
    **kwargs
):
    """Load a PyTorch C++ extension just-in-time (JIT) and return a deferred loader that compiles the extension on first use.
    
    This function is used by TorchDrug to register and JIT-compile custom native operators or performance-critical C/C++/CUDA code so that models and data-processing code can run with native performance on CPU and, when available, GPU. The function automatically chooses sensible compilation flags when they are not provided, performs lazy evaluation so compilation is deferred until the extension is actually needed, and is implemented to be safe in multi-process scenarios (for example, DataLoader worker processes or distributed training). This wrapper ultimately forwards work to the machinery used by torch.utils.cpp_extension.load and returns a LazyExtensionLoader object that encapsulates the on-demand build and import behavior.
    
    Args:
        name (str): The identifier used to name the compiled extension module. This name is used to register the compiled library so Python code, including other parts of TorchDrug, can import or reference the extension by name. The name should be chosen to avoid collisions with other extensions; it is used by the underlying PyTorch extension loader to create unique build artifacts.
        sources (list): A list of source file paths (typically strings) that implement the extension. Entries may include C, C++ and CUDA source files. If CUDA is not available and extra_cuda_cflags is None, CUDA source files are removed from this list before creating the loader so that the returned LazyExtensionLoader will not attempt to compile CUDA code on systems without CUDA.
        extra_cflags (list, optional): Extra host (C/C++) compiler flags to pass to the build. If set to None (the default), this function auto-selects flags: it starts with ["-Ofast"], appends OpenMP-related flags ["-fopenmp", "-DAT_PARALLEL_OPENMP"] when torch.backends.openmp.is_available() is True, or appends the fallback "-DAT_PARALLEL_NATIVE" when OpenMP is not available. Note that when extra_cuda_cflags is None and CUDA is available, this function will append "-DCUDA_OP" to the resolved extra_cflags list; that append mutates the list object that the caller provided if a list was passed in. Providing a non-None list disables the automatic default-selection behavior for extra_cflags (except for the possible "-DCUDA_OP" append described above).
        extra_cuda_cflags (list, optional): Extra CUDA-specific compiler flags to pass to nvcc when building CUDA sources. If set to None (the default) and torch.cuda.is_available() is True, this function sets extra_cuda_cflags to ["-O3"] and also appends "-DCUDA_OP" to extra_cflags. If extra_cuda_cflags is None and CUDA is not available, CUDA source files are removed from sources as described above. If the caller supplies a non-None list, the automatic defaulting and source filtering behavior triggered by a None value are bypassed.
        kwargs (dict): Additional keyword arguments forwarded to the LazyExtensionLoader constructor (and ultimately to the underlying torch.utils.cpp_extension.load machinery). Typical forwarded options include build-directory, verbose flags, or other controls accepted by the PyTorch C++ extension loader; these are not interpreted by this wrapper and are passed through as-is.
    
    Returns:
        LazyExtensionLoader: A deferred loader object that encapsulates the extension name, sources, and compilation flags and will perform the actual compilation and import the extension when first accessed. No compilation happens at the time of this function call (lazy evaluation). Side effects include possible mutation of the extra_cflags list argument when CUDA is available and extra_cuda_cflags is None, and filtering out CUDA source files from the provided sources list when CUDA is not available and extra_cuda_cflags is None. Compilation and import failures may raise subprocess or compiler errors propagated from the underlying build toolchain (for example, missing C/C++ compilers, incompatible CUDA/PyTorch versions, or syntax errors in source files). See torch.utils.cpp_extension.load for detailed behavior of the build and import process.
    """
    from torchdrug.utils.torch import load_extension
    return load_extension(name, sources, extra_cflags, extra_cuda_cflags, **kwargs)


################################################################################
# Source: torchdrug.utils.torch.sparse_coo_tensor
# File: torchdrug/utils/torch.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def torchdrug_utils_torch_sparse_coo_tensor(
    indices: torch.Tensor,
    values: torch.Tensor,
    size: list
):
    """Construct a sparse COO tensor without performing index validation. This function is a thin, high-performance wrapper used in the torchdrug library to build sparse tensors (COO format) for graph-structured data such as adjacency matrices or sparse node/edge features. It delegates construction to a low-level backend (torch_ext.sparse_coo_tensor_unsafe) and therefore avoids the index checks performed by torch.sparse_coo_tensor, providing faster construction for workloads in TorchDrug where indices are known to be valid in advance (for example, creating adjacency representations of molecular graphs or minibatch graph unions during training).
    
    Args:
        indices (Tensor): 2D indices describing the nonzero locations in COO format. Expected shape is (2, n), where the first row contains row indices and the second row contains column indices for n nonzero entries. In TorchDrug this is typically used to represent edge endpoints or coordinate pairs for sparse features; the function uses these indices verbatim without checking bounds or uniqueness, so the caller is responsible for ensuring they are valid for the provided size.
        values (Tensor): 1D tensor of length n containing the values corresponding to each column in indices. Each entry in values is placed at the coordinate given by the corresponding column in indices. In graph and molecular workflows within TorchDrug, values often represent edge weights, adjacency indicators, or sparse feature values. The function does not validate alignment beyond relying on the backend, so mismatched lengths (values length != n) will result in a backend error.
        size (list): List specifying the desired size (shape) of the resulting sparse tensor. This list defines the overall dimensions of the sparse COO tensor (for example, [num_rows, num_cols] for a 2D sparse matrix). The provided indices are interpreted against this size; if indices contain out-of-range entries relative to size, undefined behavior or errors from the underlying implementation may occur because no bounds checking is performed.
    
    Returns:
        Tensor: A sparse COO tensor constructed from the provided indices and values with the specified size. The returned tensor is produced by the low-level backend and reflects the inputs verbatim (no index normalization, deduplication, or bounds checking). Practical significance: this return value can be used directly in TorchDrug graph operations and PyTorch computations that accept sparse COO tensors, enabling memory-efficient representation and GPU-accelerated sparse operations when indices are already validated by the caller.
    
    Behavior, side effects, and failure modes:
        This function intentionally skips index validation to maximize performance. As a result, callers must guarantee that indices have shape (2, n), values has length n, and all index entries are within the bounds implied by size. If these conditions are violated, the underlying backend (torch_ext.sparse_coo_tensor_unsafe or the PyTorch runtime) may raise an error or produce an invalid tensor. Duplicate indices or unsorted indices are not merged or processed; behavior with duplicates depends on the backend implementation. Use this function when you can ensure input correctness (for example, after deterministic generation of graph adjacency indices) to avoid the runtime overhead of standard safe constructors like torch.sparse_coo_tensor.
    """
    from torchdrug.utils.torch import sparse_coo_tensor
    return sparse_coo_tensor(indices, values, size)


################################################################################
# Source: torchdrug.utils.torch.stack
# File: torchdrug/utils/torch.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def torchdrug_utils_torch_stack(objs: list, *args, **kwargs):
    """Stack a list of nested containers with the same structure.
    
    This utility is used in the TorchDrug codebase to merge a list of structurally identical objects (for example, per-sample graph or molecule attributes) into a single nested object where leaf tensors are combined by torch.stack. Typical TorchDrug uses include batching node features, edge attributes, atom types, or other registered node/edge/graph attributes so they can be processed efficiently by PyTorch models and moved to accelerators. The function recurses into nested containers that are either torch.Tensor, dict, list, or tuple, and preserves the original container types and nesting while stacking tensor leaves.
    
    Args:
        objs (list): A list of objects that share the same nested structure. Each element in the list must be one of the following at every corresponding position in the structure: torch.Tensor, dict, list, or tuple. For dict positions, every element must contain the same set of keys; for list/tuple positions, every element must have the same length and corresponding element types. The function inspects the first element (objs[0]) to determine the structure; passing an empty list will raise IndexError because objs[0] is accessed.
        args (tuple): Additional positional arguments forwarded directly to torch.stack when the recursion reaches torch.Tensor leaves. These arguments control the stacking operation provided by PyTorch (for example, the stacking dimension). This parameter is passed through unchanged and has no effect except when leaf objects are torch.Tensor.
        kwargs (dict): Additional keyword arguments forwarded directly to torch.stack when the recursion reaches torch.Tensor leaves. These keyword arguments control the stacking operation provided by PyTorch (for example, dim or out if supported by the PyTorch version). These are passed through unchanged and have no effect except when leaf objects are torch.Tensor.
    
    Returns:
        torch.Tensor or dict or list or tuple: If the objects in objs are torch.Tensor, returns the result of torch.stack(objs, *args, **kwargs), which is a torch.Tensor combining the input tensors along the specified dimension. If the objects are dict, returns a dict with the same keys where each value is the result of recursively stacking the list of values for that key. If the objects are list or tuple, returns a list or tuple of the same type and length where each element is the result of recursively stacking the corresponding elements across objs. The returned nested object has the same container types and nesting structure as the input elements, with leaf tensors replaced by their stacked tensor. If the input structure is inconsistent across elements (mismatched keys, differing list/tuple lengths, or differing types at corresponding positions), the function will raise standard Python errors (for example KeyError, IndexError, or TypeError) depending on the mismatch. If a leaf type is encountered that is not torch.Tensor, dict, list, or tuple, the function raises TypeError with message "Can't perform stack over object type `<type>`". There are no in-place side effects on the input objects; the function constructs and returns new containers and tensors.
    """
    from torchdrug.utils.torch import stack
    return stack(objs, *args, **kwargs)


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
