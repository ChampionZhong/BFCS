"""
Regenerated Google-style docstrings for module 'cebra'.
README source: others/readme/cebra/README.md
Generated at: 2025-12-02T00:30:01.703091Z

Total functions: 14
"""


import numpy
import torch

################################################################################
# Source: cebra.data.assets.calculate_checksum
# File: cebra/data/assets.py
# Category: valid
################################################################################

def cebra_data_assets_calculate_checksum(file_path: str):
    """cebra.data.assets.calculate_checksum calculates the MD5 checksum (hexadecimal string) of a file's raw bytes for use in CEBRA's data and asset management workflows, enabling integrity checks of dataset files, model weight files, and other binary assets used in neuroscience and behavioral-data experiments.
    
    This function reads the specified file in binary mode using a fixed 4096-byte chunk size and incrementally updates an MD5 digest. The incremental, chunked read is memory-efficient and suitable for large files commonly encountered in CEBRA pipelines (for example, large recordings, preprocessed datasets, or serialized model checkpoints). The returned value is a deterministic lowercase hexadecimal string representation of the MD5 digest for the exact file contents.
    
    Args:
        file_path (str): The filesystem path to the file whose MD5 checksum will be computed. This must be a path to a regular file that can be opened for reading in binary mode. In the context of the CEBRA library, typical uses include verifying downloaded datasets, cached preprocessed data, or exported model weights to ensure reproducibility and detect accidental corruption or modification.
    
    Returns:
        str: A lowercase hexadecimal MD5 checksum string computed from the file's bytes (the value of checksum.hexdigest()). This string uniquely identifies the file contents under the MD5 algorithm: identical byte sequences produce identical checksums, and any change to the file bytes will produce a different checksum.
    
    Behavior, side effects, defaults, and failure modes:
        The function opens the file at file_path with open(..., "rb") and reads it in 4096-byte chunks using iter(lambda: file.read(4096), b"") to avoid loading the entire file into memory. There are no persistent side effects: the file is only read (not modified) and is closed automatically when the context manager exits. The function uses the MD5 hashing algorithm from the hashlib module; while MD5 is suitable for quick integrity checks in data management and experiment reproducibility workflows in CEBRA, it is not recommended for cryptographic security purposes. If file_path does not refer to an accessible file or the process lacks permissions, the function will propagate built-in I/O exceptions such as FileNotFoundError, PermissionError, or OSError raised by open/read operations. Ensure the calling code handles or documents these exceptions where appropriate.
    """
    from cebra.data.assets import calculate_checksum
    return calculate_checksum(file_path)


################################################################################
# Source: cebra.data.assets.download_file_with_progress_bar
# File: cebra/data/assets.py
# Category: valid
################################################################################

def cebra_data_assets_download_file_with_progress_bar(
    url: str,
    expected_checksum: str,
    location: str,
    file_name: str,
    retry_count: int = 0
):
    """Download a file from a given URL with a tqdm progress bar and MD5 checksum verification.
    
    This function is used by the CEBRA library to fetch remote dataset assets (for example, neuroscience or behavioral data files referenced in the README) and to ensure file integrity for reproducible analyses. It first checks for an existing local file at the provided location and file_name and verifies its MD5 checksum using calculate_checksum. If the local file is absent or has a mismatched checksum, the function issues a streamed HTTP GET request to url, expects the server to provide a Content-Disposition header containing the filename, writes the response to disk in chunks (1024 bytes) while updating a tqdm progress bar (units in bytes), computes an MD5 checksum on the fly, compares it against expected_checksum, and either returns the downloaded file path on success or deletes the corrupt file and retries the download up to a configured global _MAX_RETRY_COUNT. The function has important side effects: it creates the destination directory (including parents), writes the downloaded file, may delete files that fail checksum verification, and emits warnings and printed messages for progress and failures. The checksum algorithm used is MD5 (hashlib.md5), so expected_checksum must be the hex MD5 digest corresponding to the file content.
    
    Args:
        url (str): The URL pointing to the remote file to download. In the CEBRA context this typically references a hosted dataset or asset necessary for training or analysis.
        expected_checksum (str): The expected MD5 checksum (hexadecimal string) of the target file. This is used to verify data integrity after download; a mismatch triggers deletion of the downloaded file and a retry sequence.
        location (str): The local directory path where the file will be saved. The function will create this directory and any missing parent directories as a side effect if they do not exist.
        file_name (str): The filename used to check for an existing local copy before downloading. If an existing file at location/file_name has a matching checksum, the function returns immediately. Note: when downloading, the implementation prefers the filename extracted from the server's Content-Disposition header; the file_name parameter is used only for the pre-download existence/checksum check.
        retry_count (int): The current retry attempt count; defaults to 0. This integer is incremented internally on checksum failures and used to determine whether the function has exceeded the global _MAX_RETRY_COUNT. The default value (0) indicates the initial attempt with no prior retries.
    
    Returns:
        Optional[str]: On success, returns a string representing the path to the downloaded (and checksum-verified) file. If the function determines a valid local file already exists it returns that file path immediately. If the download process fails or raises an exception (for example after exceeding allowed retries), the function may raise an exception instead of returning; callers should treat the return value as optional and be prepared to handle raised exceptions.
    
    Raises:
        RuntimeError: If retry_count is already greater than or equal to the global _MAX_RETRY_COUNT before attempting a new download, indicating the function will not attempt further retries.
        requests.HTTPError: If the HTTP GET request returns a non-200 status code; the exception message includes the response code.
        ValueError: If the HTTP response does not contain a Content-Disposition header or if a filename cannot be extracted from that header; the function requires the header to determine the server-provided filename used during download.
        OSError / IOError: If file system operations (directory creation, file write, file deletion) fail due to permissions or other I/O errors, the underlying exception may be propagated.
    
    Behavioral notes and failure modes:
        The function streams the response in 1024-byte chunks and updates an MD5 checksum incrementally; expected_checksum must correspond to this MD5 digest. If the computed checksum does not match expected_checksum, the partially downloaded file is removed (file_path.unlink()), warnings are emitted, and the function retries by recursively invoking itself with an incremented retry_count up to _MAX_RETRY_COUNT. If a valid local file already exists and its checksum matches, no network request is made and the local path is returned. Because the implementation relies on the server-provided Content-Disposition filename for the final saved filename, callers should ensure the server supplies this header; otherwise a ValueError is raised. The function prints a completion message on success and uses tqdm to present a live progress bar while downloading large dataset files.
    """
    from cebra.data.assets import download_file_with_progress_bar
    return download_file_with_progress_bar(
        url,
        expected_checksum,
        location,
        file_name,
        retry_count
    )


################################################################################
# Source: cebra.data.load.read_hdf
# File: cebra/data/load.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def cebra_data_load_read_hdf(filename: str, key: str = None):
    """Read an HDF5 file and return its contents as a pandas.DataFrame for use by the CEBRA library. This function is intended to load tabular datasets commonly used in CEBRA workflows (for example, neural recordings, behavioral annotations, or other time-series and metadata stored in HDF5 tables). It attempts to read the requested key using pandas.read_hdf and, if that fails, falls back to reading with h5py and converting the stored dataset into a pandas.DataFrame. The loaded DataFrame is what downstream CEBRA components (embedding training, decoding, preprocessing) expect as input.
    
    Args:
        filename (str): Path to the HDF5 file on disk. This is the filesystem location of a .h5/.hdf5 file containing one or more stored datasets or tables. The function will open and read this file; for large files this may require substantial memory because the entire requested table is loaded into a pandas.DataFrame in memory.
        key (str): Optional key (dataset/table name) to read from the HDF5 file. If None, the function first tries the literal key "df_with_missing" and, if that key does not exist, falls back to using the first available key found in the file. When a specific key is provided, the function attempts to read only that key. The key corresponds to the internal HDF5 path/name under which the table was saved (for example, a pandas.DataFrame saved with pandas.DataFrame.to_hdf).
    
    Returns:
        pandas.DataFrame: The loaded tabular data from the HDF5 file, returned as a pandas.DataFrame. This DataFrame represents the table stored at the requested key and is suitable for immediate use in CEBRA data pipelines (for example, as input to preprocessing, embedding, or supervised decoding routines). Column names, dtypes, and index are preserved where possible; if the fallback h5py path is used, the resulting DataFrame will be constructed from the raw datasets and may require additional dtype or index handling.
    
    Raises:
        RuntimeError: If reading with pandas fails and the fallback using h5py also fails, a RuntimeError is raised to indicate that the file could not be loaded. Failure modes include missing files, corrupted HDF5 files, incompatible data layouts, or insufficient permissions to read the file. The caller should catch this exception and handle retries, alternative keys, or user-facing error messages as appropriate.
    """
    from cebra.data.load import read_hdf
    return read_hdf(filename, key)


################################################################################
# Source: cebra.datasets.get_datapath
# File: cebra/datasets/__init__.py
# Category: valid
################################################################################

def cebra_datasets_get_datapath(path: str = None):
    """Convert a relative dataset path into the system-dependent absolute data path used by CEBRA.
    
    This function is used throughout the CEBRA library to compute filesystem locations of datasets and other data files that the library and its dataset-loading utilities expect to find under a common data root. The root directory is determined by the helper get_data_root(), which in turn can be configured by the environment variable CEBRA_DATADIR; therefore, this function enforces the convention that dataset paths are resolved relative to that root. When given a non-None argument, the function converts the provided path to a string and joins it with the data root using os.path.join, producing the path that CEBRA will use to open or list dataset files. This function does not read or validate filesystem contents; it only computes the pathname string that other CEBRA functions will use.
    
    Args:
        path (str or pathlib.Path): The dataset path to resolve. This argument represents a path fragment relative to the CEBRA system data directory (the directory returned by get_data_root()). If path is None, the function returns the data root itself. The function accepts pathlib.Path objects and will convert them to str; other types will be coerced with str() which may produce unintended results. The caller is responsible for ensuring the intended relativity (i.e., providing a relative path when a subpath of the data root is desired).
    
    Returns:
        str: The absolute filesystem path string that results from resolving the provided path against the CEBRA data root. If path is None, returns the data root path string returned by get_data_root(). Note that this function does not expand shell-style user shorthands (e.g., "~"), does not resolve symbolic links, and does not verify that the returned pathname exists or is accessible; callers should perform os.path.exists, os.access, os.path.abspath, or pathlib.Path.resolve() if normalization or existence checks are required. Also note: if the provided path string is absolute according to the operating system, os.path.join semantics will effectively return that absolute path (i.e., the data root will be ignored), which may be unintended when the caller expects strictly relative resolution.
    """
    from cebra.datasets import get_datapath
    return get_datapath(path)


################################################################################
# Source: cebra.datasets.set_datapath
# File: cebra/datasets/__init__.py
# Category: valid
################################################################################

def cebra_datasets_set_datapath(path: str = None, update_environ: bool = True):
    """cebra.datasets.set_datapath sets the global root data directory used by the cebra.datasets module and (by implementation) updates the process environment variable CEBRA_DATADIR so other parts of the library and external tools that read this environment variable will use the new directory when locating dataset files.
    
    This function is intended to change where CEBRA looks for and stores dataset files (common in neuroscience and biology workflows described in the project README). It performs validation on the supplied path and has observable side effects at the module and process level: it assigns the module-level __DATADIR variable and writes to os.environ["CEBRA_DATADIR"].
    
    Args:
        path (str): The filesystem path to set as the new root data directory. This argument must be a path-like string pointing to an existing directory. The function validates this path: if the path does not exist, a FileNotFoundError is raised; if the path exists but is a file rather than a directory, a FileExistsError is raised. Note that although the function signature allows None as the default, passing None (or any non-path-like type) will result in a TypeError from the underlying os.path checks; therefore callers should supply a valid directory path string when invoking this function.
        update_environ (bool): A flag intended to control whether the environment variable CEBRA_DATADIR is updated with the new path. The default value is True. Important implementation note: the current implementation ignores this flag and always updates the CEBRA_DATADIR environment variable after validating the path; callers should not rely on this parameter to prevent environment modification until the implementation is changed.
    
    Returns:
        None: This function does not return a value. Instead, it has side effects: it sets the module-level __DATADIR variable to the validated path and sets the process environment variable CEBRA_DATADIR to that same value. These side effects are how other cebra.datasets functions discover the configured data directory. Exceptions raised on invalid input (FileNotFoundError, FileExistsError, or TypeError for non-path-like inputs) prevent these side effects from occurring.
    """
    from cebra.datasets import set_datapath
    return set_datapath(path, update_environ)


################################################################################
# Source: cebra.datasets.make_neuropixel.read_neuropixel
# File: cebra/datasets/make_neuropixel.py
# Category: valid
################################################################################

def cebra_datasets_make_neuropixel_read_neuropixel(
    path: str = "/shared/neuropixel/*/*.nwb",
    cortex: str = "VISp",
    sampling_rate: float = 120.0
):
    """Load Neuropixels recordings for the "movie1" stimulus, filter units by recording area and quality, and convert spike times into binned spike-count matrices per session. This function is used in the CEBRA library to prepare Neuropixels neural data (originally recorded at high temporal resolution) into time-binned spike-count representations aligned to movie frames for downstream embedding, decoding, and joint behavioral/neural analysis. It searches for .nwb files matching a glob path, reads required datasets using h5py, applies area and quality filters via internal helper functions (_spikes_by_units, _filter_units, _get_area, _get_movie1, _spike_counts), and constructs a dictionary of per-session spike-count matrices together with per-timepoint movie-frame indices.
    
    Args:
        path (str): The wildcard file path where the neuropixels .nwb files are located. Practical role: a glob pattern (for example "/shared/neuropixel/*/*.nwb") that the function passes to glob.glob to discover Neurodata Without Borders (NWB) files to read. Behavior and side effects: files matching this pattern will be opened with h5py.File and read; if no files match, the function will return empty containers. Failure modes: an unreadable or corrupt file will raise an h5py/OSError; an unexpected NWB structure (missing keys expected by the code) will raise KeyError/IndexError.
        cortex (str): The cortex where the neurons were recorded. Choose from VISp, VISal, VISrl, VISl, VISpm, VISam. Practical role: this string is compared to unit area labels (computed from electrode/peak-channel metadata) to keep only units from the specified visual cortical area. Behavior and side effects: only units with area equal to this value and that pass quality filters are kept for each session. If an unknown value is provided, the function will simply filter out all units that do not match, possibly yielding empty spike matrices.
        sampling_rate (float): The sampling rate for spike counts to process the raw data. Practical role: defines the temporal bin width used to convert spike times into integer spike counts via bin edges computed with step 1/sampling_rate. Default is 120.0, which corresponds to 120 bins per second (8.333... ms per bin) and—given the code’s bin-to-frame mapping—typically yields 4 bins per movie frame when movie frame rate is 30 Hz. Behavior and side effects: sampling_rate is used to compute bin edges (np.arange(start_time[0], end_time[8999], 1 / sampling_rate)) and also to crop each session to sampling_rate * 10 * 30 time bins (the code performs sessions[session_key] = sessions[session_key][:sampling_rate * 10 * 30]). Failure modes and important implementation notes: because the code slices arrays with sampling_rate * 10 * 30, a non-integer-valued sampling_rate (or a float not representing an integer multiple) can produce a float slice index and raise a TypeError; callers should ensure sampling_rate is a value that yields an integer slice length (e.g., an integer-valued float like 120.0) or modify the value before calling. The function relies on internal helper functions (_spikes_by_units, _filter_units, _get_area, _get_movie1, _spike_counts) and on the NWB file containing specific datasets (for example, intervals/natural_movie_one_presentations and units/* datasets); missing datasets cause KeyError/IndexError.
    
    Returns:
        sessions (dict): A dictionary mapping session identifiers (the NWB file identifier strings read from d["identifier"][...]) to numpy.ndarray spike-count matrices for that session. Each value is a 2D numpy array produced by _spike_counts where rows index time bins (bins defined by 1 / sampling_rate) and columns index filtered units from the specified cortex. Practical significance: these matrices are the time-binned neural activity used by CEBRA for embedding and decoding analyses. Note that the code crops each session to the first sampling_rate * 10 * 30 time bins (intended to build a "pseudomouse" of fixed duration); if sampling_rate * 10 * 30 is not an integer this cropping will raise an error.
        session_frames (list): A list of numpy.ndarray objects, one per processed session, giving the movie-frame index (integer) assigned to each time bin. Values are computed by digitizing bin edges against the movie start times and then taking modulo 900 (frame indices 0..899). Practical significance: this array provides alignment between each neural time bin and the corresponding movie frame for behavioral/neural alignment and joint analysis. Implementation note: after initial computation, the function replaces session_frames entries with a tiled pattern np.tile(np.repeat(np.arange(900), 4), 10) (length 36000 for the default sampling_rate of 120.0). This behavior can produce a fixed-length per-session frame index vector regardless of sampling_rate and may lead to length mismatches between session spike matrices and session_frames for non-default sampling_rate values; callers should verify that session_frames entries align in length with the corresponding sessions[session_id] arrays.
    
    Side effects and additional notes:
        - The function prints progress messages ("read one session and filter for area and quality" and "Build pseudomouse") to standard output for each file processed and when building the pseudomouse.
        - The function opens NWB files with h5py.File in read mode; file handles are closed automatically by the context manager but reading many large files can be I/O- and memory-intensive.
        - The function assumes the NWB files contain at least 9000 movie presentation timestamps (the code accesses start_time[:9000] and end_time[8999]); if this assumption does not hold, an IndexError will be raised.
        - Internal helper functions (_spikes_by_units, _filter_units, _get_area, _get_movie1, _spike_counts) must be available in the same module; their behavior (unit grouping, quality filtering, area mapping, per-unit spike extraction, and binning) directly affects the returned matrices and frame indices.
        - If no files match the provided path pattern the returned sessions dict and session_frames list will be empty.
    """
    from cebra.datasets.make_neuropixel import read_neuropixel
    return read_neuropixel(path, cortex, sampling_rate)


################################################################################
# Source: cebra.datasets.poisson.sample_parallel
# File: cebra/datasets/poisson.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def cebra_datasets_poisson_sample_parallel(
    spike_rates: torch.Tensor,
    refractory_period: float = 0.0,
    n_jobs: int = 10
):
    """cebra.datasets.poisson.sample_parallel: Generate spike counts from input spike rates using parallelized sampling across neurons.
    
    Args:
        spike_rates (torch.Tensor): The non-negative spike rates to sample from, provided as a 3-D tensor with shape neurons x trials x time. Each element represents the expected spike rate (per time bin) for a particular neuron, trial, and timepoint. This argument is required for the function to produce stochastic spike counts; negative values are not allowed and will cause the function to raise a ValueError. The number of neurons (the size of the first dimension) must be divisible by n_jobs so the tensor can be reshaped into n_jobs equal neuron batches for parallel processing.
        refractory_period (float): The minimum time (in the same time units as the input spike rates) enforced between successive spikes for the same neuron during sampling. A value of 0.0 (the default) disables any refractory constraint and allows the sampling procedure to place spikes in adjacent time bins according to the independent-rate process implemented by the underlying sampler. Providing a positive value instructs the internal batch sampler (_sample_batch) to respect that refractory interval when generating counts; if the sampler or its assumptions do not support the requested refractory behavior, sampling results may differ from a pure Poisson process.
        n_jobs (int): The number of parallel jobs to use for generating spike trains (default 10). This value is passed to joblib.Parallel and controls how many neuron-subsets are sampled concurrently. The first dimension of spike_rates (neurons) must be divisible by n_jobs; if it is not, torch.view will fail when reshaping the tensor and an exception will be raised. Using multiple jobs speeds up sampling on multi-core systems but can increase memory usage and introduce non-determinism unless random seeds are controlled externally.
    
    Returns:
        torch.Tensor: A tensor of shape neurons x trials x time containing integer spike counts sampled from the provided spike_rates. The returned tensor preserves the original neuron ordering: spike_rates is split into n_jobs contiguous neuron batches, each batch is sampled in parallel, and the resulting batch outputs are concatenated along the neuron (first) dimension. Sampling is stochastic and depends on the process-global random state and any RNGs used inside the internal sampler; results are not deterministic across runs unless RNG seeds are fixed. Possible failure modes include ValueError when any rate is negative and errors from tensor reshaping (e.g., RuntimeError from torch.view) when the number of neurons is not divisible by n_jobs.
    """
    from cebra.datasets.poisson import sample_parallel
    return sample_parallel(spike_rates, refractory_period, n_jobs)


################################################################################
# Source: cebra.datasets.save_dataset.save_allen_dataset
# File: cebra/datasets/save_dataset.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def cebra_datasets_save_dataset_save_allen_dataset(savepath: str = "data/allen_preload/"):
    """cebra.datasets.save_dataset.save_allen_dataset loads and saves a precomputed subset of the Allen Institute dataset used for calcium (Ca) imaging experiments in the CEBRA framework to reduce data-loading time for downstream decoding and embedding tasks.
    
    This function iterates over a fixed set of neuron sample sizes and random seeds to initialize dataset objects via cebra.datasets.init and persist their neural recordings to disk. It is intended as a data-preloading utility for experiments that use the shared Allen decoding data in CEBRA (neural + behavioral analysis and self-supervised embedding workflows). The saved files contain only the neural data under the "neural" key and use a filename pattern derived from the dataset name, allowing experiments to quickly load pretrained subsets rather than querying the full remote/raw source repeatedly.
    
    Args:
        savepath (str): The directory path where the function will write the serialized dataset files. Each saved file is written as "{savepath}/allen-movie1-neuropixel-{n}-{seed}.jl" and contains a single dictionary with key "neural" whose value is the neural data extracted from the initialized dataset object. This argument should be a filesystem path string pointing to an existing writable directory; the function does not create intermediate directories. Typical usage is to point this to a project data folder (for example "data/allen_preload/" in the repository), which enables experiments to load these pre-saved subsets to reduce repeated network or processing overhead.
    
    Behavior and side effects:
        The function uses a fixed enumeration of neuron counts and seeds to generate dataset names and save files. Concretely, it loops over neuron counts [10, 30, 50, 100, 200, 400, 600, 800, 900, 1000] and seed values [111, 222, 333, 444, 555], and for each combination constructs the dataset name "allen-movie1-neuropixel-{n}-{seed}". For each constructed name it calls cebra.datasets.init(dataname) to load/initialize the dataset (this typically loads neural and behavioral data associated with that dataset identifier) and then serializes only the neural component (data.neural) to a file using jl.dump with the filename pattern described above. The function prints progress messages to standard output indicating which dataset name has been initiated and the path of the file written. This pre-saving reduces later experiment runtime overhead by avoiding repeated initialization and preprocessing of the same dataset subsets.
    
    Defaults and practical significance:
        By default callers typically set savepath to a project-local directory such as "data/allen_preload/". The saved files are intended for use by CEBRA experiments that perform decoding or representation learning on neural (Ca) data; having these pre-saved subsets allows rapid repeated experiments across the enumerated neuron counts and random seeds.
    
    Failure modes and errors:
        The function will propagate exceptions raised by cebra.datasets.init if the requested dataset identifier is not available or cannot be initialized (for example due to missing source files or network access failures). It will also raise filesystem-related exceptions if savepath does not exist or is not writable, or if jl.dump fails for any reason (permission errors, disk full, serialization error). Because the function does not create directories, callers must ensure savepath exists and is writable before calling. Partial side effects may occur if an error is raised mid-loop: files saved before the error will remain on disk.
    
    Returns:
        None: This function does not return a value. Its purpose is to produce side effects on disk (creation of "{savepath}/allen-movie1-neuropixel-{n}-{seed}.jl" files containing the neural data under the "neural" key) and to print progress messages to standard output.
    """
    from cebra.datasets.save_dataset import save_allen_dataset
    return save_allen_dataset(savepath)


################################################################################
# Source: cebra.datasets.save_dataset.save_allen_decoding_dataset
# File: cebra/datasets/save_dataset.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def cebra_datasets_save_dataset_save_allen_decoding_dataset(
    savepath: str = "data/allen_preload/"
):
    """Save and persist preprocessed Allen Institute "decoding" calcium-imaging datasets for use with CEBRA experiments.
    
    This function iterates over a predefined set of neuron subsample sizes, random seeds for neuron sampling, and train/test splits and uses cebra.datasets.init to load the corresponding Allen decoding dataset for calcium imaging ("ca"). For each combination it serializes and writes the loaded dataset's neural recording array to disk using joblib (jl.dump) as a dictionary with key "neural". The files are named using the pattern "allen-movie1-ca-decoding-{n}-{split_flag}-{seed}.jl" and are written under the provided savepath directory. This pre-saving of dataset variants is intended to reduce data loading time when running downstream decoding or embedding experiments with the CEBRA library (a toolkit for learning consistent latent embeddings of neural and behavioral recordings, as described in the project README).
    
    Args:
        savepath (str): The directory path where the serialized dataset files will be written. The function uses this exact path prefix when constructing filenames of the form "{savepath}/allen-movie1-ca-decoding-{n}-{split_flag}-{seed}.jl". The default in the original implementation points to a local "allen_preload" data directory; callers should ensure the path exists and is writable. If the directory does not exist or is not writable, joblib.dump (jl.dump) will raise an exception.
    
    Behavior and side effects:
        The function loops over neuron counts n in [10, 30, 50, 100, 200, 400, 600, 800, 900, 1000], random seeds in [111, 222, 333, 444, 555], and split flags "train" and "test", and modality "ca" (calcium imaging) only. For each combination it constructs the dataset name allen-movie1-ca-decoding-{n}-{split_flag}-{seed}, calls cebra.datasets.init(dataname) to fetch the dataset object, and then writes jl.dump({"neural": data.neural}, f"{savepath}/{dataname}.jl"). The function prints simple status messages to stdout indicating which dataset was initiated and which file was written. Only the returned object's neural attribute is serialized and saved; other attributes (for example behavioral variables) are not written by this function. The primary practical significance is to produce a local cache of neural subsets used in the Allen decoding experiments so downstream CEBRA training/decoding runs can load pre-saved joblib files instead of reinitializing and subsampling repeatedly.
    
    Failure modes and errors:
        If cebra.datasets.init raises (for example because the named dataset is not available in the local installation or remote resource), this function will propagate that exception. If jl.dump fails (for example because savepath does not exist, insufficient filesystem permissions, or insufficient disk space), jl.dump will raise an IOError/OSError or joblib-specific error. The function does not create missing directories; callers must create savepath beforehand if needed.
    
    Returns:
        None: This function does not return a value. Its effect is entirely side-effecting: it writes multiple .jl files to disk under savepath (one per neuron-count/seed/split combination) containing a dictionary {"neural": data.neural} serialized with joblib, and it writes status messages to stdout.
    """
    from cebra.datasets.save_dataset import save_allen_decoding_dataset
    return save_allen_decoding_dataset(savepath)


################################################################################
# Source: cebra.datasets.save_dataset.save_monkey_dataset
# File: cebra/datasets/save_dataset.py
# Category: valid
################################################################################

def cebra_datasets_save_dataset_save_monkey_dataset(
    savepath: str = "data/monkey_reaching_preload_smth_40/"
):
    """cebra.datasets.save_dataset.save_monkey_dataset
    Load and save the monkey reaching dataset to disk for use in CEBRA experiments that jointly analyze neural and behavioral data.
    
    This function iterates over all supported session types ("active", "passive", "all") and data splits ("all", "train", "valid", "test"), calls monkey_reaching._load_data(session=session, split=split) to load the neural and behavioral data for each combination, and serializes each dataset to a file named "{session}_{split}.jl" under the provided save directory. Its primary practical purpose in the CEBRA project is to reduce data-loading time for downstream self-supervised embedding experiments by materializing shared preprocessed data to disk so multiple experiments can reuse the saved files instead of repeatedly reloading and preprocessing raw recordings.
    
    Args:
        savepath (str): The directory path where the function will write serialized dataset files. Each saved file is named using the pattern "<session>_<split>.jl" (for example, "active_train.jl") and is written to os.path.join(savepath, dataname). By default this parameter points to the repository's example data directory (used in the source code as get_datapath("monkey_reaching_preload_smth_40/"), and in some interfaces provided as 'data/monkey_reaching_preload_smth_40/'); the caller should provide an existing, writable directory if a different location is required. The function does not create intermediate directories: if savepath does not exist or is not writable, the function will raise an OSError/FileNotFoundError/PermissionError from the underlying file operations. savepath must be a filesystem path string.
    
    Returns:
        None: This function does not return a value. Instead, it has the side effect of writing up to twelve serialized dataset files (one per session/split combination) into the provided savepath and printing the full path of each file as it is written. Failures can occur during data loading (for example if monkey_reaching._load_data is unavailable, raises an AttributeError, or fails for a given session/split) or during serialization/IO (for example jl.dump raising an exception, or filesystem permission/space errors). Callers should ensure that the monkey_reaching dataset loader is available in the runtime environment and that savepath is correct and writable before invoking this function to avoid partial saves or exceptions.
    """
    from cebra.datasets.save_dataset import save_monkey_dataset
    return save_monkey_dataset(savepath)


################################################################################
# Source: cebra.helper.download_file_from_url
# File: cebra/helper.py
# Category: valid
################################################################################

def cebra_helper_download_file_from_url(url: str):
    """Download a file from a remote URL and save it to a local temporary path with an ".h5" suffix, returning the path to the saved file. This helper is used in the CEBRA library to fetch remote dataset files and model/data artifacts (for example, HDF5 recordings of neural or behavioral data) so they can be opened by downstream data loaders or analysis routines.
    
    Args:
        url (str): Url to fetch for the file. This should be a full URL (for example, an HTTP(S) address) that points to a file resource that can be downloaded for use in CEBRA workflows (commonly an HDF5 file containing recordings or precomputed embeddings). The function does not validate the semantic content of the file; it only performs an HTTP GET and streams the bytes to disk.
    
    Returns:
        str: The path to the downloaded file on the local filesystem. The path is constructed from a temporary filename obtained from tempfile.NamedTemporaryFile() with ".h5" appended, and the function writes the streamed response bytes to that path. The file is created in the system temporary directory (the same directory used by tempfile.NamedTemporaryFile()) and is not automatically removed by this function; callers are responsible for deleting the file when it is no longer needed.
    
    Behavior and side effects:
        This function performs network I/O and local disk I/O. It uses requests.get(..., stream=True) and iterates over response.iter_content(chunk_size=8192) to write the response in 8192-byte chunks, which keeps memory usage low for large files. The function opens the target file for binary write ("wb") and writes each chunk sequentially.
    
    Failure modes and exceptions:
        If the HTTP response indicates an error status, r.raise_for_status() will raise requests.exceptions.HTTPError (a subclass of requests.exceptions.RequestException). Network-related failures, timeouts or connection errors will raise requests.exceptions.RequestException. File system errors (for example, lack of write permissions or insufficient disk space) will raise built-in OSError exceptions when attempting to open or write the file. The function does not retry failed requests and does not perform content validation beyond streaming bytes to disk.
    
    Practical significance in the CEBRA domain:
        In CEBRA pipelines and demos, this helper is a convenient utility to download remote HDF5 dataset files or other artifacts referenced by URLs so that they can be loaded into analysis notebooks, dataset classes, or model evaluation routines. Because the helper returns a concrete filesystem path, callers can pass the returned string directly to file-opening utilities used across the CEBRA codebase.
    """
    from cebra.helper import download_file_from_url
    return download_file_from_url(url)


################################################################################
# Source: cebra.helper.download_file_from_zip_url
# File: cebra/helper.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def cebra_helper_download_file_from_zip_url(url: str, file: str):
    """Directly download a ZIP archive from a remote URL, extract its members into a temporary directory, and return a pathlib.Path pointing to the requested file inside the archive's top-level "data" folder. This helper is used by CEBRA to fetch remote dataset archives (for example, neural or behavioral recording archives used in demos and data loaders) so higher-level code can access a particular file inside the archive without manually saving the ZIP to disk. Note: the current implementation creates a TemporaryDirectory, reads the entire archive into memory, extracts members with zipfile.ZipFile, and returns a path constructed as <temporary-directory>/data/<file>.
    
    Args:
        url (str): HTTP(S) URL of the remote ZIP archive to download. In the CEBRA workflow this should point to an archive that contains dataset files (for example, neural or behavioral recordings and metadata). The function will open the URL using urllib.request.urlopen and read the entire response into memory via resp.read(), so the caller should ensure the URL is reachable and the archive size is acceptable for available memory.
        file (str): Relative filename (string) expected to exist under the archive's top-level "data" directory. In CEBRA this corresponds to a specific dataset file or artifact inside the ZIP (for example a CSV, npy, or HDF5 file used by downstream dataset loaders). The function returns a pathlib.Path to "<temporary-directory>/data/<file>" constructed from the extracted contents.
    
    Returns:
        pathlib.Path: A pathlib.Path pointing to the requested file inside the extracted archive at "<temporary-directory>/data/<file>". Practical significance: callers in the CEBRA codebase use this Path to open or load the dataset file into processing pipelines. Side effects: the function downloads the full archive into memory (potentially large memory usage), extracts archive members to a temporary directory on disk, and may write files with the archive-stored names and permissions. Failure modes and important implementation notes: network errors (urllib.error.URLError, HTTPError) can occur when opening the URL; invalid archives will raise zipfile.BadZipFile or related zipfile.error; individual extraction errors are caught and suppressed during extraction, so missing or failed-to-extract members will not raise but may result in the returned Path not existing. The current implementation also scopes the TemporaryDirectory such that the temporary directory may be removed before the caller can use the returned Path; therefore callers should validate existence and, if persistent access is required, copy the target file to a stable location. Additionally, because extraction uses zipfile.ZipFile.extract, archive contents that contain path traversal entries can write outside the target directory (a ZIP slip risk); callers should only use archives from trusted sources.
    """
    from cebra.helper import download_file_from_zip_url
    return download_file_from_zip_url(url, file)


################################################################################
# Source: cebra.integrations.sklearn.utils.check_device
# File: cebra/integrations/sklearn/utils.py
# Category: valid
################################################################################

def cebra_integrations_sklearn_utils_check_device(device: str):
    """Select and normalize a PyTorch compute device string based on the requested device and the execution environment.
    
    This utility is used by CEBRA's scikit-learn integration to choose a compute device for PyTorch models, tensors, and training/inference workflows involved in self-supervised embedding estimation. It interprets the caller-provided device hint (a string) and returns a canonical device string that callers can pass to torch APIs or CEBRA components. The function checks CUDA availability, per-GPU indices, Apple Metal Performance Shaders (MPS) availability via cebra.helper._is_mps_availabe(torch) and torch.backends.mps, and otherwise falls back to CPU. The returned string should be treated as authoritative for device placement decisions in downstream CEBRA training, evaluation, and embedding computation.
    
    Args:
        device (str): Device selection hint provided by the caller. Accepted values and their meanings are:
            - "cuda_if_available": request that CUDA be used if available; if CUDA is not available, MPS will be used if detected via cebra.helper._is_mps_availabe(torch) or torch.backends.mps is available, otherwise CPU is returned. This is a convenience for code that prefers GPU acceleration but must run on machines without CUDA.
            - "cuda": request a CUDA GPU; if CUDA is available the function normalizes this to "cuda:0" (the first GPU). Use this when you want GPU acceleration and are willing to accept the default GPU index 0.
            - "cuda:N" where N is an integer (e.g., "cuda:1"): request a specific CUDA device index. The function validates that N is an integer and that 0 <= N < torch.cuda.device_count(); if valid it returns the same "cuda:N" string. If N is not an integer or the index is out of range a ValueError is raised (see failure modes).
            - "mps": request Apple MPS (Metal) device; the function verifies torch.backends.mps.is_available() and torch.backends.mps.is_built() and returns "mps" if available; otherwise it raises a ValueError with an explanation. Use this on supported macOS machines that provide MPS acceleration for PyTorch.
            - "cpu": explicitly request CPU; the function returns "cpu".
            The device argument must be a str and follow one of the allowed formats above. This string determines where CEBRA places tensors and model parameters for training and inference and therefore affects performance and hardware utilization.
    
    Returns:
        str: A canonical device string that should be used for PyTorch device placement in CEBRA components. The returned value will be one of the following concrete strings depending on the input and environment: "cuda" (only returned when explicitly detected via "cuda_if_available" and CUDA is available, though "cuda" input is normalized to "cuda:0"), "cuda:N" (where N is a validated integer GPU index), "mps", or "cpu". Callers should pass this returned string to torch.device or to CEBRA APIs that accept device strings for deterministic placement.
    
    Failure modes and side effects:
        - If device is "cuda:N" and N is not an integer, a ValueError is raised indicating invalid CUDA device ID format and instructing to use 'cuda:device_id' where the suffix is an integer.
        - If device is "cuda:N" and N is an integer but N >= torch.cuda.device_count(), a ValueError is raised indicating the requested CUDA device ID is not available and reporting the valid range 0 to device_count - 1.
        - If device is "mps" but torch.backends.mps.is_available() is False, a ValueError is raised. If torch.backends.mps.is_built() is False the error explains that the current PyTorch install was not built with MPS enabled; otherwise the error explains that the macOS version or hardware does not support MPS. These exceptions are raised so calling code can fail fast and provide informative diagnostics to the user.
        - If device is not one of the accepted forms ("cuda_if_available", "cuda", "cuda:N", "mps", "cpu"), a ValueError is raised informing the caller that the device must be cuda, cpu or mps and echoing the provided value.
        - The function does not mutate global state; it only queries torch and cebra.helper to determine availability. Its only side effect is raising exceptions on invalid inputs or unavailable requested devices.
        - Behavior depends on the runtime PyTorch installation and the results of torch.cuda.is_available(), torch.cuda.device_count(), cebra.helper._is_mps_availabe(torch), and torch.backends.mps.* checks; therefore identical inputs can produce different outputs on different machines or with different PyTorch builds.
    """
    from cebra.integrations.sklearn.utils import check_device
    return check_device(device)



################################################################################
# Source: cebra.io.reduce
# File: cebra/io.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def cebra_io_reduce(data: numpy.ndarray, ratio: float = None, num_components: int = None):
    """cebra.io.reduce maps high-dimensional sample recordings (for example neural or behavioral trial windows used in the CEBRA pipeline) to a lower-dimensional representation using principal component analysis (PCA). It fits sklearn.decomposition.PCA to the provided samples and returns the PCA-transformed data; the output dimensionality is determined either by an explained-variance threshold (ratio) or by an explicit number of principal components (num_components). In the CEBRA workflow this function is typically used as a preprocessing dimensionality reduction step to obtain a compact input representation for downstream embedding, decoding, or visualization of neural and behavioral recordings.
    
    The function reshapes the input data to (N, F) using data.reshape(len(data), -1) so that each of the len(data) entries is treated as one sample and all remaining axes are flattened into features. PCA is constructed by passing num_components to sklearn.decomposition.PCA; if num_components is None, sklearn's PCA default behavior for n_components applies. The function validates inputs and can raise ValueError for missing or out-of-range parameters; when ratio is provided the function selects components based on the cumulative explained variance reported by PCA and returns the corresponding slice of the transformed data (see failure modes below for edge cases).
    
    Args:
        data (numpy.ndarray): The input recordings array. Each element along the first axis is interpreted as one sample (N samples). The function internally reshapes this array to shape (N, F) via data.reshape(len(data), -1), flattening any remaining dimensions into features F. In the CEBRA domain this typically represents e.g. time-series windows or feature vectors per sample derived from neural/behavioral data.
        ratio (float, optional): The fraction of explained variance required from the returned principal components. Must satisfy 0 < ratio <= 1. If provided, the function computes PCA.explained_variance_ratio_ (from sklearn) and finds the smallest index i where the cumulative explained variance exceeds ratio; it then returns the PCA-transformed data sliced up to (but excluding) that index. Note that the implementation uses the strict '>' comparison and Python slicing semantics, so the number of returned components may be less than or equal to the number needed to reach the requested variance threshold in some edge cases (for example, if cumulative sums equal ratio exactly or if ratio == 1.0). If ratio is None, no explained-variance-based truncation is applied and the full set of components produced by sklearn.decomposition.PCA (as configured by num_components or sklearn defaults) is returned.
        num_components (int, optional): The number of principal components passed directly to sklearn.decomposition.PCA as its n_components argument. If provided, PCA will be initialized to compute at most this many components. If num_components is None, sklearn.decomposition.PCA's default for n_components is used. This parameter controls the maximum dimensionality available in the PCA-transformed output before any further slicing due to ratio. Note that sklearn may raise its own errors if num_components is invalid for the input data (for example, larger than the number of features or samples).
    
    Returns:
        numpy.ndarray: A 2-D array of shape (N, d) containing the PCA-transformed samples. N equals len(data) (the number of input samples after reshaping). d is the returned dimensionality: if ratio is None, d equals the number of components computed by sklearn.decomposition.PCA (controlled by num_components or sklearn's default); if ratio is provided, d equals the number of columns retained by the post-transform slice described above (the implementation selects the first index i where cumulative explained variance > ratio and returns columns up to, but excluding, that i). The function does not modify the input array in-place; it fits a new PCA object and returns its transformed copy.
    
    Failure modes and side effects:
        - If both ratio and num_components are None, the function raises ValueError("Specify either a threshold on the explained variance, or a maximumnumber of principle components").
        - If ratio is provided but is <= 0 or > 1, the function raises ValueError indicating the ratio must be in (0, 1].
        - If ratio is provided and the cumulative explained variance never strictly exceeds ratio (for example, ratio == 1.0 or due to exact equality), numpy indexing used in the implementation may raise an IndexError when attempting to select the component index; users should account for this when choosing ratio values.
        - sklearn.decomposition.PCA.fit(...) or transform(...) may raise exceptions (for example if num_components is invalid given the number of features or samples); these originate from scikit-learn and are not caught in this function.
        - Computational cost and memory use depend on N and the flattened feature dimension F; fitting PCA can be expensive for large F or large num_components.
    """
    from cebra.io import reduce
    return reduce(data, ratio, num_components)


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
