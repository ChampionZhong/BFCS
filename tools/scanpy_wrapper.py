"""
Regenerated Google-style docstrings for module 'scanpy'.
README source: others/readme/scanpy/README.md
Generated at: 2025-12-02T00:35:47.652265Z

Total functions: 21
"""


import numpy

################################################################################
# Source: scanpy._utils.identify_groups
# File: scanpy/_utils/__init__.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def scanpy__utils_identify_groups(
    ref_labels: numpy.ndarray,
    pred_labels: numpy.ndarray,
    return_overlaps: bool = False
):
    """Identify which predicted label explains which reference label.
    
    This function is used in single-cell analysis workflows (Scanpy) to map clustering or prediction labels (pred_labels) back to a set of reference labels (ref_labels) such as known cell types, annotation labels, or previously computed cluster assignments. For each unique reference label, the function computes the counts of cells shared with each predicted label, normalizes these counts both by the predicted-group size and the reference-group size, and selects predicted labels that best "explain" the reference group by maximizing the minimum of these two relative overlaps. This is useful when evaluating how well a predicted clustering or classification matches a biological or annotated reference grouping.
    
    Args:
        ref_labels (numpy.ndarray): 1D array of reference group labels for each observation (e.g., cell). Each element associates an observation with its reference annotation (for example, a curated cell type or ground-truth cluster). The function requires that ref_labels and pred_labels have the same length and correspond elementwise to the same set of observations; otherwise numpy indexing will raise an error. The array is used to compute unique reference groups and their sizes and to select the subset of predicted labels that fall into each reference group.
        pred_labels (numpy.ndarray): 1D array of predicted group labels for each observation (e.g., labels produced by clustering or a classifier). This array must align elementwise with ref_labels (same length). The function counts intersections between predicted groups and reference groups by indexing pred_labels with the boolean mask (ref_label == ref_labels).
        return_overlaps (bool): If False (default), the function returns only the mapping from each reference label to the predicted labels that best explain it. If True, the function additionally returns the normalized overlap values for each reference–predicted pair. The normalized overlaps are returned as two values per predicted label: overlap normalized by predicted-group size and overlap normalized by reference-group size, respectively.
    
    Behavior and practical details:
        For each unique reference label r, the function:
            1. Finds all predicted labels present among observations with reference label r and counts the number of observations in the intersection (overlap counts).
            2. Computes relative_overlaps_pred = overlap_count / size_of_predicted_group and relative_overlaps_ref = overlap_count / size_of_reference_group.
            3. For each predicted label present in r, forms the two-element vector [relative_overlaps_pred, relative_overlaps_ref] and then computes the elementwise minimum of these two values.
            4. Sorts predicted labels in decreasing order of that minimum value and places the sorted predicted labels in associated_predictions[r]. The associated_overlaps[r] entry gives the corresponding two-column array of [relative_overlaps_pred, relative_overlaps_ref] rows in the same order.
        The decision rule (maximizing the minimum of the two normalized overlaps) balances the fraction of the predicted group explained by the reference group and the fraction of the reference group explained by the predicted group, favoring predicted labels that represent strong mutual overlap.
        The dictionaries returned have one entry per unique reference label (i.e., length equals len(numpy.unique(ref_labels))). The keys are the unique values from ref_labels and the values are numpy arrays as described below.
        Tie-breaking when multiple predicted labels have identical minimum overlap values is determined by numpy.argsort and the underlying numpy sorting behavior (not guaranteed beyond what numpy documents); ties are not further disambiguated by this function.
        The function does not modify the input arrays (no side effects on ref_labels or pred_labels).
    
    Failure modes and errors:
        If ref_labels and pred_labels do not have the same length, numpy boolean indexing will raise an indexing error (ValueError or IndexError depending on numpy version). The caller must ensure elementwise correspondence of the two arrays. If ref_labels contains labels with zero occurrences this cannot happen because unique(..., return_counts=True) ensures only present labels are considered.
        The function assumes that numpy.unique returns unique labels and counts normally; unexpected non-hashable or non-comparable label types may raise appropriate numpy errors.
    
    Returns:
        dict or (dict, dict): If return_overlaps is False, returns a dictionary mapping each reference label to a numpy.ndarray of predicted labels sorted by decreasing explanatory power (the predicted labels that best explain the reference label appear first). If return_overlaps is True, returns a tuple (associated_predictions, associated_overlaps) where:
            associated_predictions (dict): Keys are unique reference labels from ref_labels; values are 1D numpy.ndarray of predicted labels (subset of values from pred_labels) ordered by decreasing min(relative_overlaps_pred, relative_overlaps_ref). Each array contains the predicted labels that occur among observations with the given reference label.
            associated_overlaps (dict): Keys match associated_predictions. Each value is a 2D numpy.ndarray with shape (k, 2) where k is the number of predicted labels present in that reference group. For each row i, column 0 is relative_overlaps_pred (overlap_count / predicted_group_size) and column 1 is relative_overlaps_ref (overlap_count / reference_group_size). These normalized overlap values are in the closed interval [0, 1] and quantify the proportion of the predicted group and the reference group made up by their intersection.
    
    See also:
        compute_association_matrix_of_groups: an alternative routine that computes an association matrix between groups; compare its output to the mappings and normalized overlaps produced by this function for complementary analyses.
    """
    from scanpy._utils import identify_groups
    return identify_groups(ref_labels, pred_labels, return_overlaps)


################################################################################
# Source: scanpy._utils.lazy_import
# File: scanpy/_utils/__init__.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def scanpy__utils_lazy_import(full_name: str):
    """scanpy._utils.lazy_import imports a module identified by its fully qualified name using importlib utilities and a LazyLoader so that the module's execution is deferred and safely inserted into sys.modules with proper locking. In the Scanpy single-cell analysis toolkit, this utility is used to reduce startup cost and delay importing heavy or optional dependencies (for example dask or visualization libraries) until their attributes are actually accessed by downstream code paths that perform preprocessing, visualization, or large-scale analyses.
    
    Args:
        full_name (str): The fully qualified module name to import (for example "dask.array" or "matplotlib.pyplot"). This string is passed to importlib.util.find_spec to locate the module specification. The parameter represents the module identity within Python's import system and determines which module object will be returned and inserted into sys.modules. Providing an already-imported module name causes the function to return the existing module object from sys.modules immediately, avoiding re-import and duplicate side effects.
    
    Returns:
        module: The module object corresponding to full_name. On normal operation this is a module object that has been created from the module spec, had its loader wrapped with importlib.util.LazyLoader to defer execution until member access, and been executed/installed into sys.modules under the given name with appropriate locking. The returned module may behave like a lazily initialized module: its body may only be executed when attributes are first accessed. Side effects: insertion or replacement of an entry in sys.modules[full_name], and possible execution of module code (either immediately or on first attribute access depending on the loader). Failure modes: if the module name cannot be resolved importlib.util.find_spec(full_name) returns None and subsequent operations will raise an exception originating from importlib (for example TypeError or AttributeError); loader.exec_module may also raise exceptions raised by the module's code or the import machinery. This function does not swallow those import-time exceptions; callers should catch exceptions if they need to handle missing optional dependencies in Scanpy workflows.
    """
    from scanpy._utils import lazy_import
    return lazy_import(full_name)


################################################################################
# Source: scanpy._utils.moving_average
# File: scanpy/_utils/__init__.py
# Category: valid
################################################################################

def scanpy__utils_moving_average(a: numpy.ndarray, n: int):
    """scanpy._utils.moving_average: Compute the moving average of a one-dimensional NumPy array.
    
    This utility computes a simple (unweighted) moving average over a 1-D numeric array using a cumulative-sum algorithm for efficiency. In the Scanpy single-cell analysis context this is typically used to smooth a numeric vector such as a gene expression profile across cells (for example across a neighborhood of cells or along a pseudotime trajectory) to reduce high-frequency noise while preserving coarse trends. The implementation returns values as float and does not modify the input array.
    
    Args:
        a (numpy.ndarray): One-dimensional input array of numeric values to be smoothed (for example, a vector of per-cell gene expression values). The function expects a 1-D numpy.ndarray; passing multi-dimensional arrays is not supported by this function and may produce incorrect results or errors.
        n (int): Number of consecutive entries to average over. For example, n=2 averages each element with its immediate predecessor (current and previous entry). n must be a positive integer. n == 1 returns the input values cast to float (no smoothing). If n is larger than the length of a, the function returns an empty array (length zero). If n <= 0, the function will raise an error due to invalid slicing/assignment.
    
    Returns:
        numpy.ndarray: One-dimensional NumPy array (dtype float) that is a view into an internally allocated array holding the moving-average results. The length of the returned array is len(a) - n + 1 when 1 <= n <= len(a); it is empty when n > len(a). The returned array does not share memory with the input a (the input is not modified).
    """
    from scanpy._utils import moving_average
    return moving_average(a, n)


################################################################################
# Source: scanpy.datasets._ebi_expression_atlas.ebi_expression_atlas
# File: scanpy/datasets/_ebi_expression_atlas.py
# Category: valid
################################################################################

def scanpy_datasets__ebi_expression_atlas_ebi_expression_atlas(
    accession: str,
    filter_boring: bool = False
):
    """scanpy.datasets._ebi_expression_atlas.ebi_expression_atlas: Load a dataset from the EBI Single Cell Expression Atlas and return it as an anndata.AnnData object for downstream single-cell gene expression analysis with Scanpy.
    
    This function retrieves a single-cell experiment identified by an EBI accession (for example E-GEOD-98816 or E-MTAB-4888) from the EBI Single Cell Expression Atlas. It first attempts to read a cached H5AD file from the local dataset directory configured in scanpy.settings.datasetdir under a subdirectory named by the accession. If the cached file cannot be read (for example because it does not exist or an OSError occurs), the function downloads the experiment archive using download_experiment, extracts expression data from expression_archive.zip, reads sample/experimental design from experimental_design.tsv, populates the AnnData.obs dataframe with those metadata columns, and writes a compressed H5AD file (gzip) to the same location to cache the assembled dataset for future calls. The returned object is an anndata.AnnData containing the expression matrix (n_obs × n_vars) and associated metadata suitable for Scanpy workflows (preprocessing, visualization, clustering, differential expression, etc.). This operation may require an internet connection when the dataset is not already cached locally and will log the download location.
    
    Args:
        accession (str): Dataset accession identifier used by the EBI Single Cell Expression Atlas. Example values include E-GEOD-98816 or E-MTAB-4888; the accession can be found in the dataset URL on the atlas website. This string determines the local subdirectory under scanpy.settings.datasetdir where the cached H5AD is expected (scanpy.settings.datasetdir / accession) and the name of the cached file (accession.h5ad). The accession is used by download_experiment to fetch the dataset when it is not readable from disk.
        filter_boring (bool): Whether to remove "boring" columns from the returned AnnData.obs before returning. A boring column is a metadata column that provides no informative variation for downstream single-cell analyses, for example a column with the same value for all observations or a column with as many distinct values as there are observations (anndata.AnnData.n_obs). Default is False. When True, the function calls the internal helper _filter_boring on adata.obs; this mutates the obs dataframe in memory (and the cached H5AD written after assembly will reflect the filtered obs if the dataset was downloaded and rebuilt).
    
    Returns:
        anndata.AnnData: An annotated data matrix representing the single-cell experiment, containing the expression matrix (observations × variables) and associated observation metadata in .obs. Practical significance: this return value is immediately usable with Scanpy routines (preprocessing, visualization, clustering, trajectory inference, differential expression). Side effects: when the dataset was not readable from cache, the function downloads the experiment, creates a subdirectory under scanpy.settings.datasetdir named by the accession, writes a compressed H5AD file named accession.h5ad into that directory (using gzip compression) to save disk space, and logs the download location. If filter_boring is True the returned AnnData.obs will have had boring columns removed. Failure modes: reading a cached file may raise OSError which triggers a download attempt; download_experiment requires network access and may raise network or HTTP-related exceptions; reading the archive may raise ZipFile or parsing errors; pandas.read_csv may raise errors when reading experimental_design.tsv; anndata.read_h5ad and anndata.write may raise I/O or format errors. Exceptions are not swallowed except for the initial read attempt which, on OSError, proceeds to download and rebuild the dataset.
    """
    from scanpy.datasets._ebi_expression_atlas import ebi_expression_atlas
    return ebi_expression_atlas(accession, filter_boring)


################################################################################
# Source: scanpy.neighbors._connectivity.umap
# File: scanpy/neighbors/_connectivity.py
# Category: fix_docstring
# Reason: Schema parsing failed: ("Couldn't parse this type hint, likely due to a custom class or object: ", numpy.ndarray[tuple[typing.Any, ...], numpy.dtype[numpy.int32 | numpy.int64]])
################################################################################

def scanpy_neighbors__connectivity_umap(
    knn_indices: numpy.typing.NDArray[numpy.int32 | numpy.int64],
    knn_dists: numpy.typing.NDArray[numpy.float32 | numpy.float64],
    n_obs: int,
    n_neighbors: int,
    set_op_mix_ratio: float = 1.0,
    local_connectivity: float = 1.0
):
    """scanpy.neighbors._connectivity.umap: Compute a fuzzy simplicial set (sparse connectivity graph) by wrapping umap.fuzzy_simplicial_set for use in Scanpy's neighborhood graph construction for single‑cell gene expression analysis.
    
    This function constructs a global fuzzy simplicial set (represented as a sparse graph in CSR format) from precomputed k‑nearest neighbor indices and distances. It locally approximates geodesic distances at each observation, builds local fuzzy simplicial sets per observation, and combines them via a fuzzy union. This wrapper is used in Scanpy to produce the connectivity (adjacency) graph that downstream tasks rely on (for example: clustering, UMAP embedding, and graph‑based visualization) and preserves compatibility with UMAP's implementation while suppressing a common TensorFlow import warning.
    
    Args:
        knn_indices (numpy.typing.NDArray[numpy.int32 | numpy.int64]): Integer array of neighbor indices returned by a k‑nearest neighbors search. Expected to provide, for each observation, the indices of its n_neighbors nearest neighbors. This array is consumed directly by UMAP's fuzzy_simplicial_set to determine which observations are locally connected; it must be consistent with n_obs and n_neighbors (mismatches will raise an exception from the underlying UMAP or NumPy code).
        knn_dists (numpy.typing.NDArray[numpy.float32 | numpy.float64]): Floating‑point array of neighbor distances corresponding to knn_indices. Distances are used to estimate local kernel widths (sigma) and local connectivity offsets (rho) for each observation. This array is passed unchanged to UMAP's fuzzy_simplicial_set and must align elementwise with knn_indices; invalid or NaN values will cause the underlying UMAP routine to fail.
        n_obs (int): The number of observations (rows / cells) in the original dataset. This integer is used to size internal structures and to interpret knn_indices/knn_dists correctly. If n_obs does not match the actual number of distinct indices referenced by knn_indices, the underlying UMAP routine may raise an error or produce an incorrect graph.
        n_neighbors (int): Neighborhood size used when knn_indices/knn_dists were computed; this value is forwarded to UMAP to control the number of neighbors considered when approximating local geodesic distances. It must match the neighbor dimension of knn_indices and knn_dists; inconsistent values will lead to exceptions from the underlying routine.
        set_op_mix_ratio (float): Mixing ratio controlling how local fuzzy simplicial sets are combined into the global fuzzy simplicial set. In practice, this parameter trades off between more intersection‑like vs. more union‑like combination of per‑point fuzzy sets when building the global connectivity graph. The default is 1.0. This is forwarded directly to umap.fuzzy_simplicial_set; invalid numeric types will raise an error in UMAP.
        local_connectivity (float): Parameter that affects how local connectivity is approximated at each observation when estimating geodesic distances and local kernel offsets. Higher values bias the construction toward ensuring a minimum local connectivity. The default is 1.0. This value is forwarded unchanged to umap.fuzzy_simplicial_set; inappropriate values may cause the underlying algorithm to behave unexpectedly or raise an exception.
    
    Behavior and side effects:
        This function imports umap.umap_.fuzzy_simplicial_set and calls it with a dummy sparse data matrix sized by n_obs (the actual data matrix is not required because neighbors and distances are supplied). A warnings filter is applied around the import to ignore a common "Tensorflow not installed" warning emitted by UMAP; this suppression is local to the import and does not affect other warnings. The UMAP call returns connectivities together with per‑point sigma and rho arrays; sigma and rho are computed by UMAP but deliberately discarded by this wrapper. The function does not modify the input arrays in place. If the external umap dependency is missing, importing fuzzy_simplicial_set will raise an ImportError propagated to the caller. If the shapes, dtypes, or contents of knn_indices/knn_dists are incompatible with expectations, underlying NumPy/UMAP code will raise TypeError or ValueError. No GPU/device selection or additional configuration is performed here; behavior is determined by the installed UMAP implementation and its dependencies.
    
    Returns:
        CSRBase: A sparse matrix in CSR format (CSRBase) that represents the fuzzy simplicial set (connectivity graph) computed by UMAP. This object encodes pairwise fuzzy connectivity strengths between observations and is intended to be used by Scanpy as the neighborhood/connectivity graph for downstream single‑cell analyses (e.g., clustering, visualization, manifold learning). The returned matrix is produced by converting UMAP's output to CSR via .tocsr(). The per‑observation sigma and rho arrays computed by UMAP are not returned.
    """
    from scanpy.neighbors._connectivity import umap
    return umap(
        knn_indices,
        knn_dists,
        n_obs,
        n_neighbors,
        set_op_mix_ratio,
        local_connectivity
    )


################################################################################
# Source: scanpy.plotting._utils.check_projection
# File: scanpy/plotting/_utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def scanpy_plotting__utils_check_projection(projection: str):
    """Validate a projection argument used by Scanpy plotting utilities.
    
    This function is a small input validator used throughout scanpy.plotting._utils and higher-level Scanpy plotting functions to ensure that a requested visualization projection refers to a supported dimensionality for plotting high-dimensional single-cell gene expression embeddings (for example, 2D UMAP, 3D UMAP, PCA or t-SNE visualizations commonly produced in single-cell analysis workflows). It enforces the exact, case-sensitive string values that downstream plotting code expects so that plotting routines can select two- or three-dimensional plotting code paths without further checks.
    
    Args:
        projection (str): The projection specifier provided to a plotting function. This parameter indicates the desired embedding dimensionality for plotting: use "2d" to request two-dimensional visualizations and "3d" to request three-dimensional visualizations. The value is compared exactly (case-sensitive) against the allowed set {"2d", "3d"}. There is no default value in this function; callers must supply a string. The practical significance in the Scanpy single-cell analysis domain is that this argument controls whether plotting routines render one 2D plane (commonly used for UMAP/t-SNE/PCA snapshots) or a 3D projection (used when an extra spatial/latent axis is desired).
    
    Returns:
        None: On valid input ("2d" or "3d"), the function performs no side effects and returns None; it acts as a guard that allows calling code to proceed. On invalid input (any value not exactly "2d" or "3d"), the function raises a ValueError with the message "Projection must be '2d' or '3d', was '<provided value>'." This explicit failure mode prevents downstream plotting code from silently producing incorrect dimensionality or confusing errors and provides a clear, actionable error message for users of Scanpy plotting functions.
    """
    from scanpy.plotting._utils import check_projection
    return check_projection(projection)


################################################################################
# Source: scanpy.plotting._utils.fix_kwds
# File: scanpy/plotting/_utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def scanpy_plotting__utils_fix_kwds(kwds_dict: dict, **kwargs):
    """scanpy.plotting._utils.fix_kwds: Merge and prioritize plotting keyword arguments for Scanpy plotting utilities.
    
    Concise utility used by Scanpy's plotting helpers to consolidate two sources of keyword arguments into a single dictionary of parameters for plotting functions in single-cell gene expression analysis workflows. In the Scanpy plotting codebase, callers commonly have a dictionary of plot parameters (for example, defaults or parameters collected earlier in a plotting wrapper) and also accept additional keyword arguments via **kwargs. This function prevents argument duplication errors by merging those two sources and ensuring a predictable precedence: when the same key exists in both, the value coming from kwds_dict (the explicit dictionary of plot parameters typically produced earlier in the plotting call chain) takes precedence and overwrites the value from kwargs (the additional keyword arguments provided by the caller).
    
    This behavior is important in the domain of single-cell visualization (e.g., scatter, embedding, or feature plots) where many helper functions assemble parameter dictionaries and then allow callers to override or extend them with keyword arguments. Using fix_kwds ensures that parameters deliberately set earlier in the plotting pipeline are preserved while still allowing additional parameters to be added.
    
    Args:
        kwds_dict (dict): A dictionary of keyword parameters, typically produced earlier in a plotting wrapper or by a helper function. In Scanpy plotting usage this often represents default or previously assembled plotting parameters (for example, {'color': 'red', 's': 5}). This dictionary's entries will override entries with the same keys from kwargs when merged.
        kwargs (dict): Additional keyword arguments supplied by the caller (captured via **kwargs in the calling function). This dict contains supplementary plotting parameters that should be merged with kwds_dict. Keys in this dict will be preserved unless the same key also exists in kwds_dict, in which case the value from kwds_dict wins.
    
    Behavior and side effects:
        The function performs an in-place update of the kwargs mapping by applying kwds_dict on top of it (i.e., kwargs.update(kwds_dict)) and then returns the updated kwargs mapping. The effect is that the returned dictionary contains the union of keys from both inputs, and for any duplicate keys the value from kwds_dict overrides the value that was present in kwargs. This is intentionally designed so that parameters explicitly assembled earlier in Scanpy plotting code are preserved over caller-supplied keyword defaults.
        The function expects kwds_dict to be a dict; passing a non-dict or a mapping with unhashable keys may raise built-in exceptions (for example, TypeError). There are no other side effects beyond updating and returning the merged dictionary object.
    
    Returns:
        dict: The merged keyword-argument dictionary. This is the same object that was passed to the function as kwargs, after being updated with the contents of kwds_dict. The returned dict contains all keys from both inputs with kwds_dict values taking precedence on key collisions.
    """
    from scanpy.plotting._utils import fix_kwds
    return fix_kwds(kwds_dict, **kwargs)


################################################################################
# Source: scanpy.plotting._utils.savefig
# File: scanpy/plotting/_utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def scanpy_plotting__utils_savefig(writekey: str, dpi: float = None, ext: str = None):
    """Save the current matplotlib figure to a file using Scanpy's figure settings and log the action.
    
    This function is used throughout Scanpy plotting utilities to persist figures generated during single-cell gene expression analysis (for example UMAP, PCA, heatmaps, cluster plots) so that results and figures can be included in reports, publications, or downstream reproducible workflows. The filename is constructed from Scanpy's settings: settings.figdir, the provided write key, settings.plot_suffix, and the chosen file extension. The function ensures the target directory exists, may emit a one-time low-resolution warning based on matplotlib rcParams, sets/uses a DPI value for saving, writes the file with matplotlib.pyplot.savefig(..., bbox_inches="tight"), and logs the saved filename at warning level.
    
    Args:
        writekey (str): A short identifier used to construct the output filename. Practically, this is the base name for the saved figure and is combined with Scanpy settings to produce the full path: settings.figdir / f"{writekey}{settings.plot_suffix}.{ext_or_default}". In Scanpy workflows the writekey typically identifies the plot type or analysis step (for example "umap", "pca", "heatmap") and is important for organizing figure output within the project figure directory.
        dpi (float): Dots-per-inch resolution for the saved figure. If provided (non-None), this numeric value is passed directly to matplotlib.pyplot.savefig as the dpi argument and controls raster output resolution for formats that use it. If dpi is None (the default), the function inspects matplotlib.rcParams["savefig.dpi"]: if rcParams["savefig.dpi"] is not a string and is less than 150, the function emits a single warning (unless settings._low_resolution_warning is already False) advising the user to call set_figure_params(dpi_save=...) to increase save resolution; in that low-resolution case the dpi variable is left as None and matplotlib's default behavior applies. Otherwise (rcParams["savefig.dpi"] is a string or >= 150), dpi is set to rcParams["savefig.dpi"] and passed to savefig. This behavior ensures notebook- and environment-consistent DPI handling while encouraging publication-quality resolution for Scanpy figures.
        ext (str): File extension (format) to use when saving the figure, such as "pdf" or "png". If ext is None (the default), the function uses the global setting settings.file_format_figs as the extension. The chosen extension is appended to the filename when constructing the path.
    
    Returns:
        None: This function does not return a value. Its side effects are: creating the directory settings.figdir (settings.figdir.mkdir(parents=True, exist_ok=True)), constructing the output filename from writekey, settings.plot_suffix, and ext (or settings.file_format_figs), logging a warning message that the figure is being saved to that filename, possibly emitting a low-resolution warning once via logg.warning and setting settings._low_resolution_warning = False, and calling matplotlib.pyplot.savefig(filename, dpi=dpi, bbox_inches="tight") to write the file to disk. Possible failure modes include filesystem errors such as PermissionError or OSError if the target directory or file cannot be created/written, and matplotlib-related errors if an unsupported file extension or an invalid dpi value is provided; callers should handle or allow such exceptions to propagate.
    """
    from scanpy.plotting._utils import savefig
    return savefig(writekey, dpi, ext)


################################################################################
# Source: scanpy.plotting._utils.timeseries
# File: scanpy/plotting/_utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def scanpy_plotting__utils_timeseries(X: numpy.ndarray, **kwargs):
    """scanpy.plotting._utils.timeseries: Create a Matplotlib figure sized for Scanpy timeseries plots and render the provided array X by delegating drawing and layout details to timeseries_subplot.
    
    This utility is part of Scanpy's plotting helpers used in the single-cell analysis and visualization workflow (for example, to visualize expression values across an ordered axis such as time, pseudotime, or sample order). The function prepares a figure with Scanpy's preferred default size and subplot margins, then calls timeseries_subplot(X, **kwargs) to perform the actual plotting. It does not return a plotting object; instead it produces side effects on the current Matplotlib state (creates a new figure and draws onto axes).
    
    Args:
        X (numpy.ndarray): The numerical array of values to plot. In Scanpy use cases this typically represents timeseries-like data derived from single-cell experiments (for example, expression of one or more genes across an ordered set of observations). The array may be 1-D or 2-D depending on how timeseries_subplot interprets rows/columns; this function forwards X unchanged to timeseries_subplot. If X is not a numpy.ndarray, callers should convert or cast it before calling; otherwise a TypeError or an error from timeseries_subplot may occur.
        kwargs (dict): Additional keyword arguments forwarded directly to timeseries_subplot. These control plot appearance and behavior as implemented by timeseries_subplot (for example, labels, color specifications, smoothing, aggregation options, axis parameters). This function does not validate or document individual keys here; accepted keys and their meanings must be consulted in timeseries_subplot's documentation. Passing unsupported keys will result in whatever error timeseries_subplot raises.
    
    Returns:
        None: This function does not return a value. Side effects: it creates a new Matplotlib figure with figsize set to twice each value in rcParams["figure.figsize"] and subplot parameters SubplotParams(left=0.12, right=0.98, bottom=0.13), then calls timeseries_subplot(X, **kwargs) which draws the timeseries on the current axes. Errors are raised if Matplotlib is not available, if rcParams lacks the expected "figure.figsize" entry, or if timeseries_subplot encounters invalid data or unsupported keyword arguments.
    """
    from scanpy.plotting._utils import timeseries
    return timeseries(X, **kwargs)


################################################################################
# Source: scanpy.preprocessing._deprecated.zscore_deprecated
# File: scanpy/preprocessing/_deprecated/__init__.py
# Category: fix_docstring
# Reason: Schema parsing failed: Cannot generate JSON schema for zscore_deprecated because the docstring has no description for the argument 'x'
################################################################################

def scanpy_preprocessing__deprecated_zscore_deprecated(x: numpy.ndarray):
    """scanpy.preprocessing._deprecated.zscore_deprecated: Z-score standardize each variable (gene) in a data matrix x. This function implements a column-wise (per-gene) centering by the column mean and scaling by the column standard deviation as used in Scanpy preprocessing workflows and cited in Weinreb2017. This function is deprecated; use scale instead.
    
    This function is intended for single-cell gene expression data matrices where rows correspond to cells and columns correspond to genes. It computes column-wise means and standard deviations, tiles them to the shape of x, and returns (x - mean) / (std + 0.0001). The small constant 0.0001 is added to avoid division-by-zero for genes with zero variance.
    
    Args:
        x (numpy.ndarray): Data matrix input. In Scanpy preprocessing context, rows are cells and columns are genes; each element represents an observed expression value for a gene in a cell. The function expects a NumPy array; behavior for other array-like objects is not specified. If x contains NaNs, the resulting means, standard deviations, and standardized values will follow NumPy's NaN propagation rules (i.e., means/stds will be NaN and the output will contain NaNs). If x is not two-dimensional, the operation still computes means and standard deviations along axis 0, but this deviates from the intended rows=cells, columns=genes convention.
    
    Returns:
        numpy.ndarray: A new NumPy array with the same shape as x containing the z-score standardized values for each column (gene). Each output element equals (original_value - column_mean) / (column_std + 0.0001). The function does not modify x in-place; it allocates intermediate arrays for the tiled means and standard deviations and returns a newly computed array. Note that for columns with zero variance, the addition of 0.0001 prevents division-by-zero but will yield large magnitude values for those entries. The function uses NumPy's default standard deviation (ddof=0, population standard deviation) and has O(n*m) time complexity for an n-by-m input; it also incurs extra memory overhead for the tiled mean and std arrays.
    """
    from scanpy.preprocessing._deprecated import zscore_deprecated
    return zscore_deprecated(x)


################################################################################
# Source: scanpy.preprocessing._deprecated.highly_variable_genes.filter_genes_cv_deprecated
# File: scanpy/preprocessing/_deprecated/highly_variable_genes.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def scanpy_preprocessing__deprecated_highly_variable_genes_filter_genes_cv_deprecated(
    x: numpy.ndarray,
    e_cutoff: float,
    cv_filter: float
):
    """scanpy.preprocessing._deprecated.highly_variable_genes.filter_genes_cv_deprecated filters genes by coefficient of variance and mean for single‑cell gene expression matrices and returns a boolean mask of genes that pass the deprecated CV-based selection.
    
    Args:
        x (numpy.ndarray): A numeric 2-D array containing single-cell gene expression values used by Scanpy. In typical Scanpy usage this is a matrix with rows representing cells and columns representing genes. The function computes per-gene means and per-gene standard deviations from x (using numpy.std) to derive the coefficient of variation (CV = std / mean) used for filtering. This parameter is required and the array must be suitable for numeric reduction operations.
        e_cutoff (float): A floating-point mean-expression cutoff used to exclude low-mean genes before applying the CV filter. In single-cell analysis workflows this threshold removes genes with mean expression at or below this value so that CV comparisons are made only on genes with sufficient average expression. This parameter controls the practical significance of the filter by setting the minimal mean-expression level considered.
        cv_filter (float): A floating-point coefficient-of-variation threshold. After excluding genes below e_cutoff, the function computes CV per gene as std/mean (using numpy.std for the numerator) and retains genes that meet the CV criterion defined by this value. In single-cell preprocessing this parameter determines how stringently genes are selected for high variability based on dispersion relative to mean expression.
    
    Returns:
        numpy.ndarray: A 1-D boolean numpy array (mask) with length equal to the number of genes (columns of x). Each element is True for genes that pass the deprecated CV-and-mean filter (kept for downstream analysis such as highly variable gene selection) and False for genes that fail the filter. The returned mask is intended to be used to index or subset gene columns in downstream Scanpy preprocessing and analysis steps.
    
    Behavior and side effects:
        The function performs no in-place modifications to x; it computes per-gene statistics and returns a boolean mask. The CV computation uses numpy.std for the standard deviation. Genes with mean values at or below e_cutoff are excluded from consideration prior to applying the CV threshold. Division by zero can occur when a gene mean is zero; such genes will produce infinite or NaN CV values from the std/mean computation and consequently will not pass a finite cv_filter threshold (they will not be marked True). The function lives in a _deprecated module indicating this CV-based procedure is deprecated in Scanpy; prefer the current recommended highly variable gene selection utilities described in the Scanpy documentation for new analyses.
    
    Failure modes and errors:
        The function relies on numpy reduction operations and may raise exceptions (for example, TypeError or ValueError) if x is not a numeric numpy.ndarray, has incompatible dimensionality for per-column reductions (not 2-D), or contains values that cause numpy operations to fail. NaN or infinite values in x will propagate through mean/std calculations and may produce NaN or infinite CVs; these typically result in those genes failing the filter. The function does not validate argument types beyond what numpy operations enforce; callers should ensure x, e_cutoff, and cv_filter are appropriate numeric types before calling.
    """
    from scanpy.preprocessing._deprecated.highly_variable_genes import filter_genes_cv_deprecated
    return filter_genes_cv_deprecated(x, e_cutoff, cv_filter)


################################################################################
# Source: scanpy.preprocessing._deprecated.highly_variable_genes.filter_genes_fano_deprecated
# File: scanpy/preprocessing/_deprecated/highly_variable_genes.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def scanpy_preprocessing__deprecated_highly_variable_genes_filter_genes_fano_deprecated(
    x: numpy.ndarray,
    e_cutoff: float,
    v_cutoff: float
):
    """Filter genes by Fano factor and mean for single-cell gene expression matrices.
    
    This deprecated helper is part of Scanpy's preprocessing utilities for selecting
    highly variable genes in single-cell RNA-seq data. It computes, for each gene
    (column) in the input expression matrix x, the mean expression across
    observations (cells) and the Fano factor defined as variance / mean, where the
    variance is computed with numpy.var (population variance). Genes are selected
    (if and only if) their mean expression is >= e_cutoff and their Fano factor
    is >= v_cutoff. This function is provided for compatibility with older Scanpy
    workflows (module path scanpy.preprocessing._deprecated.highly_variable_genes),
    and its behavior mirrors the historical Fano-factor-based gene filtering used
    in exploratory single-cell analysis to retain genes that are both expressed
    above a minimum level and show overdispersion relative to the mean.
    
    Args:
        x (numpy.ndarray): A 2-D numeric array containing single-cell expression
            measurements. The expected orientation is observations (cells) on axis 0
            and variables (genes) on axis 1, so x.shape == (n_obs, n_vars). Each
            column is treated as the expression vector for a single gene; the
            function computes per-column mean and variance. In single-cell analysis
            this array is typically raw or normalized count/UMI values; the choice
            of scale affects the computed Fano factor. Passing arrays with NaNs or
            non-numeric dtypes may raise errors or propagate NaNs in the result.
        e_cutoff (float): Threshold on the per-gene mean expression ("E" for
            expectation). This float is the minimum mean expression required for a
            gene to be considered expressed enough to pass the filter. In practice,
            using a positive e_cutoff excludes genes not reliably detected across
            cells; choose this parameter based on the expression scale (e.g.,
            counts vs. log-transformed values).
        v_cutoff (float): Threshold on the per-gene Fano factor (variance / mean).
            This float is the minimum Fano factor required for a gene to be
            considered overdispersed relative to its mean and thus potentially
            biologically variable. In single-cell workflows, increasing v_cutoff
            makes the filter more stringent, retaining fewer genes with strong
            overdispersion.
    
    Returns:
        numpy.ndarray: A 1-D boolean mask of length n_vars (the number of columns
        in x). Each element is True if and only if the corresponding gene meets
        both criteria mean >= e_cutoff and Fano factor >= v_cutoff. The returned
        array can be used to index columns of x or to subset gene lists. No in-place
        modification of x is performed; this function returns a new boolean array.
    
    Behavior, side effects, and failure modes:
        - Variance calculation uses numpy.var (population variance). The Fano factor
          is computed as variance / mean for each gene (column).
        - If a gene's mean expression is exactly zero, division by zero can occur;
          numpy will yield inf or NaN for the Fano factor and may emit a runtime
          warning. Such genes will typically not satisfy a finite v_cutoff and thus
          will not be selected; users should pre-filter genes with zero mean if a
          different behavior is desired.
        - NaN values in x propagate to per-gene statistics and can cause the
          corresponding mask entries to be False or NaN; callers should handle or
          clean NaNs before calling this function.
        - The function assumes a 2-D array input; passing arrays with different
          dimensionality will likely raise an error. The function does not perform
          type coercion beyond what numpy does; ensure numeric dtype for meaningful
          results.
        - This function is part of a deprecated API (module path contains
          _deprecated); prefer up-to-date Scanpy routines for new analyses where
          available.
    """
    from scanpy.preprocessing._deprecated.highly_variable_genes import filter_genes_fano_deprecated
    return filter_genes_fano_deprecated(x, e_cutoff, v_cutoff)


################################################################################
# Source: scanpy.preprocessing._simple.numpy_regress_out
# File: scanpy/preprocessing/_simple.py
# Category: fix_docstring
# Reason: Schema parsing failed: Cannot generate JSON schema for numpy_regress_out because the docstring has no description for the argument 'data'
################################################################################

def scanpy_preprocessing__simple_numpy_regress_out(data: numpy.ndarray, regressor: numpy.ndarray):
    """scanpy.preprocessing._simple.numpy_regress_out: Compute and return residuals after regressing out unwanted covariates from a numeric data matrix using NumPy linear algebra.
    
    This function implements the normal-equations solution of ordinary least squares to estimate coefficients for the provided regressor (design) matrix and then computes residuals of the input data with those estimated coefficients removed. It is a low-level NumPy implementation used in Scanpy preprocessing to remove unwanted sources of variation in single-cell gene expression analysis (for example, technical covariates such as batch effects, library size, or percent mitochondrial counts). The computation follows coeff = (regressor.T @ regressor)^{-1} @ (regressor.T @ data) and then produces residuals = data - regressor @ coeff. The regressor matrix is treated as a design matrix (observations × covariates) and data is treated as an observations × features matrix (for single-cell use: observations correspond to cells and features typically correspond to genes).
    
    Args:
        data (numpy.ndarray): Numeric data matrix to be corrected for covariates. In the source-code usage this is interpreted as an array with shape (n_observations, n_features) where n_observations (rows) align with the rows of the regressor matrix and n_features (columns) are variables such as gene expression values. The values are the dependent variables in the linear model and the returned array contains the residuals after removing the linear contribution of the regressor covariates. This argument must be a NumPy ndarray compatible with matrix multiplication by regressor.T as used in the implementation.
        regressor (numpy.ndarray): Design matrix of covariates (regressors) with shape (n_observations, n_covariates). Each column represents one covariate (for example, batch indicator or technical metric) and each row corresponds to an observation (cell). The function computes the ordinary least-squares coefficients using the normal equations with this regressor matrix. The columns of regressor must be such that regressor.T @ regressor is invertible for the computation to succeed.
    
    Raises:
        numpy.linalg.LinAlgError: If the Gram matrix regressor.T @ regressor is singular or nearly singular, numpy.linalg.inv will raise a LinAlgError. This occurs when regressors are linearly dependent (perfect multicollinearity) or when numerical precision prevents a stable inverse; callers should check and, if necessary, remove or regularize collinear regressors before calling this function.
        ValueError: If the shapes of data and regressor are incompatible for the matrix operations regressor.T @ data and regressor @ coeff, a ValueError from NumPy will be raised indicating mismatched dimensions.
    
    Returns:
        numpy.ndarray: A NumPy array of the same shape as the input data (n_observations, n_features) containing the residuals after regressing out the provided covariates. Each column corresponds to the original feature with the linear effect of regressor removed. The returned array is intended to be used downstream in Scanpy preprocessing pipelines (for example, as corrected expression values prior to scaling, PCA, clustering, or differential expression). Callers should not rely on in-place modification of the input arrays; treat the returned ndarray as the canonical corrected result.
    
    Notes on behavior and numerical considerations:
        This implementation uses the normal-equations approach (explicit matrix inverse) to compute ordinary least-squares coefficients, which is simple and efficient for small numbers of regressors but can be numerically unstable or expensive when the number of covariates is large or when regressors are highly collinear. For numerically stable or large-scale use-cases consider using QR decomposition, singular-value decomposition, or regularized solvers outside this function. The function does not add an intercept term automatically; if an intercept is required it must be included explicitly in the regressor matrix.
    """
    from scanpy.preprocessing._simple import numpy_regress_out
    return numpy_regress_out(data, regressor)


################################################################################
# Source: scanpy.preprocessing._utils.sparse_mean_var_major_axis
# File: scanpy/_compat.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def scanpy_preprocessing__utils_sparse_mean_var_major_axis(
    data: numpy.ndarray,
    indptr: numpy.ndarray,
    major_len: int,
    minor_len: int,
    n_threads: int
):
    """Compute per-major-axis (per-row) mean and variance for a sparse matrix stored in CSR-like arrays.
    
    This function is intended for use in preprocessing of single-cell gene expression matrices (as in Scanpy), where large sparse matrices are common. Given the raw CSR arrays (data and indptr) that represent nonzero entries of a 2-D sparse array, this routine computes for each major-axis entry (rows) the arithmetic mean and the population variance across the minor axis (columns). The implementation is written to be JIT-compiled and parallelized with Numba and numba.prange; n_threads controls that parallelization. The results are suitable for downstream preprocessing steps such as normalization, filtering by mean/variance, or variance-stabilizing transformations in single-cell analysis workflows.
    
    Args:
        data (numpy.ndarray): 1-D array containing the nonzero values of the sparse matrix in CSR ordering. Each element is treated as a floating-point value (the code casts to numpy.float64 during accumulation). This array must align with indptr so that for each row r, the nonzero entries for that row are data[indptr[r]:indptr[r+1]]. Practical significance: in Scanpy, data typically holds observed gene expression counts or normalized expression values; this function sums these stored values and their squares to compute means and variances that include implicit zeros from the sparse representation.
        indptr (numpy.ndarray): 1-D integer index array with length rows + 1 describing row boundaries in data (CSR index pointer). For row r, the indices of its nonzero values are in the half-open interval [indptr[r], indptr[r+1]). The function computes rows = len(indptr) - 1 and iterates r in range(rows). Failure modes: if indptr is malformed (length < 2) or contains indices outside the valid range for data, indexing errors will occur.
        major_len (int): Length of the major axis (number of rows) used to allocate the output arrays. In typical use this should equal len(indptr) - 1. If major_len < len(indptr) - 1 the function will raise an IndexError when writing outputs; if major_len > len(indptr) - 1 the extra entries in the returned arrays will remain zero. Practical significance: provides the expected output vector length for downstream Scanpy code that expects one mean/variance value per row.
        minor_len (int): Length of the minor axis (number of columns) used as the denominator when converting accumulated sums into means and variances. The function treats implicit zeros from the sparse representation by dividing by minor_len (i.e., computes the population mean and variance over minor_len entries). This value must be positive; if minor_len is zero a division-by-zero will occur. In single-cell contexts, minor_len is typically the number of features (genes) or cells depending on whether rows represent cells or genes.
        n_threads (int): Number of worker threads used with numba.prange to parallelize the accumulation and normalization loops. This integer should be positive; nonpositive values will lead to undefined or no parallel execution depending on the Numba runtime. Practical significance: increasing n_threads can accelerate computation on multi-core systems when processing large sparse single-cell matrices, but observe diminishing returns and potential contention for small inputs.
    
    Returns:
        tuple[numpy.ndarray, numpy.ndarray]: A pair (means, variances), both 1-D numpy.ndarray objects of length major_len and dtype float64. means[i] is the arithmetic mean for major-axis entry i computed as sum(data for that row including implicit zeros) / minor_len. variances[i] is the population variance for major-axis entry i computed as E[x^2] - (E[x])^2 where E denotes expectation over minor_len entries (i.e., division by minor_len, not minor_len - 1). Practical significance: these arrays provide per-row summary statistics used by Scanpy preprocessing routines (for example, selecting highly variable genes, scaling, or quality-control filters). No inputs are modified; the function allocates and returns new arrays.
    
    Behavior and failure modes:
        The function accumulates the sum and sum of squares only over stored (nonzero) entries in data and then divides by minor_len to include implicit zeros. It is designed to be JIT-compiled with Numba; calling the uncompiled Python version will be slower and may not support numba.prange semantics. Ensure major_len is at least len(indptr) - 1 (preferably equal) and minor_len > 0 to avoid IndexError or division-by-zero. Malformed indptr values (out-of-bounds indices) will raise indexing errors. The function uses population variance (denominator minor_len), so if sample variance (denominator minor_len - 1) is required, compute that externally.
    """
    from scanpy.preprocessing._utils import sparse_mean_var_major_axis
    return sparse_mean_var_major_axis(data, indptr, major_len, minor_len, n_threads)


################################################################################
# Source: scanpy.preprocessing._utils.sparse_mean_var_minor_axis
# File: scanpy/_compat.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def scanpy_preprocessing__utils_sparse_mean_var_minor_axis(
    data: numpy.ndarray,
    indices: numpy.ndarray,
    indptr: numpy.ndarray,
    major_len: int,
    minor_len: int,
    n_threads: int
):
    """Compute per-minor-axis (column) means and variances for a CSR-format sparse matrix.
    
    This function is used in Scanpy preprocessing routines to compute per-feature (per-gene) summary
    statistics across observations (cells) when the expression matrix is stored in CSR (compressed sparse row)
    format. It iterates over the nonzero entries provided by the CSR arrays and returns the mean and
    population variance for each minor-axis element (column/gene) across the major axis (rows/cells).
    The implementation is numba njit-compiled and uses parallel accumulation across n_threads to reduce
    contention; intermediate per-thread accumulators are combined at the end.
    
    Args:
        data (numpy.ndarray): 1-D array of nonzero values for the CSR matrix, in row-major order.
            Each entry data[j] is the nonzero value corresponding to column index indices[j]
            in the row described by indptr. Values are treated as numeric (floating) and are used
            directly in sum and squared-sum accumulation.
        indices (numpy.ndarray): 1-D integer array of column indices for each entry in data.
            For each j in range(len(data)), indices[j] is the minor-axis (column) index of data[j].
            Indices that are greater than or equal to minor_len are ignored (skipped) by this function.
        indptr (numpy.ndarray): 1-D integer index pointer array of length rows + 1 for the CSR representation.
            For row r (0 <= r < rows) the nonzero entries are data[indptr[r]:indptr[r+1]] with
            corresponding indices[indptr[r]:indptr[r+1]]. The function computes rows = len(indptr) - 1
            and iterates r from 0 to rows-1. indptr must therefore represent the CSR row structure of the data.
        major_len (int): Number of elements along the major axis (rows). This integer is used as the
            denominator when computing means and variances: mean = sum / major_len and variance =
            (sum_of_squares / major_len) - (mean ** 2). In Scanpy usage, major_len typically equals
            the number of observations (cells). major_len must be positive; passing 0 will cause a
            division-by-zero at return time.
        minor_len (int): Number of elements along the minor axis (columns). The function computes one
            mean and one variance for each minor index c in range(minor_len). The returned arrays have
            length minor_len. Entries in indices that are >= minor_len are skipped; mismatches between
            minor_len and the actual maximum index in indices will affect which entries are aggregated.
        n_threads (int): Number of threads used for parallel accumulation with numba.prange. The function
            creates temporary per-thread accumulators of shape (n_threads, minor_len) to accumulate sums
            and squared sums independently before combining them. n_threads must be a positive integer;
            using more threads than rows is allowed (some threads will process no rows) but increases
            temporary memory proportional to n_threads * minor_len.
    
    Behavior, side effects, defaults, and failure modes:
        This function performs a pure numeric computation and has no side effects on its inputs.
        It is intended for use with CSR arrays representing sparse single-cell expression matrices
        where rows correspond to observations (cells) and columns correspond to features (genes).
        The function runs in parallel using numba.njit and numba.prange; therefore it requires that
        the environment supports numba JIT compilation and parallel execution.
        The variance returned is the population variance (division by major_len), not the unbiased
        sample variance (which would divide by major_len - 1). Small negative variance values may
        appear due to floating-point rounding error when squared sums and squared means are nearly equal.
        If major_len is zero, a division-by-zero will occur. If indptr, indices, or data are not
        consistent CSR arrays (for example, indptr values out of range for data/indices), the function
        may raise indexing errors or produce incorrect results. No input validation is performed inside
        the function beyond skipping indices >= minor_len.
    
    Returns:
        tuple (numpy.ndarray, numpy.ndarray): A pair (means, variances), where both are 1-D numpy arrays
        of length minor_len. means[c] is the mean of column c computed as the sum of values in that column
        divided by major_len. variances[c] is the population variance of column c computed as
        (sum_of_squares / major_len) - (mean ** 2). These arrays are intended for downstream Scanpy
        preprocessing steps such as normalization, scaling, and highly-variable-gene detection.
    """
    from scanpy.preprocessing._utils import sparse_mean_var_minor_axis
    return sparse_mean_var_minor_axis(
        data,
        indices,
        indptr,
        major_len,
        minor_len,
        n_threads
    )


################################################################################
# Source: scanpy.readwrite.convert_bool
# File: scanpy/readwrite.py
# Category: fix_docstring
# Reason: Schema parsing failed: Cannot generate JSON schema for convert_bool because the docstring has no description for the argument 'string'
################################################################################

def scanpy_readwrite_convert_bool(string: str):
    """scanpy.readwrite.convert_bool checks whether a given string is the literal boolean "True" or "False" and returns a pair that signals recognition and the corresponding boolean value. This utility is intended for read/write and parsing code in Scanpy (the single-cell gene expression analysis toolkit) where boolean values may be represented as text in configuration files, metadata fields, or I/O parameters; it provides a deterministic, case-sensitive test and a simple encoded result that calling code can use to decide how to interpret or coerce string-valued inputs.
    
    Args:
        string (str): The input text to test. The function performs an exact, case-sensitive equality comparison against the two Python literal strings "True" and "False". In the Scanpy I/O and metadata context this parameter represents a textual boolean candidate read from a file, user input, or annotation field.
    
    Returns:
        tuple[bool, bool]: A two-element tuple (is_bool, value). is_bool is True when the input exactly matched one of the recognized boolean literals ("True" or "False"); otherwise is_bool is False. value is the boolean interpretation when is_bool is True: it is True for the input "True" and False for the input "False". When the input is not recognized as either literal, the function returns (False, False). There are no side effects, the function is deterministic, and matching is strict and case-sensitive (e.g., "true", "FALSE", "1", or "0" are not recognized).
    """
    from scanpy.readwrite import convert_bool
    return convert_bool(string)


################################################################################
# Source: scanpy.readwrite.convert_string
# File: scanpy/readwrite.py
# Category: fix_docstring
# Reason: Schema parsing failed: ("Couldn't parse this type hint, likely due to a custom class or object: ", int | float | bool | str | None)
################################################################################

def scanpy_readwrite_convert_string(string: str):
    """scanpy.readwrite.convert_string: Convert a textual value into a native Python scalar (int, float, bool, str) or None.
    
    Args:
        string (str): A text value to convert. In the Scanpy read/write context this is typically a cell- or feature-level annotation read from a text-based file (for example CSV/TSV metadata columns or other untyped fields encountered when importing data for single-cell gene expression analysis). The function expects a Python str and interprets its content to produce a more specific Python type so downstream Scanpy code can perform numeric or boolean operations (for example filtering, numeric comparisons, or logical indexing) instead of working with raw strings.
    
    Returns:
        int | float | bool | str | None: The converted Python value. The conversion rules applied, in order of precedence, are:
        - If the string represents an integer according to the internal is_int check, an int is returned (useful for counts, indices or integer-coded annotations).
        - Else if it represents a floating-point number according to the internal is_float check, a float is returned (useful for continuous measurements or numeric metadata).
        - Else if the helper convert_bool indicates the string is a boolean representation, the corresponding bool is returned. The exact set of boolean representations is defined by convert_bool; this function defers to that helper for boolean parsing semantics.
        - Else if the string is exactly the four characters "None", the function returns Python None (useful for representing missing or undefined metadata in a native way).
        - Otherwise, the original string is returned unchanged.
    
    Behavior and side effects:
        This function is a pure, deterministic converter with no external side effects (it does not modify global state or I/O). It relies on the helper predicates is_int and is_float and on convert_bool to determine convertibility; these helpers implement the precise syntactic checks and accepted token sets. Because the function checks conversion predicates before calling int() or float(), typical malformed numeric strings will not raise exceptions from int/float conversion; instead, they will fall through and be returned as the original string. The function signature requires a str; passing non-str types is outside the intended usage and may raise a TypeError or produce undefined behavior depending on the object's implementation of the operations used.
    
    Failure modes and notes:
        - If the input is not a Python str, behavior is unspecified and may raise an exception.
        - The function does not perform locale-specific number parsing, nor does it accept binary/hex prefixes unless covered by the helper checks.
        - The exact boolean token semantics (which string values map to True/False) and any additional edge cases are defined by convert_bool; consult that helper if you need to adjust or understand boolean parsing rules.
        - The function preserves the original string when no conversion rules match, ensuring no information is lost for unrecognized formats.
    """
    from scanpy.readwrite import convert_string
    return convert_string(string)


################################################################################
# Source: scanpy.readwrite.is_float
# File: scanpy/readwrite.py
# Category: fix_docstring
# Reason: Schema parsing failed: Cannot generate JSON schema for is_float because the docstring has no description for the argument 'string'
################################################################################

def scanpy_readwrite_is_float(string: str):
    """scanpy.readwrite.is_float: Check whether a given string can be converted to a Python float.
    
    This utility is used by Scanpy read/write code and related parsing utilities to detect whether a textual token represents a floating-point numeric value before attempting numeric conversion. In the context of single-cell gene expression data ingestion (for example when parsing CSV/TSV expression matrices or metadata fields), this function helps decide whether a cell in a text field should be interpreted as a numeric value and avoids unexpected failures when scanning large datasets.
    
    Args:
        string (str): The input text to test for float convertibility. This should be a Python str representing the token extracted from a file or other textual source (for example, a field read from a CSV row). The function calls the built-in float(string) to determine convertibility, so the accepted forms include integers, decimal notation, negative numbers, scientific notation (e.g., "1e-3"), and strings that float() understands such as "nan" or "inf". Leading and trailing whitespace is tolerated because float() strips whitespace. The function does not perform locale-aware parsing (for example, commas as decimal separators are not supported).
    
    Behavior and failure modes:
        The implementation attempts to convert the provided string using the built-in float() and returns an indicator of success. If float(string) raises a ValueError (typical for non-numeric text), the function catches that and returns False. If the provided argument is not a str (for example None or an arbitrary object) or if float() raises a TypeError for that input, that exception is not caught by this function and will propagate to the caller. There are no other side effects (no I/O, no global state changes). The function is lightweight and intended for use in parsing/validation logic prior to numeric conversion; it is not intended to perform logging or corrective transformations of the input.
    
    Returns:
        float: The source code's return annotation is float. In practice, the function returns a boolean indicator: True if float(string) completes successfully (i.e., the string can be parsed as a float), and False if float(string) raises a ValueError. Note that this describes the implemented behavior; callers relying on the annotated float return type in the signature should be aware that the concrete return values are True/False.
    """
    from scanpy.readwrite import is_float
    return is_float(string)


################################################################################
# Source: scanpy.readwrite.is_int
# File: scanpy/readwrite.py
# Category: fix_docstring
# Reason: Schema parsing failed: Cannot generate JSON schema for is_int because the docstring has no description for the argument 'string'
################################################################################

def scanpy_readwrite_is_int(string: str):
    """scanpy.readwrite.is_int checks whether a text string represents an integer according to Python's built-in int() conversion.
    
    This utility is intended for use in Scanpy's read/write and parsing routines when deciding whether a token (for example, a field from a text file, a row/column label, a barcode fragment, a layer index, or other metadata encountered while loading single-cell gene expression data) should be interpreted as an integer. The function performs a single, non-destructive conversion attempt and has no side effects on program state or input objects.
    
    Args:
        string (str): The string to test for integer-ness. This should be a Python str containing the characters to evaluate (for example, "42", "-7", or "  +3  "). The function attempts to convert this exact string with Python's built-in int(string). Valid representations follow int() semantics for base-10 integers (optional leading sign, optional surrounding whitespace). Strings that include a decimal point, numeric separators like underscores, or prefixes such as "0x" or "0b" will not be accepted by int() in its default usage and therefore will cause this function to report False. Although the signature requires a str, passing a non-str value may lead to a TypeError raised by int(); such errors are not caught by this function.
    
    Returns:
        bool: True if int(string) succeeds (the string is a valid base-10 integer representation under Python's int() semantics), False if int(string) raises a ValueError (the string is not an integer representation). There are no other return values or side effects.
    """
    from scanpy.readwrite import is_int
    return is_int(string)


################################################################################
# Source: scanpy.tools._sim.check_nocycles
# File: scanpy/tools/_sim.py
# Category: fix_docstring
# Reason: Schema parsing failed: Cannot generate JSON schema for check_nocycles because the docstring has no description for the argument 'verbosity'
################################################################################

def scanpy_tools__sim_check_nocycles(Adj: numpy.ndarray, verbosity: int = 2):
    """Check_nocycles verifies that the directed graph encoded by a square adjacency matrix contains no directed cycles.
    
    Args:
        Adj (numpy.ndarray): Square adjacency matrix of shape (n, n) describing a directed graph used in scanpy.tools._sim (for example, a simulated gene regulatory network or other directed interaction graph used during single-cell data simulation). Each entry Adj[i, j] represents the weight or presence of an edge from node j to node i as used by the implementation (the function multiplies the matrix by a column vector). The matrix must be a numpy.ndarray with numeric dtype and must be square (number of rows equals number of columns). If Adj is not square or not a numpy.ndarray with compatible numeric shape, the function will raise a numpy/linear-algebra error when performing dot products. Self-loops (nonzero diagonal entries) are treated as cycles of length 1 and will be detected.
        verbosity (int): Verbosity control for diagnostic output. Default is 2. When verbosity > 2, the function emits diagnostic messages via settings.m (it prints the adjacency matrix and a message indicating the detected cycle length and starting node). When verbosity <= 2 the function performs only the cycle check without printing diagnostics. The function does not otherwise modify Adj.
    
    Returns:
        bool: True if the graph described by Adj is acyclic (contains no directed cycles), False if a directed cycle is detected. The function tests cycles by, for each node g, initializing a basis vector at g and repeatedly applying Adj (v := Adj.dot(v)) up to n iterations; if after k iterations the entry corresponding to the original node g exceeds a numerical tolerance (1e-10), a directed cycle of length k is reported and the function returns False immediately. If the loop completes without detecting any such recurrence for any starting node, the function returns True. Notes: the function uses a fixed numerical tolerance of 1e-10 to decide whether returned mass to the starting node indicates a cycle; because of floating-point arithmetic and weighted edges, very small weights below this tolerance are considered absent for the purpose of cycle detection. The algorithm has early exit on first detected cycle and otherwise performs O(n^3)-like work for an n×n matrix due to repeated matrix-vector products.
    """
    from scanpy.tools._sim import check_nocycles
    return check_nocycles(Adj, verbosity)


################################################################################
# Source: scanpy.tools._sim.sample_coupling_matrix
# File: scanpy/tools/_sim.py
# Category: valid
################################################################################

def scanpy_tools__sim_sample_coupling_matrix(dim: int = 3, connectivity: float = 0.5):
    """Sample coupling matrix.
    
    Generates a random directed coupling matrix and associated adjacency representations for use in simulation utilities in scanpy.tools._sim. This function is used to produce a small synthetic interaction topology between dim nodes (for example, cell-state groups or modules in single-cell simulation workflows described in the Scanpy project README), ensuring no self-cycles and that the directed graph is acyclic. The sampled matrices can be used by downstream simulation code that requires a numeric coupling matrix, a binary adjacency matrix, and a signed adjacency matrix.
    
    Args:
        dim (int): Dimension of the coupling matrix and number of nodes in the sampled graph. The function allocates arrays of shape (dim, dim) and iterates over all ordered pairs (gp, g) with gp != g to decide presence of a directed edge. Default is 3, which produces 3x3 matrices representing interactions among three nodes.
        connectivity (float): Fractional connectivity parameter controlling edge density. Interpreted so that for each ordered pair (gp, g) with gp != g the function draws a uniform random number and creates a directed edge if that draw is less than 0.5 * connectivity. The multiplicative factor 0.5 is applied in the implementation so that connectivity=1. corresponds to the undirected-graph intuition in the original documentation (approximately dim*(dim-1)/2 distinct undirected edges), and connectivity=0. corresponds to no edges. Values outside the [0, 1] range are not constrained by this function but will affect the sampling probability linearly; reproducible results require controlling numpy's RNG (e.g., numpy.random.seed) before calling.
    
    Behavior and side effects:
        The function builds a float coupling matrix Coupl of shape (dim, dim) initialized to zeros and sets Coupl[gp, g] = 0.7 when an edge from gp to g is sampled; diagonal entries are always left at zero (no self-cycles). It then computes Adj_signed as numpy.sign(Coupl) with dtype "int_" (yielding 1 for positive edges and 0 for no edge) and Adj as the absolute value of Adj_signed (binary adjacency). The function uses a helper check_nocycles(Adj) to ensure the sampled directed graph contains no directed cycles. It repeats the sampling process up to 10 attempts (max_attempt = 10) until it finds a graph that is acyclic and has at least one edge. If no valid graph is found within these attempts, the function raises a ValueError with a message indicating the number of trials. The function does not modify external state other than consuming numpy's random stream.
    
    Failure modes and defaults:
        If the function cannot sample an acyclic graph with at least one edge after 10 attempts, it raises ValueError("did not find graph without cycles after 10 trials"). The default dim=3 and connectivity=0.5 are conservative choices that typically allow quick discovery of a valid acyclic topology; increasing dim or connectivity changes expected edge counts and may increase or decrease the probability of sampling cycles. The exact number of edges returned is random and depends on numpy.random; to reproduce results set numpy's RNG seed prior to calling.
    
    Returns:
        tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, int]: A 4-tuple containing:
            Coupl (numpy.ndarray): The sampled coupling matrix of shape (dim, dim) with float values. Entries corresponding to sampled directed edges are set to 0.7; diagonal entries are zero. This matrix is suitable as a weighted coupling matrix in simulation code.
            Adj (numpy.ndarray): The adjacency matrix obtained as the absolute value of Adj_signed (binary, same shape (dim, dim)). Values are 1 where an edge is present and 0 otherwise.
            Adj_signed (numpy.ndarray): The signed adjacency matrix (dtype "int_") obtained via numpy.sign(Coupl). For this implementation, entries are 1 for present positive edges and 0 otherwise; no negative entries are produced by the current sampling rule.
            n_edges (int): The number of directed edges sampled (counts gp -> g entries set to 0.7). This is the total number of nonzero off-diagonal entries in Coupl and equals the sum of Adj.
    """
    from scanpy.tools._sim import sample_coupling_matrix
    return sample_coupling_matrix(dim, connectivity)


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
