"""
Regenerated Google-style docstrings for module 'Bio'.
README source: others/readme/Bio/README.rst
Generated at: 2025-12-02T04:29:05.378109Z

Total functions: 153
"""


import numpy

################################################################################
# Source: Bio.Align.substitution_matrices.load
# File: Bio/Align/substitution_matrices/__init__.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_Align_substitution_matrices_load(name: str = None):
    """Load and return a precalculated substitution matrix used for sequence alignment scoring, or list available matrix files.
    
    This function is part of the Bio.Align.substitution_matrices module in Biopython and is used in computational molecular biology workflows to obtain substitution matrices (precomputed scoring matrices) for pairwise or multiple sequence alignment algorithms. When called with no argument, it discovers and returns the available matrix filenames in the package data directory so a user or program can choose which matrix to load. When called with a filename, it reads and returns the matrix data parsed by the module's read function so the matrix can be supplied to alignment routines that expect a substitution matrix.
    
    Args:
        name (str): The filename of a precalculated substitution matrix to load, or None to list available matrices. The filename must match exactly the file stored in the package data subdirectory "data" located alongside this module (case sensitivity follows the underlying filesystem). If name is None (the default), the function inspects the "data" subdirectory and returns a sorted list of filenames found there, excluding a development README.txt file if present. If name is a string, the function will construct the full path to that file inside the "data" subdirectory and parse it using Bio.Align.substitution_matrices.read to produce the matrix object.
    
    Returns:
        list[str] or object: If name is None, returns a sorted list of filenames (list of str) present in the substitution_matrices "data" directory; this is intended for discovery (for example, to pick a filename to pass back to this function). If name is provided, returns the matrix object produced by Bio.Align.substitution_matrices.read when parsing the named file. The exact structure/type of the returned matrix is whatever the module's read function produces for a valid substitution matrix file.
    
    Behavior and side effects:
    - The function resolves the file location relative to this module's source file, specifically the "data" subdirectory next to the module. It performs filesystem access (os.listdir and opening files via read), so calling it can raise I/O related exceptions.
    - When listing (name is None), the function attempts to remove "README.txt" from the returned filenames if present; this file is used in development installs and is not considered a matrix file.
    - When loading a named file, the function calls the module-level read(path) function; any parsing errors raised by read (for example due to malformed data) will propagate to the caller.
    
    Failure modes and exceptions:
    - If name is not None but does not match a file in the data directory, a FileNotFoundError (or OSError) will be raised when attempting to open the path.
    - If there are permission issues reading the data directory or file, PermissionError or OSError may be raised.
    - If name is provided but is not a str, errors may occur when constructing the path or when calling read; callers should pass either None or a str.
    - If the file exists but is not a valid substitution matrix, the read function may raise a parsing-related exception; those exceptions are not caught here and will propagate.
    
    Usage note:
    - Typical usage patterns are to call load() with no arguments to see available matrix filenames, then call load(name) with one of those filenames to obtain the parsed substitution matrix for use in alignment scoring routines. For example, in a Biopython-based alignment pipeline, load() enables programmatic discovery of available matrices bundled with the library and load(name) supplies the actual scoring matrix to the alignment algorithms.
    """
    from Bio.Align.substitution_matrices import load
    return load(name)


################################################################################
# Source: Bio.AlignIO.PhylipIO.sanitize_name
# File: Bio/AlignIO/PhylipIO.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_AlignIO_PhylipIO_sanitize_name(name: str, width: int = None):
    """Bio.AlignIO.PhylipIO.sanitize_name: Sanitize a sequence identifier string for safe output in PHYLIP-style alignment files used by Biopython's AlignIO/PhylipIO code.
    
    This function prepares a Python string intended to be used as a sequence identifier when writing alignment files in the PHYLIP family of formats. It applies a fixed, deterministic sequence of transformations to ensure the identifier does not contain characters banned or problematic for PHYLIP writers and optionally truncates the identifier to a specified maximum width. This routine is used by Biopython's PHYLIP output code to improve interoperability of produced files with downstream tools that expect simple identifiers.
    
    Args:
        name (str): The input sequence identifier to sanitize. The function will first remove leading and trailing whitespace (via str.strip()), then delete any occurrences of the characters '[', ']', '(', ')', and ',' from the string, and finally replace any ':' or ';' characters with the pipe character '|' to avoid using punctuation that can break PHYLIP parsers. This parameter must be a Python str; passing another type that does not implement str.strip() will raise an exception (typically AttributeError).
        width (int): Optional maximum number of characters to keep from the sanitized name. If specified as an int, the sanitized name is truncated by Python slicing name[:width]. If width is None (the default), no truncation is performed. If a non-int, non-None value is provided, slicing will raise a TypeError. Negative integer values for width follow Python slicing semantics (for example, width == -1 will omit the final character), so callers should pass a non-negative int when the intent is a simple left-side truncation.
    
    Returns:
        str: A new str containing the sanitized (and possibly truncated) identifier. The original input object is not modified in-place; instead a fresh string derived from the input is returned. The returned string will have had whitespace trimmed, the characters '[', ']', '(', ')', and ',' removed, any ':' or ';' characters replaced with '|', and will be limited to at most width characters if width was provided. Exceptions raised for invalid inputs include AttributeError when name is not a str-like object supporting strip(), and TypeError when width is not an int or None (due to invalid slicing).
    """
    from Bio.AlignIO.PhylipIO import sanitize_name
    return sanitize_name(name, width)


################################################################################
# Source: Bio.Cluster.clustercentroids
# File: Bio/Cluster/__init__.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_Cluster_clustercentroids(
    data: numpy.ndarray,
    mask: numpy.ndarray = None,
    clusterid: numpy.ndarray = None,
    method: str = "a",
    transpose: bool = False
):
    """Calculate and return the centroid of each cluster.
    
    This function computes cluster centroids from a numeric data matrix and a mapping of items to cluster indices. It is intended for use in clustering workflows common in computational molecular biology and bioinformatics (for example summarising clusters of gene expression profiles, feature vectors from sequence analysis, or other high-dimensional experimental measurements). The centroid for a cluster is computed per-dimension either as the arithmetic mean or the median over all items assigned to that cluster, ignoring values marked as missing by the mask. The implementation delegates the numerical work to a compiled extension (_cluster.clustercentroids) for performance; inputs are validated and prepared (see behavior and failure modes below).
    
    Args:
        data (numpy.ndarray): nrows x ncolumns array containing the data values. Each row (when transpose is False) or each column (when transpose is True) represents one item/observation, and each column (when transpose is False) or row (when transpose is True) represents one measured dimension/feature. This array is validated by an internal helper (__check_data) and must be a numeric array compatible with the underlying compiled routine. The function does not modify this array in place; it reads from it to compute centroids.
        mask (numpy.ndarray): nrows x ncolumns array of integers showing which data are missing, or None to indicate no explicit mask. The mask uses the convention mask[i, j] == 0 to mark a missing value at that position and non-zero to mark a present value. The mask shape must match data.shape; it is validated by an internal helper (__check_mask) which may raise a ValueError or TypeError for incompatible shapes or types. Missing values (mask == 0) are ignored when computing per-dimension means or medians for a cluster.
        clusterid (numpy.ndarray): array containing the cluster number for each item, or None. The length of this array must equal the number of items: nrows if transpose is False, or ncolumns if transpose is True. Values are treated as cluster indices and should be non-negative integers; the code coerces clusterid to dtype intc and C-contiguous storage via numpy.require before use. If clusterid is None, a default vector of zeros is created (equivalent to a single cluster for all items) and nclusters is set to 1. Supplying negative cluster indices is not supported by the algorithm and may lead to incorrect behavior or runtime errors.
        method (str): specifies the aggregation used to compute each centroid along each dimension. Accepted values are 'a' to compute the arithmetic mean (default) and 'm' to compute the median. The chosen method is applied per-dimension across all items assigned to the same cluster, with missing values excluded as indicated by mask. The method string is forwarded to the compiled extension; invalid method values will typically result in an error from the underlying implementation.
        transpose (bool): if False (default), each row of data corresponds to one item (so data is nrows x ncolumns and clusterid length is nrows). If True, each column of data corresponds to one item (so the effective items are columns and clusterid length must be ncolumns). This flag controls how the input matrix is interpreted and therefore affects the shapes of the returned centroid arrays.
    
    Behavior and side effects:
        The function validates and possibly coerces inputs: data is checked via __check_data, mask via __check_mask (which enforces matching shapes with data), and clusterid is coerced to numpy.intc C-contiguous array. If clusterid is None a new clusterid array of zeros is created (no side effect on caller variables). The number of clusters nclusters is computed as max(clusterid + 1). The function allocates output arrays cdata (dtype 'd', floating point) and cmask (dtype intc) with shapes dependent on transpose (see Returns). Computation of centroids is performed by the compiled routine _cluster.clustercentroids which may raise exceptions for invalid inputs (for example mismatched shapes, unsupported method values, or cluster indices outside expected ranges). The original data and mask arrays are not modified by this function; the outputs are newly allocated and returned.
    
    Failure modes and validation:
        Passing data and mask arrays of incompatible shapes, non-numeric data, or a clusterid array whose length does not match the number of items will result in validation errors from __check_data/__check_mask or in runtime errors from the compiled routine. Providing clusterid with negative values violates the documented requirement that cluster numbers be non-negative and can produce incorrect nclusters or errors. Supplying an unsupported method string will normally cause an error from the underlying implementation. Because the heavy computation is performed in a compiled extension, Python-level tracebacks may indicate errors originating from the extension; inspect input shapes, types, and values first when debugging.
    
    Returns:
        tuple: A pair (cdata, cmask).
        cdata (numpy.ndarray): 2D array containing the computed cluster centroids with dtype 'd' (double precision float). If transpose is False, cdata has shape (nclusters, ncolumns), where each row is the centroid for a cluster across the original columns/features. If transpose is True, cdata has shape (nrows, nclusters), where each column is the centroid for a cluster across the original rows/features. Each centroid value is the per-dimension mean or median (depending on method), computed over items assigned to that cluster and excluding positions marked missing by mask.
        cmask (numpy.ndarray): 2D array of integers (dtype intc) describing which entries in cdata are valid/present versus missing, using the same convention as the input mask (0 indicates missing). The shape of cmask matches cdata. cmask is produced by the computation and indicates dimensions for which no valid item values were available to compute a centroid (thus the corresponding cdata entry should be considered missing).
    """
    from Bio.Cluster import clustercentroids
    return clustercentroids(data, mask, clusterid, method, transpose)


################################################################################
# Source: Bio.Cluster.clusterdistance
# File: Bio/Cluster/__init__.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_Cluster_clusterdistance(
    data: numpy.ndarray,
    mask: numpy.ndarray = None,
    weight: numpy.ndarray = None,
    index1: list = None,
    index2: list = None,
    method: str = "a",
    dist: str = "e",
    transpose: bool = False
):
    """Calculate and return the distance between two clusters within a data matrix used for clustering tasks in computational molecular biology (for example, pairwise distances used in hierarchical clustering of gene expression or other experimental datasets). This function validates inputs, supports handling of missing data via a mask, optional per-item weights, multiple distance metrics (Euclidean, city block, several correlation-based measures), and several ways to combine pairwise distances into a cluster-to-cluster distance (means, medians, single/complete linkage, average). The final numeric distance is computed by delegating to the underlying compiled clustering routine _cluster.clusterdistance after input checks.
    
    Args:
        data (numpy.ndarray): nrows x ncolumns array containing the data values. Each row or column is treated as an item depending on the transpose flag. In the common biological use case, rows might represent samples and columns features (or vice versa when transpose=True). The array is validated by internal checks before use; invalid types or shapes will raise an exception.
        mask (numpy.ndarray): nrows x ncolumns array of integers indicating missing data. If mask[i, j] == 0 then data[i, j] is considered missing and will be ignored in distance calculations. If None (the default), no entries are treated as missing. The mask must have the same shape as data; a mismatch will raise an error.
        weight (numpy.ndarray): 1D array of weights to be used when calculating distances. The expected length equals the number of items (ndata) along the axis being clustered (ndata is number of columns if transpose is False, otherwise number of rows). If None (the default), equal weighting is assumed. The array is validated; an incorrect length will raise an error.
        index1 (list): 1D list identifying which items belong to the first cluster. Items are indices referring to the item axis determined by transpose. If the cluster contains only one item, index1 may instead be provided as a single integer (the function will accept it after validation). Invalid or out-of-range indices will produce an error.
        index2 (list): 1D list identifying which items belong to the second cluster. Semantics and validation are the same as for index1. If the cluster contains only one item, index2 may be a single integer.
        method (str): Specifies how the distance between two clusters is defined. Acceptable single-character codes and their meanings:
            'a' (default) -- distance between the arithmetic means of the two clusters;
            'm' -- distance between the medians of the two clusters;
            's' -- the smallest pairwise distance (single linkage) between members of the two clusters;
            'x' -- the largest pairwise distance (complete linkage) between members of the two clusters;
            'v' -- the average of the pairwise distances between members of the two clusters.
            An invalid method code will result in an error. This choice determines how pairwise item distances are aggregated into the cluster-to-cluster distance used by higher-level clustering algorithms.
        dist (str): Specifies the distance (or similarity) function used to compute pairwise distances between items. Acceptable single-character codes and their meanings:
            'e' (default) -- Euclidean distance;
            'b' -- City Block (Manhattan) distance;
            'c' -- Pearson correlation (treated as a similarity that the routine converts to a distance);
            'a' -- absolute value of the Pearson correlation (sign ignored before conversion to distance);
            'u' -- uncentered correlation;
            'x' -- absolute uncentered correlation;
            's' -- Spearman's rank correlation;
            'k' -- Kendall's tau.
            Correlation-based options are interpreted so that higher correlation corresponds to smaller distance (i.e., similarity is converted into a distance measure); absolute variants ignore sign. An invalid dist code will produce an error.
        transpose (bool): If False (default), clusters are defined over rows of the data matrix; if True, clusters are defined over columns. This flips which axis is considered the collection of items (and therefore affects the expected length of weight and the interpretation of index1/index2).
    
    Returns:
        float: A scalar numeric distance between the two specified clusters as computed by the chosen dist metric and aggregation method. The value is produced by the compiled _cluster.clusterdistance routine after input validation.
    
    Behavior, side effects, defaults, and failure modes:
        - The function performs input validation via internal helpers (__check_data, __check_mask, __check_weight, __check_index). If data, mask, weight, or index arrays have incompatible shapes, types, or out-of-range indices, a ValueError or IndexError will be raised.
        - Missing data are indicated by mask entries equal to 0 and are omitted from distance calculations; when mask is None, all data are treated as present.
        - Weights are applied per item along the axis being clustered (length must equal ndata). If weight is None, equal weighting is assumed.
        - The function does not modify the input arrays (data, mask, weight, index1, index2); it only reads them and passes validated values to the underlying compiled implementation.
        - The function delegates the numerical computation to the underlying _cluster.clusterdistance implementation (a compiled extension), so numerical details and performance characteristics follow that implementation.
        - Defaults: method='a' (distance between arithmetic means), dist='e' (Euclidean), transpose=False (clusters of rows).
        - Typical usage in the Biopython domain: computing distances between clusters of biological observations (e.g., samples or genes) as part of hierarchical clustering or cluster evaluation workflows.
    """
    from Bio.Cluster import clusterdistance
    return clusterdistance(data, mask, weight, index1, index2, method, dist, transpose)


################################################################################
# Source: Bio.Cluster.distancematrix
# File: Bio/Cluster/__init__.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_Cluster_distancematrix(
    data: numpy.ndarray,
    mask: numpy.ndarray = None,
    weight: numpy.ndarray = None,
    transpose: bool = False,
    dist: str = "e"
):
    """Compute and return a triangular distance matrix from a 2-D NumPy array for use in clustering and other computational-molecular-biology analyses.
    
    Args:
        data (numpy.ndarray): nrows x ncolumns 2-D array containing the numeric data values.
            In the Bio.Cluster context this represents the matrix of observations used to
            compute pairwise distances (for example rows = samples and columns = features,
            or vice versa when transpose=True). This function calls internal validation
            (__check_data) and requires a numeric, two-dimensional numpy.ndarray.
        mask (numpy.ndarray or None): nrows x ncolumns array of integers indicating missing
            values (same shape as data). If mask[i, j] == 0 then data[i, j] is treated as
            missing and excluded from pairwise distance calculations. If None (the default),
            no values are treated as missing. The mask must align with data shape exactly;
            supplying an incorrectly shaped mask will raise an error.
        weight (numpy.ndarray or None): 1-D array of weights applied to the data when
            calculating distances. The length of weight must match the number of items over
            which pairwise distances are computed (that is, the number of columns when
            transpose=False, or the number of rows when transpose=True). If None (the
            default) equal weighting is assumed. The function validates weight length via
            __check_weight and will raise an error on length mismatch.
        transpose (bool): If False (default), distances are computed between rows of data.
            If True, distances are computed between columns of data. This is provided to
            support common analysis patterns in molecular data where either samples or
            measured features may be compared.
        dist (str): Single-character code selecting the distance or similarity metric to
            use (default 'e'). The supported codes and their meanings are:
            'e' -- Euclidean distance
            'b' -- City Block (Manhattan) distance
            'c' -- Pearson correlation
            'a' -- Absolute value of the Pearson correlation
            'u' -- Uncentered correlation
            'x' -- Absolute value of the uncentered correlation
            's' -- Spearman's rank correlation
            'k' -- Kendall's tau
            The chosen code controls the mathematical form used for pairwise comparisons;
            invalid or unsupported codes will result in an error from the underlying
            routine.
    
    Returns:
        list[numpy.ndarray]: A lower-triangular, compressed representation of the full
        symmetric distance matrix as a Python list of one-dimensional numpy.ndarray
        objects (dtype float64). The list has length N where N is the number of items
        compared (rows if transpose=False, columns if transpose=True). Entry i of the
        list is a 1-D array of length i containing the distances from item i to items
        0..i-1. For example, the first element is an empty array (no previous items),
        the second element contains the distance between item 1 and item 0, etc. The
        diagonal distances (distance of an item to itself) are implicit zeros and are
        not stored. This compact format is suitable for downstream clustering algorithms
        and mirrors the output produced by the underlying C extension call
        (_cluster.distancematrix).
    
    Behavior, defaults, side effects, and failure modes:
        - The function validates inputs using internal helpers (__check_data, __check_mask,
          __check_weight) and then delegates the bulk computation to the C extension
          _cluster.distancematrix, which fills the preallocated list-of-arrays buffer.
        - Default behavior treats all data as present (mask=None) and equally weighted
          (weight=None), computes distances between rows (transpose=False), and uses
          Euclidean distance (dist='e').
        - Missing data positions indicated by mask entries equal to 0 are excluded from
          pairwise calculations; if too many values are missing for a pairwise comparison,
          the underlying routine may produce NaN or raise an error depending on the metric.
        - The function does not modify the input data, mask, or weight arrays.
        - Typical exceptions include TypeError if inputs are not numpy.ndarray where required,
          ValueError for shape mismatches between data, mask, and weight, and errors raised
          by the underlying C routine for invalid dist codes or unrecoverable numerical
          issues. Users should ensure input arrays are correctly shaped and that dist is one
          of the documented single-character codes.
    """
    from Bio.Cluster import distancematrix
    return distancematrix(data, mask, weight, transpose, dist)


################################################################################
# Source: Bio.Cluster.kcluster
# File: Bio/Cluster/__init__.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_Cluster_kcluster(
    data: numpy.ndarray,
    nclusters: int = 2,
    mask: numpy.ndarray = None,
    weight: numpy.ndarray = None,
    transpose: bool = False,
    npass: int = 1,
    method: str = "a",
    dist: str = "e",
    initialid: numpy.ndarray = None
):
    """Perform k-means clustering on a numeric matrix, returning cluster assignments,
    the within-cluster sum of distances for the best solution found, and how many
    times that best solution was observed. This implementation is used in the
    Biopython project for clustering rows or columns of biological data matrices
    (e.g., gene expression data: genes as rows, samples as columns), and is
    suitable for computational molecular biology workflows that require repeated
    k-means runs with different initializations or a single deterministic run
    starting from a provided initial clustering.
    
    Args:
        data (numpy.ndarray): nrows x ncolumns array containing the data values.
            Rows correspond to items and columns to variables by default. The shape
            expectation is preserved: if transpose is False, the number of items
            clustered (nitems) equals data.shape[0]; if transpose is True, nitems
            equals data.shape[1]. The array must be a NumPy ndarray containing
            numeric values; internal validation will raise an error if the input
            is not compatible.
        nclusters (int): number of clusters (the "k" in k-means). This controls
            how many cluster centroids the algorithm fits. nclusters should be a
            positive integer; if it is greater than the number of items, the
            underlying implementation or validation will raise an error.
        mask (numpy.ndarray): nrows x ncolumns array of integers indicating missing
            entries. mask[i,j] == 0 marks data[i,j] as missing and that element
            will be ignored in distance and centroid calculations. If None (the
            default), all data are treated as present. The mask must have the same
            shape as data when provided.
        weight (numpy.ndarray): one-dimensional array of length equal to the
            number of variables (ndata) containing weights applied when
            calculating distances and cluster centers. If None (the default),
            equal weighting is assumed. The internal validation enforces the
            required length and numeric type.
        transpose (bool): if False (default), rows of data are treated as the
            items to be clustered and columns as variables; if True, columns are
            treated as items and rows as variables. This option lets the function
            cluster either genes (rows) or samples (columns) in biological matrices
            without copying or transposing the input externally.
        npass (int): number of times to perform the k-means algorithm with
            different random initial clusterings when initialid is None. The
            routine returns the best solution found across these npass runs and
            how many times that best solution was encountered. Default is 1.
            If initialid is provided, npass is ignored and the algorithm runs
            once deterministically from the supplied initial clustering.
        method (str): how to compute the center of a cluster. Supported codes are
            'a' for arithmetic mean (default) and 'm' for median. This affects the
            centroid update step and therefore the fitted clusters.
        dist (str): specifies the distance or similarity metric used to assign
            items to clusters. Supported codes (and their meanings) are:
            'e' for Euclidean distance (default), 'b' for City Block distance,
            'c' for Pearson correlation, 'a' for absolute value of the
            correlation, 'u' for uncentered correlation, 'x' for absolute
            uncentered correlation, 's' for Spearman's rank correlation, and
            'k' for Kendall's tau. The chosen metric determines how distances
            between items and cluster centers are computed and therefore impacts
            clustering results.
        initialid (numpy.ndarray): initial clustering assignment used to start
            the algorithm. If provided (an array of length equal to the number of
            items, containing integer cluster indices), the routine carries out the
            k-means (EM) algorithm exactly once from that initial partition and
            does not randomize the order in which items are reassigned; the run
            is therefore fully deterministic. If None (default), npass randomized
            starts are used and results may vary between runs unless an external
            random seed is set.
    
    Returns:
        tuple:
            numpy.ndarray: clusterid. Array of length nitems containing the index
                of the cluster assigned to each item in the best k-means solution
                found. The number of items depends on transpose as described above.
            float: error. The within-cluster sum of distances (or dissimilarities)
                for the returned clustering solution; this is the objective that
                k-means attempts to minimize and is useful for comparing solutions.
            int: nfound. The number of times the returned optimal solution was
                observed across the npass runs (if initialid is None). If
                initialid is provided, nfound will reflect the single deterministic
                run (typically 1 if the run completes successfully).
    
    Notes and failure modes:
        - If initialid is None, the algorithm performs npass independent runs with
          different random initial clusterings; results are non-deterministic
          between processes or Python sessions unless randomness is controlled
          externally (for example by setting the global NumPy random seed).
        - If initialid is provided, the routine runs once deterministically from
          that starting partition and does not perform npass randomized starts.
        - Input validation functions (__check_data, __check_mask, __check_weight,
          __check_initialid) are invoked internally; they will raise exceptions
          (for example, ValueError) on incompatible shapes, invalid types, or
          unsupported parameter codes for method/dist. The caller should ensure
          shapes and types match the documented expectations to avoid these
          validation errors.
        - This function is intended for numeric data matrices common in
          computational molecular biology (e.g., gene expression, proteomics),
          and the mask/weight parameters allow handling missing data and variable
          importance typical in biological datasets.
    """
    from Bio.Cluster import kcluster
    return kcluster(
        data,
        nclusters,
        mask,
        weight,
        transpose,
        npass,
        method,
        dist,
        initialid
    )


################################################################################
# Source: Bio.Cluster.kmedoids
# File: Bio/Cluster/__init__.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_Cluster_kmedoids(
    distance: numpy.ndarray,
    nclusters: int = 2,
    npass: int = 1,
    initialid: list = None
):
    """Perform k-medoids clustering on a distance matrix.
    
    This function performs k-medoids clustering (partitioning around medoids) on a set of items given a distance matrix and returns the best clustering found, the within-cluster sum of distances (objective) for that clustering, and how many times that optimal solution was discovered across repeated restarts. In the context of Biopython and computational molecular biology, this is typically used to cluster biological items (for example, sequences, profiles, or other pairwise comparisons) when a precomputed pairwise distance matrix is available. The routine accepts three different representations of a symmetric distance matrix (full 2D array, condensed 1D array, or a list of lower-triangular rows) and uses only the lower-triangular part when a full 2D NumPy array is provided. The implementation validates the distance input, prepares an initial clustering (either provided or generated randomly), and delegates the core computation to the underlying clustering routine. If an explicit initial clustering is supplied via initialid, the algorithm runs once deterministically using that initialization; otherwise it performs npass independent random restarts and returns the best solution found.
    
    Args:
        distance (numpy.ndarray): The pairwise distance matrix between items. Accepted formats are: (1) a 2D NumPy array (only the left-lower / lower-triangular part is accessed), (2) a 1D NumPy array containing the condensed distances consecutively (as in SciPy/cluster condensed format), or (3) a list of 1D arrays where each element i contains the i-th row of the lower-triangular part (the first element may be empty). This argument is required and is validated by the function; malformed shapes or incompatible dimensions will raise a ValueError. The distances represent domain-specific dissimilarities (for example, sequence edit distances, profile distances, or other biologically meaningful dissimilarity measures) and must be non-negative as expected by k-medoids algorithms.
        nclusters (int = 2): The desired number of clusters (the k in k-medoids). Must be a positive integer less than or equal to the number of items represented by distance. Default is 2. If nclusters is larger than the number of items, the routine will raise an error.
        npass (int = 1): The number of independent runs (random restarts) of the k-medoids algorithm to perform when no initialid is provided. Each pass uses a different random initial clustering to reduce the chance of converging to a poor local optimum. Default is 1. If initialid is supplied, npass is ignored and the algorithm runs exactly once using the provided initial clustering.
        initialid (list = None): An optional initial clustering assignment used to start the algorithm deterministically. When given, initialid should specify a clustering consistent with the number of items (typically as an array-like of medoid indices or cluster assignments compatible with the internal checks); the routine will perform the EM-style re-assignment/medoid update sequence once starting from this initialization and will not randomize the order in which items are assigned to clusters. Providing initialid makes the run deterministic (useful for reproducible experiments or continuing clustering from a previous result). When initialid is None (the default), the routine will perform npass randomized starts and return the best solution found.
    
    Returns:
        tuple: A 3-tuple containing the results of the k-medoids clustering:
            clusterid (array): An array containing, for each item (in the original input order), the index of the cluster (specifically the index of the item chosen as the medoid of that cluster) to which the item was assigned in the best solution found. The indices refer to positions in the original dataset/distance matrix and therefore preserve the mapping back to the biological items being clustered.
            error (float): The within-cluster sum of distances (the k-medoids objective) for the returned clustering solution. This scalar quantifies the compactness of clusters under the provided distance measure; lower values indicate tighter clusters according to the input dissimilarity.
            nfound (int): The number of times the exact optimal solution (with the same cluster assignments and error) was found across the performed runs. If initialid is provided, the algorithm runs once and nfound will typically be 1 (deterministic). When multiple random restarts are used (initialid is None and npass > 1), nfound indicates how many of those runs converged to the returned optimal solution.
    
    Raises:
        ValueError: If the distance argument is malformed (wrong shape, inconsistent sizes, or not representing a valid lower-triangular/condensed/full distance matrix), if nclusters is not a positive integer or exceeds the number of items, or if initialid is incompatible with the number of items. Other implementation-level errors may be raised by the underlying clustering routine if input validation fails.
    
    Notes:
        The function validates and normalizes the distance input using the module's internal checks before invoking the core clustering routine. When reproducible, deterministic behavior is required, supply initialid; otherwise use npass > 1 to reduce sensitivity to initialization. The returned clusterid uses medoid indices (item indices) as cluster identifiers, matching the semantics expected by downstream Biopython utilities that operate on clustered biological data.
    """
    from Bio.Cluster import kmedoids
    return kmedoids(distance, nclusters, npass, initialid)


################################################################################
# Source: Bio.Cluster.pca
# File: Bio/Cluster/__init__.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_Cluster_pca(data: numpy.ndarray):
    """Bio.Cluster.pca performs principal component analysis (PCA) on a two-dimensional numeric data matrix.
    
    Args:
        data (numpy.ndarray): nrows x ncolumns array containing the data values. In the Biopython/biological data analysis context this represents observations (rows) such as experimental samples, sequences, or measurements, and variables/features (columns) such as gene expression levels, alignment-derived scores, or other numeric descriptors. The array must be a two-dimensional numpy.ndarray of numeric values (floating-point convertible). The function internally mean-centers the columns prior to computing principal components. The practical role of this argument is to provide the raw data from which the function computes the column means, principal component axes, projections of each observation onto those axes, and the associated variances (eigenvalues).
    
    Returns:
        tuple: A 4-tuple of numpy.ndarray objects describing the PCA decomposition, returned in the following order.
            columnmean (numpy.ndarray): 1-D array of length ncolumns containing the mean of each column of the input data. These column means are the centering offsets used by the algorithm; adding this vector (broadcast across rows) back to the projected reconstruction restores the original scale of the data.
            coordinates (numpy.ndarray): 2-D array of shape (nrows, nmin) giving the coordinates (projections) of each row (observation) onto the principal components. Each row corresponds to the projection of the corresponding input data row onto the reduced PCA basis. nmin is min(nrows, ncolumns).
            pc (numpy.ndarray): 2-D array of shape (nmin, ncolumns) containing the principal component vectors (component axes). Each row is a principal component expressed in the original column/feature space. The principal components are ordered by descending associated eigenvalue magnitude (largest first).
            eigenvalues (numpy.ndarray): 1-D array of length nmin containing the eigenvalues associated with each principal component, sorted in descending order. These eigenvalues quantify the variance explained by the corresponding principal components.
        Practical significance: In computational molecular biology workflows (as supported by Biopython), these return values allow dimensionality reduction (using a subset of the top components), visualization of samples in reduced space (using coordinates), assessment of variance explained (using eigenvalues), and accurate reconstruction of the original data using columnmean + dot(coordinates, pc) within numerical precision.
    
    Behavior and notes:
        - nrows and ncolumns refer to data.shape, and nmin is the smaller of nrows and ncolumns; all returned arrays use nmin in their reduced dimension.
        - The principal components, coordinates, and eigenvalues are sorted by the magnitude of the eigenvalue so that the first component explains the largest portion of the variance.
        - The reconstruction relation holds (within floating point tolerance): data ≈ columnmean + dot(coordinates, pc), where columnmean is broadcast across rows.
        - The function delegates low-level computation to an internal routine after validating the input array; the input is mean-centered per column as part of the PCA computation.
    
    Failure modes and exceptions:
        - TypeError: may be raised if the provided data is not a numpy.ndarray.
        - ValueError: may be raised if data is not two-dimensional, has zero size in one dimension, or contains non-finite values (NaN or Inf) that prevent a meaningful PCA decomposition.
        - Numerical issues: if the input matrix is singular or has extremely ill-conditioned covariance structure, some principal components or eigenvalues may be numerically unstable; callers should inspect eigenvalues to assess usable component count.
    
    Side effects:
        - No external side effects (files, global state) are produced. The function returns newly allocated numpy.ndarray objects; it does not modify the input array in-place as observed by callers.
    
    Domain usage example (conceptual): Use Bio.Cluster.pca to reduce dimensionality of a gene expression matrix prior to clustering or visualization: compute the top k principal components (rows 0..k-1 of pc and corresponding columns of coordinates) and use coordinates[:, :k] as input to a clustering algorithm or a scatter plot to explore sample relationships.
    """
    from Bio.Cluster import pca
    return pca(data)


################################################################################
# Source: Bio.Cluster.somcluster
# File: Bio/Cluster/__init__.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_Cluster_somcluster(
    data: numpy.ndarray,
    mask: numpy.ndarray = None,
    weight: numpy.ndarray = None,
    transpose: bool = False,
    nxgrid: int = 2,
    nygrid: int = 1,
    inittau: float = 0.02,
    niter: int = 1,
    dist: str = "e"
):
    """Calculate a Self-Organizing Map (SOM) on a rectangular grid and return the cluster assignments and cluster centroids.
    
    Args:
        data (numpy.ndarray): Two-dimensional numeric array containing the data values to cluster. For typical use in computational molecular biology (e.g., clustering gene expression profiles, microarray columns/rows, or other high-dimensional experimental measurements), data is an nrows x ncolumns array: rows conventionally represent items (e.g., genes, samples) and columns represent measured features. This function will validate and may convert the input via internal checks; the input array itself is treated as read-only by callers, but an internal copy or view may be created for computation.
        mask (numpy.ndarray or None): Optional two-dimensional integer array with the same shape as data indicating missing values. A mask element equal to 0 signals that the corresponding data element is missing and should be ignored during distance and centroid calculations. If None (the default), no missing entries are indicated. The function will validate the mask shape against the data and may raise an error if incompatible.
        weight (numpy.ndarray or None): Optional one-dimensional numeric array of length equal to the number of features used when calculating distances (the number of columns when clustering rows, or the number of rows when clustering columns). These weights scale the contribution of each feature to the distance metric. If None (the default), all features are treated equally. The function validates the length of weight against the data dimensionality.
        transpose (bool): If False (default), the function clusters rows of data (each row is an item to be assigned to a SOM cell). If True, the function clusters columns of data (each column is an item). This flag determines how the input array is interpreted and therefore the shapes of outputs: when clustering rows, the number of features used for distance computations is the number of columns; when clustering columns, it is the number of rows.
        nxgrid (int): Horizontal dimension of the rectangular SOM map (number of cells along the x axis). Must be a positive integer; the function raises ValueError if nxgrid < 1. Default is 2. In practice this controls the granularity of the 2D map used to organize items.
        nygrid (int): Vertical dimension of the rectangular SOM map (number of cells along the y axis). Must be a positive integer; the function raises ValueError if nygrid < 1. Default is 1. Together nxgrid and nygrid define the total number of map cells (nxgrid * nygrid) used to represent clusters.
        inittau (float): Initial value of the neighborhood function parameter tau which determines how strongly nearby SOM cells influence each other's centroids during training. The parameter is a positive floating point scalar; default is 0.02. Smaller or larger values change the smoothing/neighborhood size during the iterative training and therefore affect topology preservation.
        niter (int): Number of iterations of the SOM training algorithm to perform. Each iteration refines the centroids across the grid; higher values increase computation time but may improve map organization. Default is 1. The routine performs the specified number of iterations using the chosen distance metric and neighborhood schedule starting from inittau.
        dist (str): One-character code selecting the distance function used to compare items and centroids. Accepted values and their meanings are: 'e' for Euclidean distance; 'b' for City Block (Manhattan) distance; 'c' for Pearson correlation (centered correlation); 'a' for the absolute value of the Pearson correlation; 'u' for uncentered correlation; 'x' for the absolute value of the uncentered correlation; 's' for Spearman's rank correlation; 'k' for Kendall's tau. The string must match one of these documented codes; invalid codes will result in an error in the underlying implementation.
    
    Returns:
        tuple: A two-element tuple (clusterids, celldata).
            clusterids (numpy.ndarray): Integer array of shape (nitems, 2) and dtype equivalent to "intc" where nitems is the number of items being clustered (rows if transpose is False, columns if transpose is True). Each row contains the x and y integer coordinates (0-based) of the SOM grid cell to which the corresponding item was assigned. This mapping is the primary clustering output used to group items by their assigned grid cell for downstream analysis or visualization.
            celldata (numpy.ndarray): Floating-point array with shape (nxgrid, nygrid, nfeatures) where nfeatures equals the number of features used for distance calculations (the number of columns when rows are clustered, or the number of rows when columns are clustered). Each element celldata[ix, iy] is a 1-D vector (of length nfeatures) containing the centroid (prototype) values for the SOM cell at coordinates [ix, iy]. These centroids summarize the typical feature values for items assigned to each map cell and are commonly used for interpreting cluster profiles in bioinformatics analyses.
    
    Behavior, side effects, defaults, and failure modes:
        The function performs internal validation of data, mask, and weight shapes and types and may convert inputs to numeric numpy arrays suitable for computation. It raises ValueError if nxgrid or nygrid are not positive integers. Shape mismatches between data and mask or weight will raise errors during validation. The implementation delegates the computationally intensive work to an internal (compiled) routine; therefore runtime and memory usage scale with the size of data, the number of iterations (niter), and the grid dimensions (nxgrid * nygrid). Typical use in computational molecular biology includes organizing high-dimensional experimental data (for example, gene expression measurements across samples) into a 2D topology-preserving map for visualization, cluster assignment, and centroid-based downstream interpretation. The function returns the cluster assignment and centroids and does not write to disk; it does not modify caller-held input arrays in-place (but internal conversions or copies may occur).
    """
    from Bio.Cluster import somcluster
    return somcluster(
        data,
        mask,
        weight,
        transpose,
        nxgrid,
        nygrid,
        inittau,
        niter,
        dist
    )


################################################################################
# Source: Bio.Cluster.treecluster
# File: Bio/Cluster/__init__.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_Cluster_treecluster(
    data: numpy.ndarray,
    mask: numpy.ndarray = None,
    weight: numpy.ndarray = None,
    transpose: bool = False,
    method: str = "m",
    dist: str = "e",
    distancematrix: numpy.ndarray = None
):
    """Perform hierarchical (agglomerative) clustering and return a Tree object that describes the resulting dendrogram. This function implements pairwise single, complete (maximum), centroid, and average linkage hierarchical clustering and is provided by Biopython for clustering numerical biological data (for example gene expression matrices or other feature matrices used in computational molecular biology). The function can compute distances from raw data or accept a precomputed distance matrix; it validates inputs and raises errors for invalid argument combinations.
    
    Args:
        data (numpy.ndarray): nrows x ncolumns numeric array containing the data values used to compute pairwise distances when a distance matrix is not supplied. In the Biopython domain this typically represents measurements such as expression levels (rows as samples and columns as features or vice versa). If transpose is False (the default) clustering is done over the columns (features); if transpose is True clustering is done over the rows (samples). If data is provided it is validated (via internal checks) for shape and type and may raise ValueError on invalid input.
        mask (numpy.ndarray): nrows x ncolumns integer array indicating missing data positions in data. A mask element equal to 0 indicates that the corresponding data element is missing and should be ignored when computing distances. This argument is meaningful only when data is provided and will be ignored if a distancematrix is supplied, in which case providing mask raises a ValueError. The mask is used to support real biological datasets that contain missing measurements.
        weight (numpy.ndarray): 1D array of weights applied when calculating distances from data. The length and meaning of the weights are validated against the data shape; weights are ignored and cause a ValueError if supplied together with distancematrix. Weights allow certain rows or columns (depending on transpose) to contribute more or less to the pairwise distance computation, which is useful when some features are more reliable or biologically relevant.
        transpose (bool): If False (default) cluster the columns of data when data is provided; if True cluster the rows of data. This flag controls whether items to be clustered are data columns or data rows, matching common biological use cases where either genes (rows) or samples (columns) are clustered.
        method (str): Linkage method to form clusters. Allowed values are 's' for single linkage, 'm' for complete (maximum) linkage (default), 'c' for centroid linkage, and 'a' for average linkage. Note that centroid linkage ('c') can be performed only when data is provided (not from a distancematrix).
        dist (str): Specifies the distance (or similarity) measure used when computing pairwise distances from data. Allowed codes are 'e' for Euclidean distance (default), 'b' for City Block (Manhattan) distance, 'c' for Pearson correlation, 'a' for absolute value of Pearson correlation, 'u' for uncentered correlation, 'x' for absolute uncentered correlation, 's' for Spearman rank correlation, and 'k' for Kendall's tau. These options allow clustering to reflect different biologically meaningful relationships between samples or features.
        distancematrix (numpy.ndarray): An alternative to supplying data: the pairwise distance matrix between items to cluster. There are three acceptable representations for the distance matrix as allowed by the original implementation: a full 2D NumPy array (only the lower-left triangle is accessed), a 1D NumPy array containing the distances consecutively in the order expected by the algorithm, or a list of 1D arrays representing the rows of the lower-triangular distance matrix. If distancematrix is provided, data must be None. Supplying mask or weight together with distancematrix raises ValueError because mask and weight apply only when distances are computed from raw data. Be aware that the clustering routine may reorder or shuffle values inside the provided distance matrix during computation; save a copy beforehand if you need to reuse the original matrix later.
    
    Returns:
        Tree: A Tree object that encodes the hierarchical clustering result (the dendrogram). The returned Tree contains the hierarchical relationships produced by the specified linkage method and distance measure and can be used by other Biopython routines to traverse, visualize, or export the clustering result. If no Tree can be produced due to invalid arguments, the function raises ValueError before returning.
    
    Behavior and failure modes:
        Either data or distancematrix must be supplied but not both. If both data and distancematrix are None, a ValueError is raised. If both are provided, a ValueError is raised. If distancematrix is supplied together with mask or weight, a ValueError is raised because mask and weight are ignored in that mode. Centroid linkage cannot be computed from a distancematrix; attempting to use centroid linkage without providing data will result in an error. Inputs are validated by internal helper routines (__check_data, __check_mask, __check_weight, __check_distancematrix) which may raise ValueError on mismatched shapes, invalid types, or other inconsistencies. The routine performs in-memory computations and returns a Tree object; it does not persist files or modify external state beyond returning the Tree, but it may reorder the provided distance matrix in-place during processing.
    """
    from Bio.Cluster import treecluster
    return treecluster(data, mask, weight, transpose, method, dist, distancematrix)


################################################################################
# Source: Bio.Data.CodonTable.list_ambiguous_codons
# File: Bio/Data/CodonTable.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_Data_CodonTable_list_ambiguous_codons(
    codons: list,
    ambiguous_nucleotide_values: dict
):
    """Bio.Data.CodonTable.list_ambiguous_codons: Extend a list of codon strings by adding unambiguous and ambiguous codon codes that represent only the provided codons, used for example when constructing codon tables in Biopython (e.g., to extend a list of stop codons with ambiguity codes like 'TAR' or 'URA').
    
    This function is used in the Biopython codon table utilities to compute ambiguous codon symbols that are safe to add to an existing set of codons (for example, stop codons). It examines the nucleotides observed at each of the three codon positions in the provided codons and identifies ambiguous nucleotide letters (from ambiguous_nucleotide_values) whose concrete nucleotide meanings are fully represented at that position across the input codons. It then forms candidate ambiguous codons by combining such letters for the three positions and retains only those candidates whose full expansion (all concrete codons implied by the ambiguous letters) are present in the input codons. The function returns a new list containing the original codons (in their original order) followed by any added ambiguous codons (in a deterministic order produced by the algorithm). The function does not modify the input codons list or the ambiguous_nucleotide_values mapping.
    
    Args:
        codons (list): A list of codon strings to extend. Each element is expected to be a three-character nucleotide triplet (DNA or RNA letters such as 'A', 'C', 'G', 'T' or 'U'), for example ['TAG', 'TAA'] or ['UAG', 'UGA']. In Biopython this argument is typically a set of codons with a specific biological meaning (for example, stop codons). The function uses these codons as the authoritative set: ambiguous codons will only be added if every concrete codon they represent is already present in this list. The input list is not modified; it is copied into the returned list.
        ambiguous_nucleotide_values (dict): A mapping from single-letter ambiguous nucleotide codes to the string (or sequence) of concrete nucleotides they represent, for example {'R': 'AG', 'Y': 'CT'} (or using 'U' for RNA). This mapping is used to (1) determine which ambiguous letters cover all observed nucleotides at each codon position, and (2) expand candidate ambiguous codons into their concrete codons for validation. Keys must be single-character strings corresponding to ambiguity codes used in the candidates; values must be iterable strings of concrete nucleotide characters. If a required ambiguity code is missing from this mapping, a KeyError will be raised during expansion.
    
    Returns:
        list: A new list of codon strings containing the original codons (copied) followed by any ambiguous codons that were determined to be valid additions. Valid ambiguous codons are those whose per-position ambiguous letters cover only nucleotides already observed at that position in the input codons and whose full expansion (all concrete codons implied by the ambiguous letters, as given by ambiguous_nucleotide_values) are all present in the input codons. The order of appended ambiguous codons is deterministic: the algorithm sorts candidate letters per position and iterates in nested loops to preserve a stable order. No in-place modification of the input codons or ambiguous_nucleotide_values occurs.
    
    Behavior, defaults, and failure modes: The function constructs per-position lists of ambiguous letters whose meaning sets are subsets of the concrete nucleotides observed in codons at that position. It then forms Cartesian-product candidates from these per-position lists and filters out any candidate already present in the input codons. For each remaining candidate ambiguous codon, it expands the candidate using ambiguous_nucleotide_values and checks that every expanded concrete codon exists in the input codons; only then is the candidate appended to the result. If codons contain strings that are not length three or include characters not represented in ambiguous_nucleotide_values or not expected as nucleotide letters, the behavior is undefined and may result in incorrect candidates or exceptions (for example, KeyError when expanding a candidate whose ambiguous letter is not present in ambiguous_nucleotide_values). The function preserves duplicate entries from the original codons list (it copies codons[:] into the returned list) and only prevents adding ambiguous candidates that would duplicate existing entries.
    """
    from Bio.Data.CodonTable import list_ambiguous_codons
    return list_ambiguous_codons(codons, ambiguous_nucleotide_values)


################################################################################
# Source: Bio.Data.CodonTable.list_possible_proteins
# File: Bio/Data/CodonTable.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_Data_CodonTable_list_possible_proteins(
    codon: str,
    forward_table: dict,
    ambiguous_nucleotide_values: dict
):
    """Bio.Data.CodonTable.list_possible_proteins returns the set of amino acid residues that a possibly ambiguous three-nucleotide codon can encode, or raises an exception if the codon can encode stop codons (termination) in addition to proteins, or if it is a definitive stop codon. This function is used in the Biopython project for translating nucleotide triplets (codons) into amino acids when working with DNA/RNA sequences that may contain IUPAC ambiguous nucleotide codes; it helps downstream code decide how to handle ambiguous translation outcomes in computational molecular biology workflows (for example, variant interpretation, consensus sequence handling, or translating sequencing reads with ambiguous bases).
    
    Args:
        codon (str): A three-character nucleotide string representing a codon. Each character is expected to be a single IUPAC nucleotide symbol (for example 'A', 'T', 'G', 'C' or an ambiguous code such as 'R', 'Y', etc.). The function unpacks this string into three positions (c1, c2, c3), so a string of length other than three will raise a ValueError at unpacking. The codon is the unit of translation: the function explores all concrete codons compatible with this ambiguous codon to determine possible encoded amino acids.
        forward_table (dict): A mapping from concrete codon strings (three-letter strings such as 'ATG') to the encoded amino acid symbol (typically a one-character string, e.g. 'M' for methionine). The forward_table is treated as the set of sense codons: a missing key for a concrete codon is interpreted as that concrete codon being a stop (termination) codon for the purposes of this routine. The mapping is consulted for each concrete codon generated from the ambiguous positions to determine which amino acid (if any) it encodes.
        ambiguous_nucleotide_values (dict): A mapping from a single-character nucleotide symbol (including ambiguous IUPAC codes) to an iterable (such as a list or tuple) of concrete single-character nucleotide strings that the symbol can represent. For example, an entry for 'R' might map to ['A', 'G']. This dictionary is used to expand each character position of the input codon into all possible concrete nucleotides, enabling enumeration of all concrete codons compatible with the ambiguous input.
    
    Returns:
        list: A list of unique amino acid symbols (strings) that the ambiguous codon can encode, obtained by looking up each concrete codon expansion in forward_table and collecting the mapped amino acids. The list contains each amino acid at most once. No particular ordering of the amino acids should be relied upon by callers.
    
    Raises:
        ValueError: If codon does not contain exactly three characters, the unpacking "c1, c2, c3 = codon" in the implementation will raise a ValueError. This indicates incorrect input length rather than a biological translation issue.
        KeyError: Raised in two distinct situations. If any character of codon is not a key in ambiguous_nucleotide_values, a KeyError for that nucleotide will propagate. If all concrete codons generated from the ambiguous input are interpreted as stop codons (i.e. none of them are present in forward_table), the function raises KeyError(codon) to signal that the ambiguous codon is a definite stop codon in the current coding table.
        TranslationError: If some concrete expansions of the ambiguous codon map to amino acids (present in forward_table) and other expansions map to stops (missing from forward_table), the function raises TranslationError with a message indicating that the ambiguous codon codes for both proteins and stop codons. This signals a mixed translation outcome that the caller must resolve.
    
    Behavior and side effects:
        The function expands the input codon by iterating over the Cartesian product of the concrete nucleotide sets for each of the three positions obtained from ambiguous_nucleotide_values. For each concrete three-nucleotide sequence, it attempts to look up the encoded amino acid in forward_table. If an amino acid is found, it is collected (duplicates are deduplicated). If a concrete codon is not found in forward_table, it is treated as a stop codon and recorded. If any stops are encountered together with any protein-coding results, the function raises TranslationError to avoid silently mixing termination and sense translations. If only stops are encountered, a KeyError containing the original ambiguous codon is raised to indicate a true stop. The function has no other side effects (it does not modify its inputs or external state).
    """
    from Bio.Data.CodonTable import list_possible_proteins
    return list_possible_proteins(codon, forward_table, ambiguous_nucleotide_values)


################################################################################
# Source: Bio.Data.CodonTable.make_back_table
# File: Bio/Data/CodonTable.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_Data_CodonTable_make_back_table(table: dict, default_stop_codon: str):
    """Back a back-table (naive single-codon mapping) for a codon-to-amino-acid table.
    
    This function constructs a "back table" that maps each amino acid (the values from a forward codon table) to a single representative codon (one of the keys from the forward table). It is intended for simple, deterministic back-translation or for producing an example nucleotide codon for each amino acid when using Bio.Data.CodonTable codon tables in Biopython (computational molecular biology and bioinformatics workflows). The function is intentionally naive: when an amino acid is encoded by multiple codons in the input table, only one codon is returned per amino acid. The selection is deterministic and based on the sorted order of the codon keys as iterated by Python: the codon that appears last in the ascending sorted order of the codon strings is chosen. The function also ensures a mapping exists for the stop symbol by assigning the provided default_stop_codon to the None key in the returned mapping.
    
    Args:
        table (dict): A forward codon table mapping codon strings (keys) to amino-acid identifiers (values) used within Bio.Data.CodonTable. Typical codon keys are three-letter nucleotide strings (e.g., "ATG") and typical values are single-letter amino-acid strings (e.g., "M") or None for stop codons. The function does not validate codon formatting; it sorts the dictionary keys to produce a deterministic choice when multiple codons map to the same amino acid. If the keys in table are not mutually orderable (so that sorting them raises a TypeError), the call will propagate that exception.
        default_stop_codon (str): A codon string to use as the representative stop codon in the returned back table. This value is assigned to the None key in the result and will override any stop codon previously selected by iterating the sorted table keys. This parameter is required and is not type-checked by the function beyond being used as a dictionary value.
    
    Returns:
        dict: A mapping from amino-acid identifiers (the values taken from the input table, including None for stop) to a single representative codon string (one of the keys from the input table). For amino acids encoded by multiple codons in the input, the representative codon is the one that sorts last in ascending order among that amino acid's codons. The returned mapping always contains a None key mapped to default_stop_codon; the function has no other side effects and does not modify the input table.
    """
    from Bio.Data.CodonTable import make_back_table
    return make_back_table(table, default_stop_codon)


################################################################################
# Source: Bio.Data.CodonTable.register_ncbi_table
# File: Bio/Data/CodonTable.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_Data_CodonTable_register_ncbi_table(
    name: str,
    alt_name: str,
    id: int,
    table: dict,
    start_codons: list,
    stop_codons: list
):
    """Bio.Data.CodonTable.register_ncbi_table registers a single NCBI-style codon table definition with the CodonTable module by converting the provided raw table data into multiple internal codon table objects used throughout Biopython for DNA/RNA translation and ambiguous-base handling. This function is intended for internal (PRIVATE) use when populating the module-level codon table registries used by sequence translation and related utilities in computational molecular biology workflows.
    
    Args:
        name (str): The primary name string for the codon table as provided by NCBI. This string may contain multiple synonymous names separated by "; ", ", " or the word " and ". The function parses this string into a list of individual names (splitting on "; ", converting " and " to "; ", and trimming whitespace) and uses those names when registering the created codon table objects in the module-level name-to-table mappings. The parsed names are important for lookups by human-readable table names in Biopython translation functions.
        alt_name (str): An alternate or legacy single name for the codon table (may be None). If not None, this alternate name is appended to the parsed names list and also used when registering the table objects. alt_name provides backward compatibility with older naming conventions and is stored with the generated objects and registries.
        id (int): The numeric NCBI identifier for the codon table (for example, 1 for the standard code). This integer is used as the key when storing the generated table objects in the module-level id-indexed registries (e.g., unambiguous_dna_by_id, unambiguous_rna_by_id, generic_by_id, ambiguous_*_by_id). If id equals 1, the function also sets the module-level globals standard_dna_table and standard_rna_table to the created DNA and RNA objects respectively. Care should be taken because registering a table with an id that already exists will overwrite the existing entries in these registries.
        table (dict): A mapping of codon strings to encoded amino acid values representing the unambiguous DNA codon-to-amino-acid assignments as provided by NCBI. Keys are expected to be codon strings using the letter "T" for thymine (e.g., "ATG", "TAA"); values are the corresponding amino acid symbol(s) or stop indicators as used by Biopython. The function uses this dictionary to build three variants of unambiguous lookup tables: a DNA-specific table (T-based), an RNA-specific table (T replaced by U), and a generic table that contains both T- and U-based codons for flexible lookup across DNA/RNA sequences.
        start_codons (list): A list of codon strings (using "T" for thymine) that are considered valid translation initiation (start) codons for this genetic code. These are used to populate corresponding start-codon lists for the DNA-specific, RNA-specific (with "T" replaced by "U"), and generic tables. The implementation takes care to add both T- and U-forms where appropriate but only adds the U-form when the original contains "T" to reduce accidental duplicates.
        stop_codons (list): A list of codon strings (using "T" for thymine) that are considered translation termination (stop) codons for this genetic code. These are used to populate the stop-codon lists for the DNA-specific, RNA-specific (with "T" replaced by "U"), and generic tables in the same manner as start_codons, with both T- and U-forms added where applicable.
    
    Behavior and side effects:
        This routine converts the supplied DNA-based codon table into multiple internal Biopython objects:
        - NCBICodonTableDNA: an unambiguous DNA-specific codon table object using "T".
        - AmbiguousCodonTable wrapping the DNA table: handles ambiguous DNA letters using IUPACData mappings.
        - NCBICodonTable (generic): contains both "T"- and "U"-variants of codons for lookups that should accept either nucleotide alphabet.
        - AmbiguousCodonTable wrapping the generic table: handles ambiguous nucleotides after merging ambiguous RNA values and mapping "T" to "U" where needed.
        - NCBICodonTableRNA: an unambiguous RNA-specific codon table object using "U".
        - AmbiguousCodonTable wrapping the RNA table: handles ambiguous RNA letters using IUPACData mappings.
        The function builds the RNA versions by replacing "T" with "U" in keys and in start/stop codon lists, and constructs merged ambiguous mappings so both DNA and RNA ambiguous codes are handled consistently.
    
        It registers these objects in module-level dictionaries so they can be retrieved by id or by any of the parsed names. Specifically it assigns entries (and may overwrite existing ones) in the following registries: unambiguous_dna_by_id, unambiguous_rna_by_id, generic_by_id, ambiguous_dna_by_id, ambiguous_rna_by_id, ambiguous_generic_by_id, and the corresponding name-indexed registries unambiguous_dna_by_name, unambiguous_rna_by_name, generic_by_name, ambiguous_dna_by_name, ambiguous_rna_by_name, ambiguous_generic_by_name. If id == 1, it also sets the module globals standard_dna_table and standard_rna_table to the created DNA and RNA objects. These side effects make the newly created tables available to Biopython translation and sequence utilities.
    
    Defaults and data handling details:
        - Name parsing: the primary name string is split on separators to derive multiple names; " and " is converted to "; " prior to splitting, and commas followed by space are treated as separators as well. The resulting names list has whitespace trimmed.
        - RNA generation: every codon in the provided table is added to the generic table in both its original form and, if it contains a "T", with "T" replaced by "U". The RNA table contains only U-based codons. Start and stop codon lists are similarly expanded to include both T- and U-based forms as applicable.
        - Ambiguous mappings: ambiguous codon tables are created using IUPAC ambiguous DNA/RNA letter sets and associated value mappings provided by IUPACData; for the generic ambiguous table, the function merges mappings and explicitly maps "T" to "U" to ensure compatibility between DNA and RNA ambiguous representations.
    
    Failure modes and constraints:
        - The function expects table keys and codons (in table, start_codons, stop_codons) to be strings supporting .replace() and consistent use of uppercase nucleotide letters. If non-string keys or entries are provided, a TypeError or AttributeError may be raised when attempting string operations.
        - Registering a table with an id or name already present in the module registries will overwrite the previously stored objects without warning; callers should avoid duplicate registrations unless intentional.
        - The function is internal to Biopython (PRIVATE); its behavior, including the exact structure of the created objects and the registries it modifies, is intended for use by the CodonTable module and may change across releases. For external use, higher-level APIs in Biopython should be preferred.
    
    Returns:
        None: This function does not return a value. Instead, it creates multiple codon table objects and registers them in module-level dictionaries and globals so that other Biopython components (e.g., sequence translation functions) can look up codon tables by NCBI id or by any of the parsed names.
    """
    from Bio.Data.CodonTable import register_ncbi_table
    return register_ncbi_table(name, alt_name, id, table, start_codons, stop_codons)


################################################################################
# Source: Bio.Entrez.efetch
# File: Bio/Entrez/__init__.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_Entrez_efetch(db: str, **keywords):
    """Fetch records from the NCBI Entrez EFetch utility and return a file-like handle to the results.
    
    This function is used within the Bio.Entrez module of Biopython to retrieve records from NCBI Entrez databases (for example "nucleotide", "protein", "pubmed") in the format requested by the caller. It constructs an HTTP request to the EFetch endpoint (https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi) and returns a handle (an open, file-like object) from which the caller can read the response. Typical use in molecular biology and bioinformatics workflows is to download sequence records, GenBank entries, PubMed abstracts, or other Entrez-stored data for downstream parsing or analysis by Biopython modules. See the NCBI EFetch documentation for parameter meanings and database-specific behaviour: http://www.ncbi.nlm.nih.gov/books/NBK25499/#chapter4.EFetch
    
    Args:
        db (str): Name of the Entrez database to query. This identifies which NCBI database to fetch records from (for example "nucleotide", "protein", "pubmed"). The value is passed directly to the Entrez EFetch "db" parameter and determines the kinds of records and formats available via other keyword parameters (such as "rettype" and "retmode"). Selecting the correct db is essential for retrieving the expected record types in computational molecular biology pipelines.
        keywords (dict): Mapping of additional Entrez EFetch parameters to their values, provided as keyword arguments (for example "id" for one or more record identifiers, "rettype" to request a specific return format such as "gb" for GenBank, and "retmode" to request "text" or "xml"). These are sent to the NCBI EFetch endpoint and control which records are returned and in what format. If the supplied identifiers exceed NCBI's recommended limits (over 200 identifiers), this function will automatically switch to using an HTTP POST rather than GET, as recommended by NCBI, to avoid URL length limits. Typical keys and their interpretation are documented by NCBI; callers should follow NCBI usage policies (for example setting Entrez.email) and be aware that database defaults can change (see note below about retmode).
    
    Returns:
        handle (file-like object): An open, file-like handle to the HTTP response containing the requested records. The handle behaves like a readable stream (for example you can call readline(), read(), or iterate over it) and should be explicitly closed by the caller (handle.close()) when no longer needed to release network resources. The response may be large and is streamed; do not assume the entire content is loaded into memory.
    
    Raises:
        urllib.error.URLError: If there is a network-level error while contacting the NCBI EFetch service (for example DNS failure, connection refused, or timeout). Callers should catch this exception to implement retries, backoff, or to provide a clear error message in automated workflows.
    
    Behavior, side effects, defaults, and failure modes:
        - Network request: This function performs a network call to NCBI's EFetch service and therefore depends on internet availability and NCBI service uptime. It will block until the request completes or a network error occurs.
        - HTTP method selection: If the combined parameters (notably the "id" parameter when listing identifiers) would produce a very long query string (more than approximately 200 identifiers), the function automatically uses HTTP POST instead of GET to comply with NCBI recommendations and avoid URL length limits.
        - Response format: The format of the returned data depends on the database and the "rettype" and "retmode" keywords. Note that NCBI changed the default "retmode" in February 2012; many databases that previously returned plain text may now return XML by default. Specify "retmode" explicitly if your downstream code expects a particular format.
        - Resource management: The returned handle must be closed by the caller to free network and file resources. The handle should be treated as a streaming reader; for large downloads read incrementally rather than loading all data into memory.
        - Internal request construction: The function builds the request to the efetch.fcgi CGI endpoint and sends it using the module's internal request builder and opener; callers need not construct the URL themselves but must ensure keyword parameters follow the EFetch parameter conventions.
        - Usage policy: When using Entrez utilities in automated or large-scale analyses, follow NCBI usage policies (for example set Entrez.email to a valid contact and respect rate limits). Failure to follow policies may lead to request throttling or blocking by NCBI.
        - Error responses from NCBI (for example malformed parameters or invalid identifiers) are returned as part of the HTTP response body; the function does not convert HTTP error content into Python exceptions beyond network-level URLError. Callers should inspect the contents of the handle for service-provided error messages.
    """
    from Bio.Entrez import efetch
    return efetch(db, **keywords)


################################################################################
# Source: Bio.Entrez.epost
# File: Bio/Entrez/__init__.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_Entrez_epost(db: str, **keywds):
    """Bio.Entrez.epost posts a file or list of NCBI unique identifiers (UIDs) to the NCBI E-utilities EPost endpoint so they can be stored on the NCBI side (in a WebEnv) for use by subsequent Entrez queries (for example, efetch, esummary or elink). This function is part of the Bio.Entrez module in Biopython and is used to programmatically submit identifier lists when building multi-step Entrez workflows in computational molecular biology and bioinformatics. It constructs an HTTP POST request to the EPost CGI endpoint used by NCBI and returns a handle to the HTTP response for downstream parsing.
    
    Args:
        db (str): The NCBI database name to which the identifiers apply (for example "pubmed", "nucleotide", "protein"). This selects which Entrez database will store the posted identifiers and is sent as the "db" CGI parameter to the NCBI EPost service. This parameter is required by the NCBI EPost API and must match one of the supported Entrez databases.
        keywds (dict): Additional keyword parameters to include in the EPost request. Typical usage is to include the identifier list using the "id" parameter (a comma-separated list of UIDs or other formats accepted by NCBI EPost) or other EPost-specific parameters described in the NCBI EPost documentation (for example query-key related parameters returned by EPost). The entries of this mapping are added as CGI variables to the POST request; keys and values should be strings as expected by the NCBI E-utilities. No implicit defaults are supplied by this function — you must provide the appropriate EPost parameters required by NCBI (for example, an "id" parameter containing the UIDs).
    
    Behavior and side effects:
        This function builds an HTTP POST request to "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/epost.fcgi" and sends the provided database name and additional parameters to NCBI. On success, NCBI creates a temporary server-side result set (a WebEnv) that can be referenced in subsequent Entrez calls. The function itself does not parse the response; it returns a file-like handle (an HTTP response handle) which contains the raw server response stream. It is the caller's responsibility to read from and close the returned handle, and to parse the response (for example using Bio.Entrez.read or other XML/text parsing) to extract WebEnv, QueryKey, or error information. Because this performs network I/O, it may take time and depends on network and NCBI availability.
    
    Failure modes and exceptions:
        On network connectivity problems this function raises urllib.error.URLError. Other exceptions raised by the underlying HTTP or I/O libraries may also be propagated to the caller. If NCBI returns an error response, that response is provided via the returned handle and should be inspected by the caller; this function does not raise on HTTP error status codes by itself other than network-level URLError.
    
    Returns:
        handle: A file-like HTTP response handle to the raw results returned by the NCBI EPost service. The handle provides the response body (typically XML or plain text) that callers must read and parse to extract the WebEnv/QueryKey or error messages. Close the handle after use to release network resources.
    """
    from Bio.Entrez import epost
    return epost(db, **keywds)


################################################################################
# Source: Bio.Entrez.esearch
# File: Bio/Entrez/__init__.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_Entrez_esearch(db: str, term: str, **keywds):
    """Run an NCBI Entrez ESearch query and return a handle to the XML results.
    
    This function issues a request to the NCBI E-utilities ESearch endpoint to locate primary identifiers and term translations for records in a specified Entrez database. The results are intended for downstream use with other Entrez utilities (for example EFetch, ELink and ESummary) or for parsing with Bio.Entrez.read. The function builds an HTTP(S) request to the ESearch CGI (https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi), encoding the provided db and term together with any additional keyword parameters supplied via keywds into the query. The returned object is a handle to the raw XML response; callers should parse the XML (for example with Bio.Entrez.read) and close the handle when finished to release network resources.
    
    Args:
        db (str): The Entrez database to search, passed as the "db" parameter to NCBI ESearch. Typical values include domain-specific names such as "nucleotide", "protein", "pubmed", etc. In the context of computational molecular biology (the Biopython project), db selects which curated NCBI database will be queried for primary IDs that can later be retrieved or linked to with EFetch, ELink, or summarized with ESummary.
        term (str): The search term string to send to ESearch, passed as the "term" parameter to NCBI. This is the user-visible query expression (which can include field tags, boolean operators, date limits, organism qualifiers, etc.) used to match records in the chosen database. The term determines which primary IDs are returned and therefore directly controls the biological or literature subset retrieved (for example a gene name limited to an organism or a publication date range).
        keywds (dict): Additional keyword parameters to forward to the ESearch CGI. These correspond to documented ESearch parameters (for example "retmax" to limit the number of IDs returned, "retstart" for paging, "usehistory" to retain results on NCBI for later EFetch/ELink requests, "idtype" as shown in the example, and others documented at the NCBI E-utilities online documentation). The function creates an initial variables mapping with "db" and "term" and then updates it with keywds; therefore any keys in keywds that duplicate "db" or "term" will override the corresponding positional argument values.
    
    Behavior and side effects:
        This function performs a network request to the NCBI ESearch service and returns a handle to the XML response. The response is always in XML format. If "usehistory" is set in keywds, the ESearch response may include WebEnv and QueryKey values enabling server-side retention of the search results for subsequent requests; this has the side effect of creating a transient record on NCBI servers. Callers are responsible for parsing the XML (for example with Bio.Entrez.read) and for closing the returned handle to free network resources. Because the function transmits queries to a remote service, request rate limits and NCBI usage policies may apply; it is recommended to include identifying information such as Entrez.email in the Entrez module configuration when using the service.
    
    Failure modes:
        Network-related errors (for example DNS failures, connection timeouts, or HTTP errors returned by the server) will result in exceptions raised by the underlying network layer (urllib.error.URLError or its subclasses). A malformed query, unexpected changes in the NCBI response format, or temporary service outages may lead to parse errors when reading the returned XML with Bio.Entrez.read.
    
    Returns:
        Handle: A file-like handle to the HTTP response containing the ESearch results in XML format. The handle provides raw response data that can be parsed (for example with Bio.Entrez.read) and must be closed by the caller to release network resources.
    """
    from Bio.Entrez import esearch
    return esearch(db, term, **keywds)


################################################################################
# Source: Bio.ExPASy.get_prodoc_entry
# File: Bio/ExPASy/__init__.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_ExPASy_get_prodoc_entry(
    id: str,
    cgi: str = "https://prosite.expasy.org/cgi-bin/prosite/get-prodoc-entry"
):
    """Get a text handle to a PRODOC entry at the ExPASy PROSITE web service in HTML format.
    
    This function is used in computational molecular biology and bioinformatics workflows (see the Biopython project README) to retrieve the PRODOC documentation page for a given PROSITE entry identifier from the ExPASy web server. The returned handle provides the raw HTML text of the PRODOC entry, which can be read and stored or parsed to extract human-readable documentation about sequence motifs, patterns, and associated annotations curated in the PROSITE/PRODOC resource.
    
    Args:
        id (str): The PROSITE PRODOC identifier to fetch, for example "PDOC00001". This identifier is appended to the CGI endpoint as a query string and identifies the specific PRODOC record to retrieve from the remote PROSITE database. The value must be a string representing a valid PRODOC key known to the ExPASy service.
        cgi (str): The base CGI URL of the ExPASy PROSITE "get-prodoc-entry" endpoint. Default is "https://prosite.expasy.org/cgi-bin/prosite/get-prodoc-entry". This string is used verbatim and the function will construct the full request URL by appending "?{id}" to this CGI string. Override this parameter only to point to an alternative compatible CGI endpoint (for testing or mirrored services); do not change the format (it must accept the identifier as a query string).
    
    Returns:
        io.TextIOBase or file-like object: A readable text file-like handle to the HTML-formatted PRODOC entry. The handle supports .read() to obtain the HTML string for the record and is intended to be used as a context manager (with statement) so it is properly closed after use. The HTML returned contains the human-readable PRODOC documentation; for non-existing identifiers the ExPASy service typically returns an HTML page containing the phrase "There is currently no PROSITE entry for", so callers should check the returned text to detect missing entries.
    
    Behavior, side effects, defaults, and failure modes:
        The function performs network I/O by issuing an HTTP(S) request to the provided cgi endpoint and therefore depends on network connectivity and the remote ExPASy server being available. It does not write any local files itself; callers are responsible for saving data if persistence is required. The default cgi parameter points to the official ExPASy PROSITE service for production use. If the remote server is unreachable, the request may raise network-related exceptions from the underlying opener (for example connection or HTTP errors); such exceptions will propagate to the caller. Because the function returns a handle, callers should close it (preferably via a with statement) to free network resources. When running tests or automated workflows that must be offline, prefer to mock or avoid calls to this function (see Biopython README guidance about offline testing and network-dependent tests).
    """
    from Bio.ExPASy import get_prodoc_entry
    return get_prodoc_entry(id, cgi)


################################################################################
# Source: Bio.ExPASy.get_prosite_entry
# File: Bio/ExPASy/__init__.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_ExPASy_get_prosite_entry(
    id: str,
    cgi: str = "https://prosite.expasy.org/cgi-bin/prosite/get-prosite-entry"
):
    """Get a text handle to a PROSITE entry on the ExPASy PROSITE service, returned in HTML format.
    
    This function is part of Biopython's ExPASy utilities for computational molecular biology and is used to fetch PROSITE database entries (protein families, domains, and functional site annotations) over the web. It performs an HTTP GET request by constructing the URL f"{cgi}?{id}" (concatenating the provided CGI endpoint, a question mark, and the PROSITE entry identifier) and returns a text-mode file-like handle for the HTTP response body. The returned handle supports reading the HTML content (for example, via handle.read()) and can be used as a context manager in a with statement. Typical use is to request a PROSITE accession such as "PS00001" to retrieve the human- and machine-readable HTML documentation for that PROSITE pattern or profile, which can then be inspected, archived, or parsed for downstream sequence analysis or annotation tasks.
    
    Args:
        id (str): The PROSITE entry identifier to request from the ExPASy PROSITE service. This is the query string portion appended to the CGI endpoint (for example, "PS00001"). The function will form the request URL by appending '?' and this id to the cgi parameter exactly as provided, so callers should supply only the identifier (not a leading '?' or additional query parameters).
        cgi (str): The base ExPASy PROSITE CGI endpoint used to retrieve entries. Defaults to "https://prosite.expasy.org/cgi-bin/prosite/get-prosite-entry". Providing a different URL can be used to query mirrors or alternate endpoints; the function will append '?' and the id to this string to form the final request URL.
    
    Returns:
        object: A text-mode file-like handle (a readable, closeable object) for the HTML response returned by the ExPASy PROSITE service. The caller is responsible for closing the handle (or using it in a with statement). Reading from this handle returns the raw HTML text for the requested PROSITE entry. For a non-existing PROSITE id, ExPASy typically returns an HTML page containing the phrase "There is currently no PROSITE entry for", so callers should check the HTML content to distinguish a missing entry from a successful entry. This function performs a network request as a side effect; network-related problems (connection failures, DNS errors, HTTP errors from the underlying network library) will propagate as exceptions raised by the underlying open operation.
    """
    from Bio.ExPASy import get_prosite_entry
    return get_prosite_entry(id, cgi)


################################################################################
# Source: Bio.ExPASy.get_prosite_raw
# File: Bio/ExPASy/__init__.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_ExPASy_get_prosite_raw(id: str, cgi: str = None):
    """Get a text handle to a raw PROSITE or PRODOC record at the ExPASy web service.
    
    This function is used in Biopython to retrieve the plain-text PROSITE/PRODOC entry for a given PROSITE accession (for example "PS00001") from the ExPASy host (https://prosite.expasy.org/). The returned handle is a readable text file-like object that can be parsed by Biopython parsers such as Bio.ExPASy.Prosite.read to produce a Prosite record object used for motif, pattern, and protein family annotations in computational molecular biology workflows. The function performs an HTTP GET for the URL https://prosite.expasy.org/{id}.txt using the package-internal opener. Because the handle represents a network resource, prefer using it as a context manager (with statement) so it is closed automatically when no longer needed.
    
    Args:
        id (str): The PROSITE or PRODOC identifier to fetch, for example "PS00001". This value is formatted into the ExPASy PROSITE URL as {id}.txt. It is the primary lookup key and must match an existing PROSITE accession; otherwise the function will report the entry as not found.
        cgi (str): Deprecated and ignored. Historically used to select CGI-style access on the ExPASy site, but due to changes in the ExPASy website this argument is no longer used; pass None or omit it. The default value is None.
    
    Behavior and side effects:
        The function issues a network request to the ExPASy PROSITE service and returns a file-like text handle on success. The handle supports reading methods such as read() and readline() and typically has a .url attribute reflecting the final URL after any redirects. The caller is responsible for closing the handle when finished; using the handle in a with statement is recommended to ensure proper resource cleanup. The cgi parameter is accepted for backward compatibility but has no effect. Other than checking for a missing entry, the function does not modify local state.
    
    Failure modes and error handling:
        If the server returns an HTTP 404 (Not Found) response for the requested {id}.txt, the function raises ValueError with a message indicating the entry was not found on ExPASy. If the request is redirected to the ExPASy main page (handle.url equals "https://www.expasy.org/"), this is treated as the entry not being found and also raises ValueError. Other HTTP-related errors (for example urllib.error.HTTPError with codes other than 404) and network exceptions are propagated to the caller and are not converted to ValueError by this function; callers should be prepared to handle those exceptions when performing network I/O.
    
    Returns:
        A readable text file-like handle: a file-like object opened on the raw PROSITE/PRODOC text record retrieved from ExPASy. The handle provides methods for reading the entry contents and should be closed when no longer needed (use a with statement to ensure closure).
    
    Raises:
        ValueError: If the specified identifier does not exist on ExPASy (HTTP 404 or redirection to the ExPASy main page).
        HTTPError (or other network-related exceptions): Propagated for HTTP errors other than 404 or other network failures during the request.
    """
    from Bio.ExPASy import get_prosite_raw
    return get_prosite_raw(id, cgi)


################################################################################
# Source: Bio.ExPASy.get_sprot_raw
# File: Bio/ExPASy/__init__.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_ExPASy_get_sprot_raw(id: str):
    """Get a text handle to a raw SwissProt (UniProt) entry from ExPASy/UniProt.
    
    This function constructs the UniProt text URL for the given identifier and opens it using the module's internal _open helper. For an identifier XXX the function fetches the resource at http://www.uniprot.org/uniprot/XXX.txt (as per ExPASy/UniProt location conventions) and returns a readable text-mode handle to the raw SwissProt entry. In practical bioinformatics workflows within Biopython, the returned handle is typically passed to Bio.SwissProt.read or other parsers to obtain a parsed SwissProt record containing protein sequence, feature and annotation data for downstream analysis.
    
    Args:
        id (str): A UniProt/SwissProt identifier (for example an accession like "O23729" or an entry name) that names the SwissProt record to retrieve. This parameter is required and must be a string exactly as accepted by the UniProt website; it is interpolated directly into the URL http://www.uniprot.org/uniprot/{id}.txt to locate the raw text entry.
    
    Returns:
        handle: A file-like, text-mode handle opened for reading the raw SwissProt entry. The handle supports .read() and can be used as a context manager (for example, via "with ExPASy.get_sprot_raw(id) as handle:"). The caller is responsible for closing the handle if not using a context manager. The handle provides the plain text UniProt/SwissProt record suitable for parsing by Bio.SwissProt and other text parsers.
    
    Behavior and side effects:
        This function performs a network HTTP GET request to the UniProt server and may block while waiting for the response; it requires an active Internet connection and may be subject to network latency or server-side delays. The returned text reflects the current UniProt/SwissProt entry at the time of the request. No caching is performed by this function.
    
    Failure modes and exceptions:
        If the server responds with an HTTP 400 or 404 status (indicating the identifier was not found), the function raises ValueError with the message "Failed to find SwissProt entry '<id>'" (where <id> is replaced by the provided identifier). For other HTTP errors, the underlying HTTPError is re-raised so callers can inspect the HTTP status and handle it as needed. Other network-related exceptions raised by the underlying _open implementation (for example connection errors or timeouts) will propagate to the caller. Users running automated tests should be aware that network issues can cause tests that call this function to fail or be skipped.
    """
    from Bio.ExPASy import get_sprot_raw
    return get_sprot_raw(id)


################################################################################
# Source: Bio.ExPASy.ScanProsite.scan
# File: Bio/ExPASy/ScanProsite.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_ExPASy_ScanProsite_scan(
    seq: str = "",
    mirror: str = "https://prosite.expasy.org",
    output: str = "xml",
    **keywords
):
    """Bio.ExPASy.ScanProsite.scan executes a ScanProsite search against the ExPASy ScanProsite web service to locate PROSITE patterns or profiles in a protein sequence or UniProtKB accession. This function is used in computational molecular biology workflows (as described in the Biopython README and module source) to programmatically query the remote ScanProsite service, obtain the raw search results over HTTP, and then parse those results (for example, using Bio.ExPASy.ScanProsite.read when XML output is requested).
    
    Args:
        seq (str): The query supplied to ScanProsite. This may be a raw protein sequence string or a UniProtKB accession (Swiss-Prot or TrEMBL) that ScanProsite will search for PROSITE matches. The default is the empty string "", which results in a request containing an empty seq parameter; the remote service may reject or return an error/message if a sequence or accession is required. The seq parameter determines which protein data the ScanProsite server examines and so directly controls the biological search performed.
        mirror (str): The base URL of the ScanProsite mirror to contact (default: "https://prosite.expasy.org"). This string is used as the prefix to construct the request URL (the function appends the fixed path "/cgi-bin/prosite/scanprosite/PSScan.cgi" and a URL-encoded query string). The mirror therefore specifies which ExPASy server is contacted; supplying an incorrect or unreachable mirror will result in network-level errors.
        output (str): The requested format of the search results as accepted by the ScanProsite service (default: "xml"). The output value is passed verbatim to the remote server and controls the representation of results returned (for example XML when set to "xml"), which affects how the caller should parse the response. See the ScanProsite programmatic documentation for allowed output formats.
        keywords (dict): Additional ScanProsite query parameters passed as keyword arguments. Each key/value pair is URL-encoded and added to the request query string; any keyword whose value is None is omitted from the encoded parameters (the implementation filters out None values). Refer to the ScanProsite programmatic documentation at https://prosite.expasy.org/scanprosite/scanprosite_doc.html for the names and meanings of supported parameters (for example pattern selection, thresholds, or advanced options).
    
    Behavior and side effects:
        The function builds a URL-encoded query from the provided seq, output, and keywords, appends it to the mirror path, and performs an HTTP GET request using urllib (urlencode and urlopen in the implementation). This is a synchronous network call that may block until the remote server responds. The remote server processes the request and returns the raw search results. Because this function performs network I/O, it can raise network-related exceptions (for example urllib.error.URLError or HTTPError raised by urllib.request.urlopen) and can fail if the mirror is unreachable, the request parameters are invalid, or the remote service is down. The caller is responsible for handling such exceptions. The function does not validate ScanProsite-specific parameter values beyond excluding None values; validation of allowed parameter names/values must be done according to the ScanProsite documentation.
    
    Returns:
        A file-like handle to the HTTP response produced by the ScanProsite server. This handle provides a readable stream for the raw search results returned by the remote service and is the primary output used in downstream processing in bioinformatics pipelines. For XML results (the default), the returned handle can be passed to Bio.ExPASy.ScanProsite.read to parse the response into Python objects. The handle is an open network/HTTP response stream provided by urllib.request.urlopen, so the caller should read from it and close it when finished (for example by using a context manager or calling close()) to release network resources.
    
    Failure modes and recommendations:
        Network failures, DNS resolution errors, timeouts, HTTP error responses, or malformed server replies can cause exceptions or invalid results. If the remote server returns an error page, the handle may contain HTML or error text rather than the requested format. When integrating into automated pipelines, catch and handle urllib exceptions, verify HTTP status when applicable, and validate the returned content before parsing. Use the ScanProsite documentation referenced above to determine valid values for output and other query parameters to avoid server-side errors.
    """
    from Bio.ExPASy.ScanProsite import scan
    return scan(seq, mirror, output, **keywords)


################################################################################
# Source: Bio.Geo.Record.out_block
# File: Bio/Geo/Record.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_Geo_Record_out_block(text: str, prefix: str = ""):
    """Bio.Geo.Record.out_block formats a long text string into fixed-width blocks of up to 80 characters per block (slicing the input text every 80 characters) and returns the assembled lines with an optional prefix prepended to each line. This function is used in the Biopython codebase (for example in record formatting within Bio.Geo.Record) to produce human-readable and file-format-friendly multiline fields when writing out record annotations or other long text fields in computational molecular biology workflows.
    
    Behavior: The input text is split by simple character slices of length 80 (text[0:80], text[80:160], ...). Each slice is emitted as a separate line with a trailing newline character. After all blocks are emitted an additional newline is appended. If text is empty the function returns a single newline. Note that the slicing is performed only on the text content; the prefix is prepended to each output line after slicing and therefore can increase the printed line length beyond 80 characters. The function does not perform word-aware wrapping and may split words across block boundaries. This routine does not perform any I/O operations; it builds and returns a string suitable for inclusion in file output or display.
    
    Args:
        text (str): The input text to be formatted into 80-character blocks. In the Biopython context this typically holds long annotation strings or record field content that must be broken into fixed-width lines for textual record output. The function treats this as plain Python text (Unicode string) and slices it by character indices; multi-codepoint or multi-byte characters count as individual Python characters for slicing. Supplying a non-string value may cause a runtime error when slicing is attempted.
        prefix (str): Optional string to prepend to every output line. By default this is the empty string (prefix = ""), meaning no extra characters are added before each block. In practical use within Biopython this prefix can be used to add field labels, indentation, or format-specific line headers. Because prefix is added after slicing the text into 80-character chunks, a non-empty prefix will make the visible line length exceed 80 characters.
    
    Returns:
        str: A new string built from the input text split into consecutive 80-character slices, each followed by a newline, with the provided prefix prepended to every line. For non-empty input the returned string ends with an extra blank line (two consecutive newline characters at the end of the last block). For an empty input string the function returns a single newline character. The function has no external side effects.
    """
    from Bio.Geo.Record import out_block
    return out_block(text, prefix)


################################################################################
# Source: Bio.KEGG.REST.kegg_conv
# File: Bio/KEGG/REST.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_KEGG_REST_kegg_conv(target_db: str, source_db: str, option: str = None):
    """KEGG conv - convert KEGG identifiers to/from outside identifiers using the KEGG REST API.
    
    Performs a conversion request against the KEGG REST "conv" endpoint (https://rest.kegg.jp/conv/<target_db>/<source_db>[/<option>]) to map identifiers between KEGG databases (for example organism gene IDs, compound/drug/glycan IDs) and outside databases (for example NCBI GI, NCBI GeneID, UniProt, PubChem, ChEBI). This function is intended for use in computational molecular biology workflows (annotation, cross-referencing, data integration) that require translating identifiers between KEGG and external resources. The function constructs and issues the REST query via the internal helper _q and returns that helper's response. It validates the requested conversion pair and the optional output serialization format before issuing the request.
    
    Args:
        target_db (str): Target database name for the conversion in the KEGG REST call. This is the database to which identifiers should be converted (the left-hand component in the REST URL). Practical values include KEGG gene/org codes (a KEGG organism code or T number) when converting gene identifiers, or "drug", "compound", or "glycan" when converting chemical substance identifiers. It is used directly in the REST path as <target_db> and must form an allowed conversion pair with source_db (see failure modes below).
        source_db (str): Source database name or database entries used as the input for conversion. This becomes the right-hand component in the REST URL as <source_db>. Typical values include external databases such as "ncbi-gi", "ncbi-geneid", "uniprot", "pubchem", or "chebi"; for gene conversions this can also be a KEGG organism code or T number. Although the function signature documents a str, the implementation also accepts a list of strings (e.g., multiple database entries) in which case the elements will be joined using "+" to form the REST path segment (database entries are combined as "<entry1>+<entry2>+..."). The value is validated together with target_db to ensure it represents a permitted conversion direction before the request is made.
        option (str): Optional output serialization format for the KEGG service. When provided, it must be exactly "turtle" or "n-triple" to request RDF serializations from KEGG; these are appended to the REST URL as the optional <option> path segment. If None (the default), no serialization option is requested and KEGG returns its default plain text conversion listing.
    
    Returns:
        object: The response returned by the internal query helper _q after issuing the KEGG REST "conv" request. This object is the direct result of _q("conv", target_db, source_db[, option]) and represents the HTTP/API response or parsed result produced by that helper. Callers should inspect the returned object according to the conventions of the surrounding Bio.KEGG.REST module (for example to read text content, check status, or parse records). Network-level errors, timeouts, or exceptions raised by the internal _q helper will propagate to the caller.
    
    Behavior, side effects, defaults, and failure modes:
        This function validates the option argument and the conversion pair before contacting the KEGG REST service. If option is not None and is not one of "turtle" or "n-triple", the function raises ValueError("Invalid option arg for kegg conv request."). The function only permits conversion pairs documented in the implementation: conversions involving gene identifiers (where one side is a KEGG organism code/T number and the other is "ncbi-gi", "ncbi-geneid", or "uniprot") and conversions between KEGG chemical substance databases and external chemical databases (examples include allowed pairs where target_db or source_db are in {"drug","compound","glycan","pubchem","chebi"} according to the combinations enforced in the code). If the provided target_db and source_db do not match one of the allowed patterns, the function raises ValueError("Bad argument target_db or source_db for kegg conv request."). If source_db is provided as a list, it is joined with "+" before forming the request path. The primary side effect is a network/API call to the KEGG REST service via the internal _q helper; therefore callers should be prepared for network latency and for exceptions propagated from _q (such as connection errors or HTTP errors).
    """
    from Bio.KEGG.REST import kegg_conv
    return kegg_conv(target_db, source_db, option)


################################################################################
# Source: Bio.KEGG.REST.kegg_find
# File: Bio/KEGG/REST.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_KEGG_REST_kegg_find(database: str, query: str, option: str = None):
    """KEGG find - Data search.
    Finds KEGG entries whose keywords or other query data match the provided query string(s) by issuing a call to the KEGG REST API. This function is part of Biopython's KEGG.REST client and is used in computational molecular biology workflows to locate KEGG database entries (for example, compounds, drugs, pathways, enzymes, organisms) that match textual keywords, chemical formulae, or mass values.
    
    Args:
        database (str): The KEGG database or organism code to search. This argument selects which KEGG collection to query (for example "pathway", "module", "disease", "drug", "environ", "ko", "genome", a KEGG organism code or T number for an organism, "compound", "glycan", "reaction", "rpair", "rclass", "enzyme", "genes", "ligand"). The special cases "compound" and "drug" permit an additional option (see the option parameter). The value is passed directly to the KEGG REST endpoint and determines the domain of entries returned, so choosing the correct database is important when mapping experimental results to KEGG identifiers or when looking up biochemical entities.
    
        query (str or list of str): The search terms to match in the selected database. Typical uses are keyword queries (e.g., enzyme name, pathway keyword), chemical formulae, or numeric mass values depending on the database and option. If a list of strings is provided (handled by the implementation), the list elements are joined into a single query with the "+" character (equivalent to KEGG REST multi-term queries) before sending the request; each element must therefore be a string. The query content directly affects which KEGG entries are matched and returned by the KEGG server.
    
        option (str): Optional search modifier used only when database is "compound" or "drug". Accepted values are "formula", "exact_mass", or "mol_weight". When option is "formula", the search matches chemical formulae as a partial match and ignores atom order (useful for looking up compounds by their molecular formula). When option is "exact_mass" or "mol_weight", numeric matching is performed by rounding the KEGG-stored value to the same decimal precision as the provided query; a numeric range may be specified in the query using the minus ("-") sign to find entries within a range. The default is None, meaning no option-specific field search is requested. If option is provided for a database other than "compound" or "drug", the function raises a ValueError (see Failure modes).
    
    Returns:
        str: The raw response returned by the internal KEGG REST helper (the _q function). This is typically the plain-text body returned by the KEGG REST "find" endpoint and contains the matching KEGG entry lines (for example, "compound:C00031\tGlucose"). The function does not parse the returned text into structured Python objects; callers that need structured results must parse the returned string themselves.
    
    Behavior, side effects, defaults, and failure modes:
    - The function issues a network request to the KEGG REST API (endpoint form: https://rest.kegg.jp/find/<database>/<query>[/<option>]) and therefore may raise network-related exceptions or propagate HTTP errors raised by the internal helper _q; callers should handle such exceptions in networked or offline environments.
    - If query is a list, the function joins items with "+" to form the query string; items should be strings, otherwise a TypeError may occur during joining.
    - If database is "compound" or "drug" and option is one of "formula", "exact_mass", or "mol_weight", the option is included in the request and the search targets the corresponding field in KEGG. The chemical formula search performs partial matching irrespective of atom order. Exact mass and molecular weight searches compare by rounding to the same decimal place as the query; numeric ranges are supported using the "-" sign in the query string.
    - If an option argument is supplied for a database other than "compound" or "drug", the function raises ValueError("Invalid option arg for kegg find request.").
    - The function does not modify any local state; its observable effect is the network request and the returned response string. There is a noted TODO in the implementation suggesting future return of structured data (for example, a list of tuples), but currently the function returns the raw text from the KEGG server.
    """
    from Bio.KEGG.REST import kegg_find
    return kegg_find(database, query, option)


################################################################################
# Source: Bio.KEGG.REST.kegg_get
# File: Bio/KEGG/REST.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_KEGG_REST_kegg_get(dbentries: list, option: str = None):
    """Bio.KEGG.REST.kegg_get — Retrieve KEGG database entries via the KEGG REST "get" endpoint and return a handle to the response. This function is used in the KEGG REST client portion of Biopython to request textual or binary data for one or more KEGG entries (for example genes, pathways, compounds, glycans, reactions, enzymes, or organism-specific entries) and to obtain specific views of those entries (for example amino-acid sequences, nucleotide sequences, molecular files, KCF, pathway images, KGML, or JSON when supported).
    
    Args:
        dbentries (str or list): KEGG identifiers to retrieve. This may be a single identifier string (for example "hsa:10458", "cpd:C00031", or a pathway id like "path:hsa00010") or a list of such identifier strings. When passed as a list, the function will join the identifiers with '+' to form the KEGG REST query (the REST API expects multiple entries joined by '+'). The function enforces a maximum of 10 identifiers per request and will raise ValueError if a list longer than 10 is provided. Valid kinds of entries include KEGG database names shown in the code comments (pathway, brite, module, disease, drug, environ, ko, genome, <org>, compound, glycan, reaction, rpair, rclass, enzyme); <org> may be a KEGG organism code or a T number. Practical significance: callers should supply the exact KEGG identifiers or codes required by the KEGG REST API and may combine up to 10 entries in one call to reduce round trips.
        option (str, optional): Retrieval option to request a specific representation of the entry. Allowed values (as accepted by the function) are "aaseq", "ntseq", "mol", "kcf", "image", "kgml", and "json". If option is provided and valid, it is appended to the REST URL as /<option> (for example requesting protein sequences with "aaseq" or pathway XML with "kgml"). Semantic constraints enforced by the KEGG REST API and noted in the function: only one pathway entry may be requested when using "image" or "kgml"; only one compound, glycan, or drug entry may be requested when using "image". If an invalid non-empty option string is supplied, the function raises ValueError. If option is None (the default), the canonical text entry is requested.
    
    Raises (behavior and failure modes described):
        ValueError: If dbentries is a list with more than 10 items, a ValueError is raised with the message "Maximum number of dbentries is 10 for kegg get query". If a non-empty option string is provided that is not one of the allowed options, a ValueError is raised with the message "Invalid option arg for kegg get request." In addition, this function performs a network request via the internal _q(...) helper; network, HTTP, or underlying library errors raised by that helper (for example connection failures, timeouts, or HTTP errors) will propagate to the caller. Practical significance: callers must handle ValueError for invalid arguments and should be prepared to handle network/IO exceptions from the REST call.
    
    Side effects and defaults:
        If dbentries is a list of length between 1 and 10 inclusive, the function concatenates entries with '+' before issuing the request (this behavior implements the KEGG REST API multiple-entry syntax). The option argument defaults to None, which requests the default textual entry resource. The function issues a network call to the KEGG REST server (REST URL pattern: https://rest.kegg.jp/get/<dbentries>[/<option>]), which may incur network latency, bandwidth usage, and possible rate-limiting by the KEGG service. The returned handle should be consumed or closed by the caller according to the I/O semantics of the handle type returned by the underlying _q helper.
    
    Returns:
        handle: A handle to the response returned by the underlying KEGG REST query helper (_q). In practice this is the object produced by the internal network helper that represents the server response (a file-like or response object providing access to the retrieved data). The handle provides access to the requested KEGG data (text, sequence, molecular file, image bytes, KGML XML, or JSON) and should be read and/or closed by the caller. Practical significance: callers use this handle to parse or save the retrieved KEGG data; the function itself does not parse the content, it only performs the retrieval and returns the response handle.
    """
    from Bio.KEGG.REST import kegg_get
    return kegg_get(dbentries, option)


################################################################################
# Source: Bio.KEGG.REST.kegg_info
# File: Bio/KEGG/REST.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_KEGG_REST_kegg_info(database: str):
    """KEGG info - Retrieve current statistics for a KEGG database or organism.
    
    This function issues a synchronous REST request to the KEGG REST API endpoint
    (https://rest.kegg.jp/info/<database>) to obtain human-readable statistics and
    summary information for a KEGG database or an organism entry. In the context of
    Biopython (a toolkit for computational molecular biology and bioinformatics),
    kegg_info is used to programmatically discover the current size, versioning,
    and basic counts (for example number of entries) for resources hosted by KEGG
    before performing downstream retrieval or parsing operations. Typical uses
    include checking the number of pathway, gene, compound, or organism entries,
    or obtaining a list of supported organism codes via kegg_info('organism').
    
    Args:
        database (str): The KEGG database name, KEGG database official abbreviation,
            KEGG organism code, or T number to query. Examples: 'pathway' or 'path'
            for the KEGG pathway database, 'compound' for the compound database,
            an organism code such as 'hsa' for human, or a T number such as 'T01001'.
            The provided string is interpolated directly into the KEGG REST URL
            segment <database> and therefore must match KEGG's expected identifiers.
            There is no default value; the caller must supply a non-empty string.
            This parameter controls which KEGG resource's statistics are returned.
    
    Returns:
        file-like handle: A handle object returned by the internal _q function that
            represents the raw HTTP response stream from the KEGG REST API. The
            handle yields the plain-text output produced by the KEGG "info" service,
            typically one or more formatted text lines describing counts and summary
            metadata for the requested database or organism. The caller should read
            from the handle (for example, using read() or iterating over lines) to
            consume the response. The function does not parse the KEGG text into
            structured Python objects; parsing must be implemented by the caller if
            structured data is required.
    
    Behavior and side effects:
        This function performs network I/O by contacting the remote KEGG REST
        service; it may block until the HTTP request completes. It does not cache
        results or validate the supplied organism code or T number. The returned
        handle exposes the raw response; the function itself does not transform or
        validate the content. Because the function relies on an external service,
        callers should be prepared to handle network-related failures, service-side
        errors, or rate-limiting. On network failure or if the KEGG service
        returns an HTTP error, the underlying networking layer or the internal _q
        implementation may raise an exception (for example, connection or HTTP
        errors) or return a handle whose contents contain an error message from
        KEGG. The function does not perform retries or automatic parsing of the
        formatted text output.
    
    Examples and practical significance:
        Use this function within Biopython workflows when you need to programmatically
        inspect the current state of KEGG resources (for example, to decide whether
        to download a full database dump, to confirm an organism code exists, or to
        report the number of entries before running a large batch query). For a
        list of organism codes and T numbers, call kegg_info('organism') or visit
        https://rest.kegg.jp/list/organism.
    
    Notes:
        The implementation currently returns a raw handle rather than a decoded
        string or structured object; there are TODOs in the codebase mentioning
        potential future changes such as returning a string, caching, or parsing
        the formatted output. Callers who need parsed data should read the handle
        and implement parsing according to the KEGG "info" text format.
    """
    from Bio.KEGG.REST import kegg_info
    return kegg_info(database)


################################################################################
# Source: Bio.KEGG.REST.kegg_link
# File: Bio/KEGG/REST.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_KEGG_REST_kegg_link(target_db: str, source_db: str, option: str = None):
    """Bio.KEGG.REST.kegg_link queries the KEGG REST API to find related entries between two KEGG databases by using database cross-references. This is used within Biopython to programmatically map entries (for example, mapping genes to pathways or compounds to reactions) using the KEGG REST endpoint /link/<target_db>/<source_db>[/<option>].
    
    Args:
        target_db (str): Target database name used in the KEGG REST request. This identifies the database whose entries will appear on the left side of the returned link pairs (for example "pathway", "compound", "ko", an organism code like "hsa", or other KEGG databases). In the KEGG domain this parameter determines which database's entries will be returned as targets of the cross-reference query.
        source_db (str or list): Source database name or a list of source database entries to link from. This can be a database name (same set of KEGG database identifiers as for target_db, e.g. "genes", "enzyme", "<org>") or a list of specific dbentries. If a Python list is supplied, the function joins its elements with the '+' character to form a valid KEGG REST query (this enables querying multiple dbentries in one call). In the KEGG domain this parameter identifies the origin of the cross-references (for example genes to be mapped to pathways).
        option (str, optional): Optional output format specifier for the KEGG REST call. Accepted values are "turtle" or "n-triple" to request RDF serializations from KEGG; the default value None requests the standard tab-delimited link output. If provided and not one of the accepted strings, the function raises ValueError. Defaults to None.
    
    Behavior and side effects:
        This function constructs a KEGG REST API request of the form /link/<target_db>/<source_db> or /link/<target_db>/<source_db>/<option> and sends it via the internal helper _q, which performs the network request. If source_db is a list, its items are concatenated using '+' before sending the request. Supplying option="turtle" or option="n-triple" requests RDF serialized output from KEGG; omitting option requests the default plain link text format. The function performs network I/O and depends on external KEGG availability; network errors, timeouts, or HTTP errors from the KEGG server may propagate (or be raised) from the underlying _q implementation. The KEGG REST API accepts database identifiers drawn from KEGG nomenclature (examples include pathway, brite, module, ko, genome, organism codes, compound, glycan, reaction, rpair, rclass, enzyme, disease, drug, dgroup, environ, genes). The returned content may be empty if no cross-references exist for the given inputs.
    
    Failure modes:
        If option is provided but is not "turtle" or "n-triple", a ValueError is raised immediately with the message "Invalid option arg for kegg conv request." Network-related exceptions or HTTP errors can occur when contacting the KEGG REST service via the internal _q helper. Supplying invalid or misspelled database names will typically result in an empty response or a KEGG server error, depending on the KEGG REST API behavior.
    
    Returns:
        str: The raw response body returned by the KEGG REST API via the internal _q helper. For the default request (option is None) this is typically newline-separated tab-delimited link mappings of the form "<target_id>\t<source_id>\n". If option is "turtle" or "n-triple", the returned string is the RDF serialization in the requested format. The exact return reflects the KEGG REST payload and any behavior of the internal _q function.
    """
    from Bio.KEGG.REST import kegg_link
    return kegg_link(target_db, source_db, option)


################################################################################
# Source: Bio.KEGG.REST.kegg_list
# File: Bio/KEGG/REST.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_KEGG_REST_kegg_list(database: str, org: str = None):
    """KEGG list - Retrieve an entry list from a KEGG database or a database subset restricted to an organism.
    
    This function is part of the Bio.KEGG.REST module in Biopython and is used to query the KEGG REST API (see https://rest.kegg.jp/list/<database> and https://rest.kegg.jp/list/<database>/<org>) to obtain plain-text lists of database entries. In the computational molecular biology domain, this is useful for programmatically obtaining lists of pathways, modules, compounds, enzymes, organisms, and other KEGG resources for downstream processing (for example, mapping identifiers, filtering by organism, or constructing local indexes). Behavior depends on the `database` and `org` arguments as detailed below.
    
    Args:
        database (str): Name of the KEGG database to list entries from. Typical values include "pathway", "module", "compound", "enzyme", "organism", and others documented by KEGG. Although the function signature annotates this parameter as str, the implementation also accepts a Python list of database names; in that case the list elements are concatenated using "+" into a single query string to request multiple databases at once (see behavior notes below). When a single string is provided and `org` is supplied, the function only accepts "pathway" or "module" as the `database` value; supplying a different non-empty database string together with `org` raises ValueError. The practical significance is that `database` selects which KEGG resource you will receive an entry list for, and controls which REST endpoint is invoked.
        org (str): Optional KEGG organism code (for example "hsa" for Homo sapiens) or KEGG T number. When provided, and when `database` is "pathway" or "module", the request is directed to the KEGG REST endpoint that restricts the list to entries for that organism (i.e., /list/<database>/<org>). This parameter defaults to None, meaning no organism restriction. Passing `org` with a `database` that is a non-empty string other than "pathway" or "module" will cause a ValueError to be raised as described below.
    
    Returns:
        The value returned by the internal helper _q() used to perform the KEGG REST request. This contains the KEGG server response for the "list" request (typically the server's plain-text entry list for the requested database or database+organism). The exact Python object/type is whatever _q() returns in this Biopython implementation; callers should treat it as the raw response to be parsed or inspected by higher-level code.
    
    Behavior and side effects:
        If database is the string "pathway" or "module" and org is provided, the function performs a KEGG REST query using the endpoint /list/<database>/<org> to restrict results to that organism.
        If database is a non-empty string (other than "pathway" or "module") and org is provided, the function raises ValueError with the message "Invalid database arg for kegg list request." because organism-restricted listing is only supported for pathway and module endpoints in this implementation.
        If database is a Python list (even though the signature annotates str), the elements are joined with "+" to form a single multi-database query (e.g., ["ko","pathway"] -> "ko+pathway") and submitted to the /list/<database> endpoint. The implementation enforces a maximum of 100 elements in this list; if len(database) > 100 a ValueError is raised with the message "Maximum number of databases is 100 for kegg list query".
        The primary side effect is a network request to the KEGG REST server (https://rest.kegg.jp). As a result, the function may raise network-related exceptions or experience delays due to connectivity, server availability, or rate limiting; callers should ensure network access and handle transient errors appropriately.
        The function does not perform validation beyond the checks described; if database is of an unexpected type (neither str nor list) behavior depends on the internal _q implementation and may result in an exception from deeper code.
    
    Failure modes and exceptions:
        ValueError: Raised when a non-empty string `database` other than "pathway" or "module" is used together with a non-None `org`, with message "Invalid database arg for kegg list request."
        ValueError: Raised when `database` is a list with more than 100 elements, with message "Maximum number of databases is 100 for kegg list query".
        Network-related exceptions: The underlying _q() helper performs the HTTP request and may raise exceptions related to network errors, timeouts, or server-side failures; these are propagated to the caller.
    
    Notes:
        This function is intended for programmatic access to KEGG entry lists as part of bioinformatics workflows (for example, retrieving all pathway IDs for a given organism). For parsing or higher-level handling of KEGG data, combine this function with Biopython parsing utilities or custom parsers that consume the returned plain-text response.
    """
    from Bio.KEGG.REST import kegg_list
    return kegg_list(database, org)


################################################################################
# Source: Bio.NMR.NOEtools.predictNOE
# File: Bio/NMR/NOEtools.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_NMR_NOEtools_predictNOE(
    peaklist: list,
    originNuc: str,
    detectedNuc: str,
    originResNum: int,
    toResNum: int
):
    """Predict the i->j NOE (nuclear Overhauser effect) peak position by generating a single .xpk-format crosspeak entry derived from diagonal (self) peaks in a given peaklist. This function is used in NMR peaklist processing workflows (as in Biopython's NMR utilities) to predict a crosspeak that would arise from magnetization transfer originating on one nucleus of an originating residue and being detected on a nucleus of a target residue, based on existing diagonal/self peaks for those nuclei.
    
    Args:
        peaklist (list): A sequence representing an xpk-style peaklist object (in practice the code expects an object compatible with xpktools.Peaklist: it must provide attributes and methods accessed here, specifically .datalabels and residue_dict(nucleus) returning per-residue data lines). This parameter supplies the source self-peak data from which ppm averages and assignment labels are read. The function reads but does not modify peaklist; if peaklist lacks the expected attributes or has unexpected formats a KeyError or AttributeError may be raised.
        originNuc (str): The label/name of the originating nucleus (e.g., "N15"). This string is concatenated with suffixes like ".L" and ".P" to look up columns in the peaklist data map. It must match a nucleus label present in peaklist.datalabels; otherwise a KeyError will occur. The value determines which column supplies the originating residue assignment and originating ppm used when constructing the predicted crosspeak.
        detectedNuc (str): The label/name of the detected nucleus (e.g., "H1"). This string is used to select the per-residue lists from peaklist.residue_dict(detectedNuc) to find the detected (observed) peak lines and to compute the average detected ppm. It must match a nucleus label present in the peaklist; otherwise the detection/residue lookup will fail.
        originResNum (int): The residue index (integer) of the origin residue whose self-peak data supply the originating assignment label and ppm average. The function converts this integer to a string for dictionary lookup (peaklist.residue_dict expects string keys). If the origin residue number is not present in the peaklist for detectedNuc, no prediction is produced and an empty string is returned.
        toResNum (int): The residue index (integer) of the target (detected) residue for which a .xpk-style entry will be generated. The function converts this integer to a string for dictionary lookup. If the target residue number is not present in the peaklist for detectedNuc, no prediction is produced and an empty string is returned.
    
    Behavior and side effects:
        The function first builds a datamap from peaklist.datalabels via the internal helper _data_map and computes integer column indices for the originating assignment and ppm and the detected ppm. It then checks that both originResNum and toResNum (converted to strings) exist in peaklist.residue_dict(detectedNuc). If present, it selects the lists of data lines for the detected residue and the origin residue. It uses the first line of the detected-residue list as a template (returnLine) and computes aveDetectedPPM and aveOriginPPM using the internal helper _col_ave on the detected and origin lists respectively. It extracts the originating assignment label from the first line of the origin list. The template line is then updated by calling xpktools.replace_entry to set the assignment and ppm fields to the originating assignment and the averaged originating ppm. The function does not mutate the input peaklist object; it constructs and returns a new .xpk-format line string. If the required residue entries are missing, the function returns an empty string (the initial value of returnLine). If expected datalabel keys (originNuc + ".L", originNuc + ".P", detectedNuc + ".P") are absent, a KeyError will be raised by the datamap lookup. If the per-residue lists are empty, indexing (e.g., detectedList[0]) may raise an IndexError. No validation is performed to confirm that the input peaklist consists only of diagonal/self peaks; the algorithm assumes that the supplied peaklist is diagonal (self peaks only) and will produce results that are only meaningful under that assumption.
    
    Defaults:
        There are no default substitutions performed by this function; all five parameters are required and are used as described. The function initializes returnLine to an empty string and will return that empty string when the required residue data are not found.
    
    Failure modes:
        Missing nucleus labels in peaklist.datalabels will raise KeyError during datamap lookup. Missing attributes or incorrect peaklist object type may raise AttributeError. Empty per-residue lists will cause IndexError when attempting to read the first line. If input peaklist contains off-diagonal peaks, the predicted crosspeak may be incorrect because no check is done to ensure diagonal-only input.
    
    Examples of use in the Biopython NMR context:
        Calling predictNOE(peaklist, "N15", "H1", 10, 12) where peaklist is an xpktools-compatible peaklist produces a .xpk-format string representing a predicted crosspeak that originates on the N15 nucleus of residue 10 and is detected on the H1 nucleus of residue 12; the returned line is built from the detected-residue template with the origin assignment and origin averaged ppm substituted.
    
    Returns:
        str: The .xpk-format file entry (single-line string) representing the predicted crosspeak (returnLine). If prediction cannot be produced because the required residues are not found for the specified detected nucleus, an empty string is returned. The returned string is suitable for writing into an .xpk peaklist file or for further post-processing; no modifications are performed on the input peaklist object.
    """
    from Bio.NMR.NOEtools import predictNOE
    return predictNOE(peaklist, originNuc, detectedNuc, originResNum, toResNum)


################################################################################
# Source: Bio.NMR.xpktools.data_table
# File: Bio/NMR/xpktools.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_NMR_xpktools_data_table(fn_list: list, datalabel: str, keyatom: str):
    """Bio.NMR.xpktools.data_table generates a residue-indexed data table (as a list of text lines) by aggregating a specified data element from multiple .xpk files. This function is part of the Biopython NMR utilities and is used to compare or combine values (for example peak attributes) across several XPK-format outputs, producing one tab-separated row per residue number between the global minimum and maximum residues found in the input files.
    
    Args:
        fn_list (list): List of .xpk file names to read and aggregate. Each filename is passed to the internal helper _read_dicts which parses the XPK files and returns per-file dictionaries of entries keyed by the residue index (as string), together with a parallel list of labels (label_line_list). The order of fn_list determines the column order in the output rows; fn_list must be non-empty and reference readable XPK files for meaningful output.
        datalabel (str): The name of the data element to report from each XPK entry (for example a field name present in XpkEntry.fields). For each residue present in a given file, the function constructs an XpkEntry from the entry data and retrieves XpkEntry.fields[datalabel] to include in that file's column. If datalabel is not present in an XPK entry's fields, a KeyError may be raised by XpkEntry.fields access.
        keyatom (str): The nucleus name used as the index for entries in the per-file dictionaries returned by _read_dicts (for example "HN" or "CA" in protein NMR). _read_dicts uses keyatom to select which nucleus/atom is treated as the key when constructing the per-file dictionaries; the returned dictionaries are expected to include "minres" and "maxres" keys and mapping from residue-number strings to lists of entry data.
    
    Behavior and practical details:
        The function calls _read_dicts(fn_list, keyatom) to obtain two results: dict_list (a list of per-file dictionaries) and label_line_list (a list of labels corresponding to each input file). Each per-file dictionary is expected to contain integer bounds under the keys "minres" and "maxres" and mappings from residue numbers (string form) to lists of entry data. The function computes the global minimum and maximum residue numbers across all dictionaries and iterates over every integer residue number from the global minimum to the global maximum inclusive. For each residue number (converted to a string key), it builds a tab-separated line beginning with the residue number and then, in the order of fn_list / dict_list, appends either the requested datalabel value for that file or the literal '*' if that residue is absent in that file. Each line is terminated with a newline character and appended to the output list. The result is one row per residue number, ordered by increasing residue number, with one column per input file. No files are written by this function; it only reads via _read_dicts and returns the assembled lines.
    
    Return format:
        The returned value is a list of strings (outlist). Each string is a single table row formatted as: "<residue_number>\t<value_for_file1>\t<value_for_file2>\t...\n". A missing value for a file is represented by an asterisk '*' in that column. The length of each row (number of columns) equals 1 + len(fn_list). Rows are ordered by residue number from the global minimum to maximum found across all input files.
    
    Side effects and dependencies:
        This function depends on _read_dicts to read and parse the provided .xpk files and on XpkEntry to interpret entry data and expose the fields mapping. Thus it performs file I/O indirectly via _read_dicts and allocates the returned list of strings in memory. It does not modify the input files.
    
    Failure modes and exceptions:
        If fn_list is empty or _read_dicts returns an empty dict_list, accessing dict_list[0] will raise IndexError. If expected keys "minres" or "maxres" are missing from a per-file dictionary, a KeyError will be raised. If a residue entry exists but the associated list is empty or malformed, creating XpkEntry(dictionary[key][0], label) may raise IndexError or other exceptions from XpkEntry. Attempting to access XpkEntry.fields[datalabel] may raise KeyError if datalabel is not present for an entry. File I/O errors (e.g., FileNotFoundError, PermissionError) may be raised by _read_dicts when opening or reading the .xpk files. Callers should catch these exceptions as appropriate.
    
    Returns:
        outlist (list): A list of newline-terminated strings, one per residue between the global min and max residues across the input files. Each string is a tab-separated row beginning with the residue number and followed by the datalabel values (or '*' for missing) for each file in fn_list, in the same order as fn_list.
    """
    from Bio.NMR.xpktools import data_table
    return data_table(fn_list, datalabel, keyatom)


################################################################################
# Source: Bio.Nexus.Nexus.combine
# File: Bio/Nexus/Nexus.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_Nexus_Nexus_combine(matrices: list):
    """Combine multiple Nexus-format character matrices into a single Nexus instance.
    
    This function is used in phylogenetics and comparative sequence analysis (as provided by Biopython) to merge several Nexus matrices, each representing an aligned character matrix for a set of taxa (for example, gene alignments, morphological characters, or partitions of a supermatrix), into one combined Nexus matrix that preserves taxon names, character partitions, character sets, and taxon sets while aligning and concatenating sequences across shared taxa. The first matrix in the provided list is deep-copied and used as the base; subsequent matrices are appended to its sequences with careful handling of missing and gap symbols, and per-dataset character set and taxon set names are prefixed with the originating dataset name to avoid collisions. This routine is intended for use with Bio.Nexus Nexus instances that expose attributes such as datatype, charsets, charpartitions, taxsets, charlabels, statelabels, interleave, translate, matrix, taxlabels, nchar, ntax, gap, and missing.
    
    Args:
        matrices (list): A list of pairs (name, nexus_instance) in the order to be combined. Each element must be a two-item sequence where the first item is a string name identifying that matrix (used as a prefix when renaming character sets and taxon sets) and the second item is a Nexus instance (the object type produced/consumed by the Bio.Nexus.Nexus module). The list may be empty; if it is empty the function returns None. The function expects each nexus_instance to provide attributes and behaviors used by the implementation: matrix (mapping taxon -> Seq), taxlabels (list of taxa names), nchar (number of characters), ntax (number of taxa), gap and missing character symbols, charsets (mapping names -> list of integer character indices), charpartitions, charlabels, statelabels, taxsets, datatype, interleave, translate, and methods/semantics consistent with Bio.Nexus Nexus objects.
    
    Returns:
        Bio.Nexus.Nexus or None: If the input list is non-empty, returns a new Nexus instance representing the combined matrix. This returned Nexus is created by deep-copying the first provided nexus_instance and then extending its sequences and metadata with the subsequent matrices. If the input list is empty, returns None. The returned Nexus has the following concrete, practical properties important for downstream phylogenetic or sequence analyses: datatype is set to the original datatype if all inputs share the same datatype, otherwise it is set to the literal string "None" (the function does not raise an error for mixed datatypes); charlabels and statelabels are reinitialized and may be populated from later matrices; interleave is set to False and translate is set to None; per-dataset character sets and taxon sets are added with keys prefixed by the source dataset name (for example "datasetName.charsetName"), a top-level charpartition named "combined" maps each source dataset to its character index range within the combined matrix, and at the end a separate charset is created for each initial dataset partition.
    
    Behavior and side effects:
        The function deep-copies the first nexus_instance to avoid mutating the original objects passed in. For each subsequent (name, nexus_instance) pair, taxa present in both the current combined matrix and the new matrix have their sequences concatenated; gap and missing symbols from the incoming matrix are replaced by the corresponding symbols from the combined matrix to ensure consistent encoding. Taxa present only in the combined matrix receive appended missing-character symbols of length equal to the incoming matrix's character count. Taxa present only in the incoming matrix are prepended with missing-character symbols of length equal to the current combined matrix's character count and then have their adjusted sequence appended; these taxa are added to the combined taxlabels list. Character set indices from incoming matrices are offset by the current combined nchar when merged so that indices correctly refer to positions in the combined matrix. Taxon sets from incoming matrices are copied into the combined taxsets with the source dataset name prefixed. Character labels from incoming matrices, if present, are also offset and included in the combined charlabels mapping. The combined.nchar and combined.ntax numeric counters are incrementally updated to reflect the new total number of characters and taxa.
    
    Failure modes and cautions:
        If matrices contains items that are not two-item sequences (name, nexus_instance) or if the provided nexus_instance objects do not expose the expected attributes and behaviors (for example matrix, taxlabels, nchar, gap, missing), the function will raise attribute errors or type errors from those underlying operations. When input matrices have differing datatypes, the function will set the resulting combined.datatype to the literal string "None" rather than attempting to reconcile differing character state meanings; callers who require stricter enforcement should check datatypes before calling or implement their own merging policy. The function assumes sequence values are provided as Bio.Seq.Seq-like objects or strings convertible to str() and that concatenation semantics used in the implementation are appropriate. The function does not attempt to validate biological compatibility of concatenated partitions (for example, mixing nucleotide and amino-acid data); such domain-specific decisions are left to the caller.
    
    Practical significance:
        Use this function when constructing supermatrices or concatenated datasets for phylogenetic inference, comparative genomics, or multi-partition analyses where per-dataset character partitions and taxon sets should be preserved and provenance (via name prefixing) must be maintained. The returned Nexus instance is suitable for downstream Biopython operations that consume Nexus-format matrices, plotting or partition-aware phylogenetic tools, or for exporting to Nexus files for use with external phylogenetic software.
    """
    from Bio.Nexus.Nexus import combine
    return combine(matrices)


################################################################################
# Source: Bio.Nexus.Nexus.get_start_end
# File: Bio/Nexus/Nexus.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_Nexus_Nexus_get_start_end(sequence: str, skiplist: tuple = ('-', '?')):
    """Return the zero-based start and end indices of the first and last characters
    in a sequence that are not members of a skiplist. This function is used in
    Biopython's NEXUS/alignment parsing (Bio.Nexus.Nexus) to locate the ungapped
    region of an aligned molecular sequence (for example, to trim leading and
    trailing gap ('-') or unknown ('?') symbols before downstream processing such
    as consensus building, distance calculations, or writing trimmed alignments).
    
    Args:
        sequence (str): The input sequence to inspect. This should be a Python
            string representing a biological sequence (for example a DNA, RNA or
            protein alignment row) that may contain gap or unknown symbols. The
            function computes len(sequence) and indexes into sequence, so passing a
            non-string or a non-indexable object will raise a TypeError. An empty
            string (length 0) is treated specially and yields (None, None).
        skiplist (tuple): A tuple of characters to ignore when scanning from the
            ends of sequence. By convention in NEXUS and other alignment contexts,
            this defaults to ('-', '?') to represent gap and unknown characters.
            Elements of skiplist are compared against individual characters of
            sequence using membership (sequence[i] in skiplist). For typical use,
            provide single-character string elements that match the gap/unknown
            symbols present in your alignment.
    
    Returns:
        tuple: A 2-tuple (start, end) giving the zero-based indices of the first
        and last characters in sequence that are not in skiplist. The end index is
        inclusive (so the slice sequence[start:end+1] yields the ungapped region).
        Special return values reflect edge cases: (None, None) is returned when
        sequence is empty (len(sequence) == 0). If sequence is non-empty but every
        character belongs to skiplist (the sequence is entirely gaps/ignored
        symbols), the function returns (-1, -1). Under normal circumstances for a
        non-empty sequence containing at least one non-skiplist character, both
        start and end are integers satisfying 0 <= start <= end < len(sequence).
    
    Behavior and failure modes:
        The function scans from the end and the start in linear time relative to
        len(sequence), decrementing the end index while trailing characters are in
        skiplist and incrementing the start index while leading characters are in
        skiplist. There are no side effects (the input sequence is not modified).
        If sequence does not support len() or indexing (for example None or an
        incompatible type), a TypeError (or the underlying exception from those
        operations) will be raised. If skiplist is not an iterable of comparable
        elements, membership tests may raise a TypeError. The function does not
        perform additional validation or normalization of characters; callers are
        responsible for ensuring skiplist matches the representation used in the
        sequence.
    """
    from Bio.Nexus.Nexus import get_start_end
    return get_start_end(sequence, skiplist)


################################################################################
# Source: Bio.Nexus.Nexus.safename
# File: Bio/Nexus/Nexus.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_Nexus_Nexus_safename(name: str, mrbayes: bool = False):
    """Bio.Nexus.Nexus.safename: Return a taxon identifier formatted to meet the NEXUS file conventions used in phylogenetics and compatible with the MrBayes program when requested.
    
    This function is used when writing taxon labels into NEXUS-format files (commonly produced or consumed in computational molecular biology and phylogenetic workflows handled by Biopython). It produces a safe string representation of the provided taxon name according to the NEXUS standard: by default quoting names that contain punctuation or whitespace and escaping single quotes, or by producing a MrBayes-compatible unquoted identifier when mrbayes=True.
    
    Args:
        name (str): The input taxon name to be converted into a NEXUS-safe identifier. This should be a Python string containing the taxon label as used in downstream phylogenetic data (for example, species names, sample identifiers, or sequence labels). The function calls standard string methods on this value; if a non-str is passed, an exception will be raised by those methods.
        mrbayes (bool): If False (the default), produce a NEXUS-compliant name by doubling any single quote characters and surrounding the name with single quotes if it contains any character from the module constants WHITESPACE or PUNCTUATION. If True, produce an identifier intended for use with the MrBayes software by first replacing spaces with underscores and then retaining only characters present in the module constant MRBAYESSAFE (i.e., removing any characters not in MRBAYESSAFE). Use this flag when generating labels specifically for MrBayes input where quotes and many punctuation characters are not allowed.
    
    Returns:
        str: A new string that is a NEXUS-safe taxon identifier derived from the input name. When mrbayes is False, this will be the original name with single quotes doubled and, if any whitespace or punctuation characters (as defined by WHITESPACE and PUNCTUATION in this module) are present, the entire string wrapped in single quotes (to follow the NEXUS quoting/escaping convention). When mrbayes is True, this will be the original name with spaces converted to underscores and all characters not in MRBAYESSAFE removed. The function has no side effects (it does not modify the input object) and performs no further validation: for example, it does not check for uniqueness of the returned identifier within a dataset, nor enforce any length limits required by external tools. Note that it is possible for the returned string to be empty (for example, if mrbayes=True and none of the input characters are in MRBAYESSAFE); callers should handle such edge cases as appropriate.
    """
    from Bio.Nexus.Nexus import safename
    return safename(name, mrbayes)


################################################################################
# Source: Bio.PDB.DSSP.dssp_dict_from_pdb_file
# File: Bio/PDB/DSSP.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_PDB_DSSP_dssp_dict_from_pdb_file(
    in_file: str,
    DSSP: str = "dssp",
    dssp_version: str = "3.9.9"
):
    """Bio.PDB.DSSP.dssp_dict_from_pdb_file: Create a DSSP dictionary from a PDB file by invoking an external DSSP executable and parsing its output into a Python mapping of residues to DSSP annotations. This function is used in structural bioinformatics workflows (see the Biopython project) to obtain per-residue secondary structure assignments and solvent accessibility values from a PDB-format coordinate file by calling the DSSP program and converting its output into a convenient Python data structure.
    
    Args:
        in_file (string): Path to the input PDB file on disk. This is the file containing atomic coordinates for one or more chains/residues. The function passes this path as the structure input to the external DSSP executable. The caller is responsible for ensuring the file exists and is a valid PDB file; otherwise the external DSSP program may fail and this function may raise an exception.
        DSSP (string): Name or path of the DSSP executable to invoke (argument passed to subprocess). Typical values are "dssp" or "mkdssp". The function first attempts to call this executable; if it is not found and the value is not "mkdssp", the function will attempt to call "mkdssp" as a fallback. If you explicitly set DSSP="mkdssp" and that executable is not present, a FileNotFoundError will be raised. The value also affects command-line arguments selected depending on dssp_version.
        dssp_version (string): Version string of the DSSP executable (for decision logic only). This string is compared to "4.0.0" (using the module's version comparison helper) to decide which command-line arguments are passed to the executable: for versions older than 4.0.0 the function calls DSSP with the PDB filename only; for 4.0.0 and newer it adds "--output-format=dssp" before the filename. This parameter does not probe the actual program binary for its version; it is used by the function to choose an appropriate invocation style and should reflect the known or expected DSSP version available in your environment. The default is "3.9.9", which selects the older-style invocation.
    
    Behavior and side effects:
        The function launches an external process via subprocess.Popen to execute the specified DSSP program with universal newlines enabled so that the program's textual output can be read and parsed as text. Standard output from DSSP is captured and parsed; standard error is also captured. If the initial attempt to run the specified DSSP executable raises FileNotFoundError, the function will try the fallback executable name "mkdssp" unless DSSP was explicitly set to "mkdssp", in which case the FileNotFoundError is re-raised to the caller.
        If the DSSP process writes to stderr, the function issues a Python warnings.warn() with the stderr content. If stderr is non-empty and stdout is empty, the function raises a generic Exception("DSSP failed to produce an output") because no parsable DSSP output was produced. The function does not modify the input PDB file.
        The function calls an internal parser _make_dssp_dict to convert DSSP stdout into a mapping. The parsing expects the DSSP program to produce output in the DSSP text format (selected by the invocation style determined from dssp_version). Because the function depends on an external executable, callers must ensure that the correct DSSP binary is installed and accessible in the system PATH or specify an explicit path via the DSSP argument.
    
    Failure modes:
        FileNotFoundError if the specified DSSP executable is not present and fallback behavior does not apply (re-raised when DSSP == "mkdssp").
        warnings.warn is emitted when DSSP writes to stderr; if stderr is non-empty and stdout is empty an Exception is raised indicating DSSP produced no output.
        Parsing errors may occur if DSSP writes malformed output or an unexpected output format; such errors will propagate from the internal parser _make_dssp_dict.
        Invalid or non-existent in_file paths will typically cause the DSSP executable to fail and either emit an error on stderr or produce no stdout, leading to a warning and possibly an Exception as described above.
    
    Returns:
        tuple: A two-item tuple (out_dict, keys).
            out_dict (dict): A dictionary mapping residue identity keys to DSSP annotation tuples. Each dictionary key is a residue identifier used by Biopython PDB parsing code (for example a tuple like ('A', (' ', 1, ' ')) where 'A' is the chain identifier and (' ', 1, ' ') denotes the residue id). The corresponding dictionary value is a tuple containing at least the amino acid type, the DSSP secondary structure code, and the accessibility information as reported by the DSSP program. Practically, out_dict is the primary data structure used by downstream analysis to query per-residue secondary structure and solvent accessibility in structural biology and bioinformatics pipelines.
            keys (object): An auxiliary value produced by the internal parser _make_dssp_dict that represents the ordered list or collection of residue keys derived from the DSSP output and corresponding to the entries in out_dict. This value can be used to iterate residues in the same order as they appeared in the DSSP output or to inspect which residues were parsed. The exact type and structure of keys is determined by _make_dssp_dict and mirrors the parser's output format.
    
    Example usage (conceptual):
        Call this function with the path to a PDB file and the name/path of the DSSP executable to obtain DSSP annotations for each residue, then query out_dict with a residue key to get its amino acid, secondary structure assignment and accessibility. This is commonly used when annotating structures for secondary structure analysis, calculating per-residue solvent exposure, or preparing features for machine learning models in computational structural biology.
    """
    from Bio.PDB.DSSP import dssp_dict_from_pdb_file
    return dssp_dict_from_pdb_file(in_file, DSSP, dssp_version)


################################################################################
# Source: Bio.PDB.DSSP.make_dssp_dict
# File: Bio/PDB/DSSP.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_PDB_DSSP_make_dssp_dict(filename: str):
    """Make a DSSP dictionary mapping residue identifiers to DSSP properties.
    
    Read a DSSP-format output file produced by the DSSP program (a tool commonly used in structural bioinformatics to assign secondary structure and solvent accessibility to residues in PDB structures) and return a Python dictionary that maps residue identifiers to the primary DSSP annotations. This function is part of the Bio.PDB.DSSP utilities in Biopython and is used by downstream code or users who need to attach DSSP-derived annotations (amino acid identity, secondary structure assignment, and solvent accessibility) to residues parsed from PDB files.
    
    Args:
        filename (str): Path to a DSSP output file (text). This must be the pathname of an existing DSSP result file previously generated by the DSSP program. The function opens the file for reading using the system default text encoding and reads its contents; it does not invoke the DSSP program itself.
    
    Behavior and side effects:
        The function opens the specified file, reads it, and delegates parsing to the internal helper _make_dssp_dict, then closes the file before returning. No file handles are left open after the call returns. The function does not modify the input file or any global state. It expects the file to be in the conventional DSSP text format; if the contents do not conform to the expected format, parsing performed by _make_dssp_dict may raise an exception (for example, ValueError or other parsing-related exceptions) which will propagate to the caller.
    
    Failure modes and errors:
        If the file does not exist or is not readable, open(filename) will raise FileNotFoundError or OSError. If the file is readable but not a valid DSSP file, the internal parser _make_dssp_dict may raise parsing-related exceptions. Callers should catch these exceptions if they need to handle missing or malformed DSSP files gracefully.
    
    Returns:
        dict: A dictionary mapping keys of the form (chainid, resid) to 3-tuples (aa, ss, accessibility). The key (chainid, resid) identifies a residue as used in DSSP: chainid is the chain identifier string and resid is the residue identifier string as reported in the DSSP file (typically containing the residue sequence number and any insertion code). The value tuple contains:
            aa: the amino acid identifier reported by DSSP (commonly a one-letter amino acid code string).
            ss: the secondary structure assignment reported by DSSP (a short string/code such as 'H', 'E', etc., following DSSP conventions).
            accessibility: the solvent accessibility value reported by DSSP for that residue (as parsed from the file; typically a numeric value representing accessible surface area).
        This returned mapping is intended for use in structural bioinformatics workflows (for example, annotating residues in Bio.PDB Structure/Residue objects, statistical analysis of secondary structure content, or filtering residues by accessibility).
    """
    from Bio.PDB.DSSP import make_dssp_dict
    return make_dssp_dict(filename)


################################################################################
# Source: Bio.PDB.DSSP.ss_to_index
# File: Bio/PDB/DSSP.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_PDB_DSSP_ss_to_index(ss: str):
    """Bio.PDB.DSSP.ss_to_index converts a single-letter DSSP secondary structure symbol into a small integer index used by Bio.PDB.DSSP and other Biopython code that needs numeric labels for protein secondary structure. In the context of Biopython (tools for computational molecular biology and bioinformatics), this mapping is used to convert DSSP output symbols into compact integer codes for array indexing, statistical summaries, or machine learning feature labels.
    
    Args:
        ss (str): A single-character DSSP secondary structure symbol. This function expects the exact, case-sensitive symbols produced by DSSP: "H" for alpha-helix, "E" for beta-strand (extended), and "C" for coil or other non-regular secondary structure. The argument is compared directly to these literal strings (no trimming or case conversion is performed).
    
    Returns:
        int: An integer index corresponding to the input secondary structure symbol: "H" -> 0, "E" -> 1, "C" -> 2. These small integers are suitable for indexing into fixed-size arrays, encoding class labels for downstream analysis, or compact storage of secondary structure assignments.
    
    Behavior and failure modes:
        The function performs simple equality checks against the three accepted symbols and returns the mapped integer immediately. There are no side effects, no external I/O, and no state changes. If ss is not exactly one of the accepted symbols ("H", "E", "C"), the function triggers an assertion failure (raises AssertionError because the implementation ends with assert 0). Because the implementation is case-sensitive and does not validate string length beyond equality, callers should ensure they pass the exact expected single-character DSSP symbol to avoid the AssertionError.
    """
    from Bio.PDB.DSSP import ss_to_index
    return ss_to_index(ss)


################################################################################
# Source: Bio.PDB.DSSP.version
# File: Bio/PDB/DSSP.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_PDB_DSSP_version(version_string: str):
    """Bio.PDB.DSSP.version parses a dot-separated semantic version string and returns its numeric components as a tuple of integers for lexicographic comparison.
    
    This function is used in the Bio.PDB.DSSP context (part of Biopython's PDB/DSSP handling) to convert human-readable version identifiers (for example, a DSSP program or file format version) into a machine-friendly representation that can be compared using Python's built-in tuple comparison semantics. It performs a straightforward split on the ASCII dot character (".") and converts each resulting segment to an int, preserving the original ordering and number of components: for example, "3.0.1" becomes (3, 0, 1). There are no side effects and no external dependencies; the function does not normalize component count (it neither pads nor truncates the returned tuple).
    
    Args:
        version_string (str): A dot-separated version string to parse, expected to contain one or more ASCII integer components separated by "." (for example "1.2.3" or "2.0"). In the Bio.PDB.DSSP domain this is typically the version reported by the DSSP program or associated files. The argument must be a string because the implementation calls the string split method; supplying a non-string may raise an AttributeError.
    
    Returns:
        tuple: A tuple of integers corresponding to the parsed version components in the same order they appeared in version_string (e.g., "3.0.1" -> (3, 0, 1)). The returned tuple is intended for lexicographic comparisons (for example, (3, 1, 0) > (3, 0, 9) evaluates to True).
    
    Failure modes:
        ValueError: Raised if any component of version_string cannot be converted to an integer (for example, "3.0a.1").
        AttributeError: May be raised if version_string does not support the split method (for example, if a non-string object without split is passed).
    """
    from Bio.PDB.DSSP import version
    return version(version_string)


################################################################################
# Source: Bio.PDB.NACCESS.process_asa_data
# File: Bio/PDB/NACCESS.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_PDB_NACCESS_process_asa_data(rsa_data: list):
    """Process lines from an NACCESS .asa file and return per-atom solvent accessible surface area (ASA) values mapped to atom identifiers.
    
    This function is used in the Bio.PDB.NACCESS module of Biopython to parse the atomic-level solvent accessible surface area output produced by the NACCESS program. In computational molecular biology and structural bioinformatics, these atomic ASA values are used to quantify solvent exposure of individual atoms (for example, when analysing surface accessibility, protein–ligand docking, or calculating residue solvent exposure by aggregation). The function expects rsa_data to be the sequence of text lines comprising a single .asa output file (commonly obtained by calling file.readlines() on an .asa file) and extracts fixed-column fields from each line according to the NACCESS .asa format as implemented in the source code.
    
    Args:
        rsa_data (list): A list of strings where each string is a single line from an NACCESS .asa output file. Each element is expected to be long enough to contain the fixed-width columns accessed by the function. The function extracts the atom name from characters line[12:16], the chain identifier from line[21], the residue sequence number from line[22:26] (converted to int), the residue insertion code from line[26], and the atomic solvent accessibility substring from line[54:62]. rsa_data typically comes from reading an .asa file produced by NACCESS; providing lines in a different format or shorter lines will lead to errors (see Raises).
    
    Returns:
        dict: A dictionary mapping an atom identifier tuple to the raw ASA substring (string) extracted from the .asa file. Each key is a 3-tuple (chainid, res_id, atom_id) where chainid is the single-character chain identifier string from the .asa line, res_id is a 3-tuple in the form (" ", resseq, icode) where resseq is the integer residue sequence number and icode is the insertion code character, and atom_id is the atom name string (trimmed of surrounding whitespace). The value for each key is the 8-character substring taken from line[54:62] (the solvent accessibility in Angstrom^2 as represented in the .asa file). The ASA values are returned as the original text substrings (they may include leading/trailing spaces); to use the ASA numerically, callers should call asa_value.strip() and convert to float (for example, float(asa_value.strip())).
    
    Behavior and side effects:
        The function creates and returns a new dictionary; it does not modify rsa_data or any global state. It relies on fixed-column slicing of each input line exactly as implemented in the source code (line[12:16], line[21], line[22:26], line[26], line[54:62]). No rounding or numeric conversion of ASA values is performed by this function; it preserves the raw substring from the file. Typical usage is to pass the list of lines obtained from reading an .asa file produced by NACCESS so downstream code can map atomic identifiers to their reported ASA.
    
    Failure modes:
        If any input line is shorter than the slices used, an IndexError (or implicit empty substring) may occur. If the characters in line[22:26] cannot be converted to an integer, int(...) will raise ValueError. If rsa_data is not an iterable of strings (for example, not a list of lines), other exceptions such as TypeError may be raised by iteration or slicing operations. Callers should validate that rsa_data is a list (or other iterable) of full-length .asa lines before using this function to avoid these errors.
    """
    from Bio.PDB.NACCESS import process_asa_data
    return process_asa_data(rsa_data)


################################################################################
# Source: Bio.PDB.NACCESS.process_rsa_data
# File: Bio/PDB/NACCESS.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_PDB_NACCESS_process_rsa_data(rsa_data: list):
    """Process residue-level SASA (solvent accessible surface area) data produced by the NACCESS .rsa output and return a structured mapping of residue identifiers to numeric accessibility metrics.
    
    This function is used in the Bio.PDB.NACCESS pipeline within Biopython to interpret the per-residue results produced by the external NACCESS program. It expects rsa_data to be the lines (strings) of a .rsa file already read into a Python list. The function scans those lines, processes only the records that start with the literal prefix "RES" (the standard NACCESS residue record), extracts fixed-column fields (residue name, chain identifier, residue sequence number, insertion code and multiple absolute/relative SASA values), converts the numeric fields to floats/integers and returns a dictionary keyed by (chain_id, res_id). The res_id value follows the PDB-style residue id tuple used in Biopython: (" ", resseq, icode) where the first element is the hetero-flag placeholder used here as a single space character.
    
    Args:
        rsa_data (list): A list of strings, each string being one line from an NACCESS .rsa output file. Each line is expected to be the fixed-width plain-text format produced by NACCESS. Only lines beginning with the three characters "RES" are parsed; other lines are ignored. The caller is responsible for passing the file contents as a list of lines (for example, obtained via file.readlines()). This parameter is required and there is no default.
    
    Returns:
        dict: A dictionary mapping keys of the form (chain_id, res_id) to per-residue metric dictionaries. chain_id is the single-character chain identifier extracted from the "RES" record. res_id is a 3-tuple of the form (" ", resseq, icode) where resseq is an int parsed from the residue sequence number columns and icode is a single-character insertion code. The mapped value is itself a dict with the following keys and types, representing the residue name and a set of solvent accessibility metrics (absolute areas in surface area units as produced by NACCESS, and relative percentages as produced by NACCESS):
            "res_name" (str): Three-letter residue name as taken from the .rsa file (e.g., "ALA", "LYS").
            "all_atoms_abs" (float): Absolute solvent accessible area for all atoms of the residue.
            "all_atoms_rel" (float): Relative solvent accessible area (percentage) for all atoms of the residue.
            "side_chain_abs" (float): Absolute solvent accessible area for the residue side chain.
            "side_chain_rel" (float): Relative solvent accessible area (percentage) for the residue side chain.
            "main_chain_abs" (float): Absolute solvent accessible area for the main chain atoms of the residue.
            "main_chain_rel" (float): Relative solvent accessible area (percentage) for the main chain atoms of the residue.
            "non_polar_abs" (float): Absolute solvent accessible area for non-polar atoms of the residue.
            "non_polar_rel" (float): Relative solvent accessible area (percentage) for non-polar atoms of the residue.
            "all_polar_abs" (float): Absolute solvent accessible area for polar atoms of the residue.
            "all_polar_rel" (float): Relative solvent accessible area (percentage) for polar atoms of the residue.
    
    Behavior and side effects:
        The function has no external side effects (it does not read or write files, and does not modify global state). It only returns the constructed dictionary. If multiple "RES" records map to the same (chain_id, res_id) key, the last processed record in rsa_data will overwrite earlier entries for that key. The function relies on the fixed-column layout of NACCESS .rsa "RES" records and uses string slicing exactly as in the original implementation to extract fields.
    
    Failure modes and errors:
        If a "RES" line is too short for the expected fixed-column slices, the function may raise IndexError. If numeric fields cannot be converted to int or float (for example, due to malformed text), a ValueError will be raised. The function does not attempt to validate ranges of numeric values beyond Python numeric conversion. Callers should ensure rsa_data contains valid NACCESS .rsa lines to avoid parsing exceptions.
    
    Practical significance:
        The returned mapping is intended for downstream use in structural bioinformatics workflows within Biopython, for example to annotate Bio.PDB Structure/Residue objects with solvent accessibility, to classify residues as buried or exposed, or to compute per-chain or per-structure exposure statistics from NACCESS output. The numeric values correspond to the areas and percentages reported by NACCESS for each residue.
    """
    from Bio.PDB.NACCESS import process_rsa_data
    return process_rsa_data(rsa_data)


################################################################################
# Source: Bio.PDB.PICIO.pdb_date
# File: Bio/PDB/PICIO.py
# Category: fix_docstring
# Reason: Schema parsing failed: Cannot generate JSON schema for pdb_date because the docstring has no description for the argument 'datestr'
################################################################################

def Bio_PDB_PICIO_pdb_date(datestr: str):
    """Bio.PDB.PICIO.pdb_date: Convert a date string from the ISO-like format yyyy-mm-dd to the PDB file convention dd-MMM-yy (three-letter uppercase month, two-digit year).
    This function is used in the PDB input/output utilities of the Bio.PDB.PICIO module within Biopython to format dates for PDB header fields and other PDB-formatted outputs when generating or writing structure files in computational molecular biology and bioinformatics workflows.
    
    The function accepts a single string and, if it matches the pattern of four-digit year, two-digit month, and two-digit day separated by hyphens, returns a reformatted date string where the day is preserved, the month is converted to the PDB three-letter uppercase code (JAN, FEB, ..., DEC), and the year is truncated to its last two digits. If the input is empty or does not match the expected pattern, the original string is returned unchanged. The function has no side effects and does not modify external state.
    
    Args:
        datestr (str): A date string expected in the form "YYYY-MM-DD" (four-digit year, two-digit month, two-digit day, with hyphen separators). In the context of Bio.PDB.PICIO this value typically originates from metadata or timestamp fields associated with PDB structures. If this string strictly matches the pattern, it will be converted to the PDB-style form "DD-MMM-YY" where MMM is an uppercase three-letter month code. If datestr is an empty string or does not match the required numeric pattern, it will be returned as-is, allowing calling code to detect and handle malformed or absent dates without an exception being raised.
    
    Returns:
        str: The reformatted date in the PDB convention "DD-MMM-YY" when the input matches the expected "YYYY-MM-DD" pattern. Examples of transformations performed by this function: "2023-06-01" -> "01-JUN-23". If the input is empty or does not match the pattern (for example, "2023/06/01", "June 1, 2023", or any non-matching string), the original input string is returned unchanged. The function does not raise errors for non-matching formats; it assumes callers will validate input types and content if stricter handling is required.
    """
    from Bio.PDB.PICIO import pdb_date
    return pdb_date(datestr)


################################################################################
# Source: Bio.PDB.PSEA.psea
# File: Bio/PDB/PSEA.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_PDB_PSEA_psea(pname: str):
    """Bio.PDB.PSEA.psea parses the output file produced by the PSEA secondary-structure assignment step and returns the parsed secondary-structure string for the input protein identifier.
    
    Args:
        pname (str): The string identifier passed to run_psea() that identifies the protein input for which PSEA output should be produced or located. In the Biopython PDB/PSEA workflow this typically is a PDB-related name or file identifier used by run_psea to generate or reference the PSEA output file. This function does not validate the format of pname; it simply forwards it to run_psea(pname) and then opens the filename returned by that call.
    
    Detailed behavior and side effects:
        The function calls run_psea(pname) to obtain the path to the PSEA output file (this call is a side effect of psea and may itself create or modify files depending on run_psea's implementation). It then opens that file for reading using the default system encoding and scans line-by-line until it finds a marker line that begins with the six characters ">p-sea". After this marker, psea concatenates each subsequent non-blank line into a single string, removing the last character of each line prior to concatenation (the code uses line[0:-1] for this purpose). Concatenation stops when the function encounters a blank line (a line whose first character is a newline). If the marker is never found, or if the marker is found but is immediately followed by a blank line, the function will return an empty string. Any file system or I/O errors raised while calling run_psea or opening/reading the returned filename (for example FileNotFoundError, PermissionError, or UnicodeDecodeError) will propagate to the caller.
    
    Failure modes and implementation caveats:
        Because the function strips the final character of each parsed line unconditionally (line[0:-1]), if a parsed line does not end with a newline character the last non-newline character will still be removed; callers should be aware of this behavior when interpreting the returned string. The function assumes the PSEA output file uses the ">p-sea" marker and a blank line to terminate the relevant block; files that deviate from this format will not be parsed as intended. The function does not perform further validation of the parsed content; it typically returns a per-residue secondary-structure annotation string (e.g., characters representing helix/sheet/coil) used in downstream Biopython PDB analyses.
    
    Returns:
        str: The concatenated secondary-structure data extracted from the PSEA output block following the ">p-sea" marker, with the final character of each line removed before concatenation. If no data block is found, an empty string is returned.
    """
    from Bio.PDB.PSEA import psea
    return psea(pname)


################################################################################
# Source: Bio.PDB.PSEA.psea2HEC
# File: Bio/PDB/PSEA.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_PDB_PSEA_psea2HEC(pseq: str):
    """Translate a PSEA secondary structure string into HEC codes.
    
    This function is part of the Bio.PDB.PSEA utilities in Biopython and is used to convert the one-letter codes produced by the PSEA secondary structure assignment into the HEC convention commonly used in protein structure annotation and downstream PDB processing. The PSEA input uses lowercase codes: 'a' for alpha-helix, 'b' for beta-strand, and 'c' for coil/other. This function maps those to uppercase HEC codes: 'H' for helix, 'E' for strand, and 'C' for coil. The output is suitable for storing or visualizing secondary structure annotations alongside residue-based data (one output character per input character).
    
    Args:
        pseq (str): A PSEA secondary structure string where each character is expected to be one of 'a', 'b', or 'c'. Each character in pseq represents the secondary structure assignment for a single residue in sequence order. Typical use is to pass the raw PSEA output string for a chain or whole protein; an empty string is allowed and results in an empty output list.
    
    Behavior and side effects:
        The function iterates over pseq and translates each character using the mapping 'a' -> 'H', 'b' -> 'E', 'c' -> 'C', appending the resulting single-character string to an output list. The function has no side effects (it does not modify its input or any external state) and runs in linear time proportional to len(pseq). It returns a new list with one element per input character, preserving input order. Because the implementation does not handle unexpected characters, if pseq contains any character other than 'a', 'b', or 'c', the function will fail when attempting to append an undefined variable and will raise an UnboundLocalError at runtime. Callers should validate or sanitize pseq before calling if there is any possibility of other characters.
    
    Returns:
        list[str]: A list of single-character strings, each one of 'H', 'E', or 'C', corresponding to the HEC secondary structure code for each residue in the input pseq, in the same order. An empty input string returns an empty list.
    """
    from Bio.PDB.PSEA import psea2HEC
    return psea2HEC(pseq)


################################################################################
# Source: Bio.PDB.PSEA.run_psea
# File: Bio/PDB/PSEA.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_PDB_PSEA_run_psea(fname: str, verbose: bool = False):
    """Run PSEA on a structure file and return the generated ".sea" output filename.
    
    This function is part of the Bio.PDB.PSEA integration in Biopython and is used to invoke the external P-SEA command-line program from Python to process a structure file (for example a PDB file) and produce the P-SEA tool output. It constructs and runs the external command ["psea", fname] using subprocess.run with captured text output, and then verifies that P-SEA produced an output file in the current working directory. This function assumes the Biopython workflow where an external P-SEA binary is available on the system PATH and is used to post-process or analyze protein structure files as part of PDB-related analyses.
    
    Args:
        fname (str): Path to the input file to pass to the P-SEA binary. This is the filename (or path) that will be given unchanged as the second element of the command line ["psea", fname]. The function derives the expected output filename by taking the final path component (everything after the last "/") and splitting that final component at the first "."; the text before that first "." is used as the base name. For example, "/path/to/1abc.pdb" produces a base of "1abc" and an expected output "1abc.sea". Note that if the final path component contains multiple dots (for example "a.b.pdb"), only the portion before the first dot ("a" in that example) is used for the output filename. Provide a valid file path string to an input file that the external P-SEA program accepts.
        verbose (bool): If True, print the captured standard output (stdout) produced by the P-SEA process to the current Python process standard output. Defaults to False, in which case stdout is suppressed and only captured internally. This flag does not affect whether the function checks for P-SEA errors or whether the output file is created.
    
    Behavior and side effects:
        The function runs the external program "psea" found on the system PATH. It constructs the command ["psea", fname] and executes it with subprocess.run(capture_output=True, text=True), so both stdout and stderr are captured as strings. P-SEA is expected to write an output file named "<base>.sea" into the current working directory, where <base> is computed as described above from the final component of fname. The function does not change the current working directory; the output file is therefore created in whatever directory the Python process is currently running in. If verbose is True, the function prints the captured stdout to standard output. If the "psea" binary is not on PATH, subprocess.run will raise a FileNotFoundError; this function does not catch that exception. The function does not modify the input file; it only invokes the external process and checks for the resulting .sea file.
    
    Failure modes:
        If the external process writes any non-whitespace content to stderr, or if the expected output file "<base>.sea" does not exist in the current working directory after the process completes, the function raises a RuntimeError whose message includes the captured stderr content from the P-SEA run. If the P-SEA binary is missing from PATH, subprocess.run will raise FileNotFoundError before the function can perform its checks. Any other exceptions raised by subprocess.run or os.path.exists propagate to the caller.
    
    Returns:
        str: The relative filename of the P-SEA output file created in the current working directory, exactly "<base>.sea" where <base> is derived from the final path component of the provided fname as described above. The returned value is intended to be used by downstream Biopython PDB processing code to locate and open the P-SEA results file for further parsing or analysis.
    """
    from Bio.PDB.PSEA import run_psea
    return run_psea(fname, verbose)


################################################################################
# Source: Bio.PDB.Polypeptide.index_to_one
# File: Bio/PDB/Polypeptide.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_PDB_Polypeptide_index_to_one(index: int):
    """Index to corresponding one-letter amino acid name used by Bio.PDB.Polypeptide.
    
    Convert an integer index into the corresponding one-letter amino acid code as used by the Bio.PDB.Polypeptide utilities in Biopython. This function is intended for translating numeric indices (for example, positions in an internal numeric mapping of the 20 standard amino acids) into the single-character amino acid codes commonly used in sequence representation and structural analyses. The implementation performs a direct dictionary lookup against the module-level mapping dindex_to_1. Typical usage is when building or comparing polypeptide sequences extracted from PDB structures where a compact one-letter code is required (for example, assembling a sequence string from residue indices for sequence alignment or annotation). The original doctest examples demonstrate the mapping: index_to_one(0) returns 'A' and index_to_one(19) returns 'Y'.
    
    Args:
        index (int): Integer index to convert to a one-letter amino acid code. In the context of Bio.PDB.Polypeptide this index corresponds to the position/key used by the module-level mapping dindex_to_1 that maps numeric indices to the 20 standard amino acids' one-letter codes. The argument must be an integer; passing non-integer types will not match keys in the mapping and will typically result in a lookup failure. Providing an index not present in dindex_to_1 (for example, out of the mapping's defined range) will raise a KeyError.
    
    Returns:
        str: A single-character string containing the one-letter amino acid code corresponding to the provided index (for example, 'A', 'R', 'N', ...). This return value is suitable for concatenation into polypeptide sequences, comparison with sequence databases, or downstream analyses that expect standard one-letter amino acid notation.
    
    Raises:
        KeyError: If index is not a key in the internal dindex_to_1 mapping (e.g., an out-of-range integer), a KeyError is raised by the underlying dictionary lookup.
        TypeError: If index is of a type that is not hashable or not compatible with the dictionary keys, the lookup may raise a TypeError.
    """
    from Bio.PDB.Polypeptide import index_to_one
    return index_to_one(index)


################################################################################
# Source: Bio.PDB.Polypeptide.index_to_three
# File: Bio/PDB/Polypeptide.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_PDB_Polypeptide_index_to_three(i: int):
    """Bio.PDB.Polypeptide.index_to_three maps an integer index to the corresponding three-letter amino acid residue name used in PDB-style representations and by the Bio.PDB.Polypeptide utilities for sequence and structure handling. This function is used in the Biopython PDB module to convert numeric residue indices (for example, indices in the range 0–19 for the 20 standard amino acids) into the conventional three-letter residue codes (e.g., 0 -> 'ALA', 19 -> 'TYR') that appear in PDB files and are required by downstream PDB/structure-processing routines.
    
    Args:
        i (int): Integer index representing an amino acid. In the context of Bio.PDB.Polypeptide, this integer is used as a lookup key into the module-level mapping dindex_to_3 to retrieve the canonical three-letter, uppercase residue name. The value should correspond to an index present in that mapping; typical use is with indices for the twenty standard amino acids.
    
    Returns:
        str: The three-letter, uppercase amino acid name corresponding to the input index (for example, 'ALA' for alanine). This string is intended for use in PDB residue naming, sequence annotation, and any Bio.PDB code paths that require the three-letter residue code.
    
    Behavior and side effects:
        This function performs a direct lookup in the internal mapping dindex_to_3 and has no side effects: it does not modify input arguments or global state itself. The return value is derived solely from the current contents of dindex_to_3, so if that module-level mapping is mutated elsewhere in the program, the result will reflect those changes.
    
    Failure modes and errors:
        If i is not a valid key/index for the internal mapping dindex_to_3, the lookup will fail and an exception will be raised (for example, KeyError if dindex_to_3 is a dict, or IndexError if it is a sequence). Callers should ensure i is valid for the expected mapping (commonly 0–19 for standard amino acids) or handle these exceptions as appropriate. There are no implicit defaults; an explicit valid integer is required.
    """
    from Bio.PDB.Polypeptide import index_to_three
    return index_to_three(i)


################################################################################
# Source: Bio.PDB.Polypeptide.one_to_index
# File: Bio/PDB/Polypeptide.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_PDB_Polypeptide_one_to_index(s: str):
    """One-letter amino acid code to integer index used by Bio.PDB.Polypeptide.
    
    Converts a single-character amino acid one-letter code into the integer index used
    by the Bio.PDB.Polypeptide module. This function is used in Biopython (a toolkit
    for computational molecular biology) to map residue one-letter codes to numeric
    indices so residues can be looked up or used to index arrays, tables or feature
    vectors that store per-residue properties in structural bioinformatics code.
    The mapping is provided by the module-level dictionary d1_to_index; examples
    from the original implementation include one_to_index('A') == 0 and
    one_to_index('Y') == 19.
    
    Args:
        s (str): A one-character string containing a standard amino acid one-letter
            code. This parameter represents the residue identity in the common
            biological convention (for example, 'A' for alanine, 'Y' for tyrosine).
            The function expects a str and uses it as a key into the module-level
            mapping d1_to_index to produce the corresponding integer index. Passing
            strings that are not valid single-letter amino acid codes (or keys not
            present in d1_to_index) will result in a KeyError; passing values of
            non-hashable types will raise the usual Python TypeError prior to
            lookup.
    
    Returns:
        int: The integer index corresponding to the provided one-letter amino acid
            code as defined by d1_to_index. The returned index is intended for use
            as a 0-based index into arrays or lists that represent data for the
            standard amino acid alphabet in Bio.PDB.Polypeptide. There are no side
            effects; the function performs a dictionary lookup and returns the
            mapped integer or raises an exception if the input key is not found.
    """
    from Bio.PDB.Polypeptide import one_to_index
    return one_to_index(s)


################################################################################
# Source: Bio.PDB.Polypeptide.three_to_index
# File: Bio/PDB/Polypeptide.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_PDB_Polypeptide_three_to_index(s: str):
    """Convert a three-letter amino acid residue name to its integer index used by Bio.PDB.Polypeptide.
    
    This function is part of the Bio.PDB.Polypeptide utilities in Biopython, a toolkit for computational molecular biology. It maps a PDB-style three-letter amino acid code (as used in PDB files and in Biopython's Polypeptide handling) to the corresponding integer index from the module's internal mapping (d3_to_index). This integer index is used in downstream polypeptide processing tasks such as array indexing, residue ordering, sequence conversion, and compatibility with other algorithms in Bio.PDB that rely on a canonical ordering of the standard amino acids. For example, three_to_index('ALA') returns 0 and three_to_index('TYR') returns 19 in the standard mapping used here.
    
    Args:
        s (str): A three-character amino acid residue name string. This must match the keys of the module-level mapping exactly (case-sensitive), following the PDB three-letter convention (for example, 'ALA', 'GLY', 'TYR'). There are no default values; the caller must supply a string.
    
    Returns:
        int: The integer index corresponding to the provided three-letter residue name from the internal d3_to_index mapping. This integer can be used for indexing arrays or for converting between residue name representations in Bio.PDB code.
    
    Raises:
        KeyError: If the provided string is not present in the internal d3_to_index mapping (for example a non-standard residue name, incorrect case, or an unexpected string), a KeyError will be raised because the function performs a direct dictionary lookup.
        TypeError: If a non-hashable or otherwise invalid object is passed such that dictionary lookup is not possible, a TypeError may be raised by the underlying dictionary access.
    
    Side effects:
        None. The function performs a read-only lookup on the module-level mapping d3_to_index and does not modify global state or its inputs.
    
    Notes:
        The mapping d3_to_index is defined elsewhere in the Bio.PDB.Polypeptide module; users relying on a specific ordering should consult that mapping if reproducible index values across Biopython versions are required.
    """
    from Bio.PDB.Polypeptide import three_to_index
    return three_to_index(s)


################################################################################
# Source: Bio.PDB.ResidueDepth.min_dist
# File: Bio/PDB/ResidueDepth.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_PDB_ResidueDepth_min_dist(coord: numpy.ndarray, surface: numpy.ndarray):
    """Return the minimum Euclidean distance between a single 3D coordinate and a set of surface points.
    
    Args:
        coord (numpy.ndarray): A 1-dimensional array representing a single 3D point (an atomic or residue coordinate) in Cartesian space used by Bio.PDB.ResidueDepth. This function expects coord to contain the x, y, z coordinates of the atom/residue whose distance to the molecular surface is being measured. The numeric values are interpreted in the same units as the surface coordinates (for example, angstroms if the structure uses angstroms). The value is not modified by the function.
        surface (numpy.ndarray): A 2-dimensional array of shape (N, 3) where each row is a 3D Cartesian coordinate for a point on the molecular surface (N surface sample points). In the context of Bio.PDB.ResidueDepth, this is typically a precomputed set of surface points used to assess how buried an atom or residue is relative to the solvent-accessible surface. The array is not modified by the function.
    
    Returns:
        float: The minimum Euclidean distance between coord and any row in surface. This is computed by subtracting coord from each surface point, summing squared differences over the three Cartesian components, taking the minimum squared distance, then returning its square root. The returned distance has the same units as the input coordinates and is useful as a scalar measure of how close the given coordinate is to the molecular surface (lower values indicate more exposed positions; larger values indicate more buried positions).
    
    Behavior, side effects, and failure modes:
        The computation is vectorized using NumPy and has O(N) time complexity in the number of surface points N. The function does not modify coord or surface. coord is broadcast against surface for the subtraction; therefore coord must be compatible with subtraction from surface (practically, coord should be 1-D with three elements and surface should have shape (N, 3)). If surface is empty (zero rows), calling min on the distance array will raise a ValueError. If coord and surface shapes are incompatible for the required elementwise operations or for summation over axis 1, NumPy will raise an AxisError or a related indexing/shape error. Users should ensure inputs are numeric NumPy arrays with matching 3-component Cartesian coordinates for meaningful results.
    """
    from Bio.PDB.ResidueDepth import min_dist
    return min_dist(coord, surface)


################################################################################
# Source: Bio.PDB.Selection.get_unique_parents
# File: Bio/PDB/Selection.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_PDB_Selection_get_unique_parents(entity_list: list):
    """Translate a list of entities to a list of their unique parent objects.
    
    This function is intended for use in the Bio.PDB selection and traversal workflows in Biopython, where one often has a set of lower-level PDB entities (for example Atom or Residue objects) and needs the corresponding parent-level entities (for example Residue or Chain objects) without duplicates. The function iterates over the provided list, calls each element's get_parent() method to obtain its parent, and returns a list containing each distinct parent exactly once.
    
    Args:
        entity_list (list): A list of entity objects that implement a get_parent() method. In the Bio.PDB domain these are typically objects such as Atom, Residue, or Chain instances; the role of each element is to serve as a child whose parent will be retrieved. The function does not modify the objects in entity_list.
    
    Returns:
        list: A list containing the unique parent objects of the input entities. Uniqueness is determined by Python set semantics (hash and equality) because the implementation builds a set of parents internally. The returned list therefore contains one representative for each distinct parent object found among the inputs.
    
    Behavior, side effects, defaults, and failure modes:
        The function has no side effects: it does not alter the input list or the entities themselves. If entity_list is empty, the function returns an empty list. The order of elements in the returned list is not guaranteed and may be arbitrary because a set is used to remove duplicates; callers that require a specific order must reorder the result themselves after calling this function. Each element of entity_list must have a callable get_parent() method; if an element lacks this method, an AttributeError will be raised. The parent objects must be hashable (implement __hash__ and __eq__ consistently); if a parent object is unhashable, a TypeError will be raised when the set is constructed. The notion of "parent" and the practical significance of this function follow the Bio.PDB usage pattern: for example, obtaining Residue parents from a list of Atom objects so that downstream operations (such as selection, counting, or writing out higher-level entities) operate on unique parent units rather than repeated children.
    """
    from Bio.PDB.Selection import get_unique_parents
    return get_unique_parents(entity_list)


################################################################################
# Source: Bio.PDB.Selection.uniqueify
# File: Bio/PDB/Selection.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_PDB_Selection_uniqueify(items: list):
    """Bio.PDB.Selection.uniqueify returns a list containing the unique elements from the provided list. This function is used in the Bio.PDB selection and processing code within Biopython (a toolkit for computational molecular biology) to remove duplicate entries when assembling or filtering PDB-related collections (for example, lists of atom identifiers, residue objects, or other selection results). The deduplication is performed via Python's set, so the operation is fast for hashable items but does not preserve the original input order.
    
    Args:
        items (list): A Python list of elements to deduplicate. Each element should be a hashable Python object (for example, ints, strings, tuples, or user-defined objects that implement __hash__ and __eq__). In the Biopython PDB domain, typical contents are atom or residue identifiers or lightweight objects representing selections. The order of elements in this input list is ignored by this function; duplicates are removed based on each element's hash and equality semantics.
    
    Returns:
        list: A new list containing one instance of each distinct element present in the input list. The order of elements in the returned list is not defined and will not match the order of the input list. Uniqueness is determined using the elements' __hash__ and __eq__ behavior as required by Python sets.
    
    Behavior, side effects, and failure modes:
        This function has no side effects on the input list and returns a freshly constructed list. Internally it converts the input list to a set and then back to a list (list(set(items))). As a consequence, any unhashable elements in items (for example, lists or dicts) will cause Python to raise a TypeError when attempting to build the set. Additionally, two distinct objects that compare equal (via __eq__) and have the same hash will be considered the same element and only one will appear in the result; conversely, objects with identity equality but differing hash/equality implementations may both appear. If preserving the original input order is required for downstream PDB processing, use an alternative order-preserving deduplication approach instead of this function.
    """
    from Bio.PDB.Selection import uniqueify
    return uniqueify(items)


################################################################################
# Source: Bio.PDB.alphafold_db.get_predictions
# File: Bio/PDB/alphafold_db.py
# Category: fix_docstring
# Reason: Schema parsing failed: ("Couldn't parse this type hint, likely due to a custom class or object: ", collections.abc.Iterator[dict])
################################################################################

def Bio_PDB_alphafold_db_get_predictions(qualifier: str):
    """Get all AlphaFold predictions for a UniProt accession.
    
    This function is part of Bio.PDB.alphafold_db and is used by Biopython users and tools to fetch AlphaFold prediction records for a given UniProt accession (for example, "P00520") from the AlphaFold public API. It performs an HTTPS request to the AlphaFold endpoint for the supplied qualifier, decodes the JSON response, and yields each prediction record as a Python dictionary. This enables downstream code in computational molecular biology and structural bioinformatics workflows to iterate over prediction records for integration with PDB parsing, annotation, or analysis.
    
    Args:
        qualifier (str): A UniProt accession string used to identify the protein for which AlphaFold predictions are requested, e.g. "P00520". The function does not validate the accession beyond inserting it into the request URL; callers should provide a valid UniProt accession as required by the AlphaFold API.
    
    Returns:
        collections.abc.Iterator[dict]: An iterator that yields dictionaries. Each yielded dict is a single prediction record exactly as returned by the AlphaFold API JSON response for the given UniProt accession. The function yields zero items if the API returns an empty list.
    
    Behavior and side effects:
        The function constructs the request URL "https://alphafold.com/api/prediction/{qualifier}", performs a synchronous network request using urllib (urlopen), reads the response bytes, decodes to text, parses the text as JSON with json.loads, and yields the resulting sequence elements. This is blocking network I/O and will contact an external service (alphafold.com). There is no internal caching, no automatic retries, and no authentication handling.
    
    Failure modes and exceptions:
        The function can raise exceptions originating from network I/O and JSON decoding, for example urllib.error.URLError or urllib.error.HTTPError on connection/HTTP errors, UnicodeDecodeError if the response bytes cannot be decoded as text, or json.JSONDecodeError if the response is not valid JSON. Callers should handle these exceptions and consider network availability and API changes. If the API returns an empty JSON list, the function returns an empty iterator (i.e., no yielded dicts).
    """
    from Bio.PDB.alphafold_db import get_predictions
    return get_predictions(qualifier)


################################################################################
# Source: Bio.PDB.internal_coords.set_accuracy_95
# File: Bio/PDB/internal_coords.py
# Category: valid
################################################################################

def Bio_PDB_internal_coords_set_accuracy_95(num: float):
    """Reduce floating point accuracy to "9.5" (format xxxx.xxxxx) by rounding to five decimal places and returning a float.
    
    This helper is used by the IC_Residue class in Bio.PDB.internal_coords when writing PIC and SCAD files. In that domain, limiting numeric precision to five digits after the decimal reduces file size, improves human readability of coordinate files, and ensures consistent numeric formatting required by downstream tools that consume PIC/SCAD output.
    
    The implementation formats the input using Python string formatting with a fixed field width and five decimal places, then converts the formatted string back to float (float(f"{num:9.5f}")). This yields a numeric value rounded to five decimal places; the specified field width (9) ensures consistent formatting when producing text files, although any leading spaces are removed by the float conversion. A previously used alternative, round(num, 5), was commented out in the source because it was measured to be slower in this code path.
    
    Args:
        num (float): Input floating-point number representing a coordinate or other numeric value used in internal coordinate calculations. This function expects a Python float as documented in the function signature; passing a non-float that cannot be formatted as a floating-point number may raise a TypeError or ValueError at format-time. Special IEEE float values such as NaN or infinity are propagated by the formatting and returned as their corresponding float values.
    
    Returns:
        float: A new float value representing the input rounded to five decimal places (the "9.5" accuracy). There are no side effects; the function is pure and does not modify its input. If the input cannot be formatted as a float, a runtime exception (TypeError or ValueError) will be raised.
    """
    from Bio.PDB.internal_coords import set_accuracy_95
    return set_accuracy_95(num)


################################################################################
# Source: Bio.PDB.qcprot.qcp
# File: Bio/PDB/qcprot.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_PDB_qcprot_qcp(coords1: numpy.ndarray, coords2: numpy.ndarray, natoms: int):
    """Bio.PDB.qcprot.qcp implements the Quaternion Characteristic Polynomial (QCP) algorithm in Python to compute the optimal rigid-body rotation and root-mean-square deviation (RMSD) that aligns two sets of 3D coordinates. This function is used in structural bioinformatics (as provided by Biopython) to superpose molecular structures (for example, protein atomic coordinates) and to obtain the rotation matrix and quaternion that maps the mobile coordinate set onto the reference coordinate set. The implementation follows the C implementation and Theobald et al.'s formulation (root-finding of a quartic via Newton–Raphson) and preserves variable naming to aid comparison with the original code.
    
    Args:
        coords1 (numpy.ndarray): Reference coordinate array with shape (N, 3). Each row is the x,y,z coordinates of one atom. In this implementation coords1 is treated as the reference structure (i.e., the target to which coords2 will be rotated). The array must be two-dimensional, numeric (floating point), and centered at the origin (the centroid of the coordinates removed) before calling this function; the algorithm assumes zero-centered coordinates for correct computation of cross-covariance and traces.
        coords2 (numpy.ndarray): Mobile coordinate array with shape (N, 3). Each row is the x,y,z coordinates of the corresponding atom in the mobile structure. coords2 is the structure that the computed rotation will map onto coords1. Like coords1, coords2 must be two-dimensional, numeric (floating point), and centered at the origin. coords1 and coords2 must have the same number of rows N (same ordering/correspondence of atoms).
        natoms (int): Number of atoms N used in the RMSD computation. This integer is used in the RMSD normalization (the returned RMSD is computed as sqrt(2 * |E0 - lambda_max| / natoms), where E0 is the average trace term and lambda_max is the largest root found). natoms must be a positive integer equal to the number of coordinate rows in coords1 and coords2; passing zero or a non-positive value will cause a division-by-zero or incorrect RMSD and should be avoided.
    
    Detailed behavior, side effects, defaults, and failure modes:
        - Purpose and algorithm: The function computes the 3x3 rotation matrix and quaternion that minimize the RMSD between two centered coordinate sets by solving the characteristic polynomial for the quaternion scalar part (largest root of a quartic) using Newton–Raphson iterations, then forming the corresponding eigenvector (quaternion) and converting it into a rotation matrix. This is the QCP method used widely in molecular superposition tasks within computational molecular biology.
        - Input requirements: Both coords1 and coords2 must be centered at the origin and have shape (N, 3). The caller is responsible for centering (subtracting centroids) and for ensuring matching atom order between the two arrays. The function uses numpy.dot and trace operations, so the inputs must be numpy.ndarray objects; passing other types will typically raise TypeError or produce incorrect results.
        - Treatment of coords1/coords2: The implementation comment in the source notes a swap relative to an original C implementation: here, coords1 is treated as the reference and coords2 as the mobile. The computed rotation maps coords2 onto coords1.
        - Newton–Raphson root finding: The code runs up to 50 Newton–Raphson iterations (nr_it = 50) with a convergence precision evalprec = 1e-11 for the quartic root-finding step. If the iterative solver does not converge within the maximum iterations, the function prints a diagnostic message to stdout: "Newton-Rhapson did not converge after 50 iterations". This print is a side effect visible to the user; no exception is raised in that case.
        - Numerical thresholds and degenerate cases: After finding the largest root, the function computes quaternion components from cofactors of the characteristic matrix. If the computed quaternion norm squared (qsqr) is below evecprec = 1e-6, the code attempts three alternative cofactor-based fallbacks to produce a stable quaternion. If all fallbacks still yield qsqr < evecprec, the function treats the case as numerically degenerate: it returns the computed RMSD, the 3x3 identity matrix as the rotation (no rotation applied), and the current quaternion components packed as a list of four floats. In this rare degeneracy branch the returned quaternion may be unnormalized and is returned as a Python list (numerical caller should handle this possibility).
        - Normal case output normalization: In the normal (non-degenerate) case, the quaternion is normalized to unit length and returned as a tuple (q1, q2, q3, q4), where q1 is the scalar (real) component and q2,q3,q4 are the vector components (x,y,z). The rotation matrix is constructed from this scalar-first quaternion in the conventional way and is a numpy.ndarray of shape (3, 3) containing floating-point entries.
        - RMSD computation: The returned RMSD is a Python float computed from the difference between the initial energy-like E0 and the found largest root (mxEigenV) as rmsd = sqrt(2 * |E0 - mxEigenV| / natoms). Provide a positive natoms equal to N to obtain a correct RMSD.
        - Error conditions: The function does not explicitly validate shapes or natoms; if coords1 and coords2 shapes do not match or are not Nx3, numpy operations (dot, trace, indexing) will raise exceptions (for example ValueError). If natoms is zero or negative, the RMSD computation will produce a division-by-zero or nonsensical value; the caller must ensure natoms matches the number of rows and is positive. Numeric overflow, underflow, or loss of precision in extremely large or pathological coordinate sets can lead to non-convergence or the degenerate-qsqr fallback described above.
        - Determinism: Given the same floating-point inputs and environment, the function is deterministic: it uses fixed iteration counts, thresholds, and algebraic operations. Floating-point rounding may affect results at the 1e-12..1e-6 level depending on platform and data.
    
    Returns:
        tuple: A 3-tuple (rmsd, rot, quat) where:
            rmsd (float): The root-mean-square deviation between coords1 and coords2 after optimal alignment, computed as sqrt(2 * |E0 - lambda_max| / natoms) with E0 and lambda_max from the QCP computation.
            rot (numpy.ndarray): A 3x3 rotation matrix (shape (3, 3), dtype float) that maps coords2 onto coords1 when multiplied on the left (rot @ coords2.T or applying per-atom as (rot @ p.T).T). In the degenerate numeric fallback the identity matrix is returned.
            quat (tuple or list of float): The quaternion representing the rotation, given in scalar-first order (q1, q2, q3, q4) where q1 is the scalar (real) component and q2,q3,q4 are the vector components (x,y,z). In the normal case this is a normalized tuple of four floats. In the rare degenerate fallback where eigenvector precision cannot be achieved, a list of four floats may be returned and the quaternion may be unnormalized.
    
    Practical significance in Biopython/structural biology:
        - This function is intended for use within Biopython (Bio.PDB) workflows that compare or superpose macromolecular structures, such as when computing RMSD between two protein conformations, aligning model and experimental chains, or in structural clustering. It provides a numerically-efficient, pure-Python implementation of the QCP method suitable for moderate-sized coordinate sets when NumPy is available.
    
    Note:
        - The caller should pre-center both coordinate arrays (subtract centroids) and pass natoms equal to the number of rows in the arrays. The function relies on numpy for linear algebra and will raise exceptions if the inputs are incompatible with numpy.dot and indexing operations.
    """
    from Bio.PDB.qcprot import qcp
    return qcp(coords1, coords2, natoms)


################################################################################
# Source: Bio.PDB.vectors.calc_angle
# File: Bio/PDB/vectors.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_PDB_vectors_calc_angle(v1: list, v2: list, v3: list):
    """Calculate the angle at the middle point (v2) formed by three connected point vectors.
    
    This function is used in the Bio.PDB.vectors module of Biopython to compute the geometric angle defined by three points commonly encountered in structural biology (for example, the internal angle at an atom in a protein backbone or side chain defined by three atom coordinates). The routine translates the coordinate system so that v2 is treated as the vertex, forms two arm vectors (v1 - v2 and v3 - v2), and returns the angle between those two arms using the Vector.angle method.
    
    Args:
        v1 (L{Vector}): The first point as a Vector object. In the three-point ordering v1, v2, v3 this represents one end of the angle. The function computes v1 - v2 to produce the first arm vector. This parameter should be an instance of the Biopython Vector type (documented as L{Vector}) or a compatible object that implements subtraction with v2 and provides the angle() method; it is used to supply the coordinates of the first connected point in structural calculations such as bond or dihedral angle analysis.
        v2 (L{Vector}): The central vertex point as a Vector object. This is the point at which the angle is evaluated (the middle of the three connected points). The function subtracts v2 from both v1 and v3 to translate coordinates so that v2 becomes the origin before the angle calculation. v2 must support the same operations as described for v1 and v3.
        v3 (L{Vector}): The third point as a Vector object. In the three-point ordering v1, v2, v3 this represents the other end of the angle. The function computes v3 - v2 to produce the second arm vector. As with v1 and v2, v3 must be compatible with vector subtraction and the Vector.angle method.
    
    Returns:
        float: The geometric angle between the vectors (v1 - v2) and (v3 - v2) as computed by the underlying Vector.angle method. The returned value is a floating-point number produced by Vector.angle and is intended for use in structural biology computations (for example, evaluating bond angles in PDB-derived coordinates).
    
    Behavior and failure modes:
        The implementation creates local arm vectors by subtracting v2 from v1 and v3 and then calls the angle method on the first arm with the second arm as argument (v1.angle(v3) after translation). The function itself does not intentionally mutate the caller's objects; it rebinds local names to the results of the subtraction operations. However, if the provided objects implement subtraction or angle in a way that mutates inputs, side effects may occur. If the supplied objects do not implement subtraction or the angle() method, or if the angle computation is undefined for the given vectors (for example due to zero-length vectors), underlying operations will raise exceptions such as TypeError, AttributeError, ZeroDivisionError, or other errors propagated from the Vector implementation. Users should ensure inputs are valid Biopython Vector instances (L{Vector}) or compatible objects before calling this function.
    """
    from Bio.PDB.vectors import calc_angle
    return calc_angle(v1, v2, v3)


################################################################################
# Source: Bio.PDB.vectors.calc_dihedral
# File: Bio/PDB/vectors.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_PDB_vectors_calc_dihedral(
    v1: numpy.ndarray,
    v2: numpy.ndarray,
    v3: numpy.ndarray,
    v4: numpy.ndarray
):
    """Calculate the dihedral (torsion) angle defined by four connected points.
    
    This function is used in the Bio.PDB vectors utilities to compute the torsion (dihedral) angle between four connected points (for example four atom coordinates that define a backbone or side-chain torsion in a molecular structure). The routine computes bond vectors from the input points, forms two plane normals via cross products, computes the unsigned angle between those normals, and then determines the sign of the dihedral from the relative orientation of the inter-connecting vector. This is commonly used in computational molecular biology to compute protein backbone phi/psi/omega angles or other torsion angles for conformational analysis.
    
    Args:
        v1 (Vector): The first point (most commonly the atom at one end of the four-atom sequence). This object must behave like Biopython's 3D Vector: support vector subtraction (v1 - v2 yielding a Vector), the cross-product operator (a ** b), and provide an angle(other) method that returns the angle (in radians) between vectors. In the typical domain use-case, v1 contains 3D coordinates for an atom in a protein or small molecule and represents one end of the dihedral-defining sequence.
        v2 (Vector): The second point (the atom connected to v1). As for v1, this must be a 3D Vector supporting subtraction, cross product, and angle(). In molecular terms, v2 is the atom bonded to v1 and is the first internal point used to form the first bond vector.
        v3 (Vector): The third point (the central connector between the two bond vectors). This must be a 3D Vector with the same behavior as v1 and v2. In protein terms, v3 is the atom bonded to v2 and to v4 and serves as the pivot between the two planes whose normals define the dihedral.
        v4 (Vector): The fourth point (the atom at the other end of the four-atom sequence). This must be a 3D Vector as above. In practice, v4 completes the four-atom sequence used to define the torsion angle.
    
    Returns:
        float: The dihedral (torsion) angle in radians. The returned value is in the interval ]-pi, pi]; that is, greater than -pi and up to and including +pi. Positive and negative signs follow the right-hand-rule orientation determined by the ordering of the input points (v1,v2,v3,v4). The numeric algorithm computes ab = v1 - v2, cb = v3 - v2, db = v4 - v3, then u = cross(ab, cb), v = cross(db, cb), w = cross(u, v), angle = angle_between(u, v). The sign is flipped to negative if cb.angle(w) > 0.001 (the same threshold used in the implementation). If cb.angle(w) raises a ZeroDivisionError (for example when normals collapse and the intermediate cross-product is effectively zero), the function treats this as the dihedral being pi (the code catches ZeroDivisionError and leaves angle at pi), so callers will receive +pi in that degenerate case.
    
    Behavior and failure modes:
        The function has no side effects and returns a floating-point radian value. It assumes the inputs implement the vector arithmetic used in the Bio.PDB Vector implementation: subtraction (to form bond vectors), cross product using the ** operator, and an angle(other) method. If the inputs do not implement these operations, the function may raise TypeError or AttributeError. Collinear or otherwise degenerate input geometries that make one or more cross products zero are handled by the try/except in the implementation: a ZeroDivisionError encountered while determining the sign is treated as the dihedral being pi. The threshold 0.001 used when testing cb.angle(w) is part of the implementation and affects sign determination for near-degenerate geometries.
    """
    from Bio.PDB.vectors import calc_dihedral
    return calc_dihedral(v1, v2, v3, v4)


################################################################################
# Source: Bio.PDB.vectors.coord_space
# File: Bio/PDB/vectors.py
# Category: fix_docstring
# Reason: Schema parsing failed: ("Couldn't parse this type hint, likely due to a custom class or object: ", numpy.ndarray | None)
################################################################################

def Bio_PDB_vectors_coord_space(
    a0: numpy.ndarray,
    a1: numpy.ndarray,
    a2: numpy.ndarray,
    rev: bool = False
):
    """Bio.PDB.vectors.coord_space generates a 4x4 homogeneous transformation matrix that maps 3D Cartesian coordinates into a new right-handed coordinate space defined by three input points. This is used in the Bio.PDB module to align atomic coordinates or define a local coordinate frame for residues, atoms, or structural motifs: the resulting coordinate space places a1 at the origin, a2 on the positive Z axis, and a0 in the XZ plane.
    
    Args:
        a0 (numpy.ndarray): First defining point for the target coordinate frame. This is a NumPy column array containing the X, Y, Z Cartesian coordinates of the point to be placed onto the XZ plane after transformation. In the PDB/molecular-structure domain this typically is the position of an atom whose projection should lie in the plane used to define the X axis direction.
        a1 (numpy.ndarray): Second defining point and the origin of the new coordinate space. This is a NumPy column array containing the X, Y, Z Cartesian coordinates that will be translated to the origin (0,0,0) by the forward transform. In practice this is often the central atom or pivot for the local frame.
        a2 (numpy.ndarray): Third defining point that determines the +Z direction of the new coordinate space. This is a NumPy column array containing the X, Y, Z Cartesian coordinates used, together with a1, to compute the polar and azimuth angles whose rotations align the line a1->a2 with the positive Z axis.
        rev (bool): If False (default), the function returns only the forward transformation matrix that maps world coordinates into the new coordinate space. If True, the function also computes and returns the reverse transformation matrix that maps coordinates back from the new coordinate space into the original/world coordinates. The reverse matrix is produced by composing the inverse rotations and the translation back to a1.
    
    Returns:
        tuple[numpy.ndarray, numpy.ndarray | None]: A pair where the first element is a 4x4 NumPy array representing the forward homogeneous transformation matrix (mt). This matrix is intended for left-multiplication of 4-element column homogeneous coordinate vectors (e.g., mt.dot([x, y, z, 1])) to obtain coordinates in the new frame where a1 is the origin, a2 lies on +Z, and a0 lies on the XZ plane. The second element is either a 4x4 NumPy array representing the reverse transformation matrix (mr) when rev=True, or None when rev=False. The reverse matrix, when provided, composes the inverse rotations and translation so that mr.dot(mt.dot(v)) returns v (subject to numerical precision) for homogeneous vectors v.
    
    Behavior and side effects:
        The function computes spherical coordinates for the vector a2 - a1, constructs translation and rotation homogeneous matrices, and composes them to form the final transform. Internally it uses and updates module-level reusable matrices (gtm, gmry, gmrz, gmrz2) via the helper setter functions set_homog_trans_mtx, set_Y_homog_rot_mtx, and set_Z_homog_rot_mtx. Because these global matrices are reused, callers should not rely on their contents outside this function unless explicitly intended; the function itself returns the composed matrices (mt and optionally mr) needed for coordinate transformations. The default rev=False avoids computing the extra reverse matrix for efficiency.
    
    Failure modes and errors:
        The function expects numeric NumPy column arrays representing 3D Cartesian coordinates. If the inputs do not provide at least three numeric components accessible as a0[0], a0[1], a0[2], etc., the code may raise IndexError or TypeError. Degenerate or nearly collinear input configurations (for example a1 == a2, or a0, a1, a2 collinear) lead to undefined or numerically unstable azimuth/polar angle computations, which can produce NaNs or singular rotations; the function does not explicitly handle these degeneracies. Users should validate input geometry before calling when such cases are possible.
    
    Practical significance:
        In computational structural biology workflows (as in Biopython's PDB handling), coord_space is useful for defining local coordinate frames for residues, aligning bonds or functional groups, computing dihedral/angle measures in a stable local frame, and performing deterministic rotations/translations of atom sets. The produced homogeneous matrices enable transforming coordinates into and out of that local frame using standard homogeneous coordinate multiplication.
    """
    from Bio.PDB.vectors import coord_space
    return coord_space(a0, a1, a2, rev)


################################################################################
# Source: Bio.PDB.vectors.get_spherical_coordinates
# File: Bio/PDB/vectors.py
# Category: valid
################################################################################

def Bio_PDB_vectors_get_spherical_coordinates(xyz: numpy.ndarray):
    """Bio.PDB.vectors.get_spherical_coordinates computes spherical coordinates (r, azimuth, polar_angle) for a 3D point given as a NumPy column vector. In the Biopython PDB/vector context this function is used to convert Cartesian atomic coordinates (X, Y, Z) from a structure (e.g. a single atom coordinate extracted from a PDB model) into radial and angular components for downstream analysis of distances, orientations, and angular distributions.
    
    Args:
        xyz (numpy.ndarray): column vector (3 row x 1 column NumPy array) containing the Cartesian coordinates X, Y, Z of a single point. The array must provide the components via xyz[0], xyz[1], and xyz[2] as used in the implementation. The function expects exactly three components in this column-vector form; providing an array with a different shape may produce an IndexError or other exceptions.
    
    Detailed behavior: The function computes r as the Euclidean norm using numpy.linalg.norm. If r == 0 (the zero vector), the function returns (0, 0, 0) immediately to avoid division by zero. The azimuth is computed by calling the helper _get_azimuth(x, y) with the X and Y components; this yields the in-plane angular coordinate around the Z axis (expressed in radians as produced by the helper). The polar_angle is computed as numpy.arccos(z / r), i.e. the angle (in radians) measured from the positive Z axis (colatitude). There are no side effects: the function does not modify the input array or any global state.
    
    Failure modes: If xyz is not a numpy.ndarray with at least three accessible elements, the function may raise IndexError or TypeError. If the input is the exact zero vector, the function returns the zero tuple instead of attempting to compute angles, avoiding a division-by-zero error.
    
    Returns:
        tuple[float, float, float]: A 3-tuple (r, azimuth, polar_angle). r is the radial distance (a float) in the same linear units as the input coordinates. azimuth and polar_angle are floats giving angles in radians; azimuth is the in-plane angle computed by _get_azimuth(x, y), and polar_angle is the arccos(z / r) value in radians.
    """
    from Bio.PDB.vectors import get_spherical_coordinates
    return get_spherical_coordinates(xyz)


################################################################################
# Source: Bio.PDB.vectors.homog_rot_mtx
# File: Bio/PDB/vectors.py
# Category: valid
################################################################################

def Bio_PDB_vectors_homog_rot_mtx(angle_rads: float, axis: str):
    """Bio.PDB.vectors.homog_rot_mtx generates a 4x4 homogeneous rotation matrix for a single principal axis (x, y or z) used when transforming 3D coordinates in the Bio.PDB molecular-structure workflow.
    
    This function constructs a homogeneous transformation matrix suitable for rotating 3D coordinates of atoms or coordinate frames in computational molecular biology applications (for example, when manipulating PDB atom coordinates or building transformation chains). The returned matrix is a NumPy array whose upper-left 3x3 block is the rotation matrix for a single principal axis and whose last row and column make it a 4x4 homogeneous transform (so it can be used with 4-component coordinate vectors [x, y, z, 1]). The rotation follows the standard right-handed convention (positive angles produce counterclockwise rotation in the XY plane when looking from +Z toward the origin). The numeric type of the returned array is numpy.float64.
    
    Args:
        angle_rads (float): the desired rotation angle in radians. This floating-point input specifies the magnitude and sign of rotation using standard radian measure; positive and negative values are supported and rotations are periodic with period 2*pi. Typical use is to supply a Python float or NumPy scalar representing an angle computed from geometry or conversion utilities within Biopython.
        axis (str): single-character string specifying the rotation axis. Accepted values are 'x', 'y' or 'z' (lowercase). The function compares axis to 'z' and 'y' explicitly; if axis == 'z' the rotation about the Z axis is returned, elif axis == 'y' the rotation about the Y axis is returned, otherwise the function returns the rotation about the X axis. The comparison is case-sensitive, so using uppercase letters or other strings will not raise an exception but will fall back to the X-axis rotation behavior.
    
    Returns:
        numpy.ndarray: a new 4x4 NumPy array (dtype numpy.float64) containing the homogeneous rotation matrix. The matrix layout is:
            - upper-left 3x3: the standard rotation matrix for the selected principal axis using the supplied angle (right-handed convention),
            - last column [0, 0, 0, 1]^T and last row [0, 0, 0, 1] making it a homogeneous transform.
        The function has no side effects; it does not modify inputs or global state and always returns a freshly allocated NumPy array. No exceptions are raised for invalid axis strings; invalid or unexpected axis values simply result in the X-axis rotation being returned as described above.
    """
    from Bio.PDB.vectors import homog_rot_mtx
    return homog_rot_mtx(angle_rads, axis)


################################################################################
# Source: Bio.PDB.vectors.homog_scale_mtx
# File: Bio/PDB/vectors.py
# Category: valid
################################################################################

def Bio_PDB_vectors_homog_scale_mtx(scale: float):
    """Bio.PDB.vectors.homog_scale_mtx: Generate a 4x4 homogeneous scaling matrix suitable for 3D coordinate transformations used in the Bio.PDB vector utilities.
    
    This function constructs a homogeneous transformation matrix that applies an isotropic scale to 3D coordinates. The returned matrix is intended to be used with 4-component homogeneous coordinate vectors of the form [x, y, z, 1] (common in geometric transforms used in structural biology, molecular modelling, visualization, and coordinate manipulation in Bio.PDB). The matrix scales the X, Y and Z components by the provided scalar while leaving the homogeneous coordinate unchanged, so it can be composed with other 4x4 homogeneous transform matrices (for example rotations, translations, or other scalings) to build compound transformations. The function is pure (no side effects) and deterministic: given the same input it always returns the same NumPy array. If scale is not a finite numeric value (for example NaN or infinity), the resulting matrix will contain the corresponding IEEE floating-point values. There is no special input validation beyond NumPy's array construction.
    
    Args:
        scale (float): Scale multiplier applied isotropically to the X, Y and Z axes.
            In the domain of computational molecular biology (as used throughout Biopython),
            this parameter represents the factor by which atomic or molecular coordinates
            are enlarged (>1.0) or reduced (<1.0). The function expects a Python float
            value; passing a non-finite float (NaN or inf) will produce a matrix containing
            those values. There is no default; the argument must be provided.
    
    Returns:
        numpy.ndarray: A 4x4 NumPy array (shape (4, 4)) with dtype numpy.float64 representing
        the homogeneous scaling matrix:
        [[scale, 0,     0,     0],
         [0,     scale, 0,     0],
         [0,     0,     scale, 0],
         [0,     0,     0,     1]]
        This matrix can be multiplied with homogeneous coordinate column vectors or other
        4x4 homogeneous transformation matrices to apply the isotropic scale to 3D coordinates.
    """
    from Bio.PDB.vectors import homog_scale_mtx
    return homog_scale_mtx(scale)


################################################################################
# Source: Bio.PDB.vectors.homog_trans_mtx
# File: Bio/PDB/vectors.py
# Category: fix_docstring
# Reason: Schema parsing failed: Cannot generate JSON schema for homog_trans_mtx because the docstring has no description for the argument 'x'
################################################################################

def Bio_PDB_vectors_homog_trans_mtx(x: float, y: float, z: float):
    """Generate a 4x4 homogeneous translation matrix suitable for 3D coordinate transformations used in computational molecular biology workflows (for example, translating atom coordinates in PDB structures handled by Bio.PDB). The matrix is built for the standard homogeneous-coordinate column-vector convention: when a 4×1 column vector [X, Y, Z, 1]^T is left-multiplied by the returned matrix, the result is [X + x, Y + y, Z + z, 1]^T.
    
    Args:
        x (float): Translation along the X axis. In the domain of Bio.PDB this represents the displacement to apply to atomic or molecular coordinates along the X axis (typically in the same length units as the coordinates, e.g. Å when operating on PDB coordinates). The value is expected to be a real number; non-numeric inputs that cannot be converted to a floating-point value will cause NumPy to raise an error when constructing the array.
        y (float): Translation along the Y axis. As with x, this is the displacement applied to coordinates along the Y axis in the same units as the input coordinates. Must be a real number; invalid types that cannot be cast to float will result in an exception from NumPy.
        z (float): Translation along the Z axis. This is the displacement applied to coordinates along the Z axis in the same units as the input coordinates. Must be a real number; values that cannot be converted to float will cause NumPy to raise an error.
    
    Returns:
        numpy.ndarray: A new 4x4 NumPy array (dtype numpy.float64) representing the homogeneous translation matrix:
            [[1, 0, 0, x],
             [0, 1, 0, y],
             [0, 0, 1, z],
             [0, 0, 0, 1]]
        Practical significance: this matrix encodes no rotation or scaling (upper-left 3×3 is the identity) and places the translation components in the rightmost column consistent with the homogeneous-coordinate column-vector convention. There are no side effects (the function allocates and returns a fresh NumPy array). Failure modes: supplying inputs that cannot be converted to float will raise an exception when NumPy attempts to create the array; otherwise the function always returns a valid 4×4 translation matrix with dtype numpy.float64.
    """
    from Bio.PDB.vectors import homog_trans_mtx
    return homog_trans_mtx(x, y, z)


################################################################################
# Source: Bio.PDB.vectors.m2rotaxis
# File: Bio/PDB/vectors.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_PDB_vectors_m2rotaxis(m: numpy.ndarray):
    """Bio.PDB.vectors.m2rotaxis converts a 3x3 rotation matrix into the equivalent rotation angle and rotation axis (angle-axis representation). This function is used in Bio.PDB and other structural biology code within Biopython to interpret or decompose rigid-body rotations (for example, when analysing or applying coordinate transforms between protein models or PDB coordinate frames). The returned angle is in radians and lies in the closed interval [0, pi]; the returned axis is a unit Vector instance that defines the direction of the rotation and thus the sense of rotation for angles > 0.
    
    Args:
        m (numpy.ndarray): A 3x3 rotation matrix represented as a NumPy array. The function expects a numeric, two-dimensional array indexed as m[0,0]..m[2,2]. This matrix should represent a proper orthogonal rotation (determinant +1) in three-dimensional Euclidean space as used in molecular structural transforms. The code uses element-wise access to detect singularities and compute traces, so providing an array of a different shape or non-numeric contents will cause indexing or type errors.
    
    Behavior and practical details:
        The algorithm follows the standard matrix-to-angle/axis conversion (see sources such as euclideanspace.com). Numerical tolerance is used to detect singularities: an internal epsilon of 1e-5 is used to decide when off-diagonal skew-symmetric components are effectively zero. The computed cosine of the angle is clamped to the interval [-1, 1] before applying arccos to avoid NaNs from floating point round-off. There are three cases handled:
        1) Angle approximately 0 (identity rotation): this is a singular case because any axis is valid. The function treats this explicitly and returns angle 0.0 and the canonical axis Vector(1, 0, 0). This choice avoids ambiguity in downstream code that expects a concrete axis.
        2) 0 < angle < pi: the axis is computed from the anti-symmetric part of the matrix (components proportional to sin(angle)) as Vector(m[2,1]-m[1,2], m[0,2]-m[2,0], m[1,0]-m[0,1]) and then normalized. The axis orientation together with the angle determines the sense of rotation used in molecular transforms.
        3) angle approximately pi: this is another singular case requiring special handling. The algorithm extracts the axis from diagonal elements to avoid cancellation errors. The components are computed using square roots and divisions shown in the implementation, then normalized.
        Internally, a tiny threshold of 1e-15 is used to treat extremely small angles as zero. Axis normalization is performed via Vector.normalize(), so the returned Vector is a unit-length axis; normalizing is a side effect on the created Vector object but does not modify the input matrix m. The function always returns an angle in radians between 0 and pi inclusive, and a Vector instance representing the axis.
    
    Returns:
        tuple: A pair (angle, axis) where:
            angle (float): The rotation angle in radians corresponding to the input matrix, constrained to [0, pi]. This value is computed via clamped trace-based arccos or set explicitly to 0 or pi in singular cases detected by the numeric tolerance.
            axis (Vector): A unit Vector (Bio.PDB.vectors.Vector) giving the rotation axis. For the identity matrix (angle == 0) the canonical Vector(1, 0, 0) is returned. For other cases the axis is normalized before being returned so it can be directly used in downstream rotation or alignment computations.
    
    Failure modes and exceptions:
        If m is not a numeric 3x3 numpy.ndarray (wrong shape, wrong dimensionality, or non-numeric entries), the function will likely raise IndexError, TypeError, or ValueError during indexing or numerical operations. If m does not represent a near-orthogonal rotation matrix (e.g., contains large numerical errors or is not a rotation), the returned angle/axis will reflect the numeric decomposition performed by the algorithm but may be meaningless for non-rotation inputs. Numerical edge cases near singularities (angle ~ 0 or ~ pi) are handled as described above using tolerances (eps = 1e-5 and an angle threshold of 1e-15), but extremely ill-conditioned matrices may still produce unstable results.
    
    Side effects and guarantees:
        The input matrix m is not modified. The function constructs and normalizes a Vector for the axis; the Vector.normalize() call modifies that new Vector instance but does not affect external state. The return values are deterministic for the same numeric input given the described tolerances.
    """
    from Bio.PDB.vectors import m2rotaxis
    return m2rotaxis(m)


################################################################################
# Source: Bio.PDB.vectors.multi_coord_space
# File: Bio/PDB/vectors.py
# Category: fix_docstring
# Reason: Schema parsing failed: Cannot generate JSON schema for multi_coord_space because the docstring has no description for the argument 'a3'
################################################################################

def Bio_PDB_vectors_multi_coord_space(a3: numpy.ndarray, dLen: int, rev: bool = False):
    """Bio.PDB.vectors.multi_coord_space generates 4x4 homogeneous transformation matrices that map sets of three Cartesian points (atoms) into a local hedron coordinate space used in PDB geometric computations.
    
    This function is used in Biopython's PDB/vector routines to construct a local coordinate system from three atoms (a triad) so that downstream code can compute internal coordinates, align fragments, or transform points into a canonical hedron frame. The new coordinate space produced by the forward transform has the following conventions: the second atom (index 1) is placed at the origin, the third atom (index 2) lies on the positive Z axis, and the first atom (index 0) is constrained to the XZ plane. The implementation builds a translation to move atom 1 to the origin, then applies rotations about Z and Y to align atom 2 with +Z, and finally rotates about Z to move atom 0 into the XZ plane. If rev is True, the function also constructs the reverse transforms that map coordinates from the hedron space back to the original coordinate system.
    
    Args:
        a3 (numpy.ndarray): Array containing the input atom coordinates. Expected to contain dLen entries of three points each, provided as homogeneous 4-component vectors (x, y, z, 1). The code uses a3 with indexing a3[:, i, 0:3] and a3[:, i].reshape(-1, 4, 1), so the practical expected shape is (dLen, 3, 4) where the second axis indexes the three atoms (atom 0, atom 1, atom 2) and the last axis holds the homogeneous coordinate (x, y, z, w). Each row corresponds to one hedron instance to transform. The homogeneous w component is typically 1 for Cartesian points. Supplying arrays that do not match the expected third dimension will lead to shape or indexing errors.
        dLen (int): Number of entries (hedrons) for which to build transformation matrices. This is used to allocate and shape internal 4x4 matrix arrays (tm initialized with shape (dLen, 4, 4)). For correct behavior, dLen should equal a3.shape[0] (the number of hedron entries). If dLen does not match a3.shape[0], NumPy broadcasting or indexing can produce incorrect results or raise exceptions.
        rev (bool): If False (default), return only the forward transforms that map original coordinates into the hedron coordinate space. If True, also return the reverse transforms that map coordinates from hedron space back to the original coordinate system. When rev=True, the function returns both forward and reverse arrays together (see Returns). There are no side effects controlled by this flag beyond the additional computation and larger returned array.
    
    Returns:
        numpy.ndarray: If rev is False (default), a NumPy array of shape (dLen, 4, 4) containing the forward 4x4 homogeneous transformation matrices. Each 4x4 matrix maps coordinates from the original Cartesian frame into the local hedron coordinate frame for the corresponding entry in a3. If rev is True, a NumPy array of shape (2, dLen, 4, 4) is returned where the first element (index 0) is the forward transforms (as above) and the second element (index 1) contains the reverse transforms that map coordinates from the hedron frame back to the original Cartesian frame. All returned matrices are in homogeneous coordinates suitable for composing with 4-component column vectors (x, y, z, 1).
    
    Behavior, defaults, and failure modes:
        - The function assumes inputs are numeric NumPy arrays (floating point is typical) and that each entry in a3 supplies three points as homogeneous 4-component vectors. The default rev is False to return only forward transforms.
        - Internally, the function computes vector differences, Euclidean norms, azimuth and polar angles (using arctan2 and arccos), and constructs rotations using multi_rot_Z and multi_rot_Y helpers. Division by zero for zero-length vectors is partially guarded using NumPy's where machinery, but degenerate geometric inputs (for example, atom 1 coincident with atom 2 resulting in zero-length p, or all three atoms collinear or coincident) can produce undefined angles (NaN) or invalid rotation matrices. Such degenerate inputs will therefore produce transforms containing NaN or otherwise unusable matrices; the function does not raise a custom exception for these cases.
        - The function does not modify global state or input arrays in-place (it allocates new arrays for transforms), but it does rely on the caller to provide consistently structured inputs. Mismatched shapes between dLen and a3, incorrect homogeneous coordinate values, or non-numeric array contents will likely raise NumPy indexing or arithmetic errors.
        - Numerical precision and stability follow NumPy's floating point semantics. For best results in molecular geometry contexts (Biopython/PDB processing), provide coordinates in standard Cartesian units (e.g., Angstroms) and ensure the homogeneous coordinate component is 1.
    """
    from Bio.PDB.vectors import multi_coord_space
    return multi_coord_space(a3, dLen, rev)


################################################################################
# Source: Bio.PDB.vectors.multi_rot_Y
# File: Bio/PDB/vectors.py
# Category: valid
################################################################################

def Bio_PDB_vectors_multi_rot_Y(angle_rads: numpy.ndarray):
    """Create multiple 4x4 homogeneous rotation matrices for rotations about the Y axis.
    
    This function is part of the Bio.PDB.vectors utilities in Biopython and is used to generate a batch of homogeneous transformation matrices that rotate 3D coordinates around the Y axis by given angles. Each returned matrix is a 4x4 affine rotation matrix (no translation) in homogeneous coordinates, suitable for applying to 3D points or coordinate frames in molecular structure manipulations (for example, rotating atom coordinates in a PDB model). The function allocates and returns a new NumPy array and does not modify its input.
    
    Args:
        angle_rads (numpy.ndarray): One-dimensional NumPy array of angles in radians, shape (N,). Each element angle_rads[i] is the rotation angle (in radians) for which a corresponding 4x4 homogeneous rotation matrix about the Y axis will be produced. The array must be one-dimensional because the implementation uses angle_rads.shape[0] to determine the number of matrices. Passing a scalar or a zero-dimensional array will raise an IndexError when accessing shape[0]; passing a multi-dimensional array will typically raise a ValueError during assignment because the code expects a length-N 1D array.
    
    Returns:
        numpy.ndarray: A new NumPy array of shape (N, 4, 4) and floating-point dtype. The first dimension indexes the matrices corresponding to each input angle. For each i, the matrix at result[i] is the homogeneous rotation matrix for a right-handed rotation about the Y axis by angle_rads[i], with the layout (row-major):
            [ [cos(theta), 0,  sin(theta), 0],
              [0,          1,  0,          0],
              [-sin(theta),0,  cos(theta), 0],
              [0,          0,  0,          1] ]
        These matrices can be used to rotate 3D coordinates represented in homogeneous form. NaN or infinite values in angle_rads will propagate into the returned matrices.
    
    Behavior and failure modes:
        - The function creates and returns a newly allocated array; it does not perform in-place modification of the input angle_rads array.
        - The number of matrices N is determined as angle_rads.shape[0]; if angle_rads has length zero, an array with shape (0, 4, 4) is returned.
        - If angle_rads is not a one-dimensional numpy.ndarray (for example, is scalar, zero-dimensional, or multi-dimensional), the function may raise IndexError or ValueError as described above.
        - The function relies on NumPy functions np.cos and np.sin; the numerical precision and dtype of the output follow NumPy's standard floating-point behavior for these operations.
    """
    from Bio.PDB.vectors import multi_rot_Y
    return multi_rot_Y(angle_rads)


################################################################################
# Source: Bio.PDB.vectors.multi_rot_Z
# File: Bio/PDB/vectors.py
# Category: valid
################################################################################

def Bio_PDB_vectors_multi_rot_Z(angle_rads: numpy.ndarray):
    """Create a stack of 4x4 homogeneous rotation matrices for rotations about the Z axis.
    
    This function is used in the Bio.PDB.vectors context (part of Biopython) to build homogeneous transformation matrices that rotate 3D coordinates around the Z axis by specified angles in radians. In computational molecular biology workflows (for example, rotating atom coordinates in a PDB structure, aligning fragments, or composing rigid-body transforms), these matrices can be applied to homogeneous column vectors [x, y, z, 1] to rotate the x-y components while preserving the z coordinate and translation component.
    
    Args:
        angle_rads (numpy.ndarray): A NumPy array of angles in radians. The first dimension length, angle_rads.shape[0], determines the number of rotation matrices produced (referred to in older documentation as "entries"). The function reads angle values elementwise and computes elementwise cosine and sine; therefore angle_rads must be a numpy.ndarray-like object with numeric values accessible via its first dimension. If angle_rads.shape[0] is zero the function returns an array with shape (0, 4, 4). The input array is not modified by this function.
    
    Returns:
        numpy.ndarray: A NumPy array of shape (N, 4, 4) where N == angle_rads.shape[0]. Each entry i is a homogeneous rotation matrix that applies a rotation by angle_rads[i] about the Z axis. The upper-left 2x2 block of each 4x4 matrix is [[cos(theta), -sin(theta)], [sin(theta), cos(theta)]], the (2,2) element corresponding to the Z coordinate is 1 (preserving z), and the bottom row is [0, 0, 0, 1], making the matrix suitable for transforming homogeneous column vectors [x, y, z, 1]. The numeric dtype of the returned array follows NumPy's cosine/sine output (typically a floating point type).
    
    Behavior, side effects, defaults, and failure modes:
        The function allocates and returns a new NumPy array and does not mutate external state or the input array. It initializes each 4x4 matrix to the identity and then fills the 2D rotation components using np.cos and np.sin on angle_rads. If angle_rads is not a numpy.ndarray (for example does not have a shape attribute or shape[0]), an AttributeError or TypeError may be raised when the function attempts to access shape or compute trigonometric functions. If angle_rads contains non-finite values (NaN or Inf), those values propagate into the output matrices. Numerical precision of the rotation entries depends on the dtype of the trigonometric operations (e.g., float32 vs float64).
    """
    from Bio.PDB.vectors import multi_rot_Z
    return multi_rot_Z(angle_rads)


################################################################################
# Source: Bio.PDB.vectors.refmat
# File: Bio/PDB/vectors.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_PDB_vectors_refmat(p: numpy.ndarray, q: numpy.ndarray):
    """Return a left-multiplying 3x3 reflection matrix that mirrors vector p onto vector q.
    
    This function is part of Bio.PDB.vectors and is used in structural and geometric operations common in computational molecular biology (for example, reflecting atomic coordinate direction vectors when building symmetric structures or performing geometric transformations on protein backbone/sidechain vectors). The routine computes a Householder-style reflection matrix that, when left-multiplied with the column representation of p, produces a vector that lies in the direction of q (within numerical tolerance). The implementation normalizes the input vectors, handles the near-equality case by returning the identity matrix, and constructs the reflection as I - 2 b b^T where b is the normalized difference between p and q.
    
    Args:
        p (Vector): A 3D vector from Bio.PDB.vectors representing the source direction or point to be mirrored. This argument is normalized inside the function; the function assigns the normalized copy to a local variable and does not modify the caller's reference. Passing a zero-length Vector will cause normalization to fail (typically raising an error from the Vector normalization code).
        q (Vector): A 3D vector from Bio.PDB.vectors representing the target direction or point to which p should be mirrored. This argument is normalized inside the function in the same manner as p. Both p and q are interpreted as direction vectors in Cartesian 3-space consistent with Biopython's PDB vector utilities.
    
    Behavior and side effects:
        The function first replaces p and q with their normalized forms (unit vectors). If the Euclidean norm of (p - q) is less than 1e-5 (a built-in numerical tolerance chosen to treat nearly-equal directions as identical), the function returns the 3x3 identity matrix because no reflection is needed. Otherwise it computes the unit vector b = normalize(p - q), forms the outer product b b^T, and returns the reflection matrix I - 2 * (b b^T). The returned matrix is orthogonal with determinant -1 (a proper reflection), and is intended to be used as a left-multiplying matrix on column vectors (for example, new_vec = R.dot(old_vec) or via Vector.left_multiply(R) in Bio.PDB.vectors). The function has no other side effects on input objects.
    
    Failure modes and constraints:
        Inputs must be instances of the Bio.PDB.vectors.Vector class (or types that provide the same normalized(), norm(), normalize(), get_array(), and arithmetic semantics used in the module). Supplying a zero-length vector will typically result in a division-by-zero or a normalization error. Supplying incompatible types will raise attribute errors. Numerical precision is finite; the function uses a fixed tolerance of 1e-5 to detect near-equality of directions and will return the identity matrix in that case.
    
    Returns:
        numpy.ndarray: A 3x3 NumPy array (dtype float) representing the left-multiplying reflection matrix that mirrors p onto q. The matrix is intended to be applied as R.dot(v) for column vectors v (or via Vector.left_multiply(R)); when applied to the input direction p it produces a vector aligned with q within numerical precision.
    """
    from Bio.PDB.vectors import refmat
    return refmat(p, q)


################################################################################
# Source: Bio.PDB.vectors.rotmat
# File: Bio/PDB/vectors.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_PDB_vectors_rotmat(p: numpy.ndarray, q: numpy.ndarray):
    """Return a left-multiplying 3x3 rotation matrix that rotates the vector p onto the vector q.
    
    This function is part of the Bio.PDB.vectors utilities in Biopython and is used in computational molecular biology workflows that manipulate 3D coordinates (for example, rotating atom coordinates or direction vectors when aligning or transforming PDB structures). The routine computes a rotation matrix suitable for left-multiplication by a 3D coordinate/vector object (the same convention used by Vector.left_multiply in Bio.PDB.vectors). The result is a pure numeric transformation (no side effects) computed from the input vectors using internal helper routines (refmat) and NumPy matrix operations.
    
    Args:
        p (numpy.ndarray): moving vector to be rotated. This should be a 3-dimensional Cartesian vector (e.g. an array of three floats representing an atom coordinate or direction in Ångstroms) provided as a NumPy ndarray. In the Bio.PDB context p is the vector that will be transformed by the returned matrix; it is treated as the "source" or moving vector. Supplying an array of a different dimensionality or an ill-formed array will typically result in NumPy shape errors or invalid numeric results.
        q (numpy.ndarray): fixed target vector. This should be a 3-dimensional Cartesian vector (NumPy ndarray) representing the desired orientation or destination for p (for example, an atom coordinate or direction in a molecular structure). In the Bio.PDB context q is treated as the "target" or fixed vector that p should be rotated onto.
    
    Returns:
        numpy.ndarray: a 3x3 NumPy array representing the left-multiplying rotation matrix. When this matrix is applied to p using the same left-multiplication convention as used by Bio.PDB.vectors (for example, Vector.left_multiply or NumPy matrix multiplication), the rotated p will coincide with q within numerical precision limits. The matrix is intended for direct use on 3D coordinate vectors in structural biology tasks such as aligning atoms, rotating fragments, or transforming coordinate frames.
    
    Notes on behavior, defaults and failure modes:
        This function performs a deterministic numeric computation and has no side effects. It relies on internal helper functions (refmat) and NumPy dot-product operations. The rotation is computed for 3D vectors; if p or q are zero vectors the rotation is undefined and the computation may produce NaNs or raise runtime errors from the underlying numeric routines. If p and q are colinear but opposite in direction, the resulting rotation corresponds to a 180-degree rotation about an axis perpendicular to p/q and is handled by the numeric routine, subject to floating point tolerances. Inputs with shapes other than a length-3 vector (for example, not having three components) will likely cause NumPy shape or broadcasting errors.
    """
    from Bio.PDB.vectors import rotmat
    return rotmat(p, q)


################################################################################
# Source: Bio.PDB.vectors.set_X_homog_rot_mtx
# File: Bio/PDB/vectors.py
# Category: fix_docstring
# Reason: Schema parsing failed: Cannot generate JSON schema for set_X_homog_rot_mtx because the docstring has no description for the argument 'angle_rads'
################################################################################

def Bio_PDB_vectors_set_X_homog_rot_mtx(angle_rads: float, mtx: numpy.ndarray):
    """Set the rotation components of an existing homogeneous rotation matrix to represent a rotation
    about the X axis by the specified angle in radians. This function is used in the Bio.PDB
    vectors utilities to update the rotation submatrix used when transforming 3D coordinates of
    molecular structures (for example, atom coordinates in protein models) without reallocating
    a new matrix.
    
    Args:
        angle_rads (float): Rotation angle in radians. The angle defines the magnitude and direction
            of rotation about the X axis following the standard right-hand rule for 3D rotations.
            This value is passed to numpy.cos and numpy.sin to compute the rotation matrix entries.
        mtx (numpy.ndarray): Numpy array to be updated in-place. The function writes the computed
            cosine and sine values into the matrix entries corresponding to the Y/Z rotation
            submatrix for an X-axis rotation: mtx[1][1], mtx[2][2] are set to cos(angle_rads),
            mtx[2][1] is set to sin(angle_rads), and mtx[1][2] is set to -sin(angle_rads).
            Only these four entries are modified; all other elements of mtx are left unchanged.
            The array must support integer indexing and item assignment at indices 1 and 2.
            Practical significance in the Bio.PDB domain: callers typically provide a homogeneous
            transformation matrix (used to combine rotation and translation for 3D coordinate
            transforms) so updating these entries changes the rotation applied when transforming
            molecular coordinates.
    
    Returns:
        None: The function does not return a value. Its effect is to modify the provided numpy.ndarray
        mtx in-place so that it represents a rotation about the X axis by angle_rads. Note that
        common failure modes include IndexError if mtx does not have the required indices, TypeError
        if mtx does not support item assignment, and loss of numerical precision or silent truncation
        if mtx has an integer dtype (float results from numpy.cos/numpy.sin will be cast to the
        array dtype).
    """
    from Bio.PDB.vectors import set_X_homog_rot_mtx
    return set_X_homog_rot_mtx(angle_rads, mtx)


################################################################################
# Source: Bio.PDB.vectors.set_Y_homog_rot_mtx
# File: Bio/PDB/vectors.py
# Category: fix_docstring
# Reason: Schema parsing failed: Cannot generate JSON schema for set_Y_homog_rot_mtx because the docstring has no description for the argument 'angle_rads'
################################################################################

def Bio_PDB_vectors_set_Y_homog_rot_mtx(angle_rads: float, mtx: numpy.ndarray):
    """Set the elements of an existing homogeneous rotation matrix to represent a rotation
    about the Y axis by the specified angle (in radians). This function is used in the
    Bio.PDB.vectors module of Biopython to update the rotation block of a transform
    matrix so it can be applied to 3D coordinate data (for example, atomic coordinates
    or intermediate frames) without allocating a new array.
    
    Args:
        angle_rads (float): Rotation angle in radians. This numeric parameter is the
            angle by which to rotate about the Y axis. The function computes the
            cosine and sine of this value (via numpy.cos and numpy.sin) and writes
            those results into the matrix entries that control rotation in the X–Z
            plane. Practically, provide the rotation in radians matching the rest of
            a Biopython coordinate/transform pipeline; no conversion from degrees is
            performed.
        mtx (numpy.ndarray): Mutable NumPy array that will be updated in-place to
            contain the Y-axis rotation terms. The array must support indexing at
            mtx[0][0], mtx[0][2], mtx[2][0], and mtx[2][2]; in typical usage this
            is a 4x4 homogeneous transform matrix where the upper-left 3x3 block is
            the rotation matrix. The function sets mtx[0][0] and mtx[2][2] to cos(angle_rads),
            mtx[0][2] to +sin(angle_rads), and mtx[2][0] to -sin(angle_rads). All other
            entries of mtx are left unchanged by this call. Because the update is
            done in-place, callers relying on the original matrix values must make a
            copy beforehand if needed.
    
    Returns:
        None: This function does not return a value. Its effect is the in-place
        modification of the provided numpy.ndarray mtx so that it encodes a rotation
        about the Y axis by angle_rads. No new array is created.
    
    Raises:
        TypeError: If mtx is not a numpy.ndarray or is not writable, or if angle_rads
            cannot be interpreted as a numeric scalar suitable for numpy.cos/numpy.sin.
        IndexError: If mtx does not support the required indices (for example, if it
            has fewer than 3 rows or columns), indexing mtx[0][2] or mtx[2][0] will
            raise an IndexError.
    """
    from Bio.PDB.vectors import set_Y_homog_rot_mtx
    return set_Y_homog_rot_mtx(angle_rads, mtx)


################################################################################
# Source: Bio.PDB.vectors.set_Z_homog_rot_mtx
# File: Bio/PDB/vectors.py
# Category: fix_docstring
# Reason: Schema parsing failed: Cannot generate JSON schema for set_Z_homog_rot_mtx because the docstring has no description for the argument 'angle_rads'
################################################################################

def Bio_PDB_vectors_set_Z_homog_rot_mtx(angle_rads: float, mtx: numpy.ndarray):
    """Update an existing rotation matrix's Z-axis rotation terms in-place.
    
    This function computes the cosine and sine of a rotation angle provided in radians and stores those values into the appropriate entries of an existing NumPy array representing a homogeneous-style rotation matrix. In the Bio.PDB/vectors context (used for coordinate transforms and rigid-body rotations in computational molecular biology, e.g. rotating atom coordinates or coordinate frames when manipulating PDB structures), callers typically maintain mutable transformation matrices and call this function to set or change the rotation about the Z axis without reallocating a new matrix.
    
    Args:
        angle_rads (float): Rotation angle in radians. This numeric value is passed to NumPy's trigonometric functions (np.cos, np.sin) to compute the rotation coefficients. The function treats the value as an angle in radians (not degrees); providing values that are not real numbers will propagate NaN/inf through the computed entries.
        mtx (numpy.ndarray): A mutable NumPy array that will be updated in-place. The function assigns to the indexed elements mtx[0][0], mtx[1][1], mtx[1][0], and mtx[0][1] to set the Z-axis rotation terms (cosine on the diagonal, sine and negative sine on the off-diagonal). Callers must provide an array that supports item assignment at these indices and uses a numeric dtype suitable for floating-point values. The function does not allocate or return a new array; it only overwrites these four entries and leaves all other elements of mtx unchanged.
    
    Returns:
        None: This function has no return value. Its effect is a side effect on the provided mtx argument: the four matrix entries used for a Z rotation are updated in-place to represent a rotation by angle_rads. If mtx is too small to contain the required indices, a built-in IndexError will be raised by NumPy/item access. If mtx does not support item assignment (for example, if it is not writable), a TypeError or ValueError may be raised by NumPy. Invalid numeric inputs (NaN or infinite angle_rads) will produce corresponding NaN/inf entries in mtx.
    """
    from Bio.PDB.vectors import set_Z_homog_rot_mtx
    return set_Z_homog_rot_mtx(angle_rads, mtx)


################################################################################
# Source: Bio.PDB.vectors.set_homog_trans_mtx
# File: Bio/PDB/vectors.py
# Category: fix_docstring
# Reason: Schema parsing failed: Cannot generate JSON schema for set_homog_trans_mtx because the docstring has no description for the argument 'x'
################################################################################

def Bio_PDB_vectors_set_homog_trans_mtx(x: float, y: float, z: float, mtx: numpy.ndarray):
    """Set the translation components of an existing homogeneous transformation matrix in-place.
    
    This function is used in the Bio.PDB.vectors module of Biopython when constructing or modifying 4x4 homogeneous transformation matrices that represent rigid-body transforms for 3D molecular coordinates (for example when translating atom or residue coordinates in a protein structure). It overwrites the matrix elements that encode translation along the X, Y and Z axes without allocating a new array, so callers must be aware the provided matrix is mutated.
    
    Args:
        x (float): Translation distance to apply along the X axis. This scalar is written into the matrix element at row 0, column 3 and should be expressed in the same coordinate units used by the matrix (the units used by the calling code, e.g. positional units for atom coordinates).
        y (float): Translation distance to apply along the Y axis. This scalar is written into the matrix element at row 1, column 3 and plays the same role for the Y direction when the matrix is later used to transform 3D points.
        z (float): Translation distance to apply along the Z axis. This scalar is written into the matrix element at row 2, column 3 and represents the Z-direction translation component of the homogeneous transform.
        mtx (numpy.ndarray): The NumPy array representing a homogeneous transformation matrix to be updated in-place. The function assigns to mtx[0][3], mtx[1][3], and mtx[2][3]. The array must be a writable, indexable numpy.ndarray with at least three rows and four columns (commonly a 4x4 transform matrix). The dtype of mtx must be compatible with assignment from float values.
    
    Returns:
        None: This function does not return a value. Instead, it has the side effect of modifying the provided numpy.ndarray mtx by setting its translation components to the supplied x, y, z values. Callers who need to preserve the original matrix should pass a copy (for example, mtx.copy()) before calling this function.
    
    Behavior and failure modes:
        The assignments are performed directly on the provided array. If mtx does not support the required indexing (for example has fewer than three rows or fewer than four columns), an IndexError will be raised. If mtx is not writable (for example a read-only array), a ValueError may be raised on assignment. If mtx is not a numpy.ndarray as expected, the behavior is not guaranteed and may raise a TypeError elsewhere; callers should pass a numpy.ndarray to match the function signature. The numeric values x, y and z are assigned using standard NumPy casting rules; incompatible dtypes may raise a TypeError or ValueError during assignment. There are no default arguments; all four parameters must be provided.
    """
    from Bio.PDB.vectors import set_homog_trans_mtx
    return set_homog_trans_mtx(x, y, z, mtx)


################################################################################
# Source: Bio.Phylo.Consensus.adam_consensus
# File: Bio/Phylo/Consensus.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_Phylo_Consensus_adam_consensus(trees: list):
    """Bio.Phylo.Consensus.adam_consensus: Compute the Adam consensus tree from multiple phylogenetic trees.
    
    This function is part of the Bio.Phylo.Consensus module in Biopython and implements the construction of an Adam consensus tree by combining the root clades of multiple input trees. In the context of computational molecular biology and phylogenetics (see the Biopython README and Bio.Phylo module), this is used to summarise a collection of inferred tree topologies (for example bootstrap replicates or results from different inference runs) into a single representative rooted tree. The function collects the .root attribute from each provided tree, passes the collection to the internal helper _part to compute the consensus root clade, and returns a new rooted BaseTree.Tree built from that result.
    
    Args:
        trees (list): list of trees to produce consensus tree. This must be a Python list whose elements are tree objects (instances expected to expose a .root attribute, as used in Bio.Phylo.BaseTree trees). Each element represents a phylogenetic tree whose root clade will be used by the Adam consensus algorithm. The function does not modify the provided tree objects; it only reads their .root attributes.
    
    Returns:
        Bio.Phylo.BaseTree.Tree: A new rooted Tree instance representing the Adam consensus of the input trees. The returned Tree is constructed with rooted=True and its root is the clade produced by the internal helper _part applied to the collection of input roots.
    
    Behavior, side effects, defaults, and failure modes:
        The function builds a list of clades via [tree.root for tree in trees] and then constructs and returns BaseTree.Tree(root=_part(clades), rooted=True). There are no side effects on the input trees (they are not mutated). The function assumes the input is a non-empty list of tree-like objects exposing a .root attribute; if an element lacks .root an AttributeError will occur when accessing tree.root. If an empty list is provided, or if the internal helper _part cannot produce a valid root from the supplied clades, an exception may be raised by _part or by BaseTree.Tree; callers should ensure the input list contains suitable tree objects. The function signature requires a list; passing a non-list object may produce a runtime error. The implementation relies on the internal helper _part to implement the details of the Adam consensus algorithm.
    """
    from Bio.Phylo.Consensus import adam_consensus
    return adam_consensus(trees)


################################################################################
# Source: Bio.Phylo.Consensus.bootstrap
# File: Bio/Phylo/Consensus.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_Phylo_Consensus_bootstrap(msa: list, times: int):
    """Generate bootstrap replicates from a multiple sequence alignment (OBSOLETE). This function implements the standard phylogenetic bootstrap resampling procedure used to assess support for inferred tree clades by creating pseudo-replicate alignments: each replicate is produced by sampling alignment columns (sites) with replacement to the same alignment length, and can be passed to tree-building routines to compute support values. The implementation yields replicates lazily (as a generator) and does not modify the input alignment. This function is marked OBSOLETE in the source; it may be superseded by newer, maintained utilities in Bio.Phylo or other Biopython modules.
    
    Args:
        msa (MultipleSeqAlignment): multiple sequence alignment to generate replicates from. In practice this should be an object implementing the MultipleSeqAlignment interface used in Biopython: indexable by row and sliceable by columns (the code uses msa[0] to determine alignment length and msa[:, col:col+1] to extract a single-column subalignment). The alignment provides the biological input alignment of homologous sequences (rows) and aligned sites (columns) used in phylogenetic bootstrap analysis. The function does not modify this object; it constructs new alignment objects for replicates via slicing/concatenation.
        times (int): number of bootstrap replicates to generate. In phylogenetic practice this is the number of pseudo-replicate alignments to create so that downstream tree inference can be repeated and support values (e.g., percent bootstrap support) computed. The function will attempt to yield exactly this many replicates, producing one replicate per iteration until the count is reached.
    
    Returns:
        generator: a generator that yields MultipleSeqAlignment objects, each a bootstrap replicate alignment constructed by sampling columns from the original msa with replacement. Each yielded replicate has the same number of columns (alignment length) as the input alignment and contains the same number of rows (sequences). The generator yields at most times replicates.
    
    Behavior, side effects, defaults, and failure modes:
        The function determines the number of columns from len(msa[0]); therefore, msa must be non-empty and its first record must support __len__ to report the alignment length. For each replicate a new alignment is built by repeatedly selecting a random column index in the range [0, length-1] using Python's random.randint and concatenating the selected single-column subalignments. Because the function uses the global random module, results are non-deterministic unless the caller sets the random seed externally (for example, random.seed(...)) before calling the function. The function yields replicates lazily and does not store them all in memory at once, which is memory-efficient for large numbers of replicates or large alignments. If the input alignment is empty (zero columns) calling the function will raise an error when determining the column range (for example, random.randint will receive an invalid range), and if msa is not indexable/sliceable as expected a TypeError or IndexError may be raised. The function is marked OBSOLETE in the source; users should consult the Bio.Phylo documentation or Biopython release notes for recommended alternatives for production use.
    """
    from Bio.Phylo.Consensus import bootstrap
    return bootstrap(msa, times)


################################################################################
# Source: Bio.Phylo.Consensus.bootstrap_trees
# File: Bio/Phylo/Consensus.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_Phylo_Consensus_bootstrap_trees(
    alignment: numpy.ndarray,
    times: int,
    tree_constructor: str
):
    """Generate bootstrap replicate phylogenetic trees from a multiple sequence alignment for use in assessing clade support in phylogenetic analyses.
    
    This function implements the standard non-parametric bootstrap for sequence alignments: it produces "times" replicate alignments by sampling alignment columns (sites) with replacement and then builds a tree for each replicate using the provided tree_constructor. It supports Biopython alignment objects (Alignment or MultipleSeqAlignment) and NumPy 2-D arrays (shape: sequences x sites), matching how the source code handles both object types. The function yields each replicate tree in turn rather than returning a collection, allowing streaming of results for downstream consensus or support-value calculations commonly used in Bio.Phylo workflows.
    
    Args:
        alignment (Alignment or MultipleSeqAlignment or numpy.ndarray): multiple sequence alignment to generate replicates from. For Biopython MultipleSeqAlignment or Alignment objects, the code treats each sequence as a sequence record and samples columns by slicing (alignment[:, col:col+1]) and concatenating slices to build the bootstrapped alignment. For a numpy.ndarray, the code expects a 2-D array with shape (n_sequences, n_sites) and samples column indices to form each replicate (alignment[:, cols]). This argument is the primary input representing aligned homologous sequences used in phylogenetic bootstrap analysis.
        times (int): number of bootstrap replicates to generate. This must be a non-negative integer; if zero, the generator yields no trees. Each replicate involves sampling the same number of sites as the original alignment (sampling with replacement) and then constructing a tree from that resampled alignment. Large values increase computation time and memory usage proportionally to the cost of building trees for each replicate.
        tree_constructor (TreeConstructor): object responsible for building a phylogenetic tree from an alignment. The object must implement a build_tree(alignment) method that accepts the same alignment type produced for each replicate and returns a tree object (as used by Bio.Phylo). In practice this is a TreeConstructor implementation from Bio.Phylo (for example one wrapping distance or character-based methods). The tree_constructor controls the tree-building algorithm and therefore directly affects the resulting bootstrap trees used for downstream consensus or support calculations.
    
    Returns:
        generator: a Python generator that yields one tree object per bootstrap replicate. Each yielded tree is the result of calling tree_constructor.build_tree on a bootstrapped alignment. The generator is lazy: trees are produced on-the-fly and not stored by this function, allowing downstream code to consume or aggregate results incrementally (for example, to compute a consensus tree or support values).
    
    Behavior, side effects, defaults, and failure modes:
        The function samples alignment columns with replacement using the Python standard library random module (random.randint). Because it uses random, results are non-deterministic unless the caller sets the global random seed (random.seed) before calling this function. For Alignment/MultipleSeqAlignment inputs, slices are concatenated iteratively which may be less memory-efficient than vectorized approaches; for numpy.ndarray inputs, a list of column indices is constructed and used to index the array per replicate. If alignment has zero sites, sampling will raise an IndexError or produce invalid slices; callers should validate alignment length beforehand. If tree_constructor does not provide a compatible build_tree method, a AttributeError or TypeError will be raised when attempting to build the first replicate. The function does not catch exceptions from the tree constructor, so errors from tree building (for example, due to malformed input or algorithm-specific failures) propagate to the caller.
    """
    from Bio.Phylo.Consensus import bootstrap_trees
    return bootstrap_trees(alignment, times, tree_constructor)


################################################################################
# Source: Bio.Phylo.Consensus.majority_consensus
# File: Bio/Phylo/Consensus.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_Phylo_Consensus_majority_consensus(trees: list, cutoff: float = 0):
    """Search majority-rule consensus tree from multiple phylogenetic trees.
    
    This function implements an extended majority-rule consensus algorithm used in phylogenetic analysis (Bio.Phylo module of Biopython) to combine multiple input trees into a single consensus topology. It accepts a collection of input trees that must share the same set and order of terminal taxa and produces a rooted Bio.Phylo.BaseTree.Tree whose internal clades represent groups of taxa that meet the specified majority cutoff. The method is "extended" because the cutoff may be any value between 0 and 1 (inclusive), not only the classical 0.5 threshold. For each consensus clade included, the returned clade.confidence attribute is set to the percentage (0.0–100.0) of input trees containing that clade, and clade.branch_length is set to the average branch length computed across only those input trees that contained the clade.
    
    Args:
        trees (list): list of input trees to produce the consensus from. Each item must be a phylogenetic tree object compatible with Bio.Phylo.BaseTree API (for example, objects that implement get_terminals() and which _count_clades can process). All trees must contain the same set of terminal taxa in a consistent order; otherwise the function will raise ValueError. The function consumes this list to count clades across trees, so providing an empty list will raise StopIteration (no trees available). Practical significance: users supply multiple inferred phylogenies (e.g., bootstrap replicates or trees from different methods) in order to summarise recurring clades into a single consensus tree for downstream evolutionary or comparative analyses.
        cutoff (float): frequency threshold between 0 and 1 (default 0). This specifies the minimum fraction of input trees that must contain a clade for that clade to be included in the consensus. The function converts this to a percentage internally (confidence = 100.0 * count_in_trees / tree_count) and compares confidence against cutoff * 100.0. A cutoff of 0 (the default) produces a relaxed binary consensus when at least one input tree is binary, meaning many compatible clades may be included; a cutoff of 0.5 reproduces the usual majority-rule consensus behavior. Values outside the [0, 1] interval are not validated explicitly by the code and may lead to unexpected results.
    
    Returns:
        BaseTree.Tree: a rooted Bio.Phylo.BaseTree.Tree representing the majority-rule consensus. The returned Tree.root is a BaseTree.Clade whose clades contain terminal and internal clade objects assembled according to compatibility and the specified cutoff. For each internal clade included: clade.confidence is a float giving the support as a percentage (0.0–100.0) across the input trees, and clade.branch_length is the arithmetic mean of branch lengths observed for that clade across the input trees that contained it. Terminal clade objects are the original terminal taxon objects taken from the first input tree.
    
    Behavior, side effects, defaults, and failure modes:
        This function counts clades across the provided trees (via an internal _count_clades helper), sorts clade patterns by occurrence and size, and iteratively inserts compatible clades into the consensus tree from highest to lowest support until the cutoff criterion fails or a complete set of clades has been assembled. Side effects are limited to constructing and returning a new BaseTree.Tree; the input tree objects are not modified. The default cutoff of 0 yields a relaxed binary consensus when one of the supplied trees is binary, as noted above. Common failure modes include: StopIteration if trees is empty (the function calls next() on an iterator of trees), ValueError if the terminal taxa across trees are inconsistent, and potential AttributeError/TypeError if supplied items do not implement the expected tree/clade interface (for example missing get_terminals(), index_one(), or compatibility-test methods). The function does not perform extensive validation of cutoff beyond using it as a fraction; callers should ensure cutoff is within the intended [0, 1] range. Performance depends on the number of trees and number of clades counted; memory and time scale with the total number of clades across input trees.
    """
    from Bio.Phylo.Consensus import majority_consensus
    return majority_consensus(trees, cutoff)


################################################################################
# Source: Bio.Phylo.Consensus.strict_consensus
# File: Bio/Phylo/Consensus.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_Phylo_Consensus_strict_consensus(trees: list):
    """Bio.Phylo.Consensus.strict_consensus searches for the strict consensus tree from multiple phylogenetic trees and returns a new Bio.Phylo.BaseTree.Tree that represents clades present in every input tree. In the context of the Biopython Bio.Phylo module (used in computational molecular biology for representing and analyzing phylogenetic trees), this function is used to summarise agreement across a set of inferred trees by including only those clades (groups of terminal taxa) that appear in all provided trees.
    
    Args:
        trees (iterable): iterable of tree objects to produce the consensus tree from. Each element should be a tree-like object compatible with Bio.Phylo (for example, trees returned by Bio.Phylo parsers) and must implement get_terminals() and provide a consistent ordering of terminals across all trees. The function consumes an iterator of these trees (calls iter(trees) and advances it), so if an iterator object is passed it will be advanced. Passing an empty iterable will cause a StopIteration to be raised when attempting to read the first tree.
    
    Returns:
        Bio.Phylo.BaseTree.Tree: a newly constructed tree whose root is a Bio.Phylo.BaseTree.Clade containing the strict clades (clades present in every input tree). The returned tree is built from new Clade objects and does not modify the structure of the input tree objects; it encodes the intersection of clades across the input set.
    
    Behavior and practical details:
        The function counts clades across all provided trees using an internal bitstring representation (_count_clades). A clade is considered "strict" if it appears in every input tree; such clades are selected and sorted by size (number of terminals) in descending order to reconstruct a hierarchical tree. The first tree's terminal ordering (first_tree.get_terminals()) is used as the reference list of terminals; all bitstring operations and clade reconstructions use this ordering. If the bitstring representing the union of terminals indicates that all terminals are present in the root candidate, the function attaches terminals/clades under that root and returns the constructed Tree.
    
    Failure modes and errors:
        If the provided iterable is empty, StopIteration will be raised when attempting to obtain the first tree. If the provided trees do not have a consistent set or ordering of taxa (terminals), reconstruction will fail and a ValueError is raised with the message "Taxons in provided trees should be consistent". If a non-iterable is passed, a TypeError will be raised by the iter() call. Other exceptions may propagate from the tree objects' methods (for example, if get_terminals() is missing or behaves unexpectedly).
    
    Side effects:
        The function does not modify the input tree objects; it constructs new Bio.Phylo.BaseTree.Clade and Bio.Phylo.BaseTree.Tree objects for the consensus. However, note that passing an iterator will consume elements from that iterator.
    """
    from Bio.Phylo.Consensus import strict_consensus
    return strict_consensus(trees)


################################################################################
# Source: Bio.Phylo.NeXMLIO.cdao_to_obo
# File: Bio/Phylo/NeXMLIO.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_Phylo_NeXMLIO_cdao_to_obo(s: str):
    """Converts a CDAO-prefixed CURIE/URI into an OBO-prefixed CURIE used in NeXML/CDAO parsing.
    
    This function is used by the CDAO parser in Bio.Phylo.NeXMLIO (the optional CDAO/RDF parsing code referenced in the Biopython README) to translate a short CDAO-prefixed identifier into the corresponding OBO-prefixed identifier expected in downstream ontology-annotated NeXML data structures. It expects a string beginning with the literal prefix "cdao:" and uses the module-level mapping cdao_elements to look up the local CDAO term name (the substring after "cdao:") and produce an OBO-prefixed CURIE of the form "obo:<mapped_value>". The result is intended for use where OBO-prefixed URIs/CURIEs are required by other code that consumes ontology terms (for example, when normalizing RDF/ontology annotations while parsing NeXML files).
    
    Args:
        s (str): Input CDAO-prefixed string to convert. This should be a Python str that begins with the exact characters "cdao:" followed by the local CDAO term name (for example "cdao:SomeTerm"). The parameter represents a CDAO CURIE/URI fragment as produced or encountered by the CDAO/RDF NeXML parser.
    
    Returns:
        str: A new Python str in OBO-prefixed form, constructed by prepending "obo:" to the value obtained from the module-level mapping cdao_elements for the local CDAO name. For example, if s == "cdao:foo" and cdao_elements["foo"] == "FOO_0001", the function returns "obo:FOO_0001". The function has no side effects and does not mutate its input.
    
    Raises:
        KeyError: If the substring after the "cdao:" prefix is not present as a key in the module-level mapping cdao_elements, a KeyError will be raised when looking up the mapping. This is the most common failure mode when the input uses an unknown or unexpected CDAO local name.
        TypeError: If s is not a str, operations in the function (such as slicing and mapping lookup) may raise a TypeError; callers should pass a str as required by the function signature.
    """
    from Bio.Phylo.NeXMLIO import cdao_to_obo
    return cdao_to_obo(s)


################################################################################
# Source: Bio.Phylo.NeXMLIO.matches
# File: Bio/Phylo/NeXMLIO.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_Phylo_NeXMLIO_matches(s: str):
    """Check for matches in both CDAO and OBO namespaces for a NeXML metadata identifier.
    
    This function is used in Bio.Phylo.NeXMLIO when processing NeXML metadata and ontology terms attached to phylogenetic trees. Its role is to detect identifiers that use the CDAO namespace prefix and, when present, produce a pair containing the original CDAO identifier and its mapped OBO equivalent (via the module-level cdao_to_obo lookup). This aids interoperability between CDAO-prefixed terms commonly found in NeXML files and OBO ontology identifiers used elsewhere in the Bio.Phylo codebase and downstream analyses.
    
    Args:
        s (str): The input identifier string to check. This is expected to be a namespaced metadata term from NeXML (for example, a string starting with the prefix "cdao:"). The practical significance is that callers will pass metadata keys or ontology identifiers extracted while parsing or writing NeXML; this function examines that string to determine whether a CDAO-to-OBO mapping should be produced. The function requires a Python str; passing a non-str object will likely raise an exception from the attempted string operation (for example, AttributeError or TypeError).
    
    Returns:
        tuple: A tuple of one or two str elements. If the input string s begins with the literal prefix "cdao:", the function returns a two-element tuple (s, mapped) where mapped is the OBO identifier string returned by the module-level cdao_to_obo(s) mapping function. If s does not start with "cdao:", the function returns a single-element tuple (s,). Both elements are the original input and, when present, its mapped equivalent as plain Python str objects. There are no other side effects; the function does not modify external state. Any exceptions raised by the cdao_to_obo lookup (for example, lookup or mapping errors) are propagated to the caller.
    """
    from Bio.Phylo.NeXMLIO import matches
    return matches(s)


################################################################################
# Source: Bio.Phylo.NeXMLIO.qUri
# File: Bio/Phylo/NeXMLIO.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_Phylo_NeXMLIO_qUri(s: str):
    """Bio.Phylo.NeXMLIO.qUri converts a prefixed (CURIE-like) URI into its full, absolute URI (IRI) using the module's namespace mapping, for use when reading or writing NeXML phylogenetic metadata where XML-style namespace expansion is required.
    
    This function is a thin wrapper around resolve_uri(s, namespaces=NAMESPACES, xml_style=True) defined in the same module: it looks up the prefix in the module-level NAMESPACES mapping and expands the input into an XML-style full URI suitable for inclusion in NeXML files and other XML-based phylogenetic exchange formats. It performs no I/O and has no side effects beyond calling resolve_uri.
    
    Args:
        s (str): A prefixed URI string (commonly a CURIE-like form such as "ex:term" or "dc:creator") to be expanded. In the Bio.Phylo.NeXMLIO domain this parameter is expected to identify ontology terms, metadata keys, or namespace-prefixed identifiers used in NeXML files. The caller must provide a Python str; passing a non-str will typically result in a TypeError propagated from resolve_uri.
    
    Returns:
        str: The expanded full URI (IRI) corresponding to the prefixed input, as produced by resolve_uri with the module-level NAMESPACES mapping and xml_style=True. This returned string is intended for use in NeXML documents and other XML-style metadata contexts where a complete URI is required. If the input is already a full URI resolve_uri will normally return it unchanged; if the prefix cannot be resolved the behavior depends on resolve_uri (it may return the input unchanged or propagate an error such as ValueError). Callers should handle such exceptions originating from resolve_uri.
    """
    from Bio.Phylo.NeXMLIO import qUri
    return qUri(s)


################################################################################
# Source: Bio.Phylo.PAML._parse_baseml.parse_basics
# File: Bio/Phylo/PAML/_parse_baseml.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_Phylo_PAML__parse_baseml_parse_basics(lines: list, results: dict):
    """Parse the basics that should be present in most baseml results files.
    
    This function scans an iterable of text lines produced by PAML's baseml program and extracts a small set of commonly required summary values used in downstream phylogenetic analyses. It is intended to be used as a helper in the Bio.Phylo.PAML parsing pipeline described in the Biopython project README: callers typically supply the lines of a baseml results file and a dictionary to be populated with parsed values. The parsed fields are useful in molecular evolution workflows: the PAML version identifies the software used, lnL (log-likelihood) values and "lnL max" are used for model comparison and inference, "tree length" reports the total branch length of the estimated tree, and "tree" provides the Newick tree string including branch lengths for downstream tree-based analyses. The function also attempts to extract the number of free parameters (np) reported by baseml.
    
    Args:
        lines (list): An iterable (typically a list) of strings, each string a single line from a baseml results file. Each element should be a text line exactly as produced by PAML baseml (for example from file.read().splitlines()). The function iterates over these lines and applies several regular-expression checks and numeric extractions to identify version information, log-likelihood values, tree length, and the estimated tree with branch lengths. Supplying non-string elements or a non-iterable will raise a TypeError or other exceptions when the code attempts string operations.
        results (dict): A mutable dictionary supplied by the caller that will be updated in-place with parsed fields. The function will set any of the following keys when the corresponding information is found in the input lines: "version", "lnL max", "lnL", "tree length", and "tree". These keys match those used elsewhere in the Bio.Phylo.PAML parsing code and are intended to be consumed by later stages of the baseml parsing pipeline. The caller should provide a dict (empty or pre-populated); if a non-dict is provided a TypeError may be raised.
    
    Behavior and details:
        - The function uses a precompiled floating-number regular expression (line_floats_re defined elsewhere in the module) to find numeric tokens on each line and converts those tokens to Python floats. If those tokens are not valid numeric strings a ValueError may be raised during conversion.
        - To extract the PAML version it matches lines like "BASEML (in paml version 4.3, August 2009)  alignment.phylip" and stores the captured version string under results["version"].
        - To extract the maximum log-likelihood it looks for a line containing "ln Lmax" and exactly one floating-point number on that line (for example "ln Lmax (unconstrained) = -316.049385"). That single float is stored under results["lnL max"].
        - To extract the reported lnL it looks for lines containing "lnL(ntime:" (for example "lnL(ntime: 19  np: 22):  -2021.348300      +0.000000"). If any floats are found on that line the first float is recorded under results["lnL"]. The same line is also tested with an np-capturing regex to extract the integer number of parameters; if present that integer is returned as num_params (see Returns). The function compiles and uses an np-specific regex matching the typical baseml format "lnL(ntime: <int>  np: <int>)".
        - To extract tree length it looks for a line containing "tree length" with exactly one floating-point number on that line and stores that float under results["tree length"] (for example "tree length =   1.71931").
        - To extract the estimated tree string it accepts lines that begin with an opening parenthesis (a Newick-like tree) and only records the tree if the line contains a colon character ":" indicating the presence of branch lengths; the line (with surrounding whitespace stripped) is stored under results["tree"].
        - The function scans lines in order; later matches overwrite earlier values for the same keys. The function does not validate the full Newick syntax beyond checking for an opening parenthesis and a colon.
        - Default behavior when items are not found: no corresponding key is added to results for that item, and num_params defaults to -1 to indicate "not found".
        - Side effects: results is modified in-place. The function also returns the updated results object (see Returns).
        - Failure modes and errors: If expected module-level helpers (for example the line_floats_re pattern) are missing, a NameError will arise. If lines contains non-string items, string operations and regex matches may raise TypeError. If numeric substrings matched by line_floats_re are not valid floats, float(...) may raise ValueError. The caller should ensure the input is a list of text lines from a baseml output file to avoid these errors.
    
    Returns:
        tuple: A two-element tuple (results, num_params). results is the same dictionary object passed in, updated in-place with any parsed keys ("version", "lnL max", "lnL", "tree length", "tree") that were found in the input lines. num_params is an int giving the number of free parameters (np) parsed from a line like "lnL(ntime: ... np: <int>)", or -1 if no such value was found. The returned results dict and num_params are intended to be used immediately by other Bio.Phylo.PAML parsing functions to assemble a complete representation of the baseml run.
    """
    from Bio.Phylo.PAML._parse_baseml import parse_basics
    return parse_basics(lines, results)


################################################################################
# Source: Bio.Phylo.PAML._parse_baseml.parse_freqs
# File: Bio/Phylo/PAML/_parse_baseml.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_Phylo_PAML__parse_baseml_parse_freqs(lines: list, parameters: dict):
    """Parse and extract basepair and branch frequency parameters from baseml output lines.
    
    Args:
        lines (list): A list of text lines (strings) from a baseml/PAML output file. Each element should be a string representing one line of the program output. This function scans these lines to locate numeric frequency information using several baseml-specific headings (for example "Base frequencies", "base frequency parameters", "freq:", and "(frequency parameters for branches)"). The function expects that floating point numbers within each line can be extracted (the calling module provides a regex, line_floats_re, to find numeric substrings) and converts those substrings to Python float values for storage.
        parameters (dict): A dictionary (typically initially empty or containing other parsed results) that will be updated in-place with any frequency information found in lines. This dict is both a mutable input and the carrier of results returned by the function; callers should pass the same dict they intend to receive updated.
    
    This function is used by Bio.Phylo.PAML._parse_baseml to interpret baseml output produced by the PAML package and to populate the parser state with nucleotide/base frequency data and branch-specific frequency parameters. It recognizes different baseml formatting across versions (for example baseml 4.1, 4.3 and 4.4) and handles cases where frequencies appear on the same line as their heading or on the following line.
    
    Behavior and side effects:
        - The function iterates over the provided lines and extracts floating point numbers from each line using the module-level regular expression used elsewhere in the parser. Extracted numeric substrings are converted to float and stored in local lists before being placed into parameters.
        - When a "Base frequencies" line with numeric values is found (as in baseml 4.3), the function creates a mapping parameters["base frequencies"] = {"T": float, "C": float, "A": float, "G": float} where the order corresponds to the numerical order emitted by baseml (the source code assigns the first four floats to T, C, A, G respectively).
        - When baseml presents a heading ("base frequency parameters" or a "Base frequencies" heading with no numbers on the same line), the function sets an internal flag and uses the next line with numeric values to build the same parameters["base frequencies"] mapping (this handles baseml 4.1 and 4.4 where the frequencies can appear on the next line).
        - If a line beginning with "freq: " contains numbers, those numbers are stored as a list in parameters["rate frequencies"] (a list of floats).
        - When the heading "(frequency parameters for branches)" is encountered, the function begins collecting branch/node specific frequency parameter blocks. It creates parameters["nodes"] as a mapping from integer node number to a dict describing that node. For each subsequent node line containing numeric values, the function:
            - extracts the node number from a prefix matching "Node #<n>" and uses it as an integer key,
            - creates an entry parameters["nodes"][n] = {"root": False, "frequency parameters": [f0, f1, f2, f3]} where the first four floats are stored under "frequency parameters",
            - if more than four floats are present on the same node line, the additional four floats are interpreted as base frequencies and stored under "base frequencies" as {"T": f4, "C": f5, "A": f6, "G": f7}.
        - A later line matching "Note: node <n> is root." is used to mark the previously recorded node entry by setting parameters["nodes"][n]["root"] = True and to terminate the branch-specific collection mode.
        - The parameters dict passed in is modified in-place; the function also returns that dict (see Returns below).
    
    Defaults and assumptions:
        - The function assumes elements of lines are strings. Non-string elements may produce errors from regex operations or numeric conversion.
        - The order of floats in baseml output is assumed to follow the patterns used in the source code: for base frequencies the four values correspond to T, C, A, G in that order; for node frequency parameters the first four numbers are generic "frequency parameters" and optional trailing four numbers are node-specific base frequencies in the same T, C, A, G order.
        - The function does not validate that the numeric values sum to 1.0 or any other constraint; it records the values as provided by the baseml output.
    
    Failure modes and exceptions:
        - If expected numeric substrings cannot be converted to float (for example if the regex returns non-numeric text), a ValueError may be raised during float conversion.
        - When parsing node lines, the code assumes a "Node #<n>" prefix will be present; if that pattern is not found, attempting to access node_res.group(1) will raise an AttributeError. Similarly, marking a root node expects that the referenced node has already been recorded in parameters["nodes"]; otherwise assigning parameters["nodes"][root_node]["root"] will raise a KeyError.
        - If the input lines do not follow any of the recognized baseml headings or numeric formats, the function will not add the corresponding keys to parameters (i.e., parameters may remain unchanged with respect to frequency data).
    
    Returns:
        dict: The same parameters dictionary object passed in, updated in-place with any parsed frequency information found in lines. The following keys may be added or updated:
            - "base frequencies": a dict mapping nucleotide letters to floats, e.g. {"T": float, "C": float, "A": float, "G": float}, when global base frequencies are present in the output.
            - "rate frequencies": a list of floats, when a "freq:" line is present.
            - "nodes": a dict mapping integer node numbers to per-node information dicts. Each node dict contains:
                - "root": a bool indicating whether this node was labelled as the root (True when the output contains "Note: node <n> is root.").
                - "frequency parameters": a list of four floats extracted for that node.
                - optionally "base frequencies": a dict like the global one above, present when the node line provided four extra floats representing nucleotide base frequencies.
        If no appropriate lines are present in the input, the function will return the original parameters dict unchanged.
    """
    from Bio.Phylo.PAML._parse_baseml import parse_freqs
    return parse_freqs(lines, parameters)


################################################################################
# Source: Bio.Phylo.PAML._parse_baseml.parse_kappas
# File: Bio/Phylo/PAML/_parse_baseml.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_Phylo_PAML__parse_baseml_parse_kappas(lines: list, parameters: dict):
    """Parse out the kappa parameters from baseml/PAML output lines.
    
    Args:
        lines (list): A sequence of text lines (strings) read from a baseml/PAML
            output file. This function scans these lines to find numeric values
            that represent the kappa parameter(s) (the transition/transversion
            related parameter used by nucleotide substitution models such as F84,
            HKY85 and T92) and branch-specific statistics. The function expects
            that floating point numbers in each line can be found by a precompiled
            regular expression named line_floats_re (used as line_floats_re.findall
            in the implementation). Lines containing the literal "Parameters
            (kappa)" are treated as a header indicating that the next numeric line
            contains the model-wide kappa value(s). Lines containing the literal
            "kappa under" are treated as alternate placements of kappa values
            (e.g. for REV model output). Branch-specific lines are detected when
            a leading substring matches the pattern "\d+\.\.\d+" (for example
            "1..2"), using re.match(r"\s(\d+\.\.\d+)", line) as in the source.
        parameters (dict): A mutable dictionary (typically empty or pre-populated
            from earlier parsing steps) that will be updated in-place with parsed
            kappa information. This dictionary is used across the baseml parser
            (Bio.Phylo.PAML._parse_baseml) to accumulate model parameters and
            per-branch values. The function both mutates this dict and returns it.
            Expected keys the function may set are "kappa" and "branches". The
            "kappa" value will be set to either a single float (when exactly one
            numeric value is found for the model-wide kappa) or a list of floats
            (when multiple numeric values are present on the same line). The
            "branches" value, if created, will be a mapping from branch-range
            strings (the regex group captured, e.g. "1..2") to dictionaries with
            numeric entries for "t", "kappa", "TS", and "TV" drawn from the same
            numeric line.
    
    Behavior, side effects, defaults, and failure modes:
        This function iterates over the provided lines and performs three main
        recognition tasks reflecting common baseml/PAML output formats observed in
        practice. First, on encountering a line that contains the literal
        "Parameters (kappa)" it sets an internal flag so that the next line with
        floating point numbers will be interpreted as the model-wide kappa value
        (or values). If exactly one floating point number is found on that next
        line, parameters["kappa"] is set to that float; if multiple numbers are
        present, parameters["kappa"] is set to the list of floats. Second, the
        function recognizes branch-specific rows when a line begins with a branch
        identifier matching the pattern "\d+\.\.\d+" (detected by
        re.match(r"\s(\d+\.\.\d+)", line)). In that case the function ensures
        parameters contains a "branches" mapping (creating an empty dict if needed)
        and assigns parameters["branches"][branch] to a dictionary with numeric
        keys "t" (time/branch length), "kappa" (branch-specific kappa), "TS"
        (transitions count/proportion) and "TV" (transversions count/proportion),
        taken in order from the floating point numbers found on that line (indices
        0, 1, 2 and 3 respectively). Third, the function also recognizes lines
        containing the substring "kappa under" and treats the numeric values on
        that same line as an alternative placement of the model kappa(s), setting
        parameters["kappa"] to either a single float or a list of floats as
        described above.
        The function mutates the parameters dict in-place and also returns it for
        convenience. If no kappa-related patterns or numbers are found, the input
        parameters dict is returned unchanged.
        Failure modes include malformed numeric lines or unexpected column counts:
        if a branch-specific line is detected but contains fewer than four numeric
        values, the code will attempt to index missing elements when assigning
        "t", "kappa", "TS", and "TV", which will raise an IndexError. If the
        regular expression objects (line_floats_re or re) have been altered or are
        missing in the parsing module, float extraction or branch detection will
        fail. The function does not itself validate biological plausibility of
        numeric values (for example negative times or extremely large kappas) and
        relies on upstream code or the user to perform such checks.
    
    Returns:
        dict: The same parameters dictionary object passed in (mutated in-place),
        now possibly containing the key "kappa" mapped to a float or a list of
        floats representing model-wide kappa estimate(s), and/or the key
        "branches" mapped to a dictionary of branch-range strings to dictionaries
        with numeric keys "t", "kappa", "TS", and "TV". This return value is used
        by the higher-level baseml parsing logic in Bio.Phylo.PAML._parse_baseml to
        collect substitution-model parameters for downstream analyses (for
        example reporting model fits, performing comparative analyses, or saving
        parsed results).
    """
    from Bio.Phylo.PAML._parse_baseml import parse_kappas
    return parse_kappas(lines, parameters)


################################################################################
# Source: Bio.Phylo.PAML._parse_baseml.parse_parameter_list
# File: Bio/Phylo/PAML/_parse_baseml.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_Phylo_PAML__parse_baseml_parse_parameter_list(
    lines: list,
    parameters: dict,
    num_params: int
):
    """Parse the parameters list found in baseml-formatted text and extract the unlabeled
    numeric parameter vector and associated standard errors (SEs), storing them in
    the provided parameters dictionary for downstream use in PAML baseml workflows.
    
    Args:
        lines (list): Sequence (list) of lines (strings) representing the text output
            to be parsed from a baseml run or an input file such as in.baseml. Each
            element is treated as a line of text and inspected for floating point
            numeric tokens. In the Biopython PAML baseml parser this is typically
            the file contents split on newline characters.
        parameters (dict): Mutable dictionary used to collect parsed results. This
            function mutates and returns this dict. On success it sets the key
            "parameter list" to the original line (stripped of surrounding whitespace)
            that contained exactly num_params floating point values, and, if present,
            sets the key "SEs" to the corresponding SEs line (also stripped). This
            dictionary is used by callers to capture starting parameter strings for
            reuse in subsequent baseml runs (for example copying into in.baseml).
        num_params (int): The expected number of numeric parameters on the unlabeled
            parameter line. The function looks for the first line that contains
            exactly this many floating point numbers (as determined by the compiled
            regular expression line_floats_re in the surrounding module) and treats
            that line as the canonical parameter list. This value must match the
            number of parameters produced by the baseml configuration being parsed.
    
    Returns:
        dict: The same parameters dictionary passed in, mutated to include any
        discovered entries. If a matching parameter line is found, "parameter list"
        will be present and mapped to the exact line string (trimmed). If that
        parameter line is immediately followed by a line containing the literal
        "SEs for parameters:" and a subsequent line of SEs, "SEs" will be set to
        that subsequent line string (trimmed). If no matching parameter line is
        found the original dictionary is returned unchanged.
    
    Behavior and side effects:
        The function iterates the provided lines in order and applies a compiled
        regular expression (line_floats_re from the module) to extract candidate
        floating point tokens from each line. For the first line where the count of
        extracted floats equals num_params, the function records the original
        trimmed line in parameters["parameter list"]. It then checks the following
        two lines: if the immediate next line contains the literal substring
        "SEs for parameters:" the function will read the line after that and store
        it (trimmed) under parameters["SEs"]. After storing these values the
        function stops scanning further lines and returns the parameters dict.
        The function does not attempt to reorder, normalize, or individually store
        numeric values; it stores the parameter and SE lines as strings to preserve
        the exact formatting required for reproducible baseml runs (for example,
        copying directly into in.baseml).
    
    Failure modes and edge conditions:
        If no line containing exactly num_params floating point tokens is found,
        the function returns the original parameters dict unchanged. The function
        assumes that when it finds a candidate parameter line there are at least
        two subsequent lines available to check for the SEs header and SEs values;
        if the input lines list is truncated such that accesses to
        lines[line_num + 1] or lines[line_num + 2] would be out of range, an
        IndexError may be raised. If the module-level regular expression yields
        tokens that cannot be converted to float by float(), a ValueError may be
        raised during the temporary conversion step used to count numeric tokens.
        Callers should ensure the input lines correspond to baseml-like text output
        and may pre-validate length and format to avoid these exceptions.
    
    Domain significance:
        In the context of Bio.Phylo.PAML._parse_baseml and the Biopython PAML
        integration, preserving the exact parameter and SE lines as strings is
        important for reproducibility and for supplying starting parameter values
        to future baseml runs (e.g., by copying the saved string into in.baseml).
        This function supports automated parsing pipelines that extract those
        values from baseml output files for reporting, reusing starting values, or
        further analysis.
    """
    from Bio.Phylo.PAML._parse_baseml import parse_parameter_list
    return parse_parameter_list(lines, parameters, num_params)


################################################################################
# Source: Bio.Phylo.PAML._parse_baseml.parse_parameters
# File: Bio/Phylo/PAML/_parse_baseml.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_Phylo_PAML__parse_baseml_parse_parameters(
    lines: list,
    results: dict,
    num_params: int
):
    """Parse and collect model parameter values from lines of a baseml/PAML output file and store them into the provided results dictionary for downstream use by the Bio.Phylo.PAML baseml parser.
    
    Args:
        lines (list): The raw file content provided as a list of text lines (each item is a str). In the context of Biopython's PAML baseml parser, these lines are expected to be the lines of a baseml output or related parameter block. This argument is read but not modified; helper functions scan these lines to locate and extract parameter definitions such as scalar parameters, kappa values (transition/transversion ratios), substitution rate categories, and nucleotide/amino-acid equilibrium frequencies.
        results (dict): A mutable dictionary used to accumulate parsed sections of the baseml output. This function will add or replace the key "parameters" in this dictionary with a mapping of parsed parameter names to their values. callers typically pass a results dict that already contains other parsed sections; parse_parameters appends the parsed parameter block so later code in Bio.Phylo.PAML can access model settings for likelihood calculations, model comparison, or reporting.
        num_params (int): The expected number of parameters for the initial parameter list parsing step. This integer is forwarded to the helper parse_parameter_list function and determines how many scalar parameters to read from lines. It controls interpretation of a parameter list in the baseml output (for example, the number of estimated free parameters reported by PAML). If this value does not match the content found in lines, the helper functions may produce fewer/more entries or raise an error.
    
    Behavior and side effects:
        This function creates a local parameters dictionary (initialized empty) and populates it by calling a fixed sequence of helper parsers: parse_parameter_list(lines, parameters, num_params), parse_kappas(lines, parameters), parse_rates(lines, parameters), and parse_freqs(lines, parameters). Each helper is responsible for locating and parsing a specific portion of the baseml parameter output and updating the same parameters dictionary in place. Typical keys inserted by these helpers include entries corresponding to scalar parameters (from parse_parameter_list), kappa values (transition/transversion ratio(s)), substitution rate settings (rate categories or relative rates), and equilibrium frequencies (base or amino-acid frequencies), represented as numeric values or sequences as produced by those helpers. After parsing, the function assigns the populated parameters dictionary to results["parameters"] and returns the results dictionary. The primary side effect is the mutation of the provided results dict (adding or replacing the "parameters" key).
    
    Defaults and practical significance:
        The function initializes an empty parameters mapping and relies entirely on the helper functions to detect and extract parameter information from the provided lines. In PAML baseml workflows within Biopython, the populated results["parameters"] provides the necessary numeric and categorical information about the evolutionary model (e.g., kappa, rate heterogeneity parameters, base frequencies) that other modules use to interpret likelihood output, reproduce model settings, or perform downstream analyses such as tree re-scoring or model comparison.
    
    Failure modes:
        This function does not itself validate the contents beyond delegating to the helper parsers. If the input lines do not contain the expected parameter blocks, or if num_params is incorrect for the file content, the helper functions may return an incomplete parameters mapping or raise exceptions (for example ValueError, IndexError, or custom parsing errors raised by those helpers). Callers should either ensure the input originates from a compatible baseml/PAML output or be prepared to handle exceptions propagated from the helper parsing functions.
    
    Returns:
        dict: The same results dictionary object passed in as the results argument, after being updated to include a "parameters" entry whose value is a dictionary of parsed model parameters. This return value is provided for convenience; the primary effect is the in-place mutation of the results argument.
    """
    from Bio.Phylo.PAML._parse_baseml import parse_parameters
    return parse_parameters(lines, results, num_params)


################################################################################
# Source: Bio.Phylo.PAML._parse_baseml.parse_rates
# File: Bio/Phylo/PAML/_parse_baseml.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_Phylo_PAML__parse_baseml_parse_rates(lines: list, parameters: dict):
    """Bio.Phylo.PAML._parse_baseml.parse_rates parses rate-related lines from baseml/PAML output and extracts numeric rate parameters into a Python dictionary used by the Bio.Phylo.PAML baseml parser. This function is used when reading baseml program output (phylogenetic substitution model results) to collect rate parameters, per-category rates, the 4x4 nucleotide substitution rate matrix Q (for REV-like models), the gamma shape parameter alpha, the rho parameter for auto-discrete-gamma models, and transition probability arrays. The parsed values are returned in the same dictionary object passed in so downstream code in Bio.Phylo.PAML can access model parameters for tasks such as annotating trees, interpreting substitution rates, or performing further calculations.
    
    Args:
        lines (list): Lines of text (typically a list of str) from a baseml/PAML output file. Each element is scanned in order for substrings that indicate rate information. The function looks for the exact substrings "Rate parameters:", "rate: ", "matrix Q", "alpha", "rho", and "transition probabilities" (case-sensitive) and then extracts floating-point numbers from those lines and subsequent expected lines. Practical significance: these lines normally appear in baseml output to report estimated substitution rates, relative rates per site-class, a 4x4 nucleotide rate matrix, and transition probability matrices; providing the raw output lines allows this function to populate structured parameters for use by the Bio.Phylo.PAML baseml parser.
        parameters (dict): A dictionary (typically empty or partially populated) that the function will update in-place with parsed numeric values. The function will add or overwrite keys described below so that downstream code in Bio.Phylo.PAML can read standardized fields. Practical significance: callers should pass a dict to collect parsed model parameters; the same dict is returned for convenience.
    
    Behavior and side effects:
        The function iterates over the provided lines and uses a regular expression (line_floats_re, defined elsewhere in the module) to find all floating-point numeric substrings on each line and converts them to Python floats. When it finds recognized marker strings it stores parsed floats into the parameters dict as documented below. The function mutates the parameters dict in place and also returns it (see Returns). Specific keys the function will set (if the corresponding lines are present) are:
        - "rate parameters": list of floats parsed from a line containing "Rate parameters:"; these are the overall rate parameters reported by baseml and may be used to scale or interpret substitution rates.
        - "rates": list of floats parsed from a line containing "rate: "; these are per-category relative rates (for example discrete gamma category rates) and are commonly used in rate-heterogeneity models.
        - "Q matrix": dict with keys:
            - "matrix": list of four lists of floats representing the 4x4 nucleotide substitution rate matrix rows. The function expects and collects four rows (hard-coded for nucleotide data).
            - "average Ts/Tv": single float parsed from the same line that introduces "matrix Q" if a numeric value appears there; this represents the average transition/transversion ratio reported by baseml for REV-like models.
          The Q matrix is significant for downstream calculations that require the instantaneous rate matrix of the substitution model.
        - "alpha": float parsed from a line containing "alpha"; this is the gamma shape parameter describing among-site rate variation (K categories) used by baseml.
        - "rho": float parsed from a line containing "rho"; this is the rho parameter used for the auto-discrete-gamma model in baseml.
        - "transition probs.": list of lists of floats parsed after a line containing "transition probabilities"; each appended sub-list is a row of transition probabilities and the function stops collecting when the number of rows equals len(parameters["rates"]) (i.e., one row per rate category). These transition probabilities are used to describe per-category substitution probabilities over a branch length or per-category scaling and are meaningful when baseml prints them.
    
    Defaults and assumptions:
        The function assumes lines are provided in the same textual format emitted by baseml and that floating-point numbers can be extracted by the module-level regular expression line_floats_re. It assumes nucleotide data (4x4 Q matrix) and therefore collects exactly four rows for the Q matrix. The function treats the marker substrings as case-sensitive and looks for the literal text shown above.
    
    Failure modes and exceptions:
        If the module-level name line_floats_re is not defined or not a functioning regex, a NameError or related exception will occur. If numeric substrings cannot be converted to float (unexpected formatting), a ValueError may be raised during conversion. If transition probabilities are present but the "rates" key is not present in the parameters dict when the function tries to compare lengths, a KeyError can occur because the code compares the number of collected transition-probability rows to len(parameters["rates"]). The function does not itself validate that the Q matrix rows are valid probability rows or that the number of transition probability rows matches any external expectation beyond the check above.
    
    Returns:
        dict: The same parameters dictionary object passed in, updated in-place with any parsed keys described in the Behavior section. Callers can use the returned dict or continue using the original parameters object to access parsed rate parameters such as "rate parameters", "rates", "Q matrix", "alpha", "rho", and "transition probs.". If no relevant lines are found, the dict will be returned unchanged.
    """
    from Bio.Phylo.PAML._parse_baseml import parse_rates
    return parse_rates(lines, parameters)


################################################################################
# Source: Bio.Phylo.PAML._parse_codeml.parse_basics
# File: Bio/Phylo/PAML/_parse_codeml.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_Phylo_PAML__parse_codeml_parse_basics(lines: list, results: dict):
    """Parse basic metadata from lines of a PAML codeml output file.
    
    This function inspects an iterable of text lines produced by the PAML codeml program and extracts common, high-level information that appears in most codeml output files. It is intended for use within the Bio.Phylo.PAML codeml output parser in Biopython to gather metadata before more detailed parsing. The function looks for a program version string, the model description, evidence of multi-gene analyses, the codon substitution frequency model, the site-class model name (if present), and the maximum log-likelihood value reported in the file. The implementation uses several compiled regular expressions defined in the same module to match codeml-specific headers (for example, it recognizes both "Codon frequencies:" used in codeml 4.1 and "Codon frequency model:" used in codeml 4.3+), and it relies on a module-level regular expression named line_floats_re to extract floating-point numbers from a line when determining the maximum log-likelihood.
    
    Args:
        lines (list): An iterable (typically a list) of text lines from a codeml output file. Each element should be a string corresponding to one line of the codeml output. Practical significance: callers normally pass the result of reading a codeml output file (for example, file.readlines()) so the parser can scan program version, model descriptions, codon model headers, site-class model headers, gene-count annotations, and numeric values such as the maximum log-likelihood. If non-string elements are present, regular expression matching may fail with a TypeError.
        results (dict): A dictionary provided by the caller that will be populated with parsed metadata. The function mutates this dictionary in place and also returns it as the first element of the return tuple. Domain-relevant keys that may be added or updated are "version" (program version string, e.g. "4.9"), "model" (the model description line), "genes" (a list of empty dicts created when the output indicates multiple genes are analyzed), "codon model" (codon frequency/substitution model string), "site-class model" (the NSsites site-class model name when present), and "lnL max" (the maximum log-likelihood as a float). Callers should supply an existing dict if they want to preserve prior parsing results; otherwise pass an empty dict. Because the function modifies this dict in place, callers who reuse the same dict across multiple files should clear it between uses.
    
    Returns:
        tuple: A 3-tuple containing (results, multi_models, multi_genes).
            results (dict): The same dictionary object passed in, after being updated with any metadata found. This dictionary is suitable for downstream processing by other Bio.Phylo.PAML parsers and for inclusion in Biopython data structures representing codeml results.
            multi_models (bool): True if the function detected that the output file contains results for multiple models (in which case no single "site-class model" name was set); False if a single site-class model name was identified. This flag guides higher-level code to expect either a single model block or multiple model blocks in the file.
            multi_genes (bool): True if the output indicates a multi-gene run (the parser created a "genes" key containing an empty dict for each reported gene); False if no multi-gene declaration was found. When True, downstream code can iterate over results["genes"] to populate per-gene results.
    
    Behavioral notes and side effects: The function scans every line in the provided lines list once. It updates the provided results dictionary in place and also returns it. It does not open or read files itself; the caller must supply file contents as lines. Numeric extraction for "lnL max" uses a module-level regular expression line_floats_re to find floating-point tokens; if that name is not defined in the module, a NameError will be raised. If a keyed header is absent in the file, the corresponding key will not be added to results (for example, no "codon model" key if no codon frequency header is present). The function will convert the first floating-point token found on a line containing "ln Lmax" to a Python float and store it under "lnL max"; if no floats are found on that line, "lnL max" will not be set. Failure modes include passing an object that is not iterable for lines (raising TypeError), passing a non-dictionary for results (which will cause attribute errors on assignment), or malformed lines that prevent regex matches (in which case fewer keys are populated). The caller should validate inputs and handle exceptions if robust behavior is required.
    """
    from Bio.Phylo.PAML._parse_codeml import parse_basics
    return parse_basics(lines, results)


################################################################################
# Source: Bio.Phylo.PAML._parse_codeml.parse_branch_site_a
# File: Bio/Phylo/PAML/_parse_codeml.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_Phylo_PAML__parse_codeml_parse_branch_site_a(
    foreground: bool,
    line_floats: list,
    site_classes: list
):
    """Bio.Phylo.PAML._parse_codeml.parse_branch_site_a parses results produced for the PAML branch-site A model and annotates a list of site-class records with branch-specific numeric values. This function is used within the Bio.Phylo.PAML._parse_codeml module to incorporate per-site, per-branch numeric output (for example, posterior probabilities or parameter estimates parsed from codeml output lines) into the site_classes data structure used by downstream Biopython code that represents site class information for phylogenetic selection analyses.
    
    This function updates the provided site_classes list in place by ensuring each site class entry has a "branch types" mapping and then storing the corresponding numeric value from line_floats under the key "foreground" if foreground is True, or "background" if foreground is False. If there is nothing to do (site_classes is falsy or line_floats is empty), the function returns immediately without modifying the inputs.
    
    Args:
        foreground (bool): Flag indicating which branch category the values in line_floats correspond to. When True, values from line_floats are stored under the "foreground" key for each site class; when False, they are stored under the "background" key. This boolean is provided by the codeml parsing context to distinguish foreground branch estimates from background branch estimates for the branch-site A model.
        line_floats (list): A sequence of numeric values parsed from codeml output lines that correspond position-wise to entries in site_classes. Each element is intended to be a float-like numeric value (for example, a posterior probability or parameter estimate) to be recorded for the matching site class. The function iterates over range(len(line_floats)) and uses these values directly.
        site_classes (list): A list of mutable mapping-like site class records (e.g., dict objects) produced earlier in the codeml parsing process, where each element represents information for one site class. The function expects to be able to call .get("branch types") on each element and assign to site_classes[n]["branch types"][...] to store branch-specific values.
    
    Returns:
        list or None: The same site_classes list passed in, after mutation to include a "branch types" mapping for each processed site class with either a "foreground" or "background" key set to the corresponding value from line_floats. If the function returns None it indicates no work was performed because site_classes was falsy or line_floats was empty; in that case the original site_classes is not modified.
    
    Side effects and behavior details:
        The function mutates the site_classes list in place. For each index n in range(len(line_floats)), it ensures site_classes[n] contains a dictionary under the key "branch types" and then sets site_classes[n]["branch types"]["foreground"] = line_floats[n] when foreground is True, or site_classes[n]["branch types"]["background"] = line_floats[n] when foreground is False. The function processes as many entries as there are in line_floats; it does not truncate or extend site_classes automatically.
    
    Failure modes and exceptions:
        If site_classes is falsy (for example, an empty list or None) or line_floats is empty, the function returns immediately with None and makes no changes.
        If len(line_floats) is greater than len(site_classes), accessing site_classes[n] may raise IndexError.
        If elements of site_classes are not mapping-like objects that support .get(...) and item assignment (for example, not dict-like), AttributeError or TypeError may be raised.
        If elements of line_floats are not numeric as expected, downstream code that assumes numeric values may fail, though this function will accept and store any object from line_floats as-is.
    """
    from Bio.Phylo.PAML._parse_codeml import parse_branch_site_a
    return parse_branch_site_a(foreground, line_floats, site_classes)


################################################################################
# Source: Bio.Phylo.PAML._parse_codeml.parse_distances
# File: Bio/Phylo/PAML/_parse_codeml.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_Phylo_PAML__parse_codeml_parse_distances(lines: list, results: dict):
    """Parse amino acid sequence distance results from lines of codeml output and
    store them into the provided results dictionary under the "distances" key.
    
    This function is used in the Bio.Phylo.PAML._parse_codeml module to extract
    pairwise amino-acid distances reported by PAML's codeml program. It recognizes
    two types of reported distances in the codeml text output: raw ("AA distances")
    and maximum-likelihood ("ML distances of aa seqs."). The parser expects the
    distance data to appear as a lower-diagonal matrix: each matrix row begins
    with a sequence name followed (after 5–15 spaces) by a sequence of floating
    point distance values corresponding to previously-seen sequence names in the
    matrix. The function constructs a symmetric nested dictionary representation of
    those distances and inserts it into the supplied results dict under the key
    "distances". This representation is commonly used downstream in phylogenetic
    analysis workflows within Biopython to compute or visualize pairwise sequence
    distance matrices.
    
    Args:
        lines (list): A list of strings representing the lines of text from a
            codeml output file. Each element is expected to be one line of output.
            The parser scans these lines for the literal substrings "AA distances"
            and "ML distances of aa seqs." to determine whether subsequent matrix
            rows are raw or ML distances, respectively. Matrix rows must match the
            pattern "sequence_name" followed by 5–15 whitespace characters and
            then floating point numbers; sequence_name is used as the key for that
            row. If lines does not contain any recognized distance sections, the
            results dict will be left unchanged. Providing lines in a different
            format may cause no distances to be found or may raise exceptions when
            converting extracted numeric tokens to float.
        results (dict): A dict (typically initially empty or previously
            populated by other codeml parsers) that will be modified in-place. If
            distance data are found, this function will set results["distances"]
            to a nested dict structure (see Returns). If results already contains
            a "distances" key, it will be overwritten when new distances are
            parsed. This function also returns the same dict object for convenience.
    
    Returns:
        dict: The same results dictionary passed in, potentially modified. When any
        distances are parsed, results["distances"] will be populated with a dict
        that may contain the keys "raw" and/or "ml" (strings matching the
        literal section headers found in the codeml output). Each of these maps
        sequence names (strings) to another dict mapping other sequence names
        (strings) to floating point distance values. Distances are stored
        symmetrically so that distances["raw"][A][B] == distances["raw"][B][A]
        (and similarly for "ml"). If no distances are found, the original results
        dict is returned unmodified.
    
    Behavior, side effects, defaults, and failure modes:
        - The parser uses a local regex to identify matrix rows (sequence name
          followed by 5–15 spaces) and relies on a module-level regular expression
          named line_floats_re to extract numeric tokens. If line_floats_re is not
          defined in the module scope, a NameError will be raised.
        - Floating point tokens found by line_floats_re are converted to Python
          float objects; if the token stream is malformed such that conversion
          fails, a ValueError may be raised.
        - The function detects sections by checking for the literal line contents
          "AA distances" (for raw distances) and "ML distances of aa seqs."
          (for maximum-likelihood distances). Only rows encountered while one of
          these section flags is active are interpreted as matrix rows.
        - The parser builds a list of sequence names in the order they first
          appear in matrix rows; numeric entries on each matrix row are assigned
          to the previously seen sequence names, reflecting a lower-diagonal
          representation. The code inserts each parsed distance twice (A->B and
          B->A) to produce a symmetric matrix.
        - If no distances are parsed, results is returned unchanged. If distances
          are parsed, results["distances"] will be created or replaced with the
          parsed nested dictionary structure.
        - This function does not perform file I/O; it operates only on the list of
          text lines provided.
    """
    from Bio.Phylo.PAML._parse_codeml import parse_distances
    return parse_distances(lines, results)


################################################################################
# Source: Bio.Phylo.PAML._parse_codeml.parse_model
# File: Bio/Phylo/PAML/_parse_codeml.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_Phylo_PAML__parse_codeml_parse_model(lines: list, results: dict):
    """Parse an individual NSsites model's results from a codeml (PAML) output block.
    
    This function processes a list of text lines produced by codeml (the PAML package) for a single NSsites model and extracts numerical results and annotations commonly used in evolutionary analyses (for example, maximum log-likelihood for model comparison, estimated tree lengths, branch-wise dN/dS values, site-class proportions and omegas, kappa values, and other model parameters). It is intended to be used within Bio.Phylo.PAML._parse_codeml to convert codeml output into a structured Python dictionary suitable for downstream tasks such as likelihood-ratio tests, summarising model parameters, or exporting results for further computational molecular evolution analyses. The function mutates the supplied results dict in place and also returns it.
    
    Args:
        lines (list): A list of strings, each string is one line of codeml output corresponding to a single NSsites model section. Each line is examined for numeric tokens, specific substrings (for example "lnL(ntime:", "tree length =", "SEs for parameters:", "dS tree:", "dN tree:", "w ratios as labels for TreeView:"), Newick-like tree patterns, branch table rows, and parameter assignments. The caller should provide the exact text lines as produced by codeml (newlines may be stripped). The function expects the usual codeml formatting (e.g. "lnL(ntime: 19  np: 22):  -2021.348300", Newick trees terminating in ";" with branch length annotations, and branch table rows beginning with an index like "6..7"). If the input deviates substantially from codeml conventions, some fields may not be found.
        results (dict): A dict object to be populated with parsed results. This dict may already contain keys (for example inherited from a previous parsing step) and will be updated in place by this function. Typical keys added or updated include "lnL" (float), "tree length" (float), "tree", "dS tree", "dN tree", "omega tree" (strings for Newick lines), and "parameters" (a nested dict described below). The function returns this same dict after adding or updating keys; callers should be aware that it mutates this object.
    
    Behavior and side effects:
        The function iterates through lines and uses regular expressions and substring checks to detect and extract the following items when present:
        - Maximum log-likelihood (results["lnL"]) and number of free parameters (local num_params inferred from "lnL(ntime: ... np: N)").
        - A parameter list line (parameters["parameter list"], stored as the raw stripped string) when a line contains exactly num_params floating point numbers and SE parsing is not active.
        - Standard errors (parameters["SEs"]) parsed when a line immediately following the "SEs for parameters:" marker contains num_params numbers.
        - Tree length (results["tree length"]) parsed from lines containing "tree length =".
        - Newick-like trees (results["tree"] or specialized trees such as results["dS tree"], results["dN tree"], results["omega tree"]) when a line matches a Newick pattern and contains branch length or label annotations (detected by ":" or "#"). Which tree slot is used depends on preceding marker lines "dS tree:", "dN tree:", or "w ratios as labels for TreeView:"; those markers set internal flags so that the next qualified Newick line is stored under the corresponding key.
        - Multi-gene rate vectors (parameters["rates"]) for lines containing "rates for" followed by numeric tokens; the function inserts 1.0 at the start of the parsed float list to match the format expected elsewhere in the code.
        - Kappa values (parameters["kappa"]) from lines containing "kappa (ts/tv)".
        - Omega values (parameters["omega"]) from lines containing "omega (dN/dS)" or "w (dN/dS)". The latter may produce a list if multiple values appear on the line.
        - Per-gene kappa and omega (parameters["genes"]) when lines contain "gene # N: kappa = ... omega = ..."; results are stored as parameters["genes"][N] = {"kappa": float, "omega": float}.
        - dN and dS tree lengths (parameters["dN"], parameters["dS"]) from lines containing "tree length for dN" and "tree length for dS".
        - Site-class proportions and omegas (parameters["site classes"]) by delegating to helper functions parse_siteclass_proportions, parse_siteclass_omegas, parse_clademodelc, and parse_branch_site_a as appropriate for lines beginning with "p:", "proportion", "w:", "branch type ", "foreground w", or "background w". The exact structure of parameters["site classes"] is produced by these helper functions; callers can inspect it to obtain per-site-class proportions and omega values used in NSsites models.
        - Branch-wise estimates (parameters["branches"]) from rows matching a branch table pattern (e.g. "6..7   0.000  167.7  54.3  0.0000  0.0000  0.0000  0.0  0.0"). Each branch entry is stored as parameters["branches"][branch_id] = {"t": float, "N": float, "S": float, "omega": float, "dN": float, "dS": float, "N*dN": float, "S*dS": float}.
        - Additional model parameters (single-letter or letter+digit keys, like p, q, p0, etc.) parsed across lines by a regex and stored directly in parameters as parameter_name: float.
    
        After processing all lines, if any parameter items were found, they are bundled into a dict assigned to results["parameters"]. The function always returns the results dict (the same object passed in), allowing chaining or additional updates by the caller.
    
    Defaults and data formats:
        - Numeric values parsed and stored as Python float.
        - Trees and other raw lines are stored as stripped strings.
        - parameters is a dict containing a mixture of strings, floats, lists of floats, and nested dicts as described above; structural details for site classes are determined by the helper parsing functions.
        - If the number of parameters (num_params) cannot be determined from an "lnL(ntime: ... np: N)" line, lines that would be matched solely by equality with num_params (parameter list or SE lines) will not be captured.
    
    Failure modes and error conditions:
        - The function assumes codeml-formatted text; if lines do not conform, entries may be missing or mis-parsed. Missing fields are simply not added to results; the function does not raise for missing keys.
        - Converting tokens to float may raise ValueError if unexpected non-numeric tokens are encountered in positions where floats are expected.
        - The branch table regex and tree regex are moderately strict; branch rows or Newick trees that deviate from the anticipated formatting may not be recognized.
        - Model parameter parsing relies on a regex that expects parameter assignments in the form "name = number" separated by whitespace; unusual spacing or formats may prevent detection.
    
    Returns:
        dict: The same dict object passed in as results, updated in place with parsed fields. Typical keys added include "lnL" (float), "tree length" (float), "tree"/"dS tree"/"dN tree"/"omega tree" (str), and "parameters" (dict) which itself may contain "parameter list" (str), "SEs" (str), "rates" (list of floats), "kappa" (float), "omega" (float or list), "genes" (dict of per-gene dicts), "dN" (float), "dS" (float), "site classes" (structure from helper parsers), "branches" (dict of branch -> numeric dict), and any single-letter model parameters parsed. If no recognizable information is found, the returned dict may be unchanged or contain only a subset of the keys described above.
    """
    from Bio.Phylo.PAML._parse_codeml import parse_model
    return parse_model(lines, results)


################################################################################
# Source: Bio.Phylo.PAML._parse_codeml.parse_nssites
# File: Bio/Phylo/PAML/_parse_codeml.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_Phylo_PAML__parse_codeml_parse_nssites(
    lines: list,
    results: dict,
    multi_models: bool,
    multi_genes: bool
):
    """Determine which NSsites models are present in a codeml output and parse
    their per-model results into the provided results dictionary.
    
    This function is used within Bio.Phylo.PAML._parse_codeml to analyze the
    text output of PAML's codeml program and extract results for NSsites
    (site-class) models (for example M0, M1a, M2a, etc.). It locates model
    sections either as a single reported model for the whole result or as
    multiple "Model X: ..." blocks, delegates the textual parsing of each
    model block to the module-level helper parse_model, and then stores the
    parsed data in the supplied results dict under the "NSsites" key (and,
    when multi_genes is True, into results["genes"] for per-gene model
    results). The function returns the (mutated) results dictionary.
    
    Args:
        lines (list): Lines of text (strings) comprising the codeml output
            file to be scanned. Each entry is a single line from the output;
            this function iterates over these lines to find model and gene
            section boundaries. The list is not modified by this function.
        results (dict): Mutable dictionary holding aggregate parsing results
            already collected before calling parse_nssites. This function
            may read keys such as "site-class model" and "genes", and will
            add or update the "NSsites" key when valid NSsites model data
            are found. The same dict object is returned after in-place
            modification. In the multi-genes mode this function expects
            results["genes"] to exist and be indexable (a list-like object)
            with positions for each gene (the code writes to
            genes[gene_number - 1]).
        multi_models (bool): Boolean flag indicating whether the codeml
            output contains multiple NSsites models (True) or a single
            model only (False). When True the function scans for lines
            matching the regular expression r"Model (\d+):\s+(.+)" and
            parses each model block separately. When False the function
            determines the single model to parse from results["site-class
            model"] (or uses the default "one-ratio") and parses either a
            single model block or multiple gene blocks depending on
            multi_genes.
        multi_genes (bool): Boolean flag indicating whether the codeml
            output contains per-gene NSsites results (True) rather than a
            single result for all sequences (False). When True the function
            looks for lines matching r"Gene\s+([0-9]+)\s+.+" to split the
            input into per-gene blocks and writes parsed model results into
            results["genes"] at index gene_number - 1.
    
    Behavior, side effects, defaults, and failure modes:
        - If multi_models is False, the function reads results.get("site-class
          model") to determine which named site-class model was used. If that
          key is missing, the default description "one-ratio" is assumed. The
          name-to-model-index mapping used in the single-model branch is:
          "one-ratio": 0, "NearlyNeutral": 1, "PositiveSelection": 2,
          "discrete": 3, "beta": 7, "beta&w>1": 8, "M2a_rel": 22. The integer
          index selected (current_model) is used as the key in the NSsites
          dictionary when storing the parsed result for the single-model case.
        - When multi_genes is True and multi_models is False, the function
          expects results["genes"] to be present and to be a mutable,
          indexable container (typically a list). It locates gene section
          headers using the regex r"Gene\s+([0-9]+)\s+.+" and parses each
          gene-specific block with parse_model, assigning the returned
          per-gene model_results into results["genes"][gene_number - 1].
        - When multi_models is True, the function scans for model header
          lines matching r"Model (\d+):\s+(.+)", accumulates the lines for
          each model until the next model header, then calls parse_model on
          each model block and stores the result into an internal ns_sites
          dict keyed by the integer model number parsed from the header.
        - The module-level helper function parse_model must be available in
          the same parsing module; parse_nssites delegates the actual
          parsing of a model's text block to parse_model(lines_slice,
          model_results) and expects it to return a dict-like model_results.
          Any exceptions raised by parse_model (for example if the text
          structure is unexpected) are propagated to the caller.
        - After parsing, the function adds an "NSsites" entry to the results
          dictionary only when meaningful data were collected. It applies a
          special-case rule to avoid recording an empty Model 0 ("M0")
          result that may be present by default: if exactly one model was
          found and its index is 0, the function will only set
          results["NSsites"] if that model result is non-empty (or contains
          more than one key).
        - The function mutates the supplied results dict in-place and also
          returns it for convenience. Callers should treat results as
          modified after this function returns.
        - Possible failure modes include KeyError or IndexError when
          multi_genes is True but results["genes"] is missing or has
          insufficient length; AssertionError can be raised if internal
          state assumptions are violated (for example if a gene section is
          detected but no model_results object is present at the time of a
          boundary); and any exceptions raised by parse_model or by the
          regular-expression matching code will propagate to the caller.
        - The function does not perform file I/O itself; it operates on the
          provided list of lines and relies on the caller to supply that
          content (for example by reading a codeml output file elsewhere in
          the parsing pipeline).
    
    Returns:
        dict: The same results dictionary supplied as the results argument,
        potentially mutated to include an "NSsites" key mapping integer
        NSsites model indices (PAML model numbers) to parsed per-model
        result dictionaries, and/or updated per-gene entries in
        results["genes"] when multi_genes is True. If no NSsites model
        content is found, the returned dict may be unmodified.
    """
    from Bio.Phylo.PAML._parse_codeml import parse_nssites
    return parse_nssites(lines, results, multi_models, multi_genes)


################################################################################
# Source: Bio.Phylo.PAML._parse_codeml.parse_pairwise
# File: Bio/Phylo/PAML/_parse_codeml.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_Phylo_PAML__parse_codeml_parse_pairwise(lines: list, results: dict):
    """Parse pairwise comparison blocks from PAML/codeml output lines and update a
    results dictionary with extracted pairwise metrics.
    
    This function is used by Bio.Phylo.PAML._parse_codeml to interpret the text
    output produced by PAML/yn00 or codeml pairwise runs. It scans the provided
    sequence of output lines for pair labels (for example, lines matching the
    pattern "2 (Pan_troglo) ... 1 (Homo_sapie)"), single-value log-likelihood
    lines (lnL), and six-value distance/selection metric lines (t, S, N, omega,
    dN, dS). When found, these values are converted to Python float and stored in
    a nested mapping under the "pairwise" key of the provided results dictionary.
    This enables downstream Biopython code to programmatically access per-pair
    metrics produced by PAML for evolutionary and comparative analyses.
    
    Args:
        lines (list): A list containing the lines of text to parse (typically the
            stdout or saved output of a PAML/yn00 or codeml pairwise run). Each
            element is expected to be a single line of the program's text output.
            The function iterates over these lines to find pair label lines and
            numeric-result lines. The list itself is not modified. If the lines do
            not follow the expected codeml/yn00 formatting (for example, missing
            the lnL single-float line before the six-value metric line), some
            entries may not be created and a KeyError may occur when the code
            attempts to update a non-existent pair entry.
        results (dict): A dictionary (usually created earlier while parsing the
            same codeml output) that will be updated in place with a new key
            "pairwise" when any pairwise comparisons are found. The function
            mutates this dictionary and also returns it. Typical usage in the
            parsing pipeline is to pass a shared results dict that accumulates
            multiple parsing stages (alignments, trees, parameter estimates, etc.).
    
    Behavior and side effects:
        The function looks for three kinds of relevant lines in the provided lines:
        - A pair label line matching the regular expression of the form
          "<int> (<seq1>) ... <int> (<seq2>)" which sets the current sequence pair
          (seq1 and seq2). When such a line is found, the function ensures entries
          for those sequence names exist in an internal nested mapping.
        - A line containing exactly one floating-point number (after extraction by
          the module's float-finding regex) is interpreted as the log-likelihood
          ("lnL") for the current pair and stored as a Python float under the key
          "lnL".
        - A line containing exactly six floating-point numbers is interpreted as
          the metrics (in this order): t, S, N, omega, dN, dS. These six values are
          converted to float and stored under the corresponding keys ("t", "S",
          "N", "omega", "dN", "dS") for the current pair.
        When a pair has been populated, the reciprocal mapping is created so that
        results["pairwise"][seq1][seq2] and results["pairwise"][seq2][seq1] refer
        to the same dictionary object (they are aliased to the same dict). As a
        consequence, mutating the stored metrics via one direction (seq1->seq2)
        will be visible via the other direction (seq2->seq1).
    
    Defaults and failure modes:
        The function does not create the "pairwise" key on the results dict unless
        at least one valid pairwise block is parsed; if no pairwise content is
        found, the input results dict is returned unchanged. The parser expects the
        lnL single-value line to appear before the six-value metrics line for a
        given pair as in typical codeml output; if the six-value line appears
        before any lnL has been stored for that pair, a KeyError will occur when
        attempting to update a non-existent inner mapping. If the numeric extraction
        regex finds tokens that cannot be converted to float (unexpected input),
        a ValueError may be raised from float() conversion. Lines that yield a
        number of floats other than 1 or 6 are ignored with respect to creating or
        updating pairwise metric entries.
    
    Structure of stored results:
        When populated, results["pairwise"] is a nested dictionary mapping sequence
        names to dictionaries of sequence names to metric dictionaries, for
        example: results["pairwise"][seq1][seq2] == {"lnL": float, "t": float,
        "S": float, "N": float, "omega": float, "dN": float, "dS": float}. Some
        keys may be absent if the expected lines were missing or out of order in
        the input.
    
    Returns:
        dict: The same results dictionary passed in (results), potentially updated
        in place with a "pairwise" key containing the nested mapping of pairwise
        comparison metrics. If no pairwise data were found, the returned dict is
        unchanged.
    """
    from Bio.Phylo.PAML._parse_codeml import parse_pairwise
    return parse_pairwise(lines, results)


################################################################################
# Source: Bio.Phylo.PAML._parse_codeml.parse_siteclass_omegas
# File: Bio/Phylo/PAML/_parse_codeml.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_Phylo_PAML__parse_codeml_parse_siteclass_omegas(line: str, site_classes: list):
    """Parse and assign omega (dN/dS) estimates for PAML codeml site classes from a single formatted line.
    
    This function is used by Bio.Phylo.PAML._parse_codeml to extract omega estimates (the dN/dS ratio used in molecular evolution and selection analyses) produced by PAML's codeml for models that report multiple site classes. The function expects a single line of text produced by codeml that contains fixed-width floating point columns (the typical "w:" output). It finds numeric substrings matching the fixed-width format used by codeml output and assigns each found value to the corresponding site class dictionary in the provided site_classes list under the key "omega". This is important for downstream Biopython code that collects per-site-class selection parameter estimates for phylogenetic and molecular evolution analyses.
    
    Args:
        line (str): A single line of text from codeml output expected to contain omega estimates in fixed-width columns. In practice this is typically the "w:" line emitted by codeml when reporting omega (dN/dS) for multiple site classes. The function searches this string with the regular expression r"\d{1,3}\.\d{5}" to find numbers formatted with 1–3 digits before the decimal and exactly 5 digits after the decimal (the format observed in many codeml table outputs).
        site_classes (list): A list of dictionaries representing the model's site classes as parsed elsewhere in Bio.Phylo.PAML._parse_codeml. Each element is expected to be a dict (for example with keys like class id or proportions) that may be updated in-place by adding or replacing the "omega" key. The list represents the order of site classes as reported by codeml; the function assigns the nth parsed omega string to site_classes[n].
    
    Returns:
        list or None: If at least one omega-like numeric substring is found and site_classes is a non-empty list, returns the same site_classes list after updating each matched element's "omega" key with the corresponding numeric substring (note: the values stored are the string matches from the line, not converted float objects). If site_classes is empty or no matching numbers are found in line, the function returns None and performs no modifications.
    
    Behavior, side effects, defaults, and failure modes:
        - The function mutates the dictionaries inside the provided site_classes list in-place by setting site_classes[n]["omega"] = line_floats[n] for each parsed value index n.
        - The parsed omega values are left as strings matching the pattern r"\d{1,3}\.\d{5}" (e.g., "109.87121"), consistent with the fixed-width formatting described in codeml output; no numeric conversion is performed by this function.
        - The parsing strategy is specifically chosen because codeml prints fixed-width columns (effectively 9 characters per column), which can cause adjacent numbers to run together; the regular expression accounts for the fixed decimal width observed in such tables.
        - If fewer numeric substrings are found than the length of site_classes, only the first parsed site_classes entries are updated and the remaining entries are left unchanged.
        - If more numeric substrings are found than there are entries in site_classes, an IndexError may be raised when attempting to assign beyond the end of site_classes; callers should ensure the provided site_classes list matches the expected number of classes for the model being parsed.
        - The function does not perform validation beyond the regex match; if codeml output formatting changes (different decimal precision, scientific notation, or additional characters), this function may fail to find the intended values or may parse incorrect substrings. In that case, upstream code or callers should be adjusted to handle the new format or convert the parsed strings to numeric types as needed.
    """
    from Bio.Phylo.PAML._parse_codeml import parse_siteclass_omegas
    return parse_siteclass_omegas(line, site_classes)


################################################################################
# Source: Bio.Phylo.PAML._parse_codeml.parse_siteclass_proportions
# File: Bio/Phylo/PAML/_parse_codeml.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_Phylo_PAML__parse_codeml_parse_siteclass_proportions(line_floats: list):
    """Bio.Phylo.PAML._parse_codeml.parse_siteclass_proportions: Find and return the proportion of the alignment assigned to each site class reported by a PAML codeml model.
    
    This function is used in the PAML parser within Bio.Phylo to convert a sequence (list) of numeric proportions parsed from a codeml output line into a structured mapping that downstream code (for example, code computing sitewise likelihoods or summarising selection categories) can consume. In phylogenetic analyses with codeml, models may define multiple site classes (e.g. different dN/dS categories); each element of the input corresponds to the fraction of alignment sites assigned to that class. The function performs a direct conversion without validation or transformation of the numeric values.
    
    Args:
        line_floats (list): A list containing the proportions for each site class as parsed from a codeml output line. Each element is expected to represent the fraction of the alignment assigned to a specific site class (typically a Python float), and the order of elements corresponds to site class identifiers starting at 0. This parameter must be supplied as a list; providing an empty list indicates that no site-class proportions were reported.
    
    Returns:
        dict: A dictionary mapping site class indices (int, starting at 0) to dictionaries with a single key "proportion" whose value is the corresponding element from line_floats (typically a float). Example return for line_floats = [0.7, 0.3] is {0: {"proportion": 0.7}, 1: {"proportion": 0.3}}. If line_floats is empty or evaluates to False, an empty dict is returned.
    
    Behavior and side effects:
        The function performs no I/O and has no side effects; it only constructs and returns the mapping described above. It does not validate that the proportions sum to 1, nor does it coerce or check element numeric types beyond storing the provided values. If non-numeric values are present in line_floats they will be returned unchanged and may cause errors in downstream code that expects numeric proportions.
    
    Failure modes and errors:
        The function itself does not raise errors for malformed numeric values; however, downstream consumers expecting numeric proportions may fail if elements are not numeric or if the length of line_floats does not match the expected number of site classes for the codeml model being analysed. It is the caller's responsibility to ensure that line_floats contains the expected number and type of entries parsed from codeml output.
    """
    from Bio.Phylo.PAML._parse_codeml import parse_siteclass_proportions
    return parse_siteclass_proportions(line_floats)


################################################################################
# Source: Bio.Phylo.PAML._parse_yn00.parse_ng86
# File: Bio/Phylo/PAML/_parse_yn00.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_Phylo_PAML__parse_yn00_parse_ng86(lines: list, results: dict):
    """Parse the Nei & Gojobori (1986) section of yn00 results into Python data
    structures for downstream phylogenetic analyses.
    
    This function is used by the Bio.Phylo.PAML.yn00 parser to read the NG86
    (lower-triangular) pairwise comparison block produced by the PAML yn00
    program (Nei & Gojobori, 1986 method). The NG86 block lists sequence
    names (taxa) as row labels and, for each row, a sequence of fields
    representing pairwise statistics for previously-listed sequences. Each
    pairwise entry is recorded in the output as three floating-point values:
    omega (w), dN and dS (often shown in the form "w (dN dS)"). Typical
    input lines look like:
      seq_name 0.0000 (0.0000 0.0207) 0.0000 (0.0000 0.0421)
    This parser is robust to common pathologies of yn00 output: taxon names
    may be truncated to 30 characters, names may abut numeric fields without
    separators, and there may be citations or comment lines interleaved with
    data. The implementation uses regular expressions to detect data rows,
    extract the taxon name and locate floating-point tokens (including
    negative values when present) that form triples of (omega, dN, dS).
    
    Args:
        lines (list): A list of strings, each string being one line from the
            yn00 NG86 section. Each element should represent one textual line
            as produced by yn00; the order of lines determines the order of
            taxa and therefore the mapping of pairwise columns to previously
            seen taxa. This list is typically extracted from the full yn00
            output by splitting on newlines and selecting the NG86 block.
        results (dict): A dict to be populated with parsed NG86 pairwise
            statistics. This dict is mutated in place: for each parsed taxon
            name seq_i and previously-seen taxon seq_j, two symmetric
            entries are created:
                results[seq_i][seq_j] = {"NG86": {"omega": float, "dN": float, "dS": float}}
                results[seq_j][seq_i] = {"NG86": {"omega": float, "dN": float, "dS": float}}
            If called with an empty dict, the function will build a complete
            pairwise mapping for the NG86 block. If results already contains
            keys for some taxa, those keys may be used or overwritten as the
            function assigns results[seq_name] = {} for each parsed row.
            This object is typically the same results dict used by other
            PAML parsers within Bio.Phylo.PAML to aggregate different
            summary blocks.
    
    Behavior and side effects:
        - The function iterates over lines, using a regular expression to
          detect lines that begin with a taxon name followed by zero or more
          numeric fields (including cases where the taxon name runs into the
          numeric portion). Non-matching lines are skipped silently (useful
          for interleaved comments or citations in yn00 output).
        - For each matching line, all floating-point-like tokens are
          extracted and grouped into consecutive triples interpreted as
          (omega, dN, dS). The taxon name is appended, in encounter order, to
          an internal sequences list. The function then records the NG86
          triple for each previously-seen taxon corresponding to the lower
          triangular layout of the input block (so the k-th triple on row i
          corresponds to the k-th previously parsed taxon).
        - The results dict is updated in place as described above; therefore
          callers will observe the populated results immediately after the
          call, even if an exception occurs part-way through parsing.
        - The function preserves numeric values as Python floats.
    
    Failure modes and cautions:
        - If the numeric tokens on a data row are not present in multiples
          of three, the inner loop may attempt to access out-of-range
          indices and raise IndexError. This indicates malformed NG86 input
          (missing dN or dS values).
        - If the sequences list does not contain an expected previously-seen
          taxon (for example because lines were supplied out-of-order or
          earlier rows were skipped), indexing sequences[i // 3] may raise
          IndexError. Ensure that lines are provided in the original order
          printed by yn00 (each row corresponds to one taxon, and columns
          reference earlier rows).
        - Passing a non-list for lines or non-dict for results will raise
          standard Python TypeError or attribute errors when the code attempts
          list/dict operations.
        - The function does not validate biological plausibility of numeric
          values (e.g., non-negative dS), it only parses textual tokens.
        - Because results is mutated in place, partially parsed state may
          exist if an exception occurs; callers who require all-or-nothing
          behavior should pass a copy of results or manage transactions at a
          higher level.
    
    Returns:
        tuple: A 2-tuple (results, sequences) where:
            results (dict): The same dict object passed in, now populated
                with symmetric NG86 pairwise mappings. Each pairwise mapping
                is accessible as results[seq_i][seq_j]["NG86"] and contains
                a dict with keys "omega", "dN", and "dS" whose values are
                Python floats. This structure is suitable for downstream
                Biopython analyses and for exporting or summarizing pairwise
                selection statistics.
            sequences (list): A list of sequence (taxon) names in the order
                they were encountered while parsing the NG86 block. This
                ordering corresponds to the row order in the lower-triangular
                matrix and is used to associate column triples with the
                appropriate previously-listed taxa.
    """
    from Bio.Phylo.PAML._parse_yn00 import parse_ng86
    return parse_ng86(lines, results)


################################################################################
# Source: Bio.Phylo.PAML._parse_yn00.parse_others
# File: Bio/Phylo/PAML/_parse_yn00.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_Phylo_PAML__parse_yn00_parse_others(lines: list, results: dict, sequences: dict):
    """Parse the "other methods" section of PAML yn00 pairwise output and
    populate the provided results mapping with per-pair statistics for the
    LWL85, LWL85m and LPB93 methods.
    
    This function reads a sequence of text lines produced by PAML's yn00
    program describing pairwise species comparisons. It looks for comparison
    headers of the form "N (Name1) vs. M (Name2)" to establish the two
    sequence names for a block of following method result lines. For each
    method line that contains "dS =" it extracts contiguous "key = value"
    fields (for example "dS =  0.0227", "dN =  0.0000", "w = 0.0000") using
    the same regular-expression logic as in the source: r"[dSNwrho]{1,3} =.{7,8}?". Numeric values are converted to Python floats; values that cannot be parsed as floats (for example platform-specific NaN representations like "-1.#IND") are recorded as None. Parsed statistics for a method are stored as a dict mapping statistic name strings (e.g. "dS", "dN", "w", "S", "N", "rho") to float or None.
    
    The function is intended for use in computational molecular biology and
    bioinformatics workflows that integrate Biopython with PAML's yn00
    output. The populated results structure enables downstream code that
    performs comparative analyses of synonymous (dS) and nonsynonymous (dN)
    rates and related statistics across pairwise species comparisons. The
    function handles the special case of non-numeric NaN-like fields by
    mapping them to None so that further numerical processing in Python can
    explicitly test for missing values.
    
    Args:
        lines (list): A list of text lines (str) from a PAML yn00 output file.
            Each element is expected to be a single line of the output. The
            function scans these lines sequentially looking for comparison
            headers like "2 (Pan_troglo) vs. 1 (Homo_sapie)" to identify the
            two sequence names, and for method result lines that include the
            substring "dS =" to extract statistics. If no comparison header
            is seen before method lines, those method lines are ignored.
        results (dict): A nested dictionary that will be populated with the
            parsed statistics. It is expected to be keyed first by the first
            sequence name (str) and then by the second sequence name (str),
            with each leaf value itself a dict of method names to their
            statistics dicts. For example, after parsing, results[name1][name2]["LWL85"]
            will be a dict of statistics for the LWL85 method. This mapping
            is modified in-place: the function assigns the same statistics
            dict to both results[name1][name2][method] and
            results[name2][name1][method] to record the pairwise result in
            both directions. If the expected name keys are not present in
            results, a KeyError will be raised when attempting to assign.
        sequences (dict): A dictionary of sequence information (for example,
            a mapping from sequence name to sequence record or metadata).
            The current implementation does not read or modify this mapping;
            it is accepted for API compatibility with other parsers and for
            potential future use where sequence records might be needed to
            validate names or enrich parsed statistics. Because it is unused,
            passing an arbitrary dict is allowed, but callers should typically
            pass the same sequence mapping used elsewhere in the application.
    
    Returns:
        dict: The same results dictionary passed in (results), returned for
        convenience after in-place modification. The returned mapping will
        contain new entries (or updated entries) for keys corresponding to
        the sequence name pairs encountered in lines, with method names
        "LWL85", "LWL85m", and "LPB93" mapping to dicts of parsed statistics.
        If no relevant method lines are found, the results dict is returned
        unmodified.
    
    Behavior and failure modes:
        - The function uses a regular expression to find comparison header
          lines and another to extract fixed-width "key = value" pairs from
          method lines. It assumes the PAML yn00 output format where method
          statistics appear on a single line after a colon (e.g.
          "LWL85:  dS =  0.0227 dN =  0.0000 w = 0.0000 ...").
        - Numeric parsing uses float(value). If float() raises ValueError
          (for example for platform-specific NaN text like "-1.#IND" or other
          non-numeric tokens), the statistic value is stored as None to
          indicate a missing or unrepresentable numeric value.
        - The function assigns the parsed statistics dict to both
          results[seq1][seq2][method] and results[seq2][seq1][method] so the
          pairwise result is available in both directions. If results does
          not already contain nested dicts for these sequence name keys,
          attempting to assign will raise a KeyError or TypeError.
        - If multiple method lines for the same pair and method are present,
          later lines will overwrite earlier entries for that pair/method.
        - If input lines deviate substantially from PAML yn00 formatting,
          the function may fail to detect sequence name headers or to parse
          statistics correctly; in such cases it will either make no changes
          or raise standard parsing exceptions (e.g. KeyError, ValueError)
          depending on how the input deviates.
    """
    from Bio.Phylo.PAML._parse_yn00 import parse_others
    return parse_others(lines, results, sequences)


################################################################################
# Source: Bio.Phylo.PAML._parse_yn00.parse_yn00
# File: Bio/Phylo/PAML/_parse_yn00.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_Phylo_PAML__parse_yn00_parse_yn00(lines: list, results: dict, sequences: list):
    """Parse the Yang & Nielsen (2000) part of PAML yn00 output and insert
    pairwise statistics into an existing results dictionary.
    
    This function is part of Bio.Phylo.PAML._parse_yn00 and is used to parse
    the section of PAML's yn00 program output that lists pairwise comparisons
    between sequences (Yang & Nielsen 2000). Each row in that section is
    expected to start with two sequence indices (1-based) followed by a set
    of floating point values corresponding to the quantities S, N, t,
    kappa, omega, dN, dN SE, dS, and dS SE. The function converts the numeric
    strings to floats, maps the 1-based indices to sequence names provided
    via the sequences list, constructs a dictionary of the parsed values
    under the key "YN00" and stores it in the provided results mapping for
    both orderings of the sequence pair (results[name1][name2] and
    results[name2][name1]). Typical use is within a PAML output parsing
    pipeline where an already-initialized results structure must be
    populated with yn00-derived pairwise evolutionary statistics for downstream
    analysis (for example, computing dN/dS summaries or annotating trees in
    Bio.Phylo).
    
    Args:
        lines (list): A list of strings, each string being one line from the
            yn00 program output. The function scans each line for a leading
            pair of integers and for floating point values using regular
            expressions. It expects the nine floats for a table row to be
            present on the same line as the leading indices; if they are not,
            the function may raise an IndexError or fail to parse that row.
        results (dict): A nested dictionary mapping sequence name to another
            mapping of sequence name to per-comparison dictionaries. Concretely,
            results is expected to already contain keys for every sequence
            name in sequences and for each pair results[name1][name2] must be
            an addressable dictionary (for example, created earlier by the
            overall parser). This function will add or overwrite the "YN00"
            key in results[name1][name2] and results[name2][name1] with a
            dictionary containing the parsed numeric fields. Because the
            function writes into this structure, the caller should provide a
            mutable dict prepared to receive these entries.
        sequences (list): A list of sequence names (strings) ordered so that
            the sequence number labels used in the yn00 output (which are
            1-based integers) correspond to indices in this list (i.e.,
            sequences[0] is the name for sequence number 1). The function
            uses the 1-based indices found in the output lines to look up
            sequence names in this list and will raise an IndexError if an
            index from the file does not have a corresponding entry in
            sequences.
    
    Returns:
        dict: The same results dictionary object passed in (results), now
        modified in place with added "YN00" entries for any pairwise rows
        successfully parsed. The value stored under each pair is a dict with
        the following keys and float values corresponding to the columns in
        the yn00 table: "S", "N", "t", "kappa", "omega", "dN", "dN SE",
        "dS", and "dS SE". If no parsable pairwise rows are found the
        returned dict will be unchanged.
    
    Behavior, side effects, and failure modes:
        - The parser finds floating point values using the regular expression
          r"-*\d+\.\d+" and finds the leading pair of sequence indices using
          r"\s+(\d+)\s+(\d+)". Only lines where the index regex matches are
          processed as table rows.
        - The function assumes that nine floating point numbers follow the
          two indices on the same line in the order S, N, t, kappa, omega,
          dN, dN SE, dS, dS SE. If fewer than nine floats are present on the
          matched line, accessing the expected positions in the float list
          will raise IndexError.
        - The function mutates the provided results dict in place by setting
          results[seq_name1][seq_name2]["YN00"] and
          results[seq_name2][seq_name1]["YN00"]. If the required nested
          dictionaries are not present, a KeyError may be raised.
        - If a sequence index parsed from a line does not map to an entry in
          sequences (for example, a 1-based index greater than len(sequences)),
          an IndexError will be raised.
        - The parser does not perform unit or range checking on numeric
          values beyond converting strings to float; malformed numeric
          strings that do not match the float regular expression will be
          ignored and may cause IndexError later if insufficient numbers are
          collected.
        - Because the function uses per-line matching, it may not correctly
          parse table rows where parts of a multi-column value are split
          across multiple lines in the raw output; callers should pre-process
          such cases or ensure the lines list preserves the original table
          formatting.
    
    Example practical significance:
        - In a workflow parsing PAML yn00 output for pairwise dN/dS analyses,
          this function extracts per-pair statistics (S, N, t, kappa, omega,
          dN, dN SE, dS, dS SE) and attaches them to the results structure
          keyed by sequence names. Downstream modules (for example, tree
          annotation routines in Bio.Phylo or summary reporting tools) can
          use these populated "YN00" entries to compute evolutionary rates,
          generate figures, or filter sequence pairs for further analysis.
    """
    from Bio.Phylo.PAML._parse_yn00 import parse_yn00
    return parse_yn00(lines, results, sequences)


################################################################################
# Source: Bio.Phylo.PAML.baseml.read
# File: Bio/Phylo/PAML/baseml.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_Phylo_PAML_baseml_read(results_file: str):
    """Bio.Phylo.PAML.baseml.read parses a BASEML results file produced by the PAML baseml program and returns a dictionary of parsed results suitable for downstream phylogenetic analysis and programmatic inspection.
    
    Args:
        results_file (str): Path to a BASEML results text file produced by the PAML baseml program. In the context of Biopython (a library for computational molecular biology), this string should point to an existing plain-text output file generated by running baseml (for example via Baseml.run()). The function opens and reads this file into memory (calls handle.readlines()), so the caller should provide a filesystem path accessible to the running process and be aware that very large result files will increase memory usage.
    
    Returns:
        dict: A dictionary containing parsed information extracted from the BASEML output. The returned mapping uses string keys (for example, the parser stores the PAML version string under the "version" key) and values that represent the parsed parameters, summary statistics, and model results emitted by baseml (such as estimated substitution model parameters, likelihood values, and any other fields the internal _parse_baseml parser extracts). This dictionary is intended for programmatic use in downstream phylogenetic workflows within Biopython, for example to inspect model parameters or to convert results for further analysis.
    
    Raises:
        FileNotFoundError: If the path given by results_file does not exist. This indicates the caller supplied an incorrect filename or the baseml run did not produce output at the given location.
        ValueError: If the results file exists but is empty. The empty file likely indicates baseml did not complete successfully; the error message suggests re-running Baseml.run() with verbose=True to capture any diagnostic output.
        ValueError: If the results file is present and non-empty but does not contain the expected structure (for example, the parser fails to find a "version" field). In this case the output is treated as invalid and no dictionary is returned.
    
    Behavior and side effects:
        The function performs purely read-only file I/O: it opens the specified results_file, reads all lines into memory, and closes the file. It delegates parsing to internal helpers (_parse_baseml.parse_basics and _parse_baseml.parse_parameters) which populate the returned dictionary. The function does not invoke baseml itself; it only reads and parses output files previously produced by the baseml program. Because the implementation reads the entire file with readlines(), callers should avoid passing extremely large files unless sufficient memory is available.
    
    Failure modes and practical guidance:
        If you receive FileNotFoundError, verify the filesystem path and that baseml wrote output as expected. If you receive ValueError for an empty file, re-run the baseml analysis with verbose logging enabled (Baseml.run(verbose=True)) to diagnose failures. If ValueError indicates an invalid results file (missing "version" or other required tokens), the output format may have changed or the file may be corrupted; compare the file against a known-good baseml output or re-run baseml.
    """
    from Bio.Phylo.PAML.baseml import read
    return read(results_file)


################################################################################
# Source: Bio.Phylo.PAML.chi2.cdf_chi2
# File: Bio/Phylo/PAML/chi2.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_Phylo_PAML_chi2_cdf_chi2(df: int, stat: float):
    """Compute the upper-tail cumulative distribution function (p-value) of the chi-square
    distribution for a given degrees of freedom and observed test statistic.
    
    This function is part of Bio.Phylo.PAML.chi2 and is used in phylogenetics and
    molecular-evolution analyses (for example when processing PAML output) to convert
    an observed chi-square test statistic into a p-value. Practically, this is most
    commonly applied to likelihood ratio test (LRT) statistics (often twice the
    difference in log-likelihoods) to assess whether a more complex model provides a
    significantly better fit to sequence or tree data than a simpler model. The
    implementation computes the upper-tail probability using the relation
    prob = 1 - I_x(alpha), where x = stat / 2 and alpha = df / 2, and I_x is the
    regularized lower incomplete gamma function (implemented here as _incomplete_gamma).
    
    Args:
        df (int): Degrees of freedom for the chi-square distribution. In the
            context of model comparison in PAML/Bio.Phylo, df is the number of
            additional free parameters in the more complex model relative to the
            simpler model. Must be an integer greater than or equal to 1. If df < 1,
            the function raises a ValueError with the message "df must be at least 1".
        stat (float): Observed chi-square test statistic whose upper-tail
            probability is desired. In phylogenetic applications this commonly is the
            LRT statistic (e.g. 2 * (logL_complex - logL_simple)). Must be a
            non-negative floating-point value. If stat < 0, the function raises a
            ValueError with the message "The test statistic must be positive".
    
    Returns:
        float: The upper-tail probability (p-value) from the chi-square
        distribution corresponding to observing a value at least as large as
        stat given df degrees of freedom. The returned value is a floating-point
        probability in the closed interval [0.0, 1.0]. Numerically, it is computed
        as 1 - _incomplete_gamma(stat/2, df/2).
    
    Raises:
        ValueError: If df < 1, a ValueError is raised with the exact message
            "df must be at least 1".
        ValueError: If stat < 0, a ValueError is raised with the exact message
            "The test statistic must be positive".
    
    Behavior and numerical notes:
        - There are no side effects: the function does not modify global state or
          perform I/O; it only computes and returns a floating-point probability.
        - For very large degrees of freedom or very large test statistics, the
          result may underflow to 0.0 or round to 1.0 due to finite machine
          precision; users interpreting extreme p-values in biological analyses
          should be aware of floating-point limitations.
        - The function expects Python native numeric types as specified (int for df,
          float for stat) and does not perform automatic type coercion beyond what
          Python normally allows; passing inappropriate types will result in a
          TypeError or other standard Python error prior to the explicit ValueError
          checks.
    """
    from Bio.Phylo.PAML.chi2 import cdf_chi2
    return cdf_chi2(df, stat)


################################################################################
# Source: Bio.Phylo.PAML.codeml.read
# File: Bio/Phylo/PAML/codeml.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_Phylo_PAML_codeml_read(results_file: str):
    """Bio.Phylo.PAML.codeml.read parses a CODEML results file produced by the PAML codeml program and returns the parsed results as a Python dictionary useful in phylogenetics and molecular evolution analyses.
    
    This function opens and reads the given results file (read-only), parses its contents using the module's internal parsing routines, and assembles a dictionary of results covering basic run information, site-model (nonsynonymous/synonymous site) results, pairwise comparisons, and distance estimates. It is intended to be used after running codeml (for example via Codeml.run()) to programmatically extract the results for downstream processing in Biopython's Bio.Phylo workflows or other computational molecular biology pipelines. The function reads the entire file into memory (using readlines()) before parsing, so memory usage grows with file size. It does not modify the filesystem or the input file.
    
    Args:
        results_file (str): Path to the CODEML results file produced by PAML's codeml program. This string is interpreted as a filesystem path; the function will attempt to open the file using Python's default text encoding. The file is expected to contain a standard codeml output; if the file is empty or not a codeml output, parsing will fail. Typical usage is to pass the path returned by running codeml (for example Codeml.run()). The argument is required and has no default.
    
    Returns:
        dict: A dictionary containing the parsed CODEML results. The dictionary is populated by internal parsing stages: parse_basics (basic run metadata and flags), parse_nssites (site-model / nonsynonymous–synonymous site results, possibly multiple models/genes), parse_pairwise (pairwise comparison results), and parse_distances (distance matrices). The exact keys and nested structure reflect these parsed sections and are intended to be consumed by downstream Biopython code or user analysis scripts. If parsing succeeds, this dictionary will be non-empty.
    
    Raises:
        FileNotFoundError: If the path given by results_file does not exist on the filesystem. This matches the behavior when a codeml run did not produce an output file or the path is incorrect.
        ValueError: If the file exists but is empty (suggesting codeml did not complete successfully). In this case the function raises ValueError with a message hinting to run Codeml.run() with verbose=True to diagnose the codeml run. ValueError is also raised if the file could not be interpreted as a valid codeml results file (for example if the format differs from PAML codeml output), indicating invalid or unsupported content.
    
    Side effects and notes:
        The function opens the file for reading and reads all lines into memory (no file modifications). It delegates parsing to internal helpers _parse_codeml.parse_basics, _parse_codeml.parse_nssites, _parse_codeml.parse_pairwise, and _parse_codeml.parse_distances; any limitations or assumptions of those parsers apply. Do not pass non-codeml output files; doing so will typically result in ValueError.
    """
    from Bio.Phylo.PAML.codeml import read
    return read(results_file)


################################################################################
# Source: Bio.Phylo.PAML.yn00.read
# File: Bio/Phylo/PAML/yn00.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_Phylo_PAML_yn00_read(results_file: str):
    """Parse a YN00 results file produced by the PAML yn00 program and return the parsed results as a Python dictionary.
    
    Args:
        results_file (str): Path to the YN00 results file on disk. This is the filename produced by running the PAML Yn00 program (for example via Bio.Phylo.PAML.yn00.Yn00.run()). The function will open and read this file using the default system text encoding, so the caller must ensure the file is accessible at this path and encoded in a compatible text encoding. The file is expected to contain the standard YN00 sections identified in the output by the headings "(A) Nei-Gojobori (1986) method", "(B) Yang & Nielsen (2000) method", and "(C) LWL85, LPB93 & LWLm methods". The function reads the entire file into memory (calls readlines()), so very large files will consume memory proportional to their size.
    
    Returns:
        dict: A dictionary containing the parsed YN00 results aggregated from the file. The returned mapping is produced by internal parsers called by this function (notably _parse_yn00.parse_ng86, _parse_yn00.parse_yn00 and _parse_yn00.parse_others) and therefore contains the numeric estimates and associated metadata that YN00 reports for the different methods (for example pairwise estimates such as dN, dS, and dN/dS and sequence identifiers). Keys and exact structure follow the conventions used by the module's internal parsers: method-specific results and sequence pair identifiers are used to organize the data so downstream code can programmatically access per-method and per-pair estimates for evolutionary analyses (e.g., comparing synonymous and nonsynonymous substitution rates).
    
    Behavior and side effects:
        This function performs only read-only operations on the filesystem: it checks for the existence of the specified file and opens it for reading. It loads the file content entirely into memory (via readlines()) and then scans the lines to locate YN00 section headers. It dispatches subranges of the file to internal parsing helpers to build up the results dictionary. The function does not modify the input file. Because it relies on specific textual headers emitted by the YN00 program, it expects the file to be a valid YN00 output; files from other programs or truncated outputs may not parse correctly.
    
    Failure modes and exceptions:
        If the path given by results_file does not exist, FileNotFoundError is raised with the message "Results file does not exist." If the file exists but contains no lines (for example if YN00 failed to run or produced an empty output), ValueError is raised advising that YN00 may not have exited successfully (the original message suggests running Yn00.run() with verbose=True to diagnose). If the file is parsed but no valid results sections are found (the internal parsers do not populate the results dictionary), ValueError is raised indicating an invalid results file. Other exceptions may propagate from the underlying file I/O (for example permission errors) or from the internal parsing functions if they encounter malformed content.
    
    Domain significance:
        This parser is intended for users analyzing pairwise codon substitution statistics using PAML's yn00 program within the Bio.Phylo.PAML package. The returned dictionary provides programmatically accessible estimates (such as dN, dS, and their ratio) and identifiers required for comparative evolutionary analyses and downstream workflows in molecular evolution and phylogenetics.
    """
    from Bio.Phylo.PAML.yn00 import read
    return read(results_file)


################################################################################
# Source: Bio.Phylo._cdao_owl.resolve_uri
# File: Bio/Phylo/_cdao_owl.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_Phylo__cdao_owl_resolve_uri(
    s: str,
    namespaces: dict = {'cdao': 'http://purl.obolibrary.org/obo/cdao.owl#', 'obo': 'http://purl.obolibrary.org/obo/'},
    cdao_to_obo: bool = True,
    xml_style: bool = False
):
    """Convert a prefixed URI string to a full URI string for use by the Bio.Phylo CDAO/OWL parser. This function is used by the CDAO/OWL parsing code in Bio.Phylo to translate short, prefixed identifiers encountered in RDF/OWL input (for example during parsing with rdflib) into fully qualified URIs or XML QName style names so downstream Biopython code can match and process ontology terms consistently. Optionally, when a CDAO term name is supplied, it will be translated to the corresponding OBO numeric identifier using the module-level cdao_elements mapping.
    
    Args:
        s (str): The input string to resolve. This is expected to be a prefixed identifier such as "cdao:SomeName" or "obo:ID", or any string containing occurrences of known prefixes followed by a colon. The function performs simple string replacements and may return the same string unchanged if no known prefixes are present.
        namespaces (dict): Mapping of prefix to namespace URI to use for replacement. The default mapping is {'cdao': 'http://purl.obolibrary.org/obo/cdao.owl#', 'obo': 'http://purl.obolibrary.org/obo/'}. Each occurrence of "prefix:" in s is replaced by the corresponding namespaces[prefix] (or by "{namespaces[prefix]}" when xml_style is True). The mapping must provide the prefixes you expect; insertion order determines the iteration order in which replacements are attempted.
        cdao_to_obo (bool): If True (default), and if s begins with the exact substring "cdao:", perform an additional conversion step by looking up the term name (the part after "cdao:") in the module-level cdao_elements mapping and replacing s with "obo:<mapped-value>" before performing the namespace replacements. This conversion only occurs when s.startswith("cdao:"). If the key is not present in cdao_elements, a KeyError will be raised. Note that the recursive call used to convert "cdao:" to "obo:" does not propagate the xml_style argument and therefore uses the default xml_style=False for that conversion step.
        xml_style (bool): If False (default), each "prefix:" is replaced by the raw namespace URI string from namespaces (resulting in "namespaceURIlocalName"). If True, each "prefix:" is replaced by an ElementTree-style QName prefix "{namespaceURI}" (resulting in "{namespaceURI}localName"). Replacement uses str.replace and affects all occurrences of "prefix:" anywhere in s.
    
    Returns:
        str: The resolved URI string. This is the input s after any optional CDAO→OBO mapping and after all prefix replacements. No validation is performed that the result is a syntactically valid URI; callers should validate or normalize the output if required. Side effects: the function does not modify its inputs in place but may raise KeyError when cdao_to_obo is True and the CDAO term name is not found in the module-level cdao_elements mapping.
    """
    from Bio.Phylo._cdao_owl import resolve_uri
    return resolve_uri(s, namespaces, cdao_to_obo, xml_style)


################################################################################
# Source: Bio.Phylo._io.convert
# File: Bio/Phylo/_io.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_Phylo__io_convert(
    in_file: str,
    in_format: str,
    out_file: str,
    out_format: str,
    parse_args: dict = None,
    **kwargs
):
    """Convert between two phylogenetic tree file formats used in Bio.Phylo.
    
    Args:
        in_file (str): Path to the input tree file. In the Biopython/Bio.Phylo domain this is the filesystem path (string) to a file containing one or more phylogenetic trees encoded in a format readable by Bio.Phylo.parse (for example Newick, Nexus, or PhyloXML). The function calls Bio.Phylo.parse(in_file, in_format, **parse_args) so this path must be acceptable to that parser; if the path does not exist or is unreadable the underlying parse() call will raise an appropriate I/O or parsing exception.
        in_format (str): Name of the input file format. This string selects the parser within Bio.Phylo (for example "newick", "nexus", "phyloxml"); it determines how the bytes/text in in_file are interpreted as phylogenetic trees. Providing an incorrect or unsupported format will cause the parse operation to fail with a format or parsing error.
        out_file (str): Path to the output file to write converted trees to. In the Biopython/Bio.Phylo context this is a filesystem path (string) where the converted representation will be written by Bio.Phylo.write. The function will write (and usually overwrite) this path; ensure you have appropriate file system permissions and that choosing this path will not unintentionally overwrite important data.
        out_format (str): Name of the output file format. This string selects the writer within Bio.Phylo (for example "newick", "nexus", "phyloxml") and determines how the in-memory tree objects parsed from in_file are serialized. An unsupported out_format will cause the write operation to raise an error.
        parse_args (dict): Optional dictionary of keyword arguments forwarded to Bio.Phylo.parse(). If None (the default), an empty dictionary is used. Use this to pass parser-specific options (for example controlling comment handling or branch length parsing) documented by the specific parser implementation. Values must be valid parser keyword arguments; invalid keys or values will cause parse() to raise an error.
        kwargs (dict): Additional keyword arguments forwarded to Bio.Phylo.write(). These control writer-specific behavior (for example formatting options recognized by the chosen out_format). The keys and permitted values are determined by the writer implementation for out_format; supplying unsupported options will raise an error at write time.
    
    Behavior and side effects:
        This function parses trees from in_file using Bio.Phylo.parse and then writes them to out_file using Bio.Phylo.write. If parse_args is None it is treated as an empty dict. The primary side effect is that out_file is created or modified on disk by the write operation; any existing file at out_file may be overwritten. The function does not perform additional validation beyond what the underlying parse and write functions enforce; tree topology, metadata, and annotations are handled according to the semantics of the chosen formats and the parser/writer implementations. Typical use in computational molecular biology is to convert phylogenetic tree files between formats (for example converting Newick files produced by a tree inference program into PhyloXML for richer annotations or visualization tools).
    
    Failure modes:
        Errors raised by Bio.Phylo.parse (for example I/O errors, file-not-found, or parsing/format errors) will propagate to the caller. Errors raised by Bio.Phylo.write (for example unsupported out_format, I/O permission errors, or writer-specific validation errors) will also propagate. Because different formats support different features, some annotations present in the input may be lost or transformed during conversion; this is a semantic limitation of the chosen formats and writer implementations, not of this convenience wrapper.
    
    Returns:
        The return value produced by Bio.Phylo.write when called with the trees parsed from in_file and the provided kwargs. This return value is whatever the writer implementation documents for the chosen out_format (for example some writer implementations may return None while others may return a count or other sentinel). The primary observable effect of calling this function is the creation or modification of out_file on disk.
    """
    from Bio.Phylo._io import convert
    return convert(in_file, in_format, out_file, out_format, parse_args, **kwargs)


################################################################################
# Source: Bio.Phylo._io.write
# File: Bio/Phylo/_io.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_Phylo__io_write(trees: list, file: str, format: str, **kwargs):
    """Write a sequence of phylogenetic trees to the given destination using the writer implementation registered for the requested output format.
    
    Args:
        trees (list): An iterable (typically a list) of phylogenetic tree objects to be written. In the Bio.Phylo domain these are BaseTree.Tree or BaseTree.Clade objects representing rooted or unrooted phylogenetic trees; the function also accepts a single BaseTree.Tree or BaseTree.Clade instance (the function will wrap a single tree in a list). Each tree object encodes topology and optional branch lengths and annotations, and is the primary data structure produced and consumed by Bio.Phylo code and higher-level tools that manipulate phylogenetic trees.
        file (str): Destination for the serialized output. This is passed to Bio.File.as_handle and therefore is typically a filesystem path or any object accepted by that utility; the function will open the destination with mode "w+" for writing (which truncates existing content) and will close the handle on return. Providing a pathname will overwrite the file at that path; providing an existing writable file-like object will write to that object and the caller is responsible for its lifecycle if it was opened externally.
        format (str): The key naming the output format to use. This must match one of the supported formats registered in the module (lookups are performed via supported_formats[format]). The chosen format determines which format-specific writer implementation will be called (for example Newick, Nexus, PhyloXML writers available in Bio.Phylo) and therefore which file syntax and optional keyword arguments are recognized.
        kwargs (dict): Additional keyword arguments forwarded to the format-specific writer implementation. These are writer-dependent options (for example controlling branch length precision, indentation, or matrix output in some formats) and must match the parameters expected by the selected supported_formats[format].write function; unknown or invalid keyword arguments will typically cause the format-specific writer to raise an exception.
    
    Returns:
        int or object: The value returned by the selected format-specific write function (assigned to variable n in the implementation). In common practice this is an integer count of trees successfully written, but the exact return type and semantics are determined by the format implementation in supported_formats[format]. If no value is returned by the format writer, this function will return whatever that writer returns (including None). Side effects: the primary effect is writing serialized tree data to the provided destination (file), and the destination will be opened with mode "w+" and closed by this function when File.as_handle is used.
    
    Behavior and failure modes:
        - If a single BaseTree.Tree or BaseTree.Clade is passed in the trees parameter, the function treats it as a one-element sequence and writes that single tree.
        - If format is not a key in supported_formats, a KeyError will be raised during the lookup supported_formats[format].
        - If the selected format's writer raises an exception while serializing (for example due to an unsupported tree feature, invalid kwargs, or I/O errors), that exception will propagate to the caller.
        - The destination is opened with mode "w+" which truncates existing files; ensure this is the intended behavior to avoid accidental data loss.
        - The function relies on the format-specific writer to validate tree objects; passing objects that are not valid BaseTree.Tree or BaseTree.Clade instances (or an iterable of such) may cause type or value errors in the writer.
        - Any I/O errors encountered when opening or writing to the destination (permission errors, disk full, invalid path) will propagate as standard I/O exceptions.
    
    Practical significance:
        - This function is used in Bio.Phylo workflows to export trees for downstream analysis, visualization, or interoperation with other phylogenetics tools and file formats. Callers should choose the format key corresponding to the target tool or standard (for example "newick", "phyloxml", "nexus" when supported) and pass format-specific options via kwargs to control serialization details.
    """
    from Bio.Phylo._io import write
    return write(trees, file, format, **kwargs)


################################################################################
# Source: Bio.PopGen.GenePop.get_indiv
# File: Bio/PopGen/GenePop/__init__.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_PopGen_GenePop_get_indiv(line: str):
    """Bio.PopGen.GenePop.get_indiv extracts an individual's name and genotype marker data from a single line of a GenePop-format record, returning the parsed individual name, a list of allele tuples per locus and the per-allele code length. This function is used in population genetics workflows (as in the GenePop parsers in Bio.PopGen.GenePop) to convert a text line from a GenePop input file into structured Python objects that downstream routines can analyze (e.g., calculating allele frequencies, genotype counts, or performing population genetic tests).
    
    Args:
        line (str): A single line from a GenePop-style individual record. The expected format is "individual_name, marker1 marker2 ...", where the individual's name and the marker string are separated by a single comma. Tabs in the marker region are treated equivalently to spaces. Marker tokens are concatenated allele codes (no explicit separator between alleles) and may contain zeros to represent missing alleles. Typical allele encodings use either 2 or 3 digits per allele (or 4 digits total per locus for two 2-digit alleles). This string is parsed by splitting on the first comma into an individual name and a marker field, normalizing whitespace in the marker field, and then slicing each marker token into per-allele substrings according to the detected allele code length.
    
    Returns:
        tuple: A 3-tuple (indiv_name, allele_list, marker_len) where:
            indiv_name (str): The literal individual identifier extracted from the input line (the text before the comma). This is the label used in GenePop files to identify samples and is returned unmodified except for surrounding whitespace removed by the split operation.
            allele_list (list of tuple): A list with one entry per locus (marker). For diploid parsing each entry is a 2-tuple (allele1, allele2); for haploid parsing each entry is a 1-tuple (allele,). Each allele value is either an int parsed from the allele code substring or None when the allele code is numeric zero (e.g., "000" or "00" → None), where None represents missing data in downstream population-genetics computations. The list preserves the locus order as given in the input line.
            marker_len (int): The number of digits used to encode a single allele in this line (commonly 2 or 3). This is inferred from the length of the first non-empty marker token: if that length is 2 or 4 the function uses 2 digits per allele, otherwise it uses 3.
    
    Behavior, defaults, and failure modes:
        The function first splits line on comma into exactly two parts; if the split does not produce exactly two items (for example no comma or multiple commas), a ValueError will be raised by Python's unpacking. The marker field has any tabs replaced by spaces and is split on spaces; empty tokens are removed. The function determines marker_len by inspecting the length of the first marker token: a length of 2 or 4 implies 2 digits per allele, otherwise 3 digits per allele. It then attempts to parse each marker token as two allele substrings (diploid case) converting each substring to an int and mapping integer zero to None. If any ValueError occurs during the diploid parse (for example because the second allele substring is empty or non-numeric), the code falls back to a haploid parse and attempts to parse one allele substring per marker token. If the haploid parse also fails due to non-numeric or otherwise malformed allele substrings, that ValueError will propagate to the caller. The function does not modify external state; it only returns the parsed tuple. Users should validate that the input line conforms to their GenePop file conventions (consistent allele digit lengths and numeric allele codes) because malformed lines can be misinterpreted as haploid or cause exceptions.
    """
    from Bio.PopGen.GenePop import get_indiv
    return get_indiv(line)


################################################################################
# Source: Bio.PopGen.GenePop.FileParser.read
# File: Bio/PopGen/GenePop/FileParser.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_PopGen_GenePop_FileParser_read(fname: str):
    """Parse a single GenePop-format file and return a FileRecord representing its contents.
    
    This function is part of Biopython's population genetics utilities (Bio.PopGen.GenePop)
    and is used to read and parse a GenePop file so downstream code can examine
    populations, loci, samples and genotypes for population-genetic analyses (for
    example computing allele frequencies or F-statistics). The function creates a
    FileRecord by invoking FileRecord(fname) and returns that object for programmatic
    inspection. It performs read-only parsing and does not modify the file on disk.
    
    Args:
        fname (str): Path or file name of the file containing a single GenePop record.
            This should be a file in the GenePop text format; the string may be an
            absolute or relative file system path. The value is passed directly to
            the FileRecord constructor, which is responsible for opening and parsing
            the file contents. In the context of Biopython, this parameter is the
            primary input used to load genotype and population data for subsequent
            population genetics workflows.
    
    Returns:
        FileRecord: An instance of FileRecord constructed from the supplied file name.
            The returned FileRecord represents the parsed GenePop record and provides
            programmatic access to parsed elements such as loci definitions, sample
            names, and genotype data. Callers should use the FileRecord methods and
            attributes to inspect or transform the parsed data for analysis or
            conversion to other Biopython data structures.
    
    Behavior and side effects:
        This function performs no file writing and has no side effects other than
        reading and parsing the specified file via the FileRecord constructor.
        The function itself is a thin wrapper around FileRecord(fname) and does not
        perform additional validation beyond what FileRecord implements.
    
    Failure modes and errors:
        If the file named by fname cannot be opened, the underlying file open
        operation will raise a standard I/O exception (for example FileNotFoundError
        or OSError). If the file exists but does not conform to the expected GenePop
        format, the FileRecord constructor may raise a parsing-related exception
        (for example ValueError or a more specific parsing error defined by the
        GenePop parser). Callers should catch and handle these exceptions as
        appropriate in their application context.
    
    Memory and performance notes:
        Parsing large GenePop files may use significant memory depending on the
        implementation details of FileRecord; the function returns an in-memory
        representation of the parsed record suitable for programmatic analysis within
        Biopython.
    """
    from Bio.PopGen.GenePop.FileParser import read
    return read(fname)


################################################################################
# Source: Bio.PopGen.GenePop.LargeFileParser.get_indiv
# File: Bio/PopGen/GenePop/LargeFileParser.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_PopGen_GenePop_LargeFileParser_get_indiv(line: str):
    """Bio.PopGen.GenePop.LargeFileParser.get_indiv parses a single line from a Genepop-formatted individual record and returns the individual's name, a list of allele tuples per locus, and the detected allele digit length used to split genotype tokens.
    
    This function is used by the GenePop large-file parser in Biopython to interpret one line of individual genotype data found in a Genepop input. The expected input is a single string containing an individual identifier, followed by a comma, then one or more genotype tokens separated by spaces or tabs. Each genotype token encodes one locus by concatenating allele codes (for diploids two allele codes per locus, for haploids a single allele code). The function normalizes whitespace and determines whether allele codes are represented with two digits or three digits before converting them to integers and grouping them into tuples. This parsed output is used downstream by population genetics routines in Biopython to build genotype matrices, calculate allele frequencies, and perform other analyses.
    
    Args:
        line (str): A single line from a Genepop individual section. The line must contain an individual name and genotype tokens separated by a comma (for example: "Sample1, 0101 0202 101 ..." or with tabs). The left side of the first comma is taken as the individual's name (returned unchanged as a str). The right side is treated as marker data where tokens are separated by whitespace or tabs; multiple adjacent spaces or tabs are permitted and will be collapsed. If the input does not contain a comma, Python's split will raise a ValueError; if there are no marker tokens after the comma, subsequent access will raise an IndexError.
    
    Returns:
        tuple: A 3-tuple (indiv_name, allele_list, marker_len) where:
            indiv_name (str): The substring before the first comma from the input line, representing the individual's identifier as found in the Genepop file. This string is returned verbatim and has practical significance as the sample label used in downstream population genetic analyses.
            allele_list (list of tuple of int): A list with one entry per locus. Each entry is a tuple of integers representing allele codes for that locus. For typical diploid genotype tokens the tuple will contain two ints (allele1, allele2). If integer conversion of the expected paired allele substrings fails (caught as ValueError by the implementation), the function falls back to treating each token as a haploid genotype and produces single-int tuples [(allele,), ...]. Note that this fallback is implemented by catching ValueError raised during int() conversion; therefore non-numeric or malformed tokens will cause the haploid interpretation rather than raising an error.
            marker_len (int): The number of digits used per allele in the genotype tokens as inferred from the first marker token. The code sets marker_len to 2 when len(first_token) is 2 or 4 (indicating two digits per allele, with 4 typically being a diploid concatenation), otherwise marker_len is set to 3. This integer is used to slice each marker token deterministically into allele substrings.
    
    Behavioral notes and failure modes:
        - Whitespace handling: Tabs in the marker portion are converted to single spaces and multiple contiguous spaces are collapsed before tokenization.
        - Allele digit inference: marker_len is inferred solely from the length of the first marker token; inconsistent token lengths across loci are not checked and may lead to incorrect slicing.
        - Error handling: If int() conversion of the expected allele substrings fails for any marker, the implementation treats all markers as haploid and returns single-int tuples. If the input line lacks a comma or has no markers, the function will raise ValueError or IndexError respectively (these exceptions are not caught within the function).
        - Side effects: None. The function does not mutate external state or files; it only returns parsed values derived from the input string.
    """
    from Bio.PopGen.GenePop.LargeFileParser import get_indiv
    return get_indiv(line)


################################################################################
# Source: Bio.SCOP.cmp_sccs
# File: Bio/SCOP/__init__.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_SCOP_cmp_sccs(sccs1: str, sccs2: str):
    """Compare two SCOP concise classification strings (sccs) and return their ordering.
    
    This function is part of the Bio.SCOP utilities in Biopython, a toolkit for
    computational molecular biology. An sccs encodes a protein domain's SCOP
    concise classification using a class letter followed by numeric components
    separated by dots (for example, "a.4.5.11"). The leading letter denotes the
    SCOP class and the subsequent dot-separated fields denote fold, superfamily,
    family, etc. This function implements a deterministic ordering used when
    sorting or comparing domain classifications: first by class letter (lexicographic
    order), then by each numeric component interpreted as an integer, and finally
    by the number of components (shorter strings are ordered before longer ones
    when all compared components are equal). Example ordering: a.4.5.1 < a.4.5.11 < b.1.1.1.
    
    Args:
        sccs1 (str): The first SCOP concise classification string to compare.
            This is expected to be a dot-separated string where the first token
            is a single letter representing the SCOP class and the remaining tokens
            are decimal integer strings representing hierarchical levels (fold,
            superfamily, family, ...). In Biopython workflows this typically comes
            from parsed SCOP domain records and is used as the left-hand operand in
            the comparison. If sccs1 is not a well-formed sccs (for example, if a
            numeric component is missing or not an integer literal), a ValueError
            may be raised when converting components to int.
        sccs2 (str): The second SCOP concise classification string to compare.
            Same expectations and role as sccs1 but serving as the right-hand
            operand. Both arguments must be of type str; passing another type may
            raise a TypeError if it does not support the string split operation or
            a ValueError if numeric conversion fails.
    
    Behavior and side effects:
        The comparison is performed by splitting each input on '.' and comparing
        the first token (class letter) lexicographically. If the class letters
        differ, the usual string ordering determines the result. If the class
        letters are equal, the function compares corresponding numeric tokens
        pairwise after converting them to int; the first pair of differing integers
        determines the ordering. If all compared numeric tokens are equal but one
        sccs has additional components (is longer), the shorter sccs is considered
        to come before the longer one. The function has no side effects and is
        deterministic and pure. There are no default parameter values. Failure modes
        include ValueError when numeric components cannot be parsed as integers and
        TypeError when inputs are not strings or do not support the required string
        operations.
    
    Returns:
        int: An integer indicating the ordering of sccs1 relative to sccs2.
        Returns -1 if sccs1 should be ordered before sccs2, +1 if sccs1 should be
        ordered after sccs2, and 0 if the two sccs strings are equivalent in the
        defined ordering (same class letter, same numeric components, and same
        number of components). This return convention is suitable for use in
        comparison-based sorting or for implementing custom ordering logic in
        Biopython-based SCOP classification processing.
    """
    from Bio.SCOP import cmp_sccs
    return cmp_sccs(sccs1, sccs2)


################################################################################
# Source: Bio.SCOP.Raf.normalize_letters
# File: Bio/SCOP/Raf.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_SCOP_Raf_normalize_letters(one_letter_code: str):
    """Bio.SCOP.Raf.normalize_letters: Convert RAF one-letter amino acid codes into IUPAC standard codes.
    
    This function is part of the Biopython SCOP RAF utilities and is used when parsing or processing SCOP RAF-format residue annotations to ensure amino acid codes conform to the IUPAC one-letter standard expected by downstream Biopython code and common bioinformatics tools. The routine uppercases provided letter codes and maps the RAF-specific unknown/residue placeholder "." to the IUPAC convention "X".
    
    Args:
        one_letter_code (str): A one-letter amino acid code string as found in RAF-format data. In the SCOP/RAF domain this is typically a single-character string representing an amino acid or the RAF unknown marker ".", e.g. "." or "a". The function compares the entire string to "."; if equal it returns the IUPAC unknown code "X". For any other string value it returns the same string with all characters converted to upper case. Callers should therefore provide a str (preferably a single-character code) that represents an amino acid or the RAF "." unknown marker.
    
    Returns:
        str: An uppercased IUPAC-compatible amino acid code. If the input is ".", the function returns "X" to represent an unknown residue in IUPAC notation. For any other input string, the returned value is the input converted to upper case (e.g. "a" -> "A", "G" -> "G"). There are no side effects; the function is pure and deterministic.
    
    Behavior, defaults, and failure modes:
        This function performs an exact equality check against the string "." and otherwise calls the str.upper() method on the input before returning it. There are no hidden defaults. If the caller supplies a non-str value (for example None, an int, or an object without an upper() method), an exception will be raised when attempting to call upper(); callers should ensure the argument is a str (or an object implementing an upper() method) to avoid such errors. If a multi-character string is provided, only the exact "." value will be mapped to "X"; other multi-character strings will simply be converted to upper case, which may be unintended for callers expecting strict single-letter behavior.
    """
    from Bio.SCOP.Raf import normalize_letters
    return normalize_letters(one_letter_code)


################################################################################
# Source: Bio.SearchIO.ExonerateIO.exonerate_vulgar.parse_vulgar_comp
# File: Bio/SearchIO/ExonerateIO/exonerate_vulgar.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_SearchIO_ExonerateIO_exonerate_vulgar_parse_vulgar_comp(hsp: dict, vulgar_comp: str):
    """Parse the Exonerate "vulgar" component string and populate coordinate block and codon-split information in an HSP dictionary used by Bio.SearchIO.ExonerateIO.
    
    This function is used in the Exonerate parser within Biopython (a library of Python tools for computational molecular biology) to interpret the compact "vulgar" representation produced by Exonerate into explicit coordinate ranges and auxiliary annotations on an HSP (high-scoring pair) dictionary. The input hsp dict must contain integer sentinel positions and strand indicators (see Args). The function parses each vulgar component (using the module's _RE_VCOMP regular expression), advances internal sentinels according to component step lengths and strand orientation, identifies contiguous match/codon/gap blocks, records split-codon ranges for frameshifts, and finally writes canonical start/end ranges for both query and hit sequences into the hsp dict for downstream SearchIO consumers.
    
    Args:
        hsp (dict): A mutable dictionary representing an HSP (high-scoring pair) alignment record used by Bio.SearchIO.ExonerateIO. This dict is expected to contain at minimum the integer keys "query_start", "hit_start", "query_strand", and "hit_strand"; the function will read these values as initial sentinel positions and strand indicators (strand values >= 0 mean forward orientation, < 0 mean reverse orientation). The function will mutate this dict in place and add/overwrite the following keys: "query_split_codons" (list of (start, stop) tuples for query-side codon splits introduced by 'S' vulgar components), "hit_split_codons" (list of (start, stop) tuples for hit-side codon splits), "query_ner_ranges" and "hit_ner_ranges" (initialized as empty lists for later use by the parser), and "query_ranges" and "hit_ranges" (lists of (start, stop) tuples describing contiguous alignment blocks). If the dict already contains "query_end"/"hit_end" they may be swapped with the corresponding start values when negative strand orientation is detected. Passing a dict missing the required keys will raise KeyError; passing values of incorrect type may raise TypeError or ValueError during processing.
        vulgar_comp (str): The Exonerate "vulgar" component string to parse. This compact string encodes a sequence of alignment operations; each matched component parsed by the module regular expression yields a label (one-character code) and two integer step lengths (query step and hit step). Recognized labels are the characters in "MCGF53INS" (the Exonerate vulgar operation codes handled by this parser). The function converts the parsed numeric steps to int and advances internal sentinels accordingly. Supplying a string containing an unexpected label raises an AssertionError with the offending label; supplying a non-string will raise TypeError.
    
    Returns:
        dict: The same hsp dictionary object passed in (mutated in place) now augmented with alignment block ranges and split-codon annotations. Specifically, the returned dict will contain:
            - "query_ranges": list of (start, stop) tuples describing contiguous aligned blocks on the query sequence, adjusted for strand orientation.
            - "hit_ranges": list of (start, stop) tuples describing contiguous aligned blocks on the hit/target sequence, adjusted for strand orientation.
            - "query_split_codons": list of (start, stop) tuples for query-side split codons detected from 'S' components (each tuple is ordered so the first value is the smaller coordinate).
            - "hit_split_codons": list of (start, stop) tuples for hit-side split codons (each tuple ordered with the smaller coordinate first).
            - "query_ner_ranges" and "hit_ner_ranges": present as empty lists (reserved by the parser).
        The function returns the same dict for convenience, but callers should be aware that the primary effect is the in-place mutation of the provided hsp dict.
    
    Behavior, side effects, defaults, and failure modes:
        - The function mutates the provided hsp dict in place and also returns it; callers may rely on the returned value or on inspection of the original dict after the call.
        - Coordinate sentinels are advanced by qstep and hstep values parsed from vulgar_comp; qstep and hstep are multiplied by +1 or -1 depending on the "query_strand" and "hit_strand" values in the hsp dict to handle forward/reverse orientations.
        - Contiguous blocks of labels in "MCGS" are grouped into start/stop ranges; labels in "5", "3", "I", and "N" are used to determine inter-block positions but are not themselves stored as ranges by this function.
        - 'S' components (split codon / frameshift indicators) produce entries in "query_split_codons" and "hit_split_codons"; these ranges are stored with their endpoints ordered (min, max) to be orientation-agnostic.
        - If a vulgar component label outside "MCGF53INS" is encountered, an AssertionError is raised identifying the unexpected label.
        - If the hsp dict lacks required keys ("query_start", "hit_start", "query_strand", "hit_strand") a KeyError will be raised. If numeric conversion of component step values fails (e.g., because the vulgar string is malformed), a ValueError or TypeError may be raised.
        - After processing, start/end coordinates and block ranges are adjusted so that sequence orientation is consistent: when a strand value is negative, the function swaps the stored start and end fields for that sequence type and swaps internal start/end block lists so the resulting "query_ranges" and "hit_ranges" reflect canonical coordinate order.
        - The function relies on the module-level regular expression _RE_VCOMP to split vulgar_comp into components; malformed strings that do not match the expected pattern may lead to unexpected behavior or exceptions.
    
    Notes for users:
        - This function is intended for use inside the Exonerate parser in Bio.SearchIO; it is not a general-purpose vulgar-string parser for arbitrary formats. It provides the concrete coordinate data structures (ranges and split-codon lists) required by downstream Biopython code that represents alignments for computational molecular biology analyses.
    """
    from Bio.SearchIO.ExonerateIO.exonerate_vulgar import parse_vulgar_comp
    return parse_vulgar_comp(hsp, vulgar_comp)


################################################################################
# Source: Bio.SearchIO._utils.fragcascade
# File: Bio/SearchIO/_utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_SearchIO__utils_fragcascade(attr: str, seq_type: str, doc: str = ""):
    """Bio.SearchIO._utils.fragcascade: Return a property descriptor that implements a getter and a cascading setter for attributes on HSPFragment objects used in Biopython SearchIO parsing.
    
    This helper constructs a Python property which stores the value on an HSPFragment instance under an internal name derived from the fragment role (hit or query) and the attribute name, and which also propagates (cascades) the same value to the associated sequence object (the fragment's .hit or .query attribute) when that associated sequence is not None. In the context of Biopython's SearchIO module, HSPFragment objects represent aligned subregions (high-scoring pair fragments) extracted from sequence alignment output (for example BLAST/PSI-BLAST results). fragcascade is used to keep attributes (for example sequence strings or coordinate fields) synchronized between an HSPFragment and its parent sequence object so downstream code that inspects either object sees a consistent value.
    
    Args:
        attr (str): The attribute name on the sequence object to cascade to and the public attribute logically represented on the fragment. This is the literal attribute name used when calling setattr(seq, attr, value) on the associated sequence object; it must be a valid Python attribute name for the target objects. For example, using "seq" or "start" will cause the setter to set that attribute on the fragment's associated sequence object.
        seq_type (str): Must be either "hit" or "query". This selects which associated sequence attribute on the HSPFragment to cascade to: when "hit" the setter will attempt to set the attribute on self.hit, and when "query" it will attempt to set it on self.query. An assertion is raised (AssertionError) at runtime inside fragcascade if seq_type is not one of these exact strings.
        doc (str): Optional documentation string for the generated property. Defaults to the empty string "". This string will become the .__doc__ of the returned property object and can be used by automated documentation tools in the Biopython API.
    
    Behavior and side effects:
        The generated property uses an internal storage attribute named f"_{seq_type}_{attr}" on the HSPFragment instance. The getter returns the value of that internal attribute via getattr(self, f"_{seq_type}_{attr}"). The setter assigns the value to that internal attribute via setattr(self, f"_{seq_type}_{attr}", value) and then, if getattr(self, seq_type) is not None, calls setattr(seq, attr, value) to copy the value onto the associated sequence object (self.hit or self.query). This mutation means setting the property will modify both the fragment and, when present, the referenced sequence object; callers should be aware that the same sequence object may be shared by multiple fragments, so cascading updates can affect other fragments or code that holds references to that sequence.
        If the associated sequence attribute (self.hit or self.query) is None, only the fragment's internal attribute is updated and no cascading occurs.
        The getter will raise AttributeError if the internal storage attribute has not been set on the fragment instance prior to access.
    
    Failure modes and exceptions:
        fragcascade asserts that seq_type is exactly "hit" or "query"; providing any other value causes an AssertionError at the time fragcascade is called to create the property. Accessing the generated property's getter before the internal storage attribute has been set will raise AttributeError. The setter uses setattr on the associated sequence object, which will create or overwrite the attribute on that object; no AttributeError is raised by setattr itself, but unexpected side effects can occur if the sequence object defines property descriptors or validation logic for that attribute.
    
    Returns:
        property: A Python property object (built-in property) with a fget that returns the fragment's internal attribute and a fset that assigns the internal attribute and cascades the same value to the associated sequence object when present. This property is intended to be assigned as an attribute on the HSPFragment class definition so that instances automatically benefit from the getter/setter behavior described above.
    """
    from Bio.SearchIO._utils import fragcascade
    return fragcascade(attr, seq_type, doc)


################################################################################
# Source: Bio.SearchIO._utils.fullcascade
# File: Bio/SearchIO/_utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_SearchIO__utils_fullcascade(attr: str, doc: str = ""):
    """Bio.SearchIO._utils.fullcascade returns a Python property that provides a cascading getter and setter for a named attribute on SearchIO container items (for example HSP objects within a Hit or Query container). It is intended for use in Biopython SearchIO code to create container-level attributes that reflect the corresponding attribute on member items: reading the property retrieves the attribute from the first contained item, and writing the property sets that attribute on every contained item.
    
    Args:
        attr (str): The attribute name on the contained items (for example an HSP attribute) to be accessed and modified via the returned property. This string is used with getattr and setattr on items stored in the container (self._items[0] for reads, and iterating over self for writes). In the SearchIO domain, this is typically the name of a per-item attribute you want to expose at the container level (e.g. "evalue", "score"). The function does not validate that the attribute exists on items at creation time.
        doc (str): Optional documentation string to assign to the returned property object. Defaults to the empty string "". This becomes the property.__doc__ and should describe the purpose of the container-level attribute in the context of SearchIO containers.
    
    Behavior and side effects:
        The returned object is a built-in Python property with a getter and setter:
        - Getter behavior: fget returns getattr(self._items[0], attr). This reads the attribute from the first item in the container (self._items is expected to be the container's internal list-like storage of items). Practically, this means the container attribute reflects the first item's value. The implementation does not compare values across items, so if other items have a different value, the getter will return the first item's value without raising an error or issuing a warning; callers should ensure consistency if that is required.
        - Setter behavior: fset iterates over the container (for item in self) and calls setattr(item, attr, value) for each item. This mutates every contained item by assigning the provided value to the named attribute, effectively propagating the container-level assignment down to all members (a full cascade).
        The setter does not set the attribute on the container object itself; it only mutates the items.
    
    Failure modes and exceptions:
        If the container has no items (self._items is empty), the getter will raise IndexError when attempting to access self._items[0]. If the named attribute does not exist on the first item, the getter will raise AttributeError. During setting, setattr may raise AttributeError (for read-only descriptors or missing attribute implementations) or other exceptions raised by item attribute setters; such exceptions will occur during iteration and may leave some items already modified. The function does not perform type checking of values; any TypeError raised by the underlying attribute assignment will propagate to the caller.
    
    Returns:
        property: A Python property object whose fget returns the named attribute from the first item of a SearchIO container and whose fset assigns the given value to that attribute on every item in the container. The property's __doc__ is set to the provided doc string.
    """
    from Bio.SearchIO._utils import fullcascade
    return fullcascade(attr, doc)


################################################################################
# Source: Bio.SearchIO._utils.get_processor
# File: Bio/SearchIO/_utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_SearchIO__utils_get_processor(format: str, mapping: dict):
    """Return the processor object for a given search result format name by looking it up
    in a mapping and performing a dynamic import from the Bio.SearchIO subpackage.
    
    Args:
        format (str): The lower-case name of the file/search format to process. In the
            Bio.SearchIO context this is the canonical format identifier provided by the
            caller (for example when selecting which parser/iterator to use for a
            particular search program output). This function requires a Python string
            and expects it to already be lower case; if None, not a string, or not
            lower case a descriptive exception will be raised. The function uses this
            exact string as a key into the mapping argument.
        mapping (dict): A dictionary that maps format name strings (lower case) to a
            two-item sequence (mod_name, obj_name). mod_name is the name of a module
            located under the Bio.SearchIO package (for example the module that
            implements parsing for that format) and obj_name is the name of an
            attribute in that module (typically a parser class or factory function).
            Concretely each value is unpacked as "mod_name, obj_name" and both are
            expected to be strings. This mapping is typically provided by SearchIO
            infrastructure code to associate format identifiers with their
            implementing modules and exported processor names.
    
    Returns:
        object: The processor object retrieved by importing the module Bio.SearchIO.<mod_name>
        and returning the attribute named obj_name from that module. In the typical
        SearchIO usage this will be a parser class, iterator factory, or callable
        capable of processing files of the given format.
    
    Raises:
        ValueError: If format is None with the message "Format required (lower case string)".
        TypeError: If format is not a str, with the message "Need a string for the file format (lower case)".
        ValueError: If format is not lower case; message will indicate the provided
            format string should be lower case.
        ValueError: If format is not a key in mapping; message will indicate the unknown
            format and list the supported format keys from mapping.
        ImportError: If importing the module Bio.SearchIO.<mod_name> fails (propagated
            from the dynamic import), which may occur if the implementation module is
            missing or fails to execute at import time.
        AttributeError: If the imported module does not define the attribute named
            obj_name (propagated from getattr).
    
    Behavior and side effects:
        The function performs a dictionary lookup mapping[format] to obtain obj_info,
        then unpacks obj_info into (mod_name, obj_name). It dynamically imports the
        module using __import__("Bio.SearchIO.%s" % mod_name, fromlist=[""]) which
        executes that module's top-level code and may have side effects (module-level
        initialisation). After importing, getattr is used to retrieve the named
        attribute from the module and that object is returned to the caller. Because
        the import and attribute lookup are dynamic, errors in the implementation
        module or mismatches between mapping and the module contents will result in
        ImportError or AttributeError being raised to the caller.
    
    Practical significance:
        In the Biopython SearchIO subsystem (used for parsing bioinformatics search
        outputs such as BLAST, HMMER, etc.), this utility centralises the logic of
        resolving a user-supplied format identifier to the concrete parser/processor
        implementation. It enables SearchIO code to remain generic: callers supply a
        format name and this function returns the appropriate parser object based on
        the provided mapping. This makes it straightforward to add new formats by
        extending the mapping with the target module and exported processor name.
    """
    from Bio.SearchIO._utils import get_processor
    return get_processor(format, mapping)


################################################################################
# Source: Bio.SearchIO._utils.optionalcascade
# File: Bio/SearchIO/_utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_SearchIO__utils_optionalcascade(cont_attr: str, item_attr: str, doc: str = ""):
    """Create and return a property object whose getter reads a named attribute from the first contained item when the container has one or more items, otherwise from a named attribute on the container itself, and whose setter updates (cascades) the named container attribute and the named attribute on every item in the container.
    
    This helper is part of Bio.SearchIO._utils and is used by SearchIO container classes (for example to implement the id and description properties of result/hit/container objects used when parsing sequence search outputs such as BLAST). In that domain, containers represent query results or hits and hold zero or more item objects; this function ensures a consistent view of e.g. an identifier or description whether the container is empty or populated, and keeps container and item attributes synchronized when modified.
    
    Args:
        cont_attr (str): The attribute name on the container object to read or write when no items are present (and which is written to on every setter call). This must be the literal attribute name (for example "id" or "description") used by the container type in SearchIO. If the container does not have this attribute, accessing or setting the property will raise AttributeError.
        item_attr (str): The attribute name on each item object to read or write when the container has one or more items. When items are present, the getter returns getattr(self[0], item_attr); the setter will call setattr(item, item_attr, value) for every item in the container. This must be the literal attribute name used by the item objects (for example "id" or "description"). If an item lacks this attribute, AttributeError will be raised during get or set.
        doc (str): Optional documentation string to attach to the returned property object. Defaults to the empty string. This string becomes the property's __doc__ and is intended to document the behavior for users and API documentation generation.
    
    Returns:
        property: A Python property object with a getter and setter implementing the described cascading behavior. The getter returns the first item's item_attr value when the container has one or more items (checking truthiness of self._items), otherwise it returns the container's cont_attr value. The setter sets the container's cont_attr to the provided value and then iterates over the container to set each item's item_attr to the same value. Side effects: calling the setter mutates the container and all items by assigning the new value. Failure modes: retrieving via the getter can raise AttributeError if cont_attr or item_attr do not exist on the container or first item; attempting to access self[0] can raise IndexError or TypeError if the container does not support indexing; the setter can raise TypeError if the container is not iterable, or AttributeError if any item lacks item_attr. This returned property is intended for use on SearchIO container classes to provide a single coherent id/description-like attribute over empty and non-empty containers and to keep contained items synchronized when the container-level value is changed.
    """
    from Bio.SearchIO._utils import optionalcascade
    return optionalcascade(cont_attr, item_attr, doc)


################################################################################
# Source: Bio.SearchIO._utils.removesuffix
# File: Bio/SearchIO/_utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_SearchIO__utils_removesuffix(string: str, suffix: str):
    """Bio.SearchIO._utils.removesuffix: Remove a trailing suffix from a string, providing a small compatibility wrapper for Python 3.8 used in Biopython's SearchIO utilities.
    
    This function is used in the Bio.SearchIO._utils module to normalize or clean textual identifiers, filenames, and other string tokens commonly encountered when parsing search results (for example BLAST, HMMER, or other sequence search output). In the Biopython project this helps ensure consistent handling of trailing suffixes across supported Python versions, by delegating to the built-in str.removesuffix on Python versions that provide it and falling back to an equivalent implementation on Python 3.8.
    
    Args:
        string (str): The input text from which a trailing suffix may be removed. In the SearchIO parsing context this is typically an identifier, file base name, or other token produced by a parser. The function does not modify the original object (strings are immutable); it returns a new str when modification is required. Passing a non-str value is not supported and will result in a type-related exception.
        suffix (str): The suffix to remove from the end of string if present. In practical use this can be a file extension (for example ".txt" or ".fa"), a known trailing marker applied by a search program, or any other exact suffix to strip. An empty string is treated as "no suffix" and the original string is returned unchanged.
    
    Behavior and side effects:
    This function has no side effects beyond returning a value (it does not mutate its inputs or global state). On Python versions later than 3.8 it simply calls the standard library method string.removesuffix(suffix). On Python 3.8 it implements the equivalent logic: if suffix is truthy and string.endswith(suffix) is True, it returns string with the trailing suffix removed by slicing; otherwise it returns the original string unchanged. If suffix is empty or evaluates to False, the original string is returned unchanged. If the suffix is not present at the end of string (including the case where suffix is longer than string), the original string is returned unchanged. The operation uses a single endswith check and, when removing, a single slice, so the cost is proportional to the length of the involved strings.
    
    Failure modes:
    If either argument is not of type str, the function will raise a type-related exception (for example AttributeError or TypeError depending on the Python version and the actual types passed). The function does not perform implicit type coercion. It is intended only for exact suffix removal; it does not perform pattern matching, regular expression substitution, or case-insensitive matching.
    
    Returns:
        str: A string equal to string with the trailing suffix removed if and only if suffix is non-empty and string ends with suffix; otherwise the original string is returned unchanged. This return value is suitable for subsequent SearchIO parsing or identifier normalization steps in Biopython.
    """
    from Bio.SearchIO._utils import removesuffix
    return removesuffix(string, suffix)


################################################################################
# Source: Bio.Seq.transcribe
# File: Bio/Seq.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_Seq_transcribe(dna: str):
    """Bio.Seq.transcribe transcribes a DNA sequence into an RNA sequence (T -> U),
    using the convention that the provided sequence represents the coding strand
    of the DNA double helix. This function is intended for use in computational
    molecular biology and bioinformatics workflows (for example, preparing RNA
    sequences for downstream analysis such as translation or RNA structure
    prediction), as provided by the Biopython project.
    
    Transcription is performed by replacing thymine bases ('T' and 't') with
    uracil ('U' and 'u') without performing any complement or reverse-complement
    operation. Ambiguous or non-thymine characters (for example 'N') are left
    unchanged. The function preserves the original object: it does not modify a
    MutableSeq or string in place but returns a new object containing the
    transcribed sequence.
    
    Args:
        dna (str): Input DNA sequence to transcribe. In typical use this is a
            Python string containing IUPAC nucleotide codes (for example "ACTG").
            Per the implementation in Bio.Seq, this function also accepts objects
            from the Biopython sequence classes Seq and MutableSeq: if the runtime
            object is an instance of Bio.Seq.Seq the method dna.transcribe() is
            called and a new Seq object is returned; if it is an instance of
            Bio.Seq.MutableSeq the object is first converted to Seq and then
            transcribed. The str annotation in the function signature indicates
            the common case of passing a plain Python string, but Seq and
            MutableSeq instances are explicitly handled by the function. If an
            object without a .replace method (and which is not an instance of
            Seq/MutableSeq) is supplied, an AttributeError will be raised by the
            underlying code path.
    
    Returns:
        str or Seq: If the input was a plain Python string, returns a new string
        object with all 'T' characters replaced by 'U' and all 't' by 'u'
        (for example "ACTGN" -> "ACUGN"). If the input was an instance of
        Bio.Seq.Seq or Bio.Seq.MutableSeq, returns a new Bio.Seq.Seq object
        representing the transcribed RNA sequence. The function never modifies
        the original input object in place; it always returns a new object.
    
    Additional behavior and failure modes:
        - The function treats the provided sequence as the coding strand (not the
          template strand). It therefore performs a direct T->U substitution,
          which models transcription when the coding strand sequence is known.
        - The function does not perform complementing or reverse operations; to
          obtain the RNA transcribed from the template strand, callers must first
          compute the complement/reverse-complement as appropriate.
        - For Seq instances, the implementation delegates to Seq.transcribe(), so
          any behavior or future changes to that method affect this function.
        - For inputs that are neither Seq/MutableSeq nor string-like (i.e.,
          lack a replace method), the function will raise an AttributeError.
        - Case of bases is preserved except that both 'T' and 't' are converted to
          'U' and 'u' respectively.
    """
    from Bio.Seq import transcribe
    return transcribe(dna)


################################################################################
# Source: Bio.SeqIO.QualityIO.phred_quality_from_solexa
# File: Bio/SeqIO/QualityIO.py
# Category: fix_docstring
# Reason: Schema parsing failed: Cannot generate JSON schema for phred_quality_from_solexa because the docstring has no description for the argument 'solexa_quality'
################################################################################

def Bio_SeqIO_QualityIO_phred_quality_from_solexa(solexa_quality: float):
    """Convert a Solexa quality score to an equivalent PHRED quality score used by sequencing data parsers.
    
    This function is used in Bio.SeqIO.QualityIO within the Biopython project (a toolkit for computational molecular biology) to translate quality scores encoded using the Solexa log-odds scale into the more commonly used PHRED scale. Both Solexa and PHRED scores are log transformations of an estimated probability of error (higher score = lower error probability). The conversion assumes the underlying error probability estimates are equivalent between the two scales. The numeric mapping implemented is:
    phred_quality = 10 * log(10**(solexa_quality / 10.0) + 1, 10)
    which first converts the Solexa score back to an error probability equivalent and then expresses that probability on the PHRED scale. The function returns a floating point PHRED value; callers that need integer PHRED scores (for example, to store in certain file formats or downstream tools expecting integer values) should round or cast the returned value as appropriate.
    
    Args:
        solexa_quality (float): A Solexa-style quality score to convert. In sequencing workflows this is the log-odds quality value associated with a base call (can be zero or negative). As used in the source code and examples, None is treated as a special "missing value" indicator and is returned unchanged (see Returns). Negative Solexa values are allowed and expected (they correspond to probabilities of error greater than 0.5). Very low values (less than -5) are unexpected for typical Solexa data; passing such a value will still be converted mathematically but will also trigger a runtime warning (a BiopythonWarning via warnings.warn) to alert the user that the input is outside the normal expected range.
    
    Returns:
        float or None: The computed PHRED-style quality score as a floating point number, representing the same error probability expressed on the PHRED scale. If solexa_quality is None (used as a missing/NULL indicator in some Biopython pipelines and input parsers), None is returned to preserve that sentinel value. The function does not round the numeric result; it is the caller's responsibility to round or convert to an integer if required by downstream formats or tools.
    
    Behavior and failure modes:
        The conversion follows the exact logarithmic mapping shown above and will produce values approximately equal to the Solexa input for large positive scores (e.g., converting 80 returns ~80.00). For solexa_quality < -5 a BiopythonWarning is emitted but a numeric conversion is still returned (values for inputs below -5 map into the range 0 to ~1.19). If a non-numeric, non-None object is passed (for example a string), a TypeError or ValueError may be raised by the numeric operations or comparisons performed; callers should validate or coerce inputs when necessary. This function has no other side effects beyond possibly emitting the BiopythonWarning for very low inputs.
    """
    from Bio.SeqIO.QualityIO import phred_quality_from_solexa
    return phred_quality_from_solexa(solexa_quality)


################################################################################
# Source: Bio.SeqIO.QualityIO.solexa_quality_from_phred
# File: Bio/SeqIO/QualityIO.py
# Category: fix_docstring
# Reason: Schema parsing failed: Cannot generate JSON schema for solexa_quality_from_phred because the docstring has no description for the argument 'phred_quality'
################################################################################

def Bio_SeqIO_QualityIO_solexa_quality_from_phred(phred_quality: float):
    """Bio.SeqIO.QualityIO.solexa_quality_from_phred converts a PHRED-style quality score to the equivalent Solexa-style quality score used by some sequencing file formats and older Illumina encodings. It is part of Biopython's SeqIO/QualityIO utilities for handling per-base sequencing quality values when parsing, converting or writing sequence data.
    
    Args:
        phred_quality (float): PHRED quality score to convert. In sequencing and in this module a PHRED score is the standard log-scaled measure of per-base error probability, defined as phred_quality = -10 * log10(error). Typical practical PHRED values are in the range 0 to about 90. As a special-case documented in the source, passing None is supported as a sentinel for missing values (for example Bio.SeqIO may use None for gaps or unknown qualities); when phred_quality is None the function returns None unchanged. The function expects non-negative numeric PHRED values; negative inputs are treated as invalid (see Failure modes).
    
    Returns:
        float or None: The corresponding Solexa quality as a floating point number, or None when phred_quality is None. The Solexa quality is computed by converting the PHRED score back to an error probability and then applying the Solexa log-odds transform:
        error = 10 ** (-phred_quality / 10)
        solexa_quality = -10 * log10(error / (1 - error))
        which simplifies to
        solexa_quality = 10 * log10(10 ** (phred_quality / 10.0) - 1)
        Practically, the function applies the EMBOSS convention of a minimum Solexa value of -5.0 (so the returned value is max(-5.0, computed_value)), and maps a PHRED of 0 explicitly to -5.0. The function returns a floating point result so callers should round to an integer if an integer quality score is required by downstream code or file formats.
    
    Behavior, defaults, and failure modes:
        This function is pure (no side effects) and deterministic. If phred_quality is None the function treats it as a missing/NULL value used in SeqIO contexts and returns None. If phred_quality > 0 the function computes the Solexa score using the formula above and enforces a minimum returned value of -5.0 as used in real Solexa/EMBOSS outputs (this ensures random base calls map to -5 after rounding). If phred_quality == 0 the function returns -5.0 (special-case mapping discussed in the source). If phred_quality is negative the function raises a ValueError indicating PHRED qualities must be non-negative, because negative PHRED values are not valid in the PHRED error-probability interpretation. The returned float may be numerically equal to the input PHRED for high-quality scores; differences are important for low-quality bases. Callers are responsible for any rounding or type conversion required by file formats or downstream processing.
    """
    from Bio.SeqIO.QualityIO import solexa_quality_from_phred
    return solexa_quality_from_phred(phred_quality)


################################################################################
# Source: Bio.SeqUtils.GC123
# File: Bio/SeqUtils/__init__.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_SeqUtils_GC123(seq: str):
    """Bio.SeqUtils.GC123 calculates G+C content for a nucleotide sequence overall and separately for each codon position (first, second, third) in a coding-sequence reading frame starting at the first base. This function is part of Biopython, a toolkit for computational molecular biology, and is used when assessing sequence composition, codon position bias, or GC-related metrics in coding DNA.
    
    This function walks the input sequence in codons (triplets) from the first base, counts occurrences of the unambiguous DNA nucleotides A, T, G, and C (case-insensitive) at each codon position, and computes percentages as 100 * (G + C) / n where n is the count of A/T/G/C observed at that position. An incomplete trailing codon (length < 3) is padded internally so that its remaining positions do not match A/T/G/C and therefore do not contribute to counts. Ambiguous nucleotides (for example 'N' or other non-ATGC characters) are not treated as A/T/G/C: they are effectively ignored and reduce the denominator n for the affected positions. The function performs no input sanitization beyond treating the input as an indexable string and does not modify the input sequence.
    
    Args:
        seq (str): A nucleotide sequence supplied as a Python string. The sequence is interpreted as a coding-region sequence where codons are read from the first character in successive groups of three. Mixed upper- and lower-case letters are supported (both 'A' and 'a' are counted as adenine). The function only recognizes the characters 'A', 'T', 'G', and 'C' (and their lower-case equivalents) as valid nucleotides for counting; any other characters are treated as ambiguous and ignored in the GC percentage calculations.
    
    Returns:
        tuple: A 4-tuple of floats (overall_gc, gc_pos1, gc_pos2, gc_pos3). Each value is a percentage between 0.0 and 100.0 computed as 100 * (G + C) / n where n is the number of A/T/G/C observations contributing to that calculation. The order is: overall sequence G+C percentage across all codon positions combined, then the G+C percentage at the first, second, and third codon positions respectively.
    
    Behavior, defaults, and failure modes:
    - The reading frame is fixed to start at the first nucleotide (index 0). If your sequence requires a different frame, adjust the input before calling this function.
    - The function tolerates mixed case but does not accept nucleotide ambiguity codes as contributions to counts; ambiguous characters reduce the effective denominator or, if present throughout, can prevent a meaningful percentage being computed.
    - If for a particular codon position no A/T/G/C characters are observed, that position's percentage is returned as 0.0 (the implementation protects against division errors per-position). However, if the entire sequence contains no A/T/G/C characters (so there are no valid counts across all positions), a ZeroDivisionError will be raised when computing the overall GC percentage because the overall denominator becomes zero.
    - No side effects: the input string is not modified and no external state is changed.
    - Passing a non-str value for seq will typically raise a TypeError when the function attempts to use len or indexing operations on the input.
    """
    from Bio.SeqUtils import GC123
    return GC123(seq)


################################################################################
# Source: Bio.SeqUtils.GC_skew
# File: Bio/SeqUtils/__init__.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_SeqUtils_GC_skew(seq: str, window: int = 100):
    """Calculate GC skew (G-C)/(G+C) for non-overlapping windows along a DNA sequence.
    
    This function computes a simple per-window measure of nucleotide composition asymmetry used in computational molecular biology (for example, profiling local GC bias across chromosomes, contigs, or sequencing reads). For each contiguous, non-overlapping window of the input sequence it counts guanine (G/g) and cytosine (C/c) and returns the ratio (G - C) / (G + C). The implementation is case-insensitive for G and C, ignores ambiguous nucleotides (they are treated as neither G nor C), and explicitly handles windows with no G or C by returning 0.0 for that window to avoid division-by-zero errors.
    
    Args:
        seq (str): The DNA sequence to analyze. Must be a Python str containing nucleotide characters. The function counts only 'G' and 'C' in either uppercase or lowercase; all other characters (including ambiguous nucleotides such as 'N') are not counted toward G or C. In Biopython workflows this parameter is typically a whole-genome, contig, or read sequence supplied by SeqIO or other sequence-handling modules.
        window (int): The window size in nucleotides to use for each non-overlapping segment (default 100). The code iterates over seq in steps of this size producing contiguous, non-overlapping windows; the final window may be shorter if len(seq) is not a multiple of window. window is expected to be a positive integer; passing window == 0 will raise a ValueError from range(), and a negative window will result in no windows being produced (an empty result list). The default value of 100 provides a moderate resolution commonly used in exploratory genome-skew analyses but can be adjusted to increase or decrease spatial resolution.
    
    Returns:
        list of float: A list of skew values, one float per window in order from the start to the end of the sequence. Each float is the ratio (G - C) / (G + C) computed using counts of 'G' and 'C' (case-insensitive). For positive window sizes, the number of returned values equals ceil(len(seq) / window). Windows that contain no G or C return 0.0 (handled internally to avoid ZeroDivisionError). There are no side effects; the function does not modify the input sequence.
    """
    from Bio.SeqUtils import GC_skew
    return GC_skew(seq, window)


################################################################################
# Source: Bio.SeqUtils.nt_search
# File: Bio/SeqUtils/__init__.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_SeqUtils_nt_search(seq: str, subseq: str):
    """Search for a DNA subsequence in a DNA sequence string, interpreting IUPAC
    ambiguous nucleotide codes, and return the regular-expression pattern used
    followed by the 0-based start positions of each match on the forward strand.
    
    This function is part of Biopython's utilities for computational molecular
    biology and bioinformatics. It is used to locate occurrences of nucleotide
    subsequences (for example motifs or primer sites) in a larger DNA sequence
    string while allowing IUPAC ambiguity codes (e.g. N, R, Y) in the query.
    Ambiguous codes in subseq are expanded using IUPACData.ambiguous_dna_values
    so that the function builds a Python regular expression pattern (with
    bracketed character classes like "[ACGT]" for ambiguous positions) and then
    searches only the forward strand of seq using Python's re.search.
    
    Args:
        seq (str): Target DNA sequence to search. This is the full DNA string in
            which occurrences of subseq are sought. The matching performed is
            case-sensitive because the function constructs a regular expression
            from subseq and calls re.search on slices of seq without any
            case-folding flags; therefore, to reliably match IUPAC uppercase keys
            (as used by IUPACData.ambiguous_dna_values), provide seq in the same
            case (commonly uppercase). seq is not modified by the function.
        subseq (str): Query DNA subsequence expressed using IUPAC nucleotide
            codes (for example A, C, G, T for unambiguous bases and N, R, Y, etc.
            for ambiguous bases). Each character in subseq is looked up in
            IUPACData.ambiguous_dna_values to determine the set of allowed bases
            at that position; if the mapping yields a single base the literal
            character is used in the generated regex, otherwise a bracketed
            character class is created (e.g. "N" -> "[ACGT]"). subseq is treated
            as a pattern for the forward strand only; reverse-complement searching
            is not performed by this function.
    
    Returns:
        list: A list whose first element is the regular expression pattern (str)
        constructed from subseq by expanding IUPAC ambiguity codes (this is the
        exact pattern passed to re.search), and whose remaining elements (if any)
        are integer start positions (int) in seq where the pattern matches. The
        positions are 0-based indices corresponding to the first nucleotide of
        each match and are reported in ascending order as found by the iterative
        search. If no matches are found, the returned list contains only the
        pattern string.
    
    Behavior, side effects, defaults, and failure modes:
        The function constructs a regex pattern from subseq using the mapping
        IUPACData.ambiguous_dna_values and then iteratively calls re.search on
        progressively sliced views of seq to find successive matches. It begins
        searching at sequence index 0 and after finding a match at index i it
        resumes the next search at index i+1, so matches with different start
        positions are reported (overlapping matches whose starts differ by at
        least one are found). There are no side effects: seq and subseq are not
        modified and no global state is changed. The function is case-sensitive;
        if subseq contains characters not present in IUPACData.ambiguous_dna_values
        a KeyError will be raised. The regex created is unanchored and may match
        anywhere in seq; if you require anchored or case-insensitive matching,
        transform inputs or perform a separate regex-based search. Performance
        depends on the lengths of seq and subseq and the complexity of ambiguous
        expansions because the implementation repeatedly slices seq and calls
        re.search; for very large inputs or many ambiguous positions consider more
        optimized approaches.
    """
    from Bio.SeqUtils import nt_search
    return nt_search(seq, subseq)


################################################################################
# Source: Bio.SeqUtils.seq1
# File: Bio/SeqUtils/__init__.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_SeqUtils_seq1(seq: str, custom_map: dict = None, undef_code: str = "X"):
    """Convert a protein sequence given with three-letter amino acid codes into a string of one-letter amino acid codes.
    
    This function is part of Biopython's Bio.SeqUtils utilities used in computational molecular biology workflows to normalize protein sequences for downstream tasks (for example, sequence comparisons, alignments, or database lookups). It maps contiguous three-character tokens from the input sequence to their one-letter IUPAC amino acid codes using Biopython's IUPACData.protein_letters_3to1_extended mapping. The function is case-insensitive for the three-letter codes, supports the IUPAC ambiguous/reserved one-letter codes (B for Asx, J for Xle, X for Xaa, U for Sel, O for Pyl), and by default maps the termination code "Ter" to "*" (this default can be changed via custom_map). The function performs a simple fixed-width grouping of the input (every three characters), so any trailing characters when the input length is not a multiple of three are ignored.
    
    Args:
        seq (str): The input protein sequence to convert. In practice this should be a sequence of contiguous three-letter amino acid codes (for example "MetAlaIle..."). The original implementation accepts a Python string and has historically been documented to accept Seq or MutableSeq objects from Biopython as well; the code treats the object as indexable and sliceable and will therefore work with objects implementing the same interface. The function is case-insensitive: "met", "Met", and "MET" are equivalent. Whitespace or non-letter characters present in the three-character groups will be treated as part of the group and, if not found in the mapping (after uppercasing), will be replaced by the undef_code. If len(seq) is not divisible by three, characters after the last full three-character group are ignored.
        custom_map (dict): Optional mapping of three-letter tokens (keys) to one-letter strings (values) to override or extend the built-in mappings. If None (the default), custom_map is set to {"Ter": "*"} by the function. Keys from custom_map are uppercased before being merged with the internal mapping, making key matching case-insensitive. Values provided in custom_map are inserted verbatim into the output where the corresponding three-letter key is found. Keys in custom_map should be string-like objects providing an upper() method; non-string keys that do not support upper() will raise an AttributeError. Values are not validated for length by the function; providing multi-character values will produce multi-character output at those positions.
        undef_code (str): Single-character string used to represent any three-letter token that is not found in the combined mapping (built-in IUPAC mapping updated with custom_map). Defaults to "X". Any unknown or gap-like three-letter groups (including groups containing '-' or other non-alphabetic characters) will be replaced by this undef_code in the returned sequence.
    
    Returns:
        str: A new string composed of the one-letter amino acid codes corresponding to each consecutive three-character group in the input. The returned string length equals floor(len(seq) / 3) because trailing characters beyond the last complete three-character group are ignored. The mapping follows the IUPAC standard (including ambiguous codes and the termination symbol as given or overridden by custom_map). No in-place modification of the input occurs; the function has no side effects beyond computing and returning this string.
    
    Failure modes and edge cases:
        - If seq is not indexable/sliceable (does not implement __len__ and __getitem__), a TypeError or similar will be raised when attempting to slice/group the input.
        - If custom_map contains keys that are not string-like (lack upper()), an AttributeError will be raised when the function uppercases keys.
        - If seq contains characters or groups not present in the merged mapping, those groups are replaced by undef_code rather than raising an exception.
        - Trailing characters when len(seq) % 3 != 0 are silently ignored; this is intentional to allow concatenated three-letter tokens to be processed without error.
    """
    from Bio.SeqUtils import seq1
    return seq1(seq, custom_map, undef_code)


################################################################################
# Source: Bio.SeqUtils.seq3
# File: Bio/SeqUtils/__init__.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_SeqUtils_seq3(seq: str, custom_map: dict = None, undef_code: str = "Xaa"):
    """Convert a protein sequence from one-letter amino acid codes to concatenated three-letter codes following the IUPAC convention.
    
    This function is used in computational molecular biology workflows (as in Biopython) to translate protein sequences represented with single-letter amino acid codes into their three-letter equivalents for tasks such as human-readable output, legacy format conversion, or interfacing with tools that expect three-letter residue codes. The conversion follows the IUPAC extended mapping (including ambiguous and rare codes B -> Asx, J -> Xle, X -> Xaa, U -> Sel, O -> Pyl) and, by default, maps the stop/termination character '*' to 'Ter'. The implementation builds an internal mapping from the IUPAC standard and an optional user-supplied custom_map without mutating the global IUPAC mapping.
    
    Args:
        seq (str or Seq or MutableSeq): Input protein sequence to convert. This should be an iterable of one-letter amino acid codes (for example a Python string like "MAIVMGRWKGAR*", or a Bio.Seq.Seq / Bio.Seq.MutableSeq object). Each element is looked up as a key in the three-letter mapping. Typical practical use is converting standard protein sequences for display or downstream processing. If seq is not iterable (for example None), a TypeError will be raised by the iteration operation; if seq contains unexpected element types the behaviour depends on whether those elements can be used as dict keys and converted to strings for joining.
        custom_map (dict): Optional mapping of one-letter residue characters to replacement three-letter strings. If None (the default), custom_map is set to {"*": "Ter"} so that the termination character '*' maps to "Ter". When provided, the function constructs the internal mapping by combining the IUPAC mapping with the items from custom_map appended afterward; entries in custom_map therefore override the IUPAC mapping for matching keys. Keys in custom_map should be single-character strings corresponding to the one-letter codes to override (for example {"*": "***"} to change the terminator output). Values in custom_map must be strings (three-letter codes or any replacement strings); if they are not strings, the final join will raise a TypeError.
        undef_code (str): String to use for any unknown or undefined one-letter characters that are not present in the IUPAC mapping or custom_map. Defaults to "Xaa". This includes common gap characters such as '-' which, by this implementation, will be translated to undef_code unless explicitly provided in custom_map. The value must be a string because it is concatenated into the returned sequence; non-string values will cause a TypeError during the join operation.
    
    Returns:
        str: A single string formed by concatenating the mapped three-letter codes for each residue in the input sequence, in the same order as the input. The returned string contains no separators between three-letter codes (for example "MetAlaIle..."). Unknown characters are replaced by the value of undef_code. No global state is modified: the function does not update the IUPACData.protein_letters_1to3_extended mapping in-place. Possible failure modes include TypeError when seq is not iterable, when custom_map values or undef_code are not strings (causing the join to fail), or other exceptions raised by ill-formed input types. This function is inspired by BioPerl's seq3 and intended for use in Biopython-based bioinformatics code.
    """
    from Bio.SeqUtils import seq3
    return seq3(seq, custom_map, undef_code)


################################################################################
# Source: Bio.SeqUtils.six_frame_translations
# File: Bio/SeqUtils/__init__.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_SeqUtils_six_frame_translations(seq: str, genetic_code: int = 1):
    """Return a formatted, human-readable string that shows the six-frame translations of a nucleotide sequence together with nucleotide counts and GC content.
    
    This function is part of Bio.SeqUtils in Biopython and is intended for use in computational molecular biology and bioinformatics workflows where a quick, printable overview of all six translation frames (three forward frames and three reverse/complement frames) and basic sequence composition (A, T, G, C counts and GC fraction) is useful. The visual layout and behavior were adapted from xbbtools and are similar to the DNA Strider six-frame translation output: a header with nucleotide counts and a short sequence summary, then blocks of up to 60 nucleotides with corresponding translated amino acids for each frame. The function detects RNA input by checking for the presence of "u" (case-insensitive) in the sequence and uses an RNA-specific reverse-complement routine when appropriate. It delegates codon translation to the translate() implementation used elsewhere in Biopython and computes GC fraction via gc_fraction(..., ambiguous="ignore").
    
    Args:
        seq (str): The input nucleotide sequence to analyze. This should be a Python string containing the sequence characters (for example, "ATGCGT..." or an RNA sequence containing "U"). The function uses seq.lower() to detect RNA ("u" presence) and uses seq.count(nt.upper()) to compute the header counts for "A", "T", "G", and "C" (note: counts are computed by counting the uppercase letters in the provided string using nt.upper(), so providing an all-lowercase sequence will result in zero counts in the header unless the sequence contains uppercase letters). The sequence is sliced into 60-nt windows for display and into codon-aligned fragments for translation. If the sequence length is greater than 20, a shortened summary with the first 10 and last 10 bases is shown in the header; otherwise the full sequence is shown.
        genetic_code (int): An integer specifying the genetic code (translation table) to use for codon-to-amino-acid translation. This corresponds to the NCBI translation table identifier used by Biopython's translate implementation; the default value is 1 (the standard genetic code). Invalid or unsupported genetic_code values may cause the underlying translate() function to raise an exception.
    
    Behavior, defaults, side effects, and failure modes:
        The function returns a single multi-line string and does not modify any external state or perform file I/O. It performs the following concrete steps (matching the implementation in the source):
        - Detects whether the input appears to be RNA by evaluating "u" in seq.lower(); if true, it calls reverse_complement_rna(seq) to compute the reverse complement; otherwise it calls reverse_complement(seq). The chosen reverse-complement sequence is stored in the variable anti.
        - Computes comp = anti[::-1]; this is the reverse of the reverse-complement sequence produced above and is used for printing the complementary sequence lines aligned with the forward orientation.
        - Computes length = len(seq).
        - For each frame offset i in 0, 1, 2:
            - Computes fragment_length = 3 * ((length - i) // 3) to ensure only complete codons are translated.
            - Builds frames[i+1] by translating seq[i : i + fragment_length] with the provided genetic_code.
            - Builds frames[-(i+1)] by translating anti[i : i + fragment_length] with the provided genetic_code and then reversing that translation string with [::-1] so that the reverse-frame amino acids align with the forward orientation in the printed output.
          As a result, frames uses integer keys 1, 2, 3 for the three forward frames and -1, -2, -3 for the three reverse/complement frames.
        - Constructs a header string named "GC_Frame:" and appends counts for nt in ["a", "t", "g", "c"] using seq.count(nt.upper()). Because the implementation counts uppercase letters matching nt.upper(), users should be aware of case when interpreting these counts.
        - Computes gc = 100 * gc_fraction(seq, ambiguous="ignore") and appends a "Sequence: ..." line to the header with the short sequence (lowercased), the total number of nucleotides, and the GC fraction formatted to two decimal places (e.g., "54.17 %GC").
        - For output formatting, the function iterates over the sequence in 60-nt windows. For each window it:
            - Uses p = i // 3 to determine the codon index offset for slicing translated frames.
            - Prints three forward translated rows (frames[3], frames[2], frames[1]) with spacing to align amino acids over codons, then the subsequence lowercased with an integer percent (int(gc)) at the end of the line, then the corresponding comp (complement/reversed) subsequence lowercased, and finally the three reverse-frame translation rows (frames[-2], frames[-1], frames[-3]) aligned below.
          The header GC percentage shows two decimal places, while the per-60-nt lines append an integer percent computed by int(gc).
        Failure modes include the underlying translate() or gc_fraction() functions raising exceptions for invalid characters, unsupported genetic_code values, or other issues. Because nucleotide counting in the header uses seq.count(nt.upper()), passing sequences entirely in lowercase will lead to zero counts in that header unless the sequence contains uppercase letters. The function expects a string input and will raise the usual Python TypeError if passed non-string types that do not support the used string methods and slicing.
    
    Returns:
        str: A formatted, multi-line ASCII string containing:
            - A header starting with "GC_Frame:" followed by nucleotide counts for A, T, G and C computed using seq.count(nt.upper()), and a "Sequence: ..." line with a short lowercased sequence summary (first 10 and last 10 bases if length > 20), total nucleotide count, and GC fraction formatted with two decimal places.
            - One or more blocks covering the sequence in 60-nt windows. Each block contains three forward translation rows (frames 3, 2, 1), the lowercased subsequence and an integer GC percent, the lowercased complementary/reversed subsequence used for alignment, and three reverse-frame translation rows (frames -2, -1, -3). Forward frames are keyed 1, 2, 3 and reverse frames are keyed -1, -2, -3 in the implementation; each translation row displays amino acids derived from translate() using the provided genetic_code.
        The returned string is suitable for printing or logging to provide a quick visual inspection of all six translation frames and basic composition metrics.
    """
    from Bio.SeqUtils import six_frame_translations
    return six_frame_translations(seq, genetic_code)


################################################################################
# Source: Bio.SeqUtils.xGC_skew
# File: Bio/SeqUtils/__init__.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_SeqUtils_xGC_skew(
    seq: str,
    window: int = 1000,
    zoom: int = 100,
    r: int = 300,
    px: int = 100,
    py: int = 100
):
    """Bio.SeqUtils.xGC_skew calculates and displays both the per-window GC skew and the accumulated GC skew for a nucleotide sequence using a Tkinter graphical canvas. This function is part of Biopython's SeqUtils utilities for computational molecular biology and is intended as an interactive graphical diagnostic: it computes GC content and GC skew over sliding windows (using the helper functions GC_skew and gc_fraction called in the source) and renders a circular plot with radial lines for each window and an accumulated-skew trace useful for visual interpretation (for example, detecting shifts associated with replication origins in bacterial genomes).
    
    Args:
        seq (str): Nucleotide sequence to analyze. This is the full sequence string used by GC_skew(seq, window) and gc_fraction(seq) to compute per-window GC skew values and overall GC fraction. The sequence is not modified by this function; it is read-only input used to compute numerical values that determine graphical radii and labels.
        window (int): Window size in nucleotides for computing each GC skew value. Each step in the loop calls GC_skew(seq, window) to obtain a GC skew value computed over segments of this many bases. The default is 1000, matching the implementation's default and affecting the angular spacing and number of drawn radial segments.
        zoom (int): Scaling factor applied to individual GC skew values when mapping skew magnitude to radial displacement from the base circle (i.e., r2 = r1 - gc * zoom in the source). The default is 100 and controls how far inward or outward each per-window GC skew line is drawn relative to the circle of radius r.
        r (int): Base radius (in canvas units/pixels) of the primary circular guide drawn on the Tkinter canvas. The code uses r as the starting radius for plotting both the per-window GC skew and a separate accumulated-skew trace (with a small offset). The default is 300 as in the implementation and affects the overall size of the plotted circle within the window.
        px (int): Horizontal padding (in pixels) added to the canvas origin for placing the circle and text (used to compute the circle center X0 = r + px). The default is 100 as in the source and shifts the entire plot horizontally within the displayed Tkinter window.
        py (int): Vertical padding (in pixels) added to the canvas origin for placing the circle and text (used to compute the circle center Y0 = r + py). The default is 100 as in the source and shifts the entire plot vertically within the displayed Tkinter window.
    
    Behavior and side effects:
        This function creates a Tkinter graphical window and a Canvas with attached vertical and horizontal scrollbars, sets the toplevel window geometry to "700x700", and packs and updates the canvas. It writes textual labels showing the sequence ends and length and the overall GC fraction (via gc_fraction(seq)), draws a base circle (oval) of radius r, then iterates over the values produced by GC_skew(seq, window). For each GC skew value it:
        - computes an angle from the current start position along the sequence,
        - draws a blue radial line representing the per-window GC skew (scaled by zoom),
        - updates an accumulated skew value and draws a magenta radial line representing the accumulated GC skew (offset slightly from the main radius),
        - calls canvas.update() to render the additions progressively,
        and finally sets the canvas scrollregion to encompass all drawn items.
        Because it directly controls GUI elements, it has visible side effects (an interactive window and drawing operations) and does not return any data.
    
    Dependencies and assumptions:
        The plotting relies on GC_skew(seq, window) to yield per-window skew values and gc_fraction(seq) to compute overall GC content; it also uses trigonometric functions (pi, sin, cos) to convert sequence position to angular coordinates. The function assumes a graphical environment is available for Tkinter; it will raise a Tkinter/TclError or fail to display if run in a headless environment without an X server or equivalent GUI support.
    
    Defaults:
        Default parameter values are those in the function signature: window=1000, zoom=100, r=300, px=100, py=100. These defaults were chosen in the implementation to produce a 700x700 window with a circle of radius 300 and comfortable paddings.
    
    Failure modes and warnings:
        The function performs no explicit validation of sequence contents or numeric parameter ranges in the source; invalid or unexpected characters in seq will affect the results produced by the helper functions GC_skew and gc_fraction. Running in an environment without GUI support will typically raise a TclError from Tkinter. Because the function performs drawing and calls canvas.update() frequently, very large sequences (resulting in many windows) may cause slow rendering or high memory use in the Tkinter canvas.
    
    Returns:
        None: This function does not return a value. Its purpose is the side effect of creating and updating a Tkinter window and canvas to visualize GC skew metrics for the provided nucleotide sequence.
    """
    from Bio.SeqUtils import xGC_skew
    return xGC_skew(seq, window, zoom, r, px, py)


################################################################################
# Source: Bio.SeqUtils.CheckSum.crc64
# File: Bio/SeqUtils/CheckSum.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_SeqUtils_CheckSum_crc64(s: str):
    """Return the CRC-64 checksum for a biological sequence as used in Biopython tools for computational molecular biology.
    
    This function computes a 64-bit cyclic redundancy check (CRC-64) over the characters of the input sequence and returns a standardized ASCII label suitable for storing in sequence metadata, file headers, database fields, or for lightweight integrity checks in bioinformatics pipelines. The computation is case-sensitive (the ASCII value of each character is used via ord()) and produces a hexadecimal string prefixed with "CRC-". The function is pure (no side effects) and is intended for non-cryptographic checksum purposes such as detecting accidental corruption or quickly comparing sequences; it is not suitable as a cryptographic hash.
    
    Args:
        s (str or Seq): The input sequence to checksum. In Biopython this is typically a Python str containing IUPAC sequence characters (for example "ACGT..." for DNA) or a Bio.Seq.Seq-like object whose iteration yields single-character strings; the function iterates over s and applies ord() to each character. Case is significant (e.g. "ACGT" and "acgt" yield different checksums). The caller must provide an iterable of single-character strings; providing a non-iterable or elements that are not length-1 strings will raise a TypeError when ord() is called.
    
    Returns:
        str: A string of the form "CRC-<8-hex><8-hex>" where the two 8-hex groups represent the high and low 32-bit words of the computed 64-bit CRC in uppercase hexadecimal, zero-padded to eight characters each (for example "CRC-C4FBB762C4A87EBD"). The returned value is suitable for use in sequence metadata, file headers, or databases. No other side effects occur.
    
    Failure modes:
        The function will raise a TypeError if s is not iterable or if its elements are not single-character strings (because ord() requires a one-character string). Characters outside the ASCII range are processed by their Unicode code point truncated to the lowest 8 bits (the implementation uses ord(c) and masks/indexes using only the low 8 bits), which affects the resulting checksum; this function does not perform any normalization or encoding conversions.
    """
    from Bio.SeqUtils.CheckSum import crc64
    return crc64(s)


################################################################################
# Source: Bio.SeqUtils.CheckSum.gcg
# File: Bio/SeqUtils/CheckSum.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_SeqUtils_CheckSum_gcg(seq: str):
    """Bio.SeqUtils.CheckSum.gcg: Compute the historical GCG checksum for a biological sequence.
    
    Computes the GCG checksum (an integer) for a nucleotide or amino-acid sequence string using the same algorithm historically used by the GCG program and implemented in BioPerl (this function is an adaptation used in Biopython). This function is useful in computational molecular biology and bioinformatics workflows that need to reproduce or verify legacy GCG checksums stored in older files or interoperating tools. The algorithm processes the sequence in a single pass, is case-insensitive (all characters are converted to uppercase internally), and is deterministic: the same input string always yields the same integer result. The current function signature expects seq to be a Python str; if you have a Seq object (from Bio.Seq) convert it to str before calling this function.
    
    Behavior details: starting with index = 0 and checksum = 0, the function iterates over the characters of seq. For each character, index is incremented by 1 and checksum is increased by index * ord(character_uppercase). When index reaches 57 it is reset to 0 for the next character (so the multiplicative index cycles with period 57). After processing the entire sequence, the function returns checksum % 10000 (an integer in the range 0 to 9999). All characters are converted to uppercase prior to calling ord(), so ASCII letters differing only by case do not change the result. The function performs no I/O and has no side effects; it works in linear time O(n) with respect to the sequence length and uses constant extra memory.
    
    Failure modes and edge cases: seq must be a str. Passing a non-str object that does not behave like a string (for example, lacks an upper() method or is not iterable over single-character strings) will raise an AttributeError or TypeError when iterating or calling upper(). An empty string returns 0. Non-ASCII characters are processed with their Unicode code point values via ord(); this will affect the checksum but does not raise an error by itself. The function does not validate biological alphabet characters (A/C/G/T/N, amino-acid letters); any characters present in seq contribute according to their uppercase Unicode code point.
    
    Args:
        seq (str): The input sequence to checksum. In the domain of computational molecular biology this is intended to be a nucleotide or amino-acid sequence represented as a Python string. The function converts this string to uppercase internally and computes a legacy GCG checksum over its characters. If you have a Bio.Seq.Seq object, convert it to a str before calling this function.
    
    Returns:
        int: The GCG checksum for seq as an integer. This is the accumulated weighted sum (with index cycling every 57 positions) taken modulo 10000, and therefore will be in the range 0..9999.
    """
    from Bio.SeqUtils.CheckSum import gcg
    return gcg(seq)


################################################################################
# Source: Bio.SeqUtils.CheckSum.seguid
# File: Bio/SeqUtils/CheckSum.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_SeqUtils_CheckSum_seguid(seq: str):
    """Bio.SeqUtils.CheckSum.seguid returns the SEGUID (a Sequence Globally Unique IDentifier) for a biological sequence. The function computes a reproducible identifier for a nucleotide or amino-acid sequence (or any string/Seq-like object) by normalizing case, computing the SHA-1 digest of the normalized sequence bytes, base64-encoding the digest, and removing base64 padding and newlines to produce the final SEGUID string. This SEGUID is used in Biopython and computational molecular biology workflows to create compact, comparable identifiers for sequence deduplication, database indexing, or cross-referencing sequences in publications or tools (see http://bioinformatics.anl.gov/seguid/ and https://doi.org/10.1002/pmic.200600032).
    
    Args:
        seq (str): The input sequence to identify. In practice this is typically a Python string containing a nucleotide or amino-acid sequence (for example, "ACGTACGT" or "MTEYK..."). The implementation also accepts Biopython Seq objects (the code first attempts bytes(seq)) and, failing that, will call seq.encode() on a Python str. The function normalizes the sequence by converting it to uppercase before hashing, so letter case in the input does not affect the resulting SEGUID. Encoding of Python str objects uses the default str.encode() behavior (UTF-8 by default). If seq cannot be converted to bytes (for example, it is neither a str nor a bytes-convertible object), an exception (TypeError or AttributeError) will be raised.
    
    Returns:
        str: The SEGUID string computed from the input sequence. This is the base64 encoding of the SHA-1 digest of the uppercase sequence bytes, with newline characters removed and any trailing base64 padding characters ("=") stripped. The returned value is a compact ASCII string intended for use as a stable, comparable identifier for the given sequence.
    
    Behavior and failure modes:
        The function performs the following steps in order: (1) attempt to obtain a bytes representation via bytes(seq); (2) if that raises TypeError, assume seq is a str and call seq.encode() to get bytes; (3) convert the bytes to uppercase (canonicalizing A/a, C/c, etc.) before hashing; (4) compute SHA-1 digest; (5) base64-encode the digest; (6) remove newline characters from the base64 output and strip trailing "=" padding characters. There are no external side effects (no file, network, or global state changes). If seq is not a str and not bytes-convertible, calling seq.encode() may raise AttributeError; if bytes(seq) raises a TypeError for an unsupported type and seq.encode() is not available, an exception will propagate to the caller. The normalization to uppercase means different input character casing does not change the SEGUID.
    """
    from Bio.SeqUtils.CheckSum import seguid
    return seguid(seq)


################################################################################
# Source: Bio.SeqUtils.MeltingTemp.Tm_GC
# File: Bio/SeqUtils/MeltingTemp.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_SeqUtils_MeltingTemp_Tm_GC(
    seq: str,
    check: bool = True,
    strict: bool = True,
    valueset: int = 7,
    userset: tuple = None,
    Na: float = 50,
    K: float = 0,
    Tris: float = 0,
    Mg: float = 0,
    dNTPs: float = 0,
    saltcorr: int = 0,
    mismatch: bool = True
):
    """Return the estimated melting temperature (Tm) in degrees Celsius for a DNA/RNA primer or oligonucleotide using empirical formulas based on percent GC content and optional salt corrections. This function implements a family of simple, commonly cited Tm approximations (see Marmur & Doty, Wetmur, Primer3Plus, von Ahsen, and QuikChange variants) and is used in Biopython for quick Tm estimates when designing primers, checking primer properties, or as a component of higher-level primer design pipelines described in the Biopython documentation.
    
    Args:
        seq (str): Nucleotide sequence for which to calculate Tm. The sequence is converted to str() on entry. Ambiguous bases are allowed but handled specially: gc_fraction is computed with the "weighted" option (so e.g. "X" counts as 0.5 GC for gc_fraction), and if mismatch is True each "X" is treated as an actual mismatch and reduces the calculated %GC and final Tm by the corresponding fraction. Sequence length (len(seq)) is used as N in the empirical formulas and in mismatch percentage calculations. The function may call the internal _check(seq, "Tm_GC") when check is True to validate/standardize the sequence (see note on check below).
        check (bool): If True (default) perform an internal validity check and possible normalization of seq by calling _check(seq, "Tm_GC"). This step is intended to catch or normalize invalid characters early (see Biopython SeqUtils internal checks). If False the input is used as-is after conversion to str().
        strict (bool): If True (default) raise ValueError when the sequence contains any of the ambiguous bases "K", "M", "N", "R", "Y", "B", "V", "D", "H" (exactly those letters). This enforces unambiguous base usage for calculations where ambiguity is not acceptable. If strict is False these letters are permitted and treated according to the weighted gc_fraction routine.
        valueset (int): Integer selector (default 7) choosing one of several published empirical Tm formula variants. The implementation provides variants 1–8 (see below). The allowed range is 0–8; valueset > 8 raises ValueError. If userset is provided it overrides valueset. Default valueset 7 corresponds to the Primer3Plus style formula used for product Tm.
        userset (tuple): If provided, a tuple of four numeric values (A, B, C, D) overriding the preset values for the empirical formula Tm = A + B*(%GC) - C/N + salt_correction - D*(%mismatch). Userset must be a tuple with exactly four elements; these values are used directly as the constants in the formula. If userset is supplied it takes precedence over valueset. Incorrect length or non-iterable will result in the usual Python unpacking/type error at runtime.
        Na (float): Sodium ion concentration in mM (default 50). Used directly for salt correction when saltcorr specifies a method that uses [Na+]. If any of K, Tris, Mg or dNTPs are non-zero the function will compute a sodium-equivalent concentration (von Ahsen et al., 2001) and use that value for salt correction according to the selected method.
        K (float): Potassium ion concentration in mM (default 0). If non-zero it is included when computing a sodium-equivalent concentration for salt correction (see Na).
        Tris (float): Tris buffer concentration in mM (default 0). If non-zero it contributes to the sodium-equivalent concentration used for salt correction.
        Mg (float): Magnesium ion concentration in mM (default 0). If non-zero it contributes to the sodium-equivalent concentration used for salt correction; the salt correction routine may account for Mg-dNTP interactions when dNTPs is provided.
        dNTPs (float): Total deoxynucleotide triphosphates concentration in mM (default 0). If non-zero it is used together with Mg to compute effective free Mg2+ (per von Ahsen) when performing salt correction.
        saltcorr (int): Integer code selecting the type of salt correction to apply (default 0). saltcorr == 0 or None means no salt correction is applied. Positive integers select among salt correction methods implemented by salt_correction; note that saltcorr == 5 is explicitly not applicable to Tm_GC and will raise ValueError. If saltcorr is non-zero the function will call salt_correction(Na=Na, K=K, Tris=Tris, Mg=Mg, dNTPs=dNTPs, seq=seq, method=saltcorr) and add the returned correction (in degrees Celsius) to the base empirical Tm.
        mismatch (bool): If True (default) every 'X' in seq is treated as a mismatch: first gc_fraction is computed with the "weighted" option (where X counts as 0.5 GC) and then percent_gc is reduced by seq.count("X") * 50.0 / len(seq) before applying the formula; finally the Tm is decreased by D * (seq.count("X") * 100.0 / len(seq)) to account for the percent mismatch. If mismatch is False the function does not apply the explicit mismatch penalty D*(%mismatch) and leaves the weighted gc_fraction value as-is.
    
    Behavior and formulas:
        The general calculation is: Tm = A + B*(%GC) - C/N + salt_correction - D*(%mismatch)
        where A, B, C, D are empirical constants, N is primer length, %GC is the percent of G+C computed by SeqUtils.gc_fraction(seq, "weighted")*100 with additional adjustment if mismatch is True, and salt_correction is added only if saltcorr is non-zero.
        The function provides preset valuesets (A, B, C, D and default saltcorr) corresponding to commonly cited formulas:
        1. A=69.3, B=0.41, C=650, D=1, saltcorr=0 (Marmur & Doty 1962 and variants)
        2. A=81.5, B=0.41, C=675, D=1, saltcorr=0 (QuikChange manufacturer formula)
        3. A=81.5, B=0.41, C=675, D=1, saltcorr=1 (adds 16.6*log[Na+] style correction)
        4. A=81.5, B=0.41, C=500, D=1, saltcorr=2 (Wetmur standard approximation)
        5. A=78.0, B=0.7, C=500, D=1, saltcorr=2 (RNA)
        6. A=67.0, B=0.8, C=500, D=1, saltcorr=2 (RNA/DNA hybrids)
        7. A=81.5, B=0.41, C=600, D=1, saltcorr=1 (Primer3Plus product Tm; default)
        8. A=77.1, B=0.41, C=528, D=1, saltcorr=4 (von Ahsen et al. 2001 recommendation)
        If userset is provided it replaces A, B, C, D and the associated preset saltcorr behavior.
    
    Side effects and internal checks:
        seq is converted to str on entry. If check is True an internal _check(seq, "Tm_GC") call may validate or normalize the sequence (this is an internal Biopython helper; see SeqUtils code). If strict is True and seq contains any of the letters "K", "M", "N", "R", "Y", "B", "V", "D", "H" the function raises ValueError. If saltcorr == 5 a ValueError is raised because method 5 is not applicable in this GC-based Tm routine. If valueset > 8 a ValueError is raised. If userset is provided but does not unpack into four values, Python will raise the usual unpacking/type error.
    
    Failure modes and exceptions:
        Raises ValueError when saltcorr == 5, when strict is True and the sequence contains any of the ambiguous bases "K", "M", "N", "R", "Y", "B", "V", "D", "H", or when valueset > 8. Incorrect userset shape or non-iterable userset will raise the normal Python unpacking/type error at runtime. Other errors can arise from the called helper functions (_check, SeqUtils.gc_fraction, or salt_correction) if inputs are malformed.
    
    Returns:
        float: The estimated melting temperature in degrees Celsius computed by the selected empirical formula with any requested salt correction and mismatch penalty applied.
    """
    from Bio.SeqUtils.MeltingTemp import Tm_GC
    return Tm_GC(
        seq,
        check,
        strict,
        valueset,
        userset,
        Na,
        K,
        Tris,
        Mg,
        dNTPs,
        saltcorr,
        mismatch
    )


################################################################################
# Source: Bio.SeqUtils.MeltingTemp.Tm_NN
# File: Bio/SeqUtils/MeltingTemp.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_SeqUtils_MeltingTemp_Tm_NN(
    seq: str,
    check: bool = True,
    strict: bool = True,
    c_seq: str = None,
    shift: int = 0,
    nn_table: dict = None,
    tmm_table: dict = None,
    imm_table: dict = None,
    de_table: dict = None,
    dnac1: float = 25,
    dnac2: float = 25,
    selfcomp: bool = False,
    Na: float = 50,
    K: float = 0,
    Tris: float = 0,
    Mg: float = 0,
    dNTPs: float = 0,
    saltcorr: int = 5
):
    """Bio.SeqUtils.MeltingTemp.Tm_NN: Calculate the melting temperature (Tm) of a DNA/DNA, RNA/RNA or RNA/DNA duplex using nearest-neighbor (NN) thermodynamic parameters and optional corrections for mismatches, terminal mismatches, dangling ends and salt effects. This function is used in primer and probe design and validation workflows (e.g., PCR primer Tm estimation, oligo/oligo hybridization, RNA/DNA hybridization) by summing enthalpy (ΔH) and entropy (ΔS) contributions from initiation terms, nearest-neighbor stacks, internal mismatches, terminal mismatches and dangling ends, converting concentrations to the equilibrium constant k, applying salt corrections, and returning the predicted Tm in degrees Celsius.
    
    Args:
        seq (str): The primer or probe sequence as a string. For RNA/DNA hybridizations this must be the RNA sequence. The sequence is cast to str internally. If check is True, the sequence is validated by the internal _check() routine (e.g., allowed nucleotide codes); pass check=False to skip validation and any automatic normalization.
        check (bool): If True (default), validate seq and c_seq using the module's _check() function before calculation. Validation ensures the sequences conform to expected nucleotide symbols/format used by the nearest-neighbor tables; disabling validation may be useful for pre-validated inputs but risks undefined behavior if invalid characters are present.
        strict (bool): If True (default), treat missing thermodynamic keys (e.g., an unexpected neighbor pattern, dangling-end key or mismatch key not present in the provided tables) as an error via the internal _key_error() handling, which stops the calculation. If False, missing keys are handled more permissively (the function will attempt to continue and typically issue a warning rather than raising an exception), allowing approximate results when tables are incomplete.
        c_seq (str): Complementary/template sequence in 3'->5' orientation relative to seq (i.e., the target sequence to which the primer/probe anneals). Default=None, in which case the function computes the perfect Watson–Crick complement of seq using Bio.Seq.Seq(seq).complement(). Providing c_seq is required if you want explicit control for mismatch corrections or dangling-end corrections; when mismatches or dangling ends are present and c_seq is given, the appropriate corrections are applied automatically.
        shift (int): Integer shift of seq relative to c_seq to indicate alignment offsets and create dangling ends (default=0). Positive shift inserts leading gaps on seq, negative shift inserts leading gaps on c_seq. The function aligns seq and c_seq using shift, pads with dot placeholders (".") where needed, trims over-dangling positions, and applies dangling-end thermodynamic corrections for single-base overhangs on either end when present.
        nn_table (dict): Nearest-neighbor thermodynamic table mapping neighbor keys (e.g., "AA/TT") and initiation/symmetry keys to [ΔH, ΔS] values. If None, defaults to DNA_NN3 (Allawi & SantaLucia, 1997) for DNA/DNA hybridizations. Users may pass a custom table constructed with the module's maketable method to use alternative published parameter sets (e.g., Breslauer, Sugimoto, SantaLucia) or RNA-specific tables for RNA/RNA or RNA/DNA hybridizations.
        tmm_table (dict): Thermodynamic values for terminal mismatches mapping terminal-mismatch keys to [ΔH, ΔS]. Default is DNA_TMM1 (SantaLucia & Peyret, 2001). Terminal mismatch terms are applied to the 5'/3' ends when detected after aligning and trimming dangling ends.
        imm_table (dict): Thermodynamic values for internal mismatches (including inosine mismatches where provided), mapping internal-mismatch keys to [ΔH, ΔS]. Default is DNA_IMM1 (Allawi & SantaLucia and others). Internal mismatches are consulted during the nearest-neighbor "zipping" step and take precedence over the standard nn_table when present.
        de_table (dict): Thermodynamic values for dangling ends mapping dangling-end keys to [ΔH, ΔS]. Defaults to DNA_DE1 (Bommarito et al., 2000) for DNA; RNA_DE1 (Turner & Mathews, 2010) may be used for RNA. Dangling-end corrections are applied to single-base overhangs at duplex termini identified via shift or length differences and require accurate c_seq to be meaningful.
        dnac1 (float): Concentration of the higher-concentration strand in nanomolar (nM). Typically this is the primer or probe concentration in hybridization/PCR experiments. Default=25. This value is used to compute the effective duplex concentration k (see selfcomp).
        dnac2 (float): Concentration of the lower-concentration strand in nanomolar (nM). In PCR applications this often represents the template concentration and may be set to 0 if negligible; in symmetric oligo/oligo hybridizations set equal to dnac1. Default=25. Note: some external tools (e.g., Primer3Plus) use different conventions; to mimic Primer3Plus total-oligo behaviors divide total oligo concentration accordingly and assign values to dnac1 and dnac2 as described in the module documentation.
        selfcomp (bool): Whether seq is self-complementary (default=False). If True, the function treats the duplex as self-complementary: the effective concentration for the equilibrium constant k is set to dnac1 (converted to M), and the symmetry correction term nn_table["sym"] (ΔH and ΔS) is added to the totals. For non-self-complementary duplexes k is computed as (dnac1 - dnac2/2) converted from nM to M.
        Na (float): Sodium ion concentration used by the salt correction routine; default=50. See the Tm_GC and salt_correction routines referenced in the module for details about units and the influence of different ionic species. Na, K, Tris, Mg and dNTPs are passed to the salt_correction() method to compute a correction term based on the requested salt correction method (saltcorr).
        K (float): Potassium ion concentration used by the salt correction routine; default=0.
        Tris (float): Tris buffer concentration used by the salt correction routine; default=0.
        Mg (float): Magnesium ion concentration used by the salt correction routine; default=0.
        dNTPs (float): dNTP concentration used by the salt correction routine; default=0.
        saltcorr (int): Integer code selecting the salt-correction method (default=5). Behavior implemented in this function:
            0: no salt correction performed (salt_correction() is not called).
            1,2,3,4: call salt_correction(..., method=saltcorr) and add the returned correction value directly (in degrees C) to the computed melting temperature.
            5: call salt_correction(..., method=5) and add the returned correction (an entropy correction) to the cumulative ΔS before computing Tm (this is the module's default).
            6,7: call salt_correction(..., method=saltcorr) and apply the correction via the reciprocal-temperature formula Tm = 1/(1/(Tm+273.15) + corr) - 273.15.
        Note: salt correction requires passing Na/K/Tris/Mg/dNTPs in the expected units used by salt_correction(); consult the module's salt_correction and Tm_GC documentation for precise units and interpretations.
    
    Returns:
        float: The predicted melting temperature (Tm) in degrees Celsius. The returned value is computed from the summed enthalpy (ΔH, in cal/mol) and entropy (ΔS, in cal/(K·mol)) contributions using the formula Tm = (1000 * ΔH) / (ΔS + R * ln(k)) - 273.15 with R = 1.987 cal/(K·mol) and k the effective duplex concentration in mol/L. Depending on saltcorr, additional corrections are applied either to ΔS prior to the Tm calculation, added post hoc to Tm, or applied via the reciprocal-temperature transform. This numeric Tm is intended for use in primer/probe design decisions; users should be aware that experimental conditions (salt, divalent ions, mismatches, concentration accuracy) and the choice of thermodynamic table (nn_table, imm_table, tmm_table, de_table) materially affect the prediction.
    
    Behavior, defaults and failure modes:
        - Table defaults: If nn_table, tmm_table, imm_table or de_table are None, they default to DNA_NN3, DNA_TMM1, DNA_IMM1 and DNA_DE1 respectively (DNA parameter sets). To use RNA or hybrid parameter sets or custom values, supply appropriate dict tables created with the module's maketable.
        - c_seq handling: If c_seq is None the function builds the perfect complement of seq and uses that; however, for accurate mismatch and dangling-end corrections provide the true template sequence in 3'->5' orientation. Mismatches and dangling-end corrections are only meaningful when c_seq correctly represents the intended duplex partner.
        - Dangling ends and shift: shift is used to align seq and c_seq and to reveal single-base dangling ends. The function inserts dot placeholders (".") to align sequences, removes over-dangling positions, and applies single-base dangling-end corrections from de_table where keys exist.
        - Mismatches: Terminal mismatches are detected and matched against tmm_table; internal mismatches are detected during the neighbor iteration and matched against imm_table before falling back to nn_table. If a neighbor key is not found in any provided table, _key_error(neighbors, strict) is invoked—strict=True will typically abort with an error, strict=False will attempt to continue (usually emitting a warning).
        - Concentrations: dnac1 and dnac2 are interpreted as nanomolar (nM) and converted to molar inside the routine. Incorrect concentration units will produce incorrect Tm values.
        - Salt corrections: The function delegates salt corrections to salt_correction(..., method=saltcorr). saltcorr=0 disables salt correction. The default saltcorr=5 modifies ΔS prior to Tm calculation; other methods add directly to Tm or apply a reciprocal-temperature adjustment as described above. Providing Na/K/Tris/Mg/dNTPs in inappropriate units or ranges may produce invalid corrections.
        - Self-complementarity: If selfcomp=True, the function uses dnac1 as the effective concentration (converted to M) and adds the symmetry correction nn_table["sym"] to ΔH/ΔS. For non-self-complementary duplexes the effective concentration is computed as (dnac1 - dnac2/2) converted to M; if this yields non-positive values the ln(k) term will be invalid and a math domain error may occur.
        - Numerical/exceptional conditions: The calculation involves logarithms and divisions; invalid concentrations (e.g., resulting in k <= 0) or missing thermodynamic keys (handled according to strict) can raise exceptions (ValueError, KeyError, or math domain errors). Consumers should validate inputs (e.g., concentrations, sequences, and tables) before calling or use try/except to handle runtime errors.
        - Units and provenance: Thermodynamic tables correspond to published parameter sets (e.g., Allawi & SantaLucia for DNA_NN3) and should be chosen to match the experimental system (DNA/DNA, RNA/RNA, RNA/DNA). The README and module documentation provide references and guidance on selecting and constructing appropriate tables with maketable.
    """
    from Bio.SeqUtils.MeltingTemp import Tm_NN
    return Tm_NN(
        seq,
        check,
        strict,
        c_seq,
        shift,
        nn_table,
        tmm_table,
        imm_table,
        de_table,
        dnac1,
        dnac2,
        selfcomp,
        Na,
        K,
        Tris,
        Mg,
        dNTPs,
        saltcorr
    )


################################################################################
# Source: Bio.SeqUtils.MeltingTemp.Tm_Wallace
# File: Bio/SeqUtils/MeltingTemp.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_SeqUtils_MeltingTemp_Tm_Wallace(seq: str, check: bool = True, strict: bool = True):
    """Bio.SeqUtils.MeltingTemp.Tm_Wallace calculates the melting temperature (Tm) of a DNA oligonucleotide using the Wallace rule, a simple rule-of-thumb estimator commonly used in PCR primer design and other basic oligonucleotide annealing estimates.
    
    Args:
        seq (str): The input nucleic acid sequence for which to estimate Tm. In the Biopython context this is typically a DNA primer sequence string possibly containing whitespace or non-DNA characters; the function treats the sequence as a string and counts nucleotides to compute the Tm. The method ignores non-DNA characters (for example digits or punctuation) when computing the temperature (see examples in the source: spaces and digits do not change the result). When check=True (the default) the sequence is passed to the module's internal _check routine which performs validation and normalization (for example, converting to upper case and removing or handling whitespace) before counting bases; _check may raise an exception for unacceptable input.
        check (bool): If True (default), perform sequence validation/normalization by calling the module's _check function prior to counting bases. This typically normalizes the sequence (e.g., uppercasing, removing or handling whitespace) and may raise a ValueError or other exception for badly formed input. If False, the function uses the seq string as provided (after conversion via str(seq)) and counts characters directly, still ignoring characters that are not counted as A, T, G, C, or recognised ambiguous IUPAC codes used below.
        strict (bool): If True (default), disallow ambiguous IUPAC nucleotide codes B, D, H, K, M, N, R, V, Y in the sequence and raise a ValueError if any of these ambiguous bases are present. If False, ambiguous bases are handled by adding intermediate fractional contributions to the Tm (see detailed behavior below). Use strict=True when you require an exact base composition (typical when assessing a specific primer sequence), and strict=False when you want a rough estimate that accounts for ambiguous positions.
    
    Detailed behavior and practical significance:
        This function implements the Wallace rule: Tm = 4°C * (count of G + count of C) + 2°C * (count of A + count of T). It is intended as a simple, fast approximation of primer melting temperature and is most appropriate as a rule-of-thumb for primers roughly 14–20 nucleotides long, a common length range in PCR primer design. For ambiguous IUPAC codes, when strict is False the function adds intermediate contributions computed in the source code as:
            - add 3.0°C for each occurrence of K, M, N, R, or Y,
            - add 10/3.0°C (≈3.333...°C) for each occurrence of B or V,
            - add 8/3.0°C (≈2.666...°C) for each occurrence of D or H.
        When strict is True (the default) and any of the ambiguous bases B, D, H, K, M, N, R, V, Y are present, the function raises a ValueError with the message "ambiguous bases B, D, H, K, M, N, R, V, Y not allowed when strict=True" (as produced in the source). Non-DNA characters that are not counted as A, T, G, C or the above ambiguous IUPAC codes are ignored in the temperature calculation (for example, whitespace or numeric characters do not affect the computed Tm in typical usage).
    
    Failure modes and side effects:
        The function is pure with respect to external state (it returns a numeric result and does not modify external variables), but calling with check=True delegates to the module's _check implementation which may perform normalization and may raise exceptions (e.g., ValueError) for invalid sequences. If strict=True and ambiguous bases are present, a ValueError is raised (see message above). The numeric result is only an approximation and should not be used as a substitute for more detailed thermodynamic models when precise Tm calculations are required.
    
    Returns:
        float: The estimated melting temperature in degrees Celsius computed using the Wallace rule and the intermediate contributions for ambiguous bases when strict is False. This numeric value is an approximate Tm intended for quick, practical use in workflows such as primer design and should be interpreted accordingly.
    """
    from Bio.SeqUtils.MeltingTemp import Tm_Wallace
    return Tm_Wallace(seq, check, strict)


################################################################################
# Source: Bio.SeqUtils.MeltingTemp.chem_correction
# File: Bio/SeqUtils/MeltingTemp.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_SeqUtils_MeltingTemp_chem_correction(
    melting_temp: float,
    DMSO: float = 0,
    fmd: float = 0,
    DMSOfactor: float = 0.75,
    fmdfactor: float = 0.65,
    fmdmethod: int = 1,
    GC: float = None
):
    """Bio.SeqUtils.MeltingTemp.chem_correction corrects a melting temperature (Tm) estimate for the presence of chemical denaturants DMSO and formamide, producing an adjusted Tm used in oligonucleotide design, PCR primer planning, and nucleic acid hybridization predictions within the Biopython computational molecular biology toolkit.
    
    This function applies simple, literature-based linear corrections to a provided melting temperature. These corrections are approximate and intended to give a pragmatic adjustment to Tm values computed elsewhere in Biopython or supplied by the user. The function does not mutate inputs and returns a new float value. The default numeric factors are those coded into the function; alternative reported values from the literature are noted below for user awareness.
    
    Args:
        melting_temp (float): Input melting temperature to correct. This is the Tm value in whatever temperature units the caller is using (typically degrees Celsius) produced by melting temperature calculations or experimental estimates. The function returns a corrected value in the same units.
        DMSO (float): Percent (by volume) DMSO present in the sample (e.g., 3 for 3% DMSO). If non-zero, the function decreases melting_temp by DMSOfactor * DMSO. A value of 0 means no DMSO correction is applied.
        fmd (float): Formamide concentration. Interpreted according to fmdmethod: when fmdmethod == 1, fmd is percent formamide (e.g., 5 for 5%); when fmdmethod == 2, fmd is molar concentration (e.g., 1.25 for 1.25 M). If fmd is 0, no formamide correction is applied.
        DMSOfactor (float): Per-percent decrease in Tm applied for DMSO (multiplier used as DMSOfactor * DMSO). The function default is 0.75 (as set in the function signature). Literature reports a range of values (e.g., 0.5, 0.6, 0.65 — von Ahsen et al. 2001 — and 0.675); users can override this parameter to reflect their preferred empirical factor for their experimental conditions.
        fmdfactor (float): Per-percent decrease in Tm applied when fmdmethod == 1 (i.e., degrees per percent formamide). The function default is 0.65. Literature reports factors between approximately 0.6 and 0.72; set this parameter to match the empirical correction appropriate for your data if needed.
        fmdmethod (int): Method for applying the formamide correction. Two methods are supported:
            1: Linear percent method (default). The function subtracts fmdfactor * fmd from melting_temp where fmd is percent formamide.
            2: GC-dependent molar method. The function uses the formula Tm += (0.453 * f(GC) - 2.88) * fmd, where f(GC) is the fraction (not percent) of GC (GC / 100.0) and fmd is given in molar. This method is taken from Blake & Delcourt (1996).
        GC (float or None): GC content expressed in percent (e.g., 50 for 50% GC). Required only when fmdmethod == 2. If fmdmethod == 2 and GC is None or negative, the function raises a ValueError. If fmdmethod == 1, GC is ignored and may be left as None.
    
    Returns:
        float: The corrected melting temperature (same numerical units as the melting_temp argument, typically degrees Celsius). If neither DMSO nor formamide corrections are applied (DMSO == 0 and fmd == 0), the original melting_temp value is returned unchanged.
    
    Raises:
        ValueError: If fmdmethod == 2 and GC is None or GC < 0, a ValueError is raised with the message "'GC' is missing or negative" because the GC fraction is required for the GC-dependent molar formamide correction.
        ValueError: If fmd is non-zero and fmdmethod is not 1 or 2, a ValueError is raised with the message "'fmdmethod' must be 1 or 2".
    
    Notes:
        - The DMSO correction applied is melting_temp -= DMSOfactor * DMSO when DMSO is non-zero.
        - For formamide: if fmdmethod == 1 (percent method) the correction is melting_temp -= fmdfactor * fmd (McConaughy et al., 1969). If fmdmethod == 2 (molar, GC-dependent method) the correction is melting_temp += (0.453 * (GC / 100.0) - 2.88) * fmd (Blake & Delcourt, 1996), and GC must be provided.
        - These corrections are approximate and intended to give practical adjustments during primer design and related tasks in computational molecular biology; they are not substitutes for full thermodynamic modelling or experimental measurement.
    """
    from Bio.SeqUtils.MeltingTemp import chem_correction
    return chem_correction(
        melting_temp,
        DMSO,
        fmd,
        DMSOfactor,
        fmdfactor,
        fmdmethod,
        GC
    )


################################################################################
# Source: Bio.SeqUtils.MeltingTemp.make_table
# File: Bio/SeqUtils/MeltingTemp.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_SeqUtils_MeltingTemp_make_table(oldtable: dict = None, values: dict = None):
    """Return a dictionary table of thermodynamic parameters used by DNA melting
    temperature calculations.
    
    This function is used within the Bio.SeqUtils.MeltingTemp module to build or
    customize the lookup table of nearest-neighbor and initiation thermodynamic
    parameters (commonly enthalpy and entropy pairs) employed by melting
    temperature routines. If no existing table is provided, a default table is
    constructed with a set of standard parameter names initialized to (0, 0).
    A user can supply a pre-existing table and/or a dictionary of new or updated
    values to modify that table. This is intended for practical use cases such as
    replacing initiation parameters from one published dataset (for example,
    Sugimoto '96, stored in DNA_NN2) with values from another dataset (for
    example, Allawi & SantaLucia '97), as shown in the original example usage.
    
    Args:
        oldtable (dict): An existing dictionary of thermodynamic parameters to use
            as the starting point. The keys are parameter name strings (for
            example 'AA/TT', 'init_A/T', 'sym') and the values are two-number
            tuples (commonly interpreted as enthalpy and entropy). If None, the
            function builds and returns a fresh default table. The default table
            created when oldtable is None contains these keys initialized to
            (0, 0): "init", "init_A/T", "init_G/C", "init_oneG/C",
            "init_allA/T", "init_5T/A", "sym", "AA/TT", "AT/TA", "TA/AT",
            "CA/GT", "GT/CA", "CT/GA", "GA/CT", "CG/GC", "GC/CG", "GG/CC".
            The function makes a shallow copy of oldtable (via oldtable.copy())
            before applying any updates, so passing a mutable mapping will not
            be mutated by this function (the returned object is a separate
            dictionary). If oldtable is not a mapping with a copy() method, a
            runtime error may be raised when attempting to copy it.
    
        values (dict): A dictionary of parameter updates to apply to the table.
            Keys should be parameter name strings matching those used by the
            MeltingTemp routines (for example 'init_A/T'). Values should be
            two-number tuples consistent with the table's value format (for
            example (enthalpy, entropy) as floats or ints). If values is None or
            empty, no updates are applied. The function applies updates using
            the standard dict.update(values) operation; it does not validate keys
            or the numeric nature of tuple elements. If values is not a mapping
            with an update-compatible interface, a runtime error may be raised.
            Note that because the function only checks truthiness before calling
            update (if values: table.update(values)), an empty dict will result
            in no update call but yields the same final table as updating with an
            empty dict would.
    
    Behavior and side effects:
        The function returns a new dict and does not mutate the original
        oldtable argument (it uses oldtable.copy()). No external files are read
        or written. The function does not perform domain-specific validation of
        parameter names or numeric ranges; it simply constructs or copies a
        mapping and applies dict.update(values) when requested. Downstream code
        that consumes the returned table expects numeric two-element tuples for
        thermodynamic calculations; providing non-numeric or incorrectly shaped
        values will not be caught here and may cause errors later during
        calculations.
    
    Failure modes:
        If oldtable or values are not mapping-like objects supporting the used
        methods (copy and update respectively), AttributeError or TypeError may
        be raised at runtime. Incorrectly shaped values (for example not a
        two-number tuple) are not validated and can cause calculation errors in
        consumers of the table.
    
    Returns:
        dict: A dictionary mapping parameter name strings to two-number tuples
        (typically enthalpy and entropy values) representing thermodynamic
        parameters for DNA melting calculations. The returned dictionary is a new
        object (a shallow copy of oldtable with values applied, or a newly
        constructed default table if oldtable was None).
    """
    from Bio.SeqUtils.MeltingTemp import make_table
    return make_table(oldtable, values)


################################################################################
# Source: Bio.SeqUtils.MeltingTemp.salt_correction
# File: Bio/SeqUtils/MeltingTemp.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_SeqUtils_MeltingTemp_salt_correction(
    Na: float = 0,
    K: float = 0,
    Tris: float = 0,
    Mg: float = 0,
    dNTPs: float = 0,
    method: int = 1,
    seq: str = None
):
    """Calculate a term to correct nucleic acid melting temperature (Tm) or entropy
    for the ionic environment. This function computes a scalar correction term
    based on supplied millimolar concentrations of common ions (Na+, K+, Tris,
    Mg2+) and dNTPs, and on a selected empirical method (1-7) drawn from the
    literature (Schildkraut & Lifson 1965; Wetmur 1991; SantaLucia 1996/1998;
    Owczarzy 2004/2008, and von Ahsen 2001 for Na-equivalent). The computed
    correction is intended to be applied to a previously calculated Tm or to
    deltaS according to the method semantics described below; the function does
    not itself compute Tm or deltaS, only the ionic correction term.
    
    Args:
        Na (float): Millimolar concentration of sodium ions [Na+]. This is the
            primary input for a simple salt correction: pass only Na to apply a
            sodium-only correction. Units are millimolar (mM). Typical practical
            use: buffer [Na+] expressed in mM derived from experimental conditions.
        K (float): Millimolar concentration of potassium ions [K+]. When non-zero,
            K contributes to a sodium-equivalent concentration according to the
            von Ahsen et al. (2001) prescription used here: it is added directly
            (in mM) to the effective monovalent contribution. Defaults to 0.0 mM.
        Tris (float): Millimolar concentration of Tris base (Tris buffer). In the
            sodium-equivalent calculation Tris contributes half its molar amount
            (Tris/2) because Tris is a weak base that partially provides monovalent
            cations. Units are millimolar (mM). Defaults to 0.0 mM.
        Mg (float): Millimolar concentration of magnesium ions [Mg2+]. Mg2+ is a
            divalent cation and affects salt correction differently: it is used
            to compute a sodium-equivalent term 120*sqrt([Mg2+] - [dNTPs]) (mM)
            when appropriate (see behavior). Mg is also converted internally to
            molar units (Mg * 1e-3) for formulae that require molar concentrations.
            Defaults to 0.0 mM.
        dNTPs (float): Millimolar total concentration of dNTPs (deoxynucleotide
            triphosphates). dNTPs strongly bind Mg2+ and therefore reduce free
            Mg2+. When dNTPs >= Mg the code treats free Mg2+ as negligible in the
            sodium-equivalent calculation. Units are millimolar (mM). Defaults to
            0.0 mM.
        method (int): Integer selecting the empirical correction method to apply.
            Valid values are 0 and 1-7. Default is 1. Semantic effect of the
            returned correction depends on method:
            - method == 0: No correction; the function returns 0.0 (useful to
              disable salt correction programmatically).
            - methods 1-4: The returned correction corr is intended to be added to
              an existing Tm value: Tm(new) = Tm(old) + corr. These methods use
              simple log([Na+]) or modified log formulae (units consistent with
              Tm scales used by the cited authors).
            - method 5: The returned correction is an entropy correction (deltaS).
              It should be added to an existing deltaS: deltaS(new) = deltaS(old)
              + corr. Formula is proportional to (N-1)*ln[Na+] where N is sequence
              length.
            - methods 6 and 7: The returned correction is for the reciprocal of
              Tm; the corrected Tm is computed as Tm(new) = 1/(1/Tm(old) + corr).
            Detailed definitions and literature sources:
            1: 16.6 * log10([Na+]) (Schildkraut & Lifson 1965)
            2: 16.6 * log10([Na+]/(1 + 0.7*[Na+])) (Wetmur 1991)
            3: 12.5 * log10([Na+]) (SantaLucia et al. 1996)
            4: 11.7 * log10([Na+]) (SantaLucia 1998)
            5: 0.368 * (N - 1) * ln([Na+]) (SantaLucia 1998) — entropy correction
            6: (4.29*%GC - 3.95)*1e-5*ln([Na+]) + 9.40e-6*ln([Na+])^2
               (Owczarzy et al. 2004) — correction applied to 1/Tm
            7: Complex empirical formula with decision tree and empirical
               constants, includes explicit Mg2+ correction for dNTP binding and
               additional regimes based on the ratio sqrt([Mg2+])/[Na_eq]
               (Owczarzy et al. 2008). Note: method 7 applies a specific Mg2+
               treatment and a dissociation constant for Mg:dNTP interactions.
            Use this parameter to select which empirical model best matches your
            experimental protocol or literature source.
        seq (str): DNA sequence string used only for methods that require
            sequence-dependent quantities: methods 5, 6 and 7. For method 5 the
            sequence length (len(seq)) is used; for methods 6 and 7 the GC
            fraction (percentage GC) is used via SeqUtils.gc_fraction(seq,
            "ignore"). If seq is required by the selected method but is None or
            empty, a ValueError is raised. The function does not validate or
            canonicalize sequence letters beyond what SeqUtils.gc_fraction
            performs; pass the nucleotide string you used when computing Tm or
            deltaS.
    
    Behavior, defaults, and failure modes:
        - All ionic concentration arguments (Na, K, Tris, Mg, dNTPs) are expected
          in millimolar (mM). Internally, a monovalent-equivalent concentration
          Mon (in mM) is formed as Mon = Na + K + Tris/2. If any of K, Mg, Tris or
          dNTPs are non-zero and method != 7 and dNTPs < Mg, an additional term
          120 * sqrt(Mg - dNTPs) (mM) is added to Mon to account for Mg2+'
          contribution to effective monovalent ionic strength (von Ahsen et al.
          2001). After this, Mon is converted to molar units mon = Mon * 1e-3 for
          logarithmic formulae.
        - For method 7, if dNTPs > 0 the function computes free Mg2+ by solving a
          quadratic expression that models Mg:dNTP binding using an empirical
          dissociation constant (ka = 3e4). Method 7 then follows a decision tree
          based on the ratio R = sqrt(mg)/mon to pick the appropriate empirical
          form. Method 7 returns the correction in the form used by Owczarzy et
          al. (2008).
        - If method in (5, 6, 7) and seq is not provided, the function raises
          ValueError("sequence is missing (is needed to calculate GC content or sequence length).").
        - If method is in range(1,7) (i.e., 1..6) and the computed mon (molar
          total ion concentration) is zero, the function raises
          ValueError("Total ion concentration of zero is not allowed in this method.").
        - If method > 7 the function raises ValueError("Allowed values for parameter 'method' are 1-7.").
        - If method == 0 the function returns 0.0 (no ionic correction).
        - The function is pure (no side effects, does not modify global state or
          I/O). It relies on math and SeqUtils.gc_fraction for GC calculation.
        - The sign and magnitude of the returned correction follow the
          literature formulae; negative values typically correspond to Tm
          reductions under the given ionic conditions when the correction is
          added to an existing Tm as specified for the selected method.
    
    Returns:
        float: The computed scalar correction term. Its practical interpretation
        depends on the chosen method: for methods 1-4 it is an additive Tm
        correction (apply as Tm(new) = Tm(old) + corr); for method 5 it is an
        additive entropy (deltaS) correction (apply as deltaS(new) =
        deltaS(old) + corr); for methods 6 and 7 it is a correction of the
        reciprocal of Tm (apply as Tm(new) = 1/(1/Tm(old) + corr)). For method 0
        the function returns 0.0. The numeric value is returned as a Python float
        and carries the units implied by the selected literature method (e.g.,
        degrees for Tm corrections or entropy units where specified by the
        original publications).
    """
    from Bio.SeqUtils.MeltingTemp import salt_correction
    return salt_correction(Na, K, Tris, Mg, dNTPs, method, seq)


################################################################################
# Source: Bio.SeqUtils.lcc.lcc_mult
# File: Bio/SeqUtils/lcc.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_SeqUtils_lcc_lcc_mult(seq: str, wsize: int):
    """Calculate Local Composition Complexity (LCC) values over a sliding window for an unambiguous DNA sequence.
    
    This function implements an optimized, incremental computation of the Local Composition Complexity (LCC) over a sequence using a fixed-size sliding window. LCC is a per-window measure of nucleotide composition heterogeneity based on the Shannon entropy of nucleotide frequencies normalized by log(4) (natural logarithm), producing values in the range [0.0, 1.0]. A value of 0.0 indicates a window composed of a single nucleotide (minimum complexity); values near 1.0 indicate maximal composition diversity (approximately equal counts of A, C, G, T). This implementation is used in the Biopython project for genomic sequence analysis to detect low-complexity regions and survey local sequence composition. It is optimized relative to a naive recomputation by updating counts and entropy terms incrementally when the window slides by one base.
    
    Args:
        seq (str): The input DNA sequence as a Python string. semantically, this should be an unambiguous DNA sequence composed of the nucleotide letters A, C, G, and T (case-insensitive). The function converts the sequence to upper case internally. This parameter represents the biological nucleotide sequence to be scanned; providing non-ACGT characters may produce incorrect or undefined LCC values because the algorithm only updates counts for A, C, G and T.
        wsize (int): Window size as a positive integer. This controls the length of the sliding window used to compute each LCC value and thus the spatial resolution of the complexity measure. Typical usage provides 1 <= wsize <= len(seq). The function precomputes per-count entropy contributions for indices from 0 up to wsize and uses these to update the LCC efficiently as the window moves by one base.
    
    Returns:
        list of float: A list containing the LCC value for each sliding window position. The list length is len(seq) - wsize + 1 when 1 <= wsize <= len(seq). Each float is the normalized negative sum of per-nucleotide terms: -sum(p_i * ln(p_i)) / ln(4) computed from the nucleotide frequencies in that window, where p_i are the frequencies of A, C, G and T in the window. Values are in the closed interval [0.0, 1.0].
    
    Behavior, side effects, defaults, and failure modes:
        - The function is pure (no external side effects) and converts the provided seq to upper case internally before processing.
        - Performance: This variant is optimized for speed by maintaining nucleotide counts for A, C, G, T and updating only the terms that change when the window slides; it avoids recomputing counts and logarithms for the entire window at every step.
        - Input expectations: seq should be a non-empty string containing the DNA bases A, C, G, T (case-insensitive). wsize should be an integer >= 1. For meaningful sliding-window output, wsize should be <= len(seq).
        - Edge cases and failures: If seq is empty, or wsize < 1, the function may raise an IndexError (for example when accessing seq[0]) or otherwise behave unpredictably; callers should validate inputs before calling. If seq contains characters other than A, C, G, T, the function will ignore those as distinct symbols and will not update internal A/C/G/T counters for them; this can lead to misleading LCC values. The implementation does not raise ValueError itself for invalid ranges of wsize; callers should enforce valid ranges to avoid incorrect results.
        - Output size: The returned list size equals the number of window positions (len(seq) - wsize + 1) when wsize is between 1 and len(seq) inclusive. If wsize equals len(seq) the function returns a single LCC value for the entire sequence.
        - Numerical details: Natural logarithms (math.log) are used, and results are normalized by math.log(4) so the maximum possible LCC is 1.0 for a window with approximately equal counts of A, C, G and T.
    """
    from Bio.SeqUtils.lcc import lcc_mult
    return lcc_mult(seq, wsize)


################################################################################
# Source: Bio.SeqUtils.lcc.lcc_simp
# File: Bio/SeqUtils/lcc.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_SeqUtils_lcc_lcc_simp(seq: str):
    """Bio.SeqUtils.lcc.lcc_simp calculates the Local Composition Complexity (LCC) for a DNA sequence using a normalized Shannon-entropy style measure (log base 4) as described by Konopka (2005). This function is intended for use in computational molecular biology and bioinformatics workflows (for example, when analyzing nucleotide composition bias or segmenting genomic sequences) and returns a single scalar complexity value representing the overall base composition diversity of the provided sequence.
    
    Args:
        seq (str or Seq): An unambiguous DNA sequence to analyse. The input may be a Python string (recommended) or a Biopython Seq-like object that supports len(), upper(), and count() semantics. The sequence is treated as nucleotide characters; the function converts the sequence to upper case internally before counting bases. Practical significance: callers should provide the full sequence whose global LCC is required (for instance, a genomic region, gene, or oligonucleotide). Ambiguous or non-ACGT characters are allowed but will affect results (see behaviour and failure modes).
    
    Returns:
        float: The Local Composition Complexity (LCC) value for the entire sequence computed as the negative sum over bases A, C, T, G of (p_b * log(p_b) / log(4)), where p_b is the frequency of base b in the sequence (counts divided by sequence length). This normalized entropy-like value quantifies nucleotide composition diversity: lower values indicate more biased composition (e.g., dominated by one base), while higher values indicate more even composition across the four canonical bases. For sequences consisting only of the four canonical bases the value ranges from 0.0 (no composition complexity) up to a maximum associated with uniform base frequencies; callers should interpret the returned float as a normalized entropy measure rather than an absolute probability.
    
    Behavior, side effects, defaults, and failure modes:
        The function computes wsize = len(seq) and then seq = seq.upper() locally; it does not modify the caller's object (no external side effects). Base counts are obtained with seq.count("A"), seq.count("C"), seq.count("T"), and seq.count("G"); each base term is set to zero when that base is absent to avoid taking log(0). Because wsize (the denominator) is the total length of the provided sequence, any ambiguous or non-ACGT characters present in seq reduce the computed frequencies of A/C/T/G (they remain counted in wsize but not in base counts), which can lower the computed LCC compared to computing frequencies over only canonical bases—this may produce unintuitive results if ambiguous characters are present and callers expect them to be ignored. Performance: the implementation calls count() four times and uses math.log, so runtime scales linearly with sequence length with a small constant factor. Failure modes: passing an empty sequence (len(seq) == 0) will raise a ZeroDivisionError due to division by wsize; callers should validate non-empty input prior to calling. Invalid types that do not support len(), upper(), or count() will raise the corresponding TypeError or AttributeError. Reference: Andrzej K. Konopka (2005) Sequence Complexity and Composition, https://doi.org/10.1038/npg.els.0005260.
    """
    from Bio.SeqUtils.lcc import lcc_simp
    return lcc_simp(seq)


################################################################################
# Source: Bio.TogoWS.entry
# File: Bio/TogoWS/__init__.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_TogoWS_entry(db: str, id: list, format: str = None, field: str = None):
    """Bio.TogoWS.entry fetches a single or multiple records from the TogoWS "entry" endpoint for molecular biology resources (for example NCBI, KEGG, DDBj, EBI databases). It validates the requested database, optional field extraction and output format against cached metadata (populated on first use), constructs the proper TogoWS URL (quoting and joining identifiers as required), and returns the opened network resource produced by the module's internal _open helper. This function is used within Biopython to retrieve sequence, annotation, and other bioinformatics records programmatically (similar in purpose to NCBI Entrez EFetch, but with optional per-record field extraction supported by TogoWS).
    
    Args:
        db (str): The target database name as expected by TogoWS for the "entry" service. Practical examples include "pubmed", "nucleotide", "protein", "uniprot", "compound" (KEGG), "ddbj", etc. The function first ensures this name is among the supported databases returned by the internal cache _entry_db_names (fetched from TogoWS on demand). If the database is not supported, a ValueError is raised. The database parameter determines which record namespace, available fields, and output formats will be validated and used when constructing the request URL.
        id (list or str): One or more record identifiers for the target database. This may be a single identifier string or a list of identifier strings; if a list is supplied the identifiers are joined with commas to form a single TogoWS request. The identifier(s) are URL-quoted before being appended to the constructed entry URL. Typical identifiers are accession numbers, database-specific IDs, or PubMed IDs depending on the chosen db.
        format (str, optional): Desired output format for the returned record, for example "xml", "json", "gff", "fasta", or "ttl" (RDF Turtle), where supported by the chosen db. When provided, the function validates this format against the formats advertised by the TogoWS entry metadata for the chosen database (cached in _entry_db_formats and fetched on demand via _get_entry_formats). If the format is not supported for that database, a ValueError is raised. If omitted or None, the default format as provided by the TogoWS service for that database is used.
        field (str, optional): A database-specific field name to extract from within the returned record (for example "au" or "authors" for PubMed records). When provided, the function validates the field against the fields advertised by the TogoWS entry metadata for the chosen database (cached in _entry_db_fields and fetched on demand via _get_entry_fields). For backward compatibility, a call requesting db="pubmed" and field="ti" is translated to "title" and emits a warnings.warn message instructing callers to use "title" instead. If the field is not supported for that database, a ValueError is raised. If omitted or None, the entire record is requested.
    
    Returns:
        object: The value returned by the internal helper _open(url) after performing the HTTP request to the constructed TogoWS entry URL. This is the network resource returned by the module (typically a response/file-like object produced by the module's URL-open helper); callers should use the appropriate API provided by that object (for example read()) to obtain the raw bytes or text content. The exact runtime type and behavior of the returned object are determined by Bio.TogoWS._open.
    
    Behavior, side effects, defaults, and failure modes:
        This function performs a network request to the TogoWS entry endpoint constructed as _BASE_URL + "/entry/{db}/{quoted_id}" with optional "/{field}" and ".{format}" suffixes when those arguments are provided. On first use it populates and caches database names, available fields, and available formats by calling _get_entry_dbs, _get_entry_fields, and _get_entry_formats as needed; these caches are stored in the module-level variables _entry_db_names, _entry_db_fields, and _entry_db_formats. Passing id as a list causes the list elements to be joined with commas before quoting; passing id as a string is accepted as-is. If db, field, or format are not supported according to the remote metadata, the function raises ValueError with a descriptive message listing supported values when available. For the special case of db="pubmed" and field="ti", the function substitutes "title" and emits a runtime warning to preserve backwards compatibility with older code. The function may raise network-related exceptions or propagate exceptions raised by _open if the HTTP request fails, and callers should handle such exceptions as appropriate. The function does not modify persistent state beyond populating the module-level caches.
    """
    from Bio.TogoWS import entry
    return entry(db, id, format, field)


################################################################################
# Source: Bio.TogoWS.search
# File: Bio/TogoWS/__init__.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_TogoWS_search(
    db: str,
    query: str,
    offset: int = None,
    limit: int = None,
    format: str = None
):
    """Bio.TogoWS.search: Low-level wrapper to perform a text/web search against the TogoWS search API for biological databases.
    
    Performs a single search request to the TogoWS service and returns the raw response for the requested database and query. This function is a low-level helper used within Biopython to access remote biological database search services (for example NCBI, EBI, KEGG databases listed by TogoWS). For typical iterative retrieval of many results prefer Bio.TogoWS.search_iter(), and to obtain only the count of matches use Bio.TogoWS.search_count().
    
    Args:
        db (str): The TogoWS database name to search. This string selects which remote biological resource to query (examples include "ncbi-pubmed" or "pubmed", "ncbi-genbank" or "genbank", "ncbi-taxonomy", "ebi-uniprot" or "uniprot", "kegg-compound" or "compound"). The function maintains a cached list of supported database names in the module-level _search_db_names; if that cache is empty it will fetch the list from the TogoWS base URL (causing an initial network request). If the provided db is not present in the cached list the function will emit a warning (warnings.warn) rather than failing, because the authoritative HTML listing and the live service can differ.
        query (str): The search term to run against the selected TogoWS database. This string is URL-quoted and embedded in the request path; it determines which records the remote service will match and return. The query semantics (field names, boolean operators, etc.) are determined by the remote database as documented on the TogoWS site (http://togows.dbcls.jp/search/).
        offset (int, optional): 1-based index of the first result to return. If provided, offset is converted to an integer and validated to be at least 1. The code will raise ValueError if offset cannot be converted to an int or if offset <= 0. offset must be provided together with limit; providing only one of offset or limit will raise ValueError. If omitted (None), no offset is added to the request URL and the service default is used.
        limit (int, optional): Number of results to return starting from offset. If provided, limit is converted to an integer and validated to be at least 1. The code will raise ValueError if limit cannot be converted to an int or if limit <= 0. limit must be provided together with offset; providing only one of offset or limit will raise ValueError. If omitted (None), no limit is added to the request URL and the service default is used. Note that TogoWS applies its own default maximum (at the time of writing a default count limit of 100) which is an upper bound on the number of results a single request may return; to retrieve larger result sets use search_iter(...) to batch requests.
        format (str, optional): Requested format for the returned data, appended to the request URL as a file extension (for example "json" or "ttl" for RDF). If None, the default plain text format is requested (one result per line). The available formats depend on the selected database and TogoWS support; use the TogoWS documentation for which formats are supported for a given db.
    
    Returns:
        file-like object: The return value is the raw result of calling the module's internal _open(url) on the constructed TogoWS URL. This object is a readable handle for the HTTP response data returned by the TogoWS service and contains the search results in the requested format (plain text, JSON, TTL, etc.). The caller should read from and close this handle as appropriate. The exact concrete type depends on the underlying transport implementation used by _open.
    
    Behavior and side effects:
        - The function constructs a URL of the form "<BASE_URL>/search/{db}/{quoted query}" and conditionally appends "/{offset},{limit}" and/or ".{format}" depending on the arguments.
        - If the module-level cache of supported database names (_search_db_names) is empty, the function will perform a network fetch to populate it before validating db; this may produce network latency or transient network errors.
        - If db is not found in the cached list, a runtime warning (warnings.warn) is emitted advising the user to consult the TogoWS search listing; this does not prevent the function from attempting the request.
        - The function enforces that offset and limit are provided together and that both are positive integers; otherwise it raises ValueError with a descriptive message.
        - The function performs a network request via _open(url); any network-related exceptions raised by the underlying transport (for example connection errors, HTTP errors) propagate to the caller.
        - The TogoWS service itself imposes limits (for example a default maximum number of results per request). To obtain more records than allowed in a single request, use Bio.TogoWS.search_iter() which batches requests transparently.
    
    Failure modes:
        - ValueError is raised when offset or limit cannot be converted to int, when offset or limit <= 0, or when only one of offset/limit is supplied.
        - A warnings.warn is issued (not an exception) when the db name is not present in the discovered list of supported databases.
        - Network/transport errors raised by the internal _open call are not swallowed and are propagated to the caller.
        - The remote TogoWS service may return fewer results than requested due to service-side limits or transient errors; callers should handle partial or empty responses.
    
    Domain significance:
        - This function provides programmatic access to heterogeneous biological databases indexed by TogoWS and is intended for use within bioinformatics workflows where raw search responses are required. Because it returns the raw response, it is intended for low-level use; higher-level helpers in Bio.TogoWS (search_iter, search_count) provide more user-friendly and robust interfaces for most common tasks.
    """
    from Bio.TogoWS import search
    return search(db, query, offset, limit, format)


################################################################################
# Source: Bio.TogoWS.search_count
# File: Bio/TogoWS/__init__.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_TogoWS_search_count(db: str, query: str):
    """Bio.TogoWS.search_count — Request the TogoWS search service and return the integer count of records matching a search.
    
    This function calls the TogoWS REST search/count endpoint to determine how many records in a given TogoWS database match the provided query. It is intended for use in bioinformatics workflows where you need to estimate the size of a result set (for example to plan batched downloads using offset and limit with Bio.TogoWS.search()). The function caches the list of known searchable database names in the module-level variable _search_db_names by calling _get_fields(_BASE_URL + "/search") when needed. If the provided db is not in that cached list, the function issues a runtime warning but still attempts the request. The function performs network I/O by constructing a URL of the form _BASE_URL + "/search/{db}/{quote(query)}/count", opening it via _open(), reading the response body, and converting the response to an integer count.
    
    Args:
        db (str): The TogoWS database name to query. This should be one of the database identifiers exposed by the TogoWS search index (see the TogoWS search listing, e.g. http://togows.dbcls.jp/search). Practically, this parameter tells TogoWS which biological database (such as sequence, taxonomy, or other indexed resources) to apply the search to. If this value is not present in the cached _search_db_names, the function will emit a warning but will still attempt the request.
        query (str): The search expression passed to the TogoWS search API. This is the text query understood by the specified TogoWS database (for example an identifier, keyword, or more complex query language supported by that DB). The function URL-encodes this value with quote() before including it in the request URL.
    
    Returns:
        int: The total number of matching records reported by the TogoWS service for the given db and query. This integer is computed by stripping whitespace from the response body and converting it to an int. A typical use is to call this function first to determine how many records exist so you can download results in batches with offset/limit or choose to use Bio.TogoWS.search_iter() for streaming results instead.
    
    Raises:
        ValueError: If the HTTP response body is empty (no data returned) or if the response cannot be parsed as an integer (for example the service returned an unexpected error message or non-numeric content).
        Exception: Network, HTTP, or I/O related exceptions from the underlying _open() and read() calls (for example connection errors or HTTP error responses) are not caught here and will propagate to the caller. The caller should handle such exceptions when using this function.
    
    Side effects:
        - Performs network I/O to the remote TogoWS server at the module base URL (_BASE_URL). Calls to this function may be slow or fail depending on network connectivity and remote service availability.
        - May populate the module-level cache _search_db_names by calling _get_fields(...) when it is None.
        - May emit a runtime warning via warnings.warn() if the supplied db is not in the cached list of known searchable databases.
        - Does not modify input arguments.
    """
    from Bio.TogoWS import search_count
    return search_count(db, query)


################################################################################
# Source: Bio.TogoWS.search_iter
# File: Bio/TogoWS/__init__.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_TogoWS_search_iter(db: str, query: str, limit: int = None, batch: int = 100):
    """Call TogoWS search and iterate over the resulting identifiers in batches.
    
    This generator function is part of Biopython's TogoWS client utilities and is used to perform a search against a remote TogoWS-compatible database (for example "pubmed") and stream the matching record identifiers (strings such as PubMed IDs) without loading all results into memory. It first obtains the total hit count via Bio.TogoWS.search_count(db, query) and then repeatedly calls Bio.TogoWS.search(db, query, offset, batch) to fetch successive batches of identifiers. This is useful in computational molecular biology workflows (e.g., fetching large sets of sequence or literature identifiers) where you want to process results incrementally.
    
    Args:
        db (str): The TogoWS target database name as a string, e.g. "pubmed" or another database supported by TogoWS. This parameter determines which remote collection of records the query will search and corresponds to the first argument passed to the underlying TogoWS HTTP API used by Bio.TogoWS.search and Bio.TogoWS.search_count.
        query (str): The search query string to send to TogoWS, using the database-specific query syntax (for example "diabetes+human" for PubMed). This value is passed unchanged to the remote search service and defines which records are matched and returned as identifiers.
        limit (int): Optional upper bound on the total number of identifiers to yield. If None (the default), the function will yield up to the total hit count returned by TogoWS.search_count(db, query). If specified, the function yields at most this many identifiers even if the remote service reports more matches. Setting this allows client code to cap network usage and processing.
        batch (int): The desired number of identifiers to retrieve per HTTP request to the TogoWS search endpoint. Default is 100. This controls the granularity of network calls and memory usage: smaller values reduce memory per request but increase the number of HTTP calls. Note: the TogoWS service currently enforces a maximum batch size (currently 100) and may return an HTTP 400 Bad Request if a larger batch is requested; this function does not override server-side limits and relies on the server to enforce them.
    
    Returns:
        Iterator[str]: A generator that yields identifiers as strings, one by one, as returned by the remote TogoWS search service. Calling this function does not perform the remote requests immediately; iteration over the returned generator triggers the search_count call and subsequent batched search requests. The generator will stop after yielding the smaller of the remote hit count and the provided limit (if any).
    
    Behavior, side effects, defaults, and failure modes:
        This function issues network calls to the remote TogoWS service via the package's internal search_count and search functions; these produce HTTP requests and may raise network- or HTTP-related exceptions (e.g., connection errors, timeouts, or HTTP error responses) which propagate to the caller. If the total hit count from search_count is zero or falsy, the generator will be empty and iteration will immediately raise StopIteration. The function uses 1-based offsets (offset starts at 1) because the remote API is not zero-based.
        On each iteration it reads the batch response, splits by whitespace to obtain identifiers, and asserts that the number of identifiers returned equals the requested batch size (assertion error will be raised if this invariant is violated). If the same batch of identifiers is returned for two different offsets, or if any identifier is repeated across consecutive batches, the function raises RuntimeError to indicate inconsistent or unstable remote search results. The function relies on the remote service to enforce any per-request batch upper bound; do not pass values for batch exceeding the service limit. The caller should handle or propagate network and assertion/runtime exceptions as appropriate for their application.
    """
    from Bio.TogoWS import search_iter
    return search_iter(db, query, limit, batch)


################################################################################
# Source: Bio.UniProt.GOA.record_has
# File: Bio/UniProt/GOA.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_UniProt_GOA_record_has(inrec: dict, fieldvals: dict):
    """Check whether a UniProt GOA record contains any of the specified field values.
    
    Args:
        inrec (dict): A mapping representing a parsed UniProt Gene Ontology Annotation (GOA) record.
            Keys are field names (strings) used in the GOA record (for example 'go_id',
            'evidence', 'qualifier' depending on how the caller parsed the GOA file).
            Values are either a single string for that field or an iterable of strings
            (for example a list or set of GO IDs or evidence codes). This parameter
            provides the record being tested and is typically produced by the GOA
            parser code in Bio.UniProt.GOA; the function inspects these entries to
            determine whether the record satisfies any of the requested field-value
            criteria. The function does not modify this dictionary.
    
        fieldvals (dict): A mapping defining the filter criteria in the exact format
            {'field_name': set([val1, val2])}. Each key is a field name that should
            be present in inrec, and each value MUST be a set of candidate values
            (typically strings such as GO identifiers or evidence codes) to look for
            in that field. The role of this parameter is to supply the set of values
            that, if any are present in the corresponding inrec field, will cause the
            function to report a match. Using sets for the values enables efficient
            membership testing via set intersection.
    
    Behavior and practical details:
        The function iterates over every field name in fieldvals. For each field it
        retrieves inrec[field]. If the retrieved value is a str it is treated as a
        single value and wrapped into a one-element set; otherwise the value is
        converted into a set via set(inrec[field]) so that it can be intersected with
        the candidate set from fieldvals. If the intersection is non-empty for any
        tested field (i.e. the record contains at least one of the requested values
        for that field), the function immediately returns True. If no fields produce
        a non-empty intersection the function returns False after checking all
        fields. This behavior is used in GOA-related filtering to decide whether a
        parsed record contains any of the target annotation values and therefore
        should be selected for downstream processing.
    
    Failure modes and constraints:
        If a field name from fieldvals is not present in inrec, a KeyError will be
        raised because the function accesses inrec[field] directly. If inrec[field]
        is neither a str nor an iterable acceptable to set(), a TypeError will be
        raised when attempting set(inrec[field]). The function expects fieldvals
        values to be set objects; providing other container types for fieldvals
        values may cause unexpected errors because the code uses set intersection
        (&) with the provided value.
    
    Side effects and defaults:
        The function has no side effects: it does not modify inrec or fieldvals, nor
        does it perform any I/O. There are no default values; both parameters must be
        provided by the caller and must be dict objects as described above.
    
    Returns:
        bool: True if any field named in fieldvals has at least one value that
        matches a value in the corresponding inrec field (using set intersection);
        False otherwise. This boolean is typically used by GOA parsing/filtering
        logic to include or exclude UniProt GOA records based on specified annotation
        values.
    """
    from Bio.UniProt.GOA import record_has
    return record_has(inrec, fieldvals)


################################################################################
# Source: Bio.bgzf.make_virtual_offset
# File: Bio/bgzf.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_bgzf_make_virtual_offset(block_start_offset: int, within_block_offset: int):
    """Bio.bgzf.make_virtual_offset: Compute a BGZF "virtual offset" used by BAM/Tabix indexing by packing a BGZF block file offset and an offset within the decompressed block into a single 64-bit integer.
    
    This function implements the BAM indexing scheme representation of a file position as a 64-bit "virtual offset". The virtual offset is formed in C-style bit packing as (block_start_offset << 16) | within_block_offset, where the high 48 bits encode the file offset of the start of a BGZF block and the low 16 bits encode the byte offset inside the decompressed block. In the Biopython project (a library for computational molecular biology and bioinformatics), this value is used when creating or interpreting BAM (.bam) and Tabix index records to allow random access to alignments or annotations by genomic coordinates.
    
    Args:
        block_start_offset (int): The file offset (in bytes) of the start of a BGZF block on disk. This is the unsigned block-level file position used in BAM/Tabix virtual offsets and must satisfy 0 <= block_start_offset < 2**48 (i.e. fits in the high 48 bits of the 64-bit virtual offset). Practically, this is the byte position you would pass to an operating system file seek to reach the start of the compressed BGZF block.
        within_block_offset (int): The byte offset within the decompressed contents of the BGZF block. This is an unsigned 16-bit value used in the low 16 bits of the virtual offset and must satisfy 0 <= within_block_offset < 2**16. Practically, this is the position inside the uncompressed block where a record (for example, the start of an alignment) begins.
    
    Returns:
        int: The computed BGZF virtual offset as an integer equal to (block_start_offset << 16) | within_block_offset. The returned integer is suitable for use in BAM/Tabix index entries or for comparing/serializing virtual file positions. In Python this is a plain int (unbounded precision), but the function enforces the canonical 48/16-bit partitioning so the numeric value fits the intended BAM/Tabix encoding.
    
    Raises:
        ValueError: If within_block_offset is outside the allowed 16-bit range (less than 0 or >= 2**16), a ValueError is raised explaining the required range and the provided value. If block_start_offset is outside the allowed 48-bit range (less than 0 or >= 2**48), a ValueError is raised explaining the required range and the provided value.
    
    Behavior and side effects:
        This is a pure, deterministic function with no side effects such as I/O or mutation; it only validates its integer inputs and returns a packed integer. It must be given integer inputs that conform to the documented ranges; otherwise the function fails early with a ValueError. The packing follows the BAM specification so the result can be directly used when writing or interpreting BAM/Tabix index structures in Biopython and downstream bioinformatics tools.
    """
    from Bio.bgzf import make_virtual_offset
    return make_virtual_offset(block_start_offset, within_block_offset)


################################################################################
# Source: Bio.bgzf.open
# File: Bio/bgzf.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_bgzf_open(filename: str, mode: str = "rb"):
    """Open a BGZF-compressed file for reading, writing or appending. This routine is used in computational molecular biology workflows (for example when working with blocked GZIP files such as BAM, BCF, or Tabix-indexed VCF/GFF files) to obtain a Biopython file-like object that performs BGZF-aware I/O.
    
    Args:
        filename (str): Filesystem path to the BGZF file to open. This must be a Python str giving the path or filename of the file on disk that you intend to read from or write to. In practice this is the path to a compressed genomic data file (e.g. a BAM or BGZF-compressed VCF) that downstream Biopython modules or external tools will read or produce.
        mode (str): Mode string controlling how the file is opened. The default is "rb". The function inspects mode.lower() and:
            - If the string contains "r", a BgzfReader is returned for reading BGZF-compressed data.
            - If the string contains "w" or "a", a BgzfWriter is returned for writing or appending BGZF-compressed data.
            The mode string may also indicate binary vs text with the usual "b" or "t" characters. If text mode is requested (e.g. "rt" or "wt"), the implementation forces the "latin1" encoding to avoid multi-byte character handling and passes "\r" and "\n" through as-is (universal newline translation is not enabled). Because of this, if your data is encoded in UTF-8 or another multi-byte encoding, you must open the file in binary mode and perform any decoding yourself. The default "rb" is appropriate for most genomic file formats that are stored as BGZF.
    
    Returns:
        BgzfReader or BgzfWriter: A BGZF-aware, file-like object. If mode contains "r", the function returns a BgzfReader instance configured to read decompressed data from the given filename. If mode contains "w" or "a", the function returns a BgzfWriter instance configured to write BGZF-compressed data to the given filename. The returned object implements the usual file-like operations expected by Biopython modules (for example read, write, seek, close) and is intended to be used directly by downstream code that processes compressed genomic data.
    
    Behavior, side effects, and failure modes:
        - Opening for writing ("w") will create the target file and, by conventional semantics, will overwrite/truncate an existing file. Opening for appending ("a") will create the file if it does not exist and will add BGZF blocks to the end if it does exist. These behaviors are provided by the underlying BgzfWriter implementation.
        - Text mode uses the fixed "latin1" encoding and does not perform universal newline translation; therefore "\r" and "\n" are preserved. This is done to avoid corrupting multi-byte encodings when treating the BGZF payload as text.
        - If the mode string does not contain "r", "w", or "a" (case-insensitive), a ValueError is raised indicating a bad mode.
        - Opening may raise standard file system or I/O exceptions (e.g. FileNotFoundError, PermissionError, or exceptions raised by the BgzfReader/BgzfWriter when encountering malformed BGZF data). Callers should handle these exceptions as appropriate for their workflow.
        - This function does not itself perform format validation beyond delegating to the reader/writer classes; errors reading or writing invalid BGZF blocks will be raised by those classes.
    """
    from Bio.bgzf import open
    return open(filename, mode)


################################################################################
# Source: Bio.bgzf.split_virtual_offset
# File: Bio/bgzf.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_bgzf_split_virtual_offset(virtual_offset: int):
    """Split a 64-bit BGZF virtual offset into the BGZF block start and the offset within that block.
    
    This function is used in Bio.bgzf and related Biopython code that manipulates BGZF-compressed genomic files (for example BAM/BAI workflows). A BGZF virtual offset encodes a file-position that can be used to seek to a particular record: the high-order bits are the byte offset of the start of a BGZF block in the compressed file, and the low-order 16 bits are the offset inside the uncompressed data of that block. This function reverses that packing: it extracts the block start (the compressed-file byte offset where the BGZF block begins) and the within-block offset (the byte offset inside the uncompressed block where the desired data record begins). The result is commonly used when resolving virtual offsets stored in indexes (such as BAI) to actual file seek positions for random access in large genomic files.
    
    Args:
        virtual_offset (int): A Python int representing the packed 64-bit BGZF virtual offset value. In Biopython usage this value is expected to originate from BGZF-aware indexes or file metadata. The function interprets this integer by taking the high-order bits (shifted right by 16) as the BGZF block start (compressed-file byte offset) and the low-order 16 bits as the within-block offset. Supplying a non-integer will raise a TypeError when Python attempts bitwise/shift operations; supplying negative values or values outside the conventional 64-bit virtual-offset domain may produce results that do not correspond to valid BGZF locations (Python's arbitrary-precision ints will still be processed mathematically).
    
    Returns:
        tuple(int, int): A two-tuple (start, offset_within_block). 'start' is an int giving the BGZF block start as a byte offset into the compressed file (obtained by shifting the input right by 16 bits). 'offset_within_block' is an int giving the low 16-bit offset inside the uncompressed block (computed as the difference between the original virtual offset and start<<16). Both return values are intended to be directly usable for seeking and reading within BGZF-compressed genomic files; there are no side effects.
    """
    from Bio.bgzf import split_virtual_offset
    return split_virtual_offset(virtual_offset)


################################################################################
# Source: Bio.motifs.create
# File: Bio/motifs/__init__.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_motifs_create(instances: list, alphabet: str = "ACGT"):
    """Create a Motif object from a collection of aligned motif occurrences.
    
    This convenience function constructs an Alignment from the provided instances and then constructs and returns a Motif using that alignment and the provided alphabet. In the Biopython project, a Motif object represents a conserved sequence pattern (for example, a transcription factor binding site) derived from multiple aligned occurrences and is used in downstream analyses such as computing position-specific counts, position-specific scoring matrices (PSSMs), consensus sequences, and motif visualization.
    
    Args:
        instances (list): A list of sequence instances that represent individual occurrences of the motif to be modeled. These elements together form the multiple alignment passed to Alignment(instances). In practical use within computational molecular biology, each element corresponds to one observed occurrence of the motif (for example, a short DNA sequence from a set of binding-site hits). Typically all instances should have the same length (i.e., be already aligned); if they are not compatible with Alignment, the Alignment constructor may raise an exception (for example ValueError or a more specific error from Alignment). The exact element types accepted are those accepted by the Alignment constructor; this function does not perform additional type conversion beyond calling Alignment(instances).
        alphabet (str): A string of characters defining the alphabet used to interpret the sequences (default "ACGT"). The default value corresponds to the canonical DNA nucleotide alphabet and is used to initialize the Motif's internal representations (counts, frequency matrices, etc.). The alphabet string is forwarded to the Motif constructor, which may validate it and raise an exception if it is invalid for the Motif implementation being used.
    
    Returns:
        Motif: A newly created Motif instance initialized with the alignment built from the provided instances and the specified alphabet. The returned Motif encapsulates the multiple alignment of motif occurrences and provides methods and attributes for motif analysis (for example, computing consensus sequences, counts, and PSSMs). There are no persistent side effects (no files are written); any exceptions raised by Alignment or Motif (for example due to incompatible instance lengths or an invalid alphabet) are propagated to the caller.
    """
    from Bio.motifs import create
    return create(instances, alphabet)


################################################################################
# Source: Bio.motifs.write
# File: Bio/motifs/__init__.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_motifs_write(motifs: list, fmt: str, **kwargs):
    """Return a string representation of a collection of sequence motifs in a specified motif file format.
    
    This function is part of the Bio.motifs subpackage of Biopython, a toolkit for computational molecular biology. It converts an in-memory list of motif objects (position frequency matrices or similar motif representations used in motif discovery and transcription factor binding site analysis) into the textual file format required by downstream tools or databases. The function supports exporting to ClusterBuster, JASPAR (single or multiple PFM variants), and TRANSFAC-like formats by delegating to the corresponding format-specific writer implementations in Bio.motifs.jaspar, Bio.motifs.transfac, and Bio.motifs.clusterbuster. The fmt argument is treated case-insensitively. No files are written by this function; it returns the formatted content as a string so callers can write it to disk, send it over a network, or further process it in memory.
    
    Args:
        motifs (list): A list of motif objects to be serialized. Each element should be a motif representation compatible with Bio.motifs writers (for example, position frequency matrix / motif instances used elsewhere in the Bio.motifs API). The list order will be preserved for multi-motif formats (for example, "jaspar"), and a single-motif list is appropriate for formats that expect one PFM per output when using "pfm".
        fmt (str): The target output format name (case-insensitive). Supported values are "clusterbuster" (Cluster Buster PFM format), "pfm" (JASPAR simple single Position Frequency Matrix), "jaspar" (JASPAR multiple PFM format), and "transfac" (TRANSFAC-like files). The choice of fmt controls which format-specific writer is invoked and therefore the exact textual layout of the returned string.
        kwargs (dict): Additional keyword arguments forwarded to format-specific writer functions. These are accepted only when the underlying writer supports them (for example, Bio.motifs.clusterbuster.write may accept extra options). Do not rely on any particular keys unless documented by the specific format writer; unspecified keys will be ignored or may cause the delegated writer to raise an exception.
    
    Returns:
        str: The motifs formatted as a single string in the requested file format. The returned string contains all content that would appear in a file written in that format (header lines, matrix rows, identifiers, etc.), allowing callers to write it to disk or transmit it. No file I/O or global state is modified by this function itself.
    
    Raises:
        ValueError: If fmt is not one of the supported format names (after case normalization), a ValueError is raised with a message indicating the unknown format type.
        Exception: The function performs dynamic imports and delegates to format-specific writer functions; any exceptions raised by those writers (for example, due to invalid motif objects, missing required motif metadata, or unsupported kwargs) will propagate to the caller. Users should validate motif objects and consult the documentation for the specific format writer (Bio.motifs.jaspar, Bio.motifs.transfac, Bio.motifs.clusterbuster) for format-specific requirements and accepted keyword arguments.
    
    Behavior and side effects:
        - The fmt argument is lower-cased internally and matched against supported formats.
        - For "pfm" and "jaspar" the function delegates to Bio.motifs.jaspar.write(motifs, fmt) and returns its result.
        - For "transfac" the function delegates to Bio.motifs.transfac.write(motifs) and returns its result.
        - For "clusterbuster" the function delegates to Bio.motifs.clusterbuster.write(motifs, **kwargs) and returns its result, passing through any provided kwargs.
        - The function performs dynamic imports of the format-specific modules at call time; this may raise ImportError if those modules are unavailable.
        - The function itself performs no file writing; the returned string is the serialized representation.
    """
    from Bio.motifs import write
    return write(motifs, fmt, **kwargs)


################################################################################
# Source: Bio.motifs.clusterbuster.write
# File: Bio/motifs/clusterbuster.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_motifs_clusterbuster_write(motifs: list, precision: int = 0):
    """Bio.motifs.clusterbuster.write returns a Cluster Buster position frequency matrix representation for a collection of motif objects, suitable for saving to a file or passing to external motif analysis tools. In the context of Biopython (a library for computational molecular biology), this function serializes Biopython motif objects (each expected to provide a name, counts for A/C/G/T, and optional weight and gap attributes) into the plain-text format understood by the Cluster Buster suite for cis-regulatory motif clustering and scanning.
    
    Args:
        motifs (list): A sequence (list) of motif objects to be serialized. Each motif object is expected to have the following attributes used by this function: name (string) used as the motif identifier and written as a header line prefixed with ">" ; counts (mapping) with keys "A", "C", "G", "T" whose values are iterables of per-position counts for that nucleotide (the function iterates through the four nucleotide count lists in A, C, G, T order to produce one row per motif position); optional attributes weight and gap are queried and, if present and truthy, emitted as comment lines "# WEIGHT: <value>" and "# GAP: <value>" immediately after the header. The order of motifs in the input list is preserved in the output. If a motif lacks the required counts mapping or the expected nucleotide keys, a KeyError or other exception may be raised (see Failure modes below). This parameter is required and must be provided as a Python list as in the function signature.
        precision (int): Number of decimal places to use when formatting the per-position nucleotide counts. By default precision=0 which results in integer-style formatting (counts displayed with zero decimal places). If precision is set to a higher integer value, counts are formatted as floating point numbers with that many digits after the decimal point. This parameter controls the textual numeric representation only; it does not change or rescale the underlying numeric values in the motif objects.
    
    Returns:
        str: A single string containing the full Cluster Buster formatted representation for all motifs in the same order as provided. The string consists of one header line per motif (">" followed by motif.name and a newline), optional comment lines for weight and gap when those attributes exist and are truthy ("# WEIGHT: <value>" and/or "# GAP: <value>"), followed by one tab-separated row per motif position with counts for A, C, G, T in that order and a terminating newline. Typical usage is to take the returned string and write it to a file (for example, to create an input file for Cluster Buster) or pass it to another program; the function itself performs no I/O.
    
    Behavior, side effects, defaults, and failure modes:
        The function builds the text representation in memory and returns it; there are no side effects such as file writes or global state modification. The default behavior (precision=0) formats counts with zero decimal places producing integer-looking values; setting precision to a positive integer produces floating point values with that many decimal places. The function silently ignores missing optional attributes weight and gap by checking for AttributeError and only emitting those comment lines when the attribute exists and is truthy; however, if a motif does not have a counts attribute, or if counts does not contain the keys "A", "C", "G", "T", the function will raise a KeyError. If the count values are not numeric (i.e., cannot be formatted by the Python float-format specifier), a ValueError or TypeError may be raised during formatting. The function expects the input motifs list to be iterable and to contain objects conforming to the described minimal motif interface; it does not validate types beyond attempting to access the named attributes.
    """
    from Bio.motifs.clusterbuster import write
    return write(motifs, precision)


################################################################################
# Source: Bio.motifs.jaspar.split_jaspar_id
# File: Bio/motifs/jaspar/__init__.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_motifs_jaspar_split_jaspar_id(id: str):
    """Split a JASPAR matrix ID into its base identifier and optional version component.
    
    This function is used in the Bio.motifs.jaspar subpackage to parse JASPAR motif matrix identifiers commonly found in JASPAR databases and motif exchange files. A JASPAR ID often encodes a stable matrix identifier (the base ID) and an optional version suffix separated by a single period, for example "MA0047.2". The base ID identifies the motif (useful for looking up the canonical matrix or grouping related motifs) and the version component identifies a release or revision of that matrix (useful for tracking changes across database versions).
    
    Args:
        id (str): The JASPAR matrix identifier to split. Must be a Python string containing the identifier as it appears in JASPAR files or related metadata, for example "MA0047" or "MA0047.2". The function expects this exact parameter name and type; passing a non-string value will result in an AttributeError when the method tries to call the string split operation.
    
    Returns:
        tuple: A 2-tuple (base_id, version) describing the parsed components.
            base_id (str): The base JASPAR identifier extracted from the input. If the input contained exactly one period (".") separating two fields, base_id is the substring before that period. If the input does not match that pattern (no period or more than one period), base_id is returned as the original input string.
            version (str or None): The textual version suffix extracted from the input when the input contains exactly one period separating two fields (the substring after the period). The version is not converted to an integer; it is returned as the raw string present after the period. If no single-period-separated version is present, version is None.
    
    Behavior and failure modes:
        The function performs a simple split on the literal period character using Python's str.split("."). If the split yields exactly two components, those components are treated as (base_id, version). If the split yields any other number of components, the function treats the entire input as the base_id and returns version as None. There are no side effects (no I/O or mutation). Because the implementation calls the string split method, passing a non-str value for id will raise an AttributeError. The function does not perform validation of the semantic format of the base_id or version beyond this splitting rule (for example it does not verify that base_id matches the "MA" prefix or that version is numeric).
    """
    from Bio.motifs.jaspar import split_jaspar_id
    return split_jaspar_id(id)


################################################################################
# Source: Bio.motifs.jaspar.write
# File: Bio/motifs/jaspar/__init__.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_motifs_jaspar_write(motifs: list, format: str):
    """Bio.motifs.jaspar.write returns a text representation of one or more motif objects in either the "pfm" (position frequency matrix) or "jaspar" motif database format. This function is used in computational molecular biology workflows (as in Biopython) to serialize motif/count matrices for storage, sharing, or submission to motif databases such as JASPAR; the produced string contains ASCII lines with numeric counts formatted to two decimal places and newline termination suitable for writing to a file.
    
    Args:
        motifs (list): A list of motif-like objects to be serialized. Each motif object is expected to provide a counts attribute (mapping from DNA letter to a sequence of numeric values), a name attribute (string) and optionally a matrix_id attribute. The counts mapping must contain keys for the letters "A", "C", "G", and "T" (the function uses this fixed order) and each mapped value must be an iterable of numbers (the function formats each number with two decimal places using Python's ":6.2f" format). For the "pfm" format only the first element motifs[0] is used; for the "jaspar" format all motifs in the list are serialized in sequence. If a motif lacks the expected attributes (counts or name) or if counts values are not numeric, the function will raise the underlying exceptions (AttributeError, KeyError, TypeError, or ValueError) originating from attribute access or numeric formatting.
        format (str): A string indicating the desired output format. Valid values are the exact strings "pfm" and "jaspar" (case-sensitive). If format == "pfm", the function serializes only the first motif in motifs as four lines (one per base in the fixed order "A", "C", "G", "T") each containing space-separated floating point numbers formatted with two decimal places and a trailing newline. If format == "jaspar", the function serializes every motif in motifs: for each motif it writes a header line starting with ">" followed by the motif's matrix_id (or the literal "None" if matrix_id is absent) and the motif's name, then four lines in the fixed letter order with each line containing the letter, a space, and the list of formatted numeric counts enclosed in square brackets, each terminated with a newline. Any other value for format causes a ValueError to be raised with the message "Unknown JASPAR format %s" % format.
    
    Returns:
        str: The complete serialized representation as a single string. This string is constructed by joining all generated lines (each line ends with a newline character) and is suitable for writing directly to a text file or sending to other components that expect PFM or JASPAR formatted motif data. The function has no other side effects (it does not modify the input motif objects or perform I/O).
    """
    from Bio.motifs.jaspar import write
    return write(motifs, format)


################################################################################
# Source: Bio.motifs.pfm.write
# File: Bio/motifs/pfm.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_motifs_pfm_write(motifs: list):
    """Return a string representing motifs in the Cluster Buster position frequency matrix (PFM) format.
    
    This function is part of Biopython's motifs utilities and is used in computational molecular biology workflows to export motif models (position frequency matrices) so they can be consumed by external motif scanning tools such as Cluster-Buster or other tools expecting the four-column A C G T count format. For each motif in the input list, the function writes a header line beginning with ">" followed by the motif's name, then one line per motif position containing four tab-separated numeric counts in the order A, C, G, T. Numeric values are formatted with no decimal places (rounded according to Python's float formatting rules). The implementation concatenates the per-line strings into a single text block and returns it; it does not write to disk or modify the input objects.
    
    Args:
        motifs (list): A list of motif objects to be written in Cluster Buster PFM format. Each motif object is expected to provide a .name attribute (used verbatim after the ">" header) and a .counts mapping where the keys "A", "C", "G", and "T" map to iterable sequences of numeric counts (e.g., lists of floats or ints) for each motif position. The order of motifs in this list is preserved in the output. No validation beyond attribute/key access and formatting is performed.
    
    Behavior and practical details:
        - Output format: For each motif m in motifs, the function appends a header line of the form ">{m.name}\n". Then for each position i (determined by zipping the four iterables m.counts["A"], m.counts["C"], m.counts["G"], m.counts["T"]), it appends a line "{A}\t{C}\t{G}\t{T}\n" where A, C, G, T are the corresponding numeric counts formatted with zero decimal places using Python's "{:0.0f}" float format. Thus each position yields exactly four tab-separated columns in A, C, G, T order.
        - Truncation behavior: The per-position loop uses zip on the four count iterables; if the count sequences differ in length, positions beyond the shortest sequence are silently ignored (zip truncates to the shortest), so inconsistent count lengths will produce a shorter motif representation without an explicit error.
        - Rounding and numeric formatting: Counts are formatted with no decimal places; non-integer floats will be rounded according to Python's float formatting rules. If a count value cannot be formatted as a float (e.g., non-numeric types), a ValueError or TypeError will be raised by the format call.
        - Header handling: The motif .name value is used verbatim after the ">" character and is not escaped; embedded newlines or special characters in names will be included as-is in the output.
        - Memory and performance: The function builds a list of lines and joins them into a single string before returning. This is efficient for moderate sizes but will allocate memory proportional to the total output size; very large collections of motifs may require substantial memory.
        - Side effects: None on input objects; the function does not write to files, modify motif objects, or perform I/O. The returned string is the complete textual representation and can be written to a file by the caller if desired.
    
    Failure modes and exceptions:
        - AttributeError: If an element in motifs lacks a .name attribute or a .counts attribute, accessing these will raise AttributeError.
        - KeyError: If m.counts does not contain the keys "A", "C", "G", or "T", a KeyError will be raised.
        - ValueError/TypeError: If elements of the count iterables are not numeric or cannot be formatted with "{:0.0f}", formatting will raise ValueError or TypeError.
        - Silent truncation: If the four count iterables have different lengths, positions beyond the shortest are silently omitted (no exception is raised).
    
    Returns:
        str: A single string containing the Cluster Buster position frequency matrix representation of the input motifs. The string consists of repeated blocks, each starting with a header line ">name\n" followed by one line per motif position with four tab-separated counts (A, C, G, T) terminated by newlines. This return value is intended to be written to a file or passed to other tools as the PFM formatted text.
    """
    from Bio.motifs.pfm import write
    return write(motifs)


################################################################################
# Source: Bio.motifs.transfac.write
# File: Bio/motifs/transfac.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_motifs_transfac_write(motifs: list):
    """Write the representation of one or more motifs in TRANSFAC format.
    
    This function is part of the Bio.motifs module in Biopython, used in computational molecular biology to export motif objects (position frequency matrices and associated metadata) into the TRANSFAC flat-file format widely used to represent transcription factor binding sites and their annotation. The function assembles text blocks for each motif following TRANSFAC conventions: two-letter keys (e.g., AC, ID, DT), section separators "XX", per-motif terminators "//", a P0 frequency-matrix section for counts, and an optional top-level version block if the provided motifs container exposes a version attribute. The routine consults Motif.multiple_value_keys to determine when a key maps to multiple output lines, and it writes reference entries (RN, RX, RA, RT, RL) when a motif exposes a references sequence.
    
    Args:
        motifs (list): A list of Motif objects (as used in Bio.motifs). Each Motif object provides metadata accessible via motif.get(key) for TRANSFAC keys, may expose attributes used by this function (for example motif.length, motif.degenerate_consensus, motif.alphabet, motif.counts, motif.references), and the motifs container itself may expose a version attribute (motifs.version). The practical significance is that each Motif encodes a biologically meaningful binding model (counts per position, alphabet and consensus) and associated metadata (accession, name, organism, comments, external database links, references) which are translated into TRANSFAC fields. The function expects a Python list object (not a file or stream); if the caller supplies an object with a different type but with compatible attributes, behavior is not guaranteed.
    
    Returns:
        str: A single string containing the concatenated TRANSFAC representation for all motifs in the input list. The returned text contains optional top-level version block ("VV  <version> ... //") if motifs.version is present and not None, then one block per motif with section lines for keys defined in the code, "XX" section separators after any non-empty section, and a motif terminator line "//" followed by a newline. The P0 frequency-matrix section, when emitted, begins with a header "P0" followed by the sorted alphabet letters and one line per position formatted with a two-digit position index, numeric counts for each letter (formatted by the implementation with floating formatting) and the motif's degenerate consensus letter at the line end.
    
    Behavior and side effects:
        The function does not perform any file or network I/O; it only constructs and returns the TRANSFAC text as a Python string. It handles the presence or absence of optional data fields conservatively: if motifs.version is absent an overall version header is omitted; for each motif, if a requested metadata key is absent (motif.get raises AttributeError or returns None) that key is skipped; keys listed in Motif.multiple_value_keys are emitted as one line per value. The P0 block is only written when motif.length is greater than zero; if motif.length is zero, the frequency-matrix section is omitted. If motif.references is present, each reference dictionary is queried for RN, RX, RA, RT, RL keys and corresponding lines are emitted when values are present.
    
    Failure modes and exceptions:
        AttributeError and other exceptions may propagate if a Motif in the list lacks attributes that the code accesses directly (for example motif.length, motif.alphabet, motif.counts, or motif.degenerate_consensus) when writing the P0 section. The function internally suppresses AttributeError when checking for the container-level version and when calling motif.get(key) for metadata keys, but does not broadly validate Motif objects beyond these checks. Callers should provide Motif objects with the expected attributes to avoid runtime errors. The function assumes motif.counts is indexable by alphabet letters and positions; mismatched shapes or missing keys in counts may raise IndexError or KeyError originating from those data structures.
    
    Examples of practical use:
        - Exporting a set of Biopython Motif objects to a string for writing into a .transfac file.
        - Generating TRANSFAC-compatible text for downstream tools that accept TRANSFAC formatted motif collections for motif scanning or database submission.
    
    Note:
        The output follows the TRANSFAC conventions implemented in this function and the exact numeric formatting (position index width and numeric formatting of counts) is determined by the code's string formatting.
    """
    from Bio.motifs.transfac import write
    return write(motifs)


################################################################################
# Source: Bio.pairwise2.calc_affine_penalty
# File: Bio/pairwise2.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_pairwise2_calc_affine_penalty(
    length: int,
    open: float,
    extend: float,
    penalize_extend_when_opening: bool
):
    """Calculate the affine gap penalty score for a gap of a given length used in pairwise sequence alignment.
    
    This function implements the common affine gap penalty model used by Bio.pairwise2 and other sequence alignment algorithms in computational molecular biology: a gap penalty composed of a gap opening term plus a gap extension term multiplied by the gap length. It is intended to be called when scoring a contiguous gap (run of insertions or deletions) during dynamic programming for pairwise alignments (global, local, or semi-global) as provided by the Biopython pairwise2 machinery.
    
    Args:
        length (int): The length of the gap (number of consecutive gap positions). In the context of sequence alignment, this is the number of residues or bases missing from one sequence relative to the other. If length <= 0 the function treats this as no gap and returns 0.0. Passing a non-integer that cannot be compared to 0 may raise a TypeError from the comparison operation.
        open (float): The gap opening penalty. This numeric score represents the cost (or negative reward) for starting a new gap in an alignment. In practical use within Biopython pairwise2, this value is typically combined with the extension penalty to compute the total penalty for a contiguous gap. The function does not enforce any sign convention (penalties may be positive costs or negative scores) — the meaning depends on the scoring scheme used by the caller.
        extend (float): The gap extension penalty per gap position. This numeric value is multiplied (directly or with an adjusted factor) by the gap length to compute the extension component of the affine penalty. As with open, the function does not validate sign; callers should supply values consistent with their alignment scoring model.
        penalize_extend_when_opening (bool): Flag controlling whether the extension penalty is also applied to the first gap position (the one paired with opening the gap). If True, the penalty is computed as open + extend * length (i.e., the extension penalty applies to every gap position including the first). If False, the function subtracts one extend term so the penalty becomes open + extend * (length - 1), treating the opening penalty as already accounting for the first extension. This behaviour matches common conventions in affine gap models and allows callers to select whether the "open" term includes the first-position extension cost.
    
    Returns:
        float: The computed penalty for a gap of the given length according to the affine model described above. For length <= 0 the function returns 0.0. The returned value may be positive or negative depending on the sign convention of the supplied open and extend parameters. No side effects occur; the function is pure and deterministic.
    """
    from Bio.pairwise2 import calc_affine_penalty
    return calc_affine_penalty(length, open, extend, penalize_extend_when_opening)


################################################################################
# Source: Bio.pairwise2.format_alignment
# File: Bio/pairwise2.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_pairwise2_format_alignment(
    align1: list,
    align2: list,
    score: float,
    begin: int,
    end: int,
    full_sequences: bool = False
):
    """Format a pairwise alignment (given as two aligned sequences) into a human-readable multi-line string suitable for console display and logging.
    
    This function is used in Biopython's pairwise alignment utilities to present an alignment between two sequences (or sequence tokens) in a compact, readable form. It prints a sequence-1 line, a match-line that marks identical matches, mismatches and gaps, and a sequence-2 line, followed by a score line. This is intended for users and developers working with computational molecular biology sequence comparisons, for quick inspection of aligned regions produced by pairwise alignment algorithms.
    
    Args:
        align1 (list): The first aligned sequence provided as a list of string tokens (for example single-character bases/amino acids or multi-character tokens). Each element corresponds to one alignment column. When align1 is a list, elements will be displayed separated by spaces and may be of different lengths; the formatter centers each pairwise column using the maximum element width. The gap symbol used in these lists must be "-" (or the list element equal to ['-'] for lists) for correct start-position computation described below.
        align2 (list): The second aligned sequence provided as a list of string tokens parallel to align1. Must have alignment columns aligned to those of align1 (same logical length); the function iterates over align1[begin:end] and align2[begin:end] in parallel. As with align1, gap symbol should be "-" for correct start-position computation.
        score (float): The numerical alignment score to display. It is included on the final output line as "Score=<value>" formatted with Python's general format specifier (:g). This value is provided by the alignment algorithm (for example, a dynamic programming routine) and has no effect on the layout beyond being printed.
        begin (int): The Python 0-based index into the aligned sequences indicating the first column of the alignment to display. Note: begin/end are indices into the aligned sequences (0-based), not the original un-aligned sequence coordinates. For local alignments (when full_sequences is False) and when begin != 0, the function computes 1-based start positions for the aligned subsequences by counting non-gap tokens before this index.
        end (int): The Python 0-based end index (exclusive) into the aligned sequences indicating one-past-the-last column to display. The function slices align1 and align2 using this end index; when full_sequences is False and end != len(align1) only the aligned subrange [begin:end] is shown.
        full_sequences (bool): If False (default), and the displayed region does not span the entire aligned sequences (begin != 0 or end != len(align1)), only the aligned subsequence between begin and end is shown and 1-based start positions for the aligned subsequences are printed to the left of the respective sequence lines. If True, the historic behavior is restored: the entire sequences (from index 0 to len(align1)) are displayed, the start-position prefixes are omitted (start markers set to zero width), and non-aligned leading/trailing columns are included; in this case, the match-line shows spaces for columns outside the aligned region. The default is False.
    
    Behavior and formatting details:
        The function produces three main lines followed by a score line. The first line is the displayed portion of sequence 1 with an optional right-justified 1-based start index prefix (when a subsequence is shown and full_sequences is False). The second line is the match-line where identical tokens are shown as "|" (vertical bar), mismatches as "." (dot), and gaps as a space. The third line is the displayed portion of sequence 2 with its optional 1-based start index prefix aligned to sequence 1's prefix width. Finally a score line of the form "  Score=<score>" is appended (using the :g float format).
        For lists, each element is centered in its column using the maximum width of the two tokens in the column. If align1 is a list, a single space is appended to each element when preparing the display so columns are separated visually.
        Start positions (the numeric prefixes when showing only aligned subsequences) are 1-based positions in the original un-aligned sequences. They are computed as (number of non-gap tokens before begin) + 1. This computation assumes the gap symbol is "-" (or ['-'] for lists); if a different gap symbol is used the reported start positions will be incorrect.
        Since the function uses zip() over the slices align1[begin:end] and align2[begin:end], if the two input lists differ in length or the slices are of different lengths, the displayed alignment is truncated to the shorter slice without raising an error. Ensure align1 and align2 have matching aligned-column counts for correct output.
        The function treats equality between tokens using standard Python equality (a == b). It treats an element as a gap if a.strip() == "-" or b.strip() == "-". Leading/trailing whitespace in tokens can therefore affect gap detection.
        The formatting conventions changed in Biopython 1.71: identical matches are shown with "|" (pipe), mismatches with "." (dot), and gaps with a space. Older behavior used only the pipe character to indicate the aligned region; full_sequences=True restores the historic behavior of displaying full sequences but still uses the updated match symbols.
    
    Failure modes and edge cases:
        If the gap token is not "-" (or ['-'] for list tokens), the start-position computation for local alignments will be incorrect. If begin or end are outside the valid slicing range, Python slicing semantics apply (slices will be truncated), and no exception is raised for out-of-range indices; however, an ill-formed pairing of align1 and align2 or mismatched slice lengths will lead to truncated or misleading display. If tokens are not strings or do not support the string operations used (count, strip, len), the behavior may raise exceptions. The function has no external side effects.
    
    Returns:
        str: A formatted, multi-line string representing the (sub)alignment between align1 and align2 and the provided score. The returned string contains three aligned display lines and a trailing score line; it does not modify the input lists.
    """
    from Bio.pairwise2 import format_alignment
    return format_alignment(align1, align2, score, begin, end, full_sequences)


################################################################################
# Source: Bio.pairwise2.print_matrix
# File: Bio/pairwise2.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def Bio_pairwise2_print_matrix(matrix: list):
    """Print a two-dimensional matrix to standard output in a human‑readable, column‑aligned form for debugging pairwise alignment code in Biopython.
    
    Args:
        matrix (list): A two-dimensional matrix represented as a list of row sequences (for example, a list of lists or other indexable row objects) containing the values to print. In the Biopython/pairwise2 context this is typically a dynamic programming scoring or traceback matrix produced during pairwise sequence alignment; each inner row corresponds to one row of the matrix. The function uses the first row (matrix[0]) to determine the number of columns, so matrix must be non-empty and the first row must be indexable. Rows may be shorter than the first row, but no row may be longer than the first row (a longer row will raise IndexError). Individual matrix entries are converted to strings via str(value) and their printed width is computed from that string representation.
    
    Returns:
        None: This function does not return a value. Instead, it prints formatted output to standard output (sys.stdout) as a side effect. The output contains each matrix row on its own line, with columns right‑justified and separated by single spaces. The formatting width for each column is chosen to fit the widest stringified entry found in that column (the function transposes the matrix to compute these widths). Because it mutates no external state and only prints, it is intended solely for debugging and display.
    
    Behavior, defaults, and failure modes:
        The function computes column widths by transposing the matrix and taking the maximum length of str(entry) for each column. It then prints each row with entries right‑aligned to the corresponding column width using Python string formatting ("%*s "). If matrix is empty, accessing matrix[0] will raise IndexError. If rows are longer than the first row, an IndexError will be raised when attempting to append to the transposed column lists. If entries are not convertible to str, a TypeError may propagate from str(value) (though most Python objects support str()). If non‑sequence objects are provided for rows, indexing operations will raise appropriate exceptions. Use this function for quick, human‑readable inspection of alignment matrices during development or debugging of pairwise2 algorithms; do not rely on it for machine parsing of outputs.
    """
    from Bio.pairwise2 import print_matrix
    return print_matrix(matrix)


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
