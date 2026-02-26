"""
Regenerated Google-style docstrings for module 'pyriemann'.
README source: others/readme/pyriemann/README.md
Generated at: 2025-12-02T02:05:56.023631Z

Total functions: 100
"""


import numpy

################################################################################
# Source: pyriemann.classification.class_distinctiveness
# File: pyriemann/classification.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def pyriemann_classification_class_distinctiveness(
    X: numpy.ndarray,
    y: numpy.ndarray,
    exponent: int = 1,
    metric: dict = "riemann",
    return_num_denom: bool = False
):
    """pyriemann.classification.class_distinctiveness measures the separability of classes of symmetric/Hermitian positive definite (SPD/HPD) matrices using Riemannian geometry. It implements the class distinctiveness criterion used in biosignal and brain‑computer interface (BCI) applications to quantify how far apart class centers are relative to their within‑class dispersion when matrices represent covariance estimates (for example, covariance matrices estimated from multichannel EEG, MEG or EMG epochs). The function computes the ratio of between‑class dispersion to within‑class dispersion using a user‑specified exponent on distances and user‑selectable Riemannian metrics for mean and distance estimation.
    
    Args:
        X (numpy.ndarray): Set of SPD/HPD matrices with shape (n_matrices, n_channels, n_channels). Each entry X[i] is a covariance (or HPD) matrix estimated from multivariate time series (e.g., an EEG epoch). Matrices must be valid SPD/HPD for the chosen Riemannian metric. This array is the primary input from which class centers and within‑class dispersions are computed.
        y (numpy.ndarray): 1D array of labels with shape (n_matrices,). y[i] is the class label corresponding to X[i]. Labels determine class membership used to compute per‑class means and dispersions. The function requires at least two distinct labels; if y contains fewer than two unique classes a ValueError is raised.
        exponent (int): Exponent p applied to distances (default=1). This parameter corresponds to the p value in the class distinctiveness formulas: exponent = 1 returns the original class distinctiveness definition; exponent = 2 yields the manifold generalization of the Fisher criterion (ratio of between‑class variance to within‑class variance). Must be an integer as used to raise distances to a power in the computation.
        metric (str or dict): Metric specification used for mean estimation and distance computation (default="riemann"). Accepts the same metric identifiers as pyriemann.utils.mean.mean_covariance and pyriemann.utils.distance.distance. If a string is provided it is used for both mean and distance; if a dict is provided it can contain keys "mean" and "distance" to use different metrics for mean estimation and distance measurement respectively. Typical values include "riemann" for the affine‑invariant Riemannian metric; consult mean_covariance and distance documentation for the full supported metric list. The function internally calls check_metric(metric) to obtain (metric_mean, metric_dist).
        return_num_denom (bool): Whether to return the numerator and denominator of the class distinctiveness ratio in addition to the scalar ratio (default=False). If True the function returns a tuple (class_dis, num, denom) where num is the between‑class dispersion (numerator) and denom is the within‑class dispersion (denominator). If False only the scalar class distinctiveness value is returned.
    
    Returns:
        float or tuple: If return_num_denom is False, returns class_dis (float): the class distinctiveness scalar computed as described below. If return_num_denom is True, returns a 3‑tuple (class_dis, num, denom) where:
        class_dis (float): The ratio of between‑class dispersion to within‑class dispersion computed on the SPD/HPD manifold with the requested exponent and metrics; higher values indicate greater separability of class centers relative to within‑class scatter.
        num (float): The numerator of the ratio (between‑class term). For two classes this is d(M_K1, M_K2)^p, the pth power of the distance between class centers. For c > 2 classes this is the sum over classes of d(M_Kj, M_bar)^p, where M_bar is the mean of class centers.
        denom (float): The denominator of the ratio (within‑class term). For two classes this is 0.5*(sigma_K1^p + sigma_K2^p) where sigma_K^p is the mean of pth‑power distances from class members to their class center; for c > 2 classes this is the sum over classes of sigma_Kj^p.
    
    Behavior and implementation details:
        The function computes class centers by calling mean_covariance on the subsets X[y == c] for each unique class label c, using metric_mean derived from the metric argument. Distances are computed using metric_dist. For two classes the numerator is the distance between the two class centers raised to exponent; the denominator is half the sum of within‑class dispersions. For more than two classes the numerator is the sum of pth‑power distances from each class center to the global mean of class centers and the denominator is the sum of within‑class dispersions. The function uses check_metric(metric) internally to accept either a single metric name or a dict with "mean" and "distance" entries.
        Default behavior uses exponent=1 and metric="riemann", matching common practice in Riemannian BCI literature for covariance‑based classification and separability analysis.
    
    Side effects and performance:
        The function performs no in‑place modification of X or y and has no persistent side effects; it performs CPU computations only. Computational cost scales with the number of matrices, number of channels (matrix size), and the cost of the chosen mean and distance routines (Riemannian means and distances are typically more expensive than Euclidean metrics). Memory usage is proportional to the input arrays and temporary arrays for class means; no files are written.
    
    Defaults:
        exponent defaults to 1 to reproduce the original class distinctiveness definition from the referenced literature; metric defaults to "riemann" to use the affine‑invariant Riemannian metric for covariance matrices; return_num_denom defaults to False to return only the scalar measure.
    
    Failure modes and error conditions:
        If y contains fewer than two unique classes the function raises ValueError("y must contain at least two classes"). If within‑class dispersion is zero (denom == 0) the division num/denom will raise a ZeroDivisionError or produce an infinite result depending on floating point handling; the function does not catch this case, so callers should check denom when requesting numerator and denominator. Inputs that are not valid SPD/HPD matrices for the chosen metric may cause mean_covariance or distance to raise errors or produce invalid numerical results; ensure matrices are symmetric/Hermitian positive definite and well‑conditioned for the selected metric.
    
    Practical significance in the pyRiemann/BCI domain:
        This function provides a quantitative scalar to assess separability of classes when matrices are covariance estimates of biosignals (EEG/MEG/EMG) in BCI experiments, or covariance/HPD estimates in remote sensing applications. It can be used to compare feature extraction pipelines, channel selections, preprocessing steps, or to evaluate transfer learning strategies by measuring how changes affect between‑class vs within‑class dispersion on the SPD/HPD manifold.
    
    References:
        Implements the class distinctiveness measure described in Lotte and Jeunet (2018) and used in pyRiemann examples and research for assessing BCI skill separability.
    """
    from pyriemann.classification import class_distinctiveness
    return class_distinctiveness(X, y, exponent, metric, return_num_denom)


################################################################################
# Source: pyriemann.datasets.sampling.sample_gaussian_spd
# File: pyriemann/datasets/sampling.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def pyriemann_datasets_sampling_sample_gaussian_spd(
    n_matrices: int,
    mean: numpy.ndarray,
    sigma: float,
    random_state: int = None,
    n_jobs: int = 1,
    sampling_method: str = "auto"
):
    """Sample a Riemannian Gaussian distribution of symmetric positive definite (SPD) matrices.
    
    This function generates n_matrices SPD samples drawn from the Riemannian Gaussian distribution defined in the literature [Said et al., 2017]. It is intended for use when simulating covariance matrices or SPD observations in applications supported by pyRiemann (for example EEG/MEG covariance simulation for brain–computer interfaces, or covariance over spatial patches in hyperspectral/remote-sensing imaging). The sampling is performed in the spectral parameterization of SPD matrices: each sample is created by sampling an orthogonal eigenvector matrix (U-parameters) and the log of eigenvalues (r-parameters), forming a centered SPD sample in the tangent space, and then transporting it to be centered at the provided mean via the congruence mean_sqrt @ X @ mean_sqrt where mean_sqrt is the matrix square root of mean. Internally the dispersion sigma is corrected by 1/sqrt(n_dim) to account for dimensional scaling. The function performs an SPD validity check and warns if some generated matrices are numerically ill-conditioned.
    
    Args:
        n_matrices (int): Number of SPD matrices to generate. This controls the first dimension of the returned array and represents how many independent samples of the Riemannian Gaussian distribution are drawn.
        mean (numpy.ndarray): Center of the Riemannian Gaussian distribution. Must be a symmetric positive definite matrix of shape (n_dim, n_dim); n_dim is inferred as mean.shape[0]. The returned samples are generated around this matrix by applying the congruence transform mean_sqrt @ samples_centered @ mean_sqrt, where mean_sqrt is the matrix square root of mean. Providing a non-SPD or incorrectly shaped mean will lead to incorrect behaviour or errors when computing the matrix square root.
        sigma (float): Dispersion (scale) parameter of the Riemannian Gaussian distribution. This scalar controls the spread of samples around mean. Note that the implementation rescales sigma internally by dividing by sqrt(n_dim) (i.e., effective dispersion used is sigma / sqrt(n_dim)) to correct for dimensional effects, so users should account for this when choosing sigma.
        random_state (int | numpy.random.RandomState | None): Seed or random number generator to ensure reproducible sampling. Pass an int to seed the RNG for reproducible output across calls, pass a numpy.random.RandomState instance to control RNG state directly, or pass None (default) to use the global RNG. Values outside these options will raise an error when the RNG is initialized.
        n_jobs (int): Number of parallel jobs to use for the internal sampling computation. If set to 1 (default), sampling is done sequentially. If greater than 1, sampling of centered samples is parallelized across jobs. If set to -1, all available CPUs are used. Note: parallelism affects only the internal sampling stage and may increase memory usage and introduce nondeterminism unless random_state is fixed appropriately.
        sampling_method (str): Method used to sample eigenvalues (r-parameters). Valid values are "auto", "slice", or "rejection". The default "auto" selects "slice" when n_dim != 2 and "rejection" when n_dim == 2, matching the implementation heuristics. Choosing an unsupported string will raise a ValueError. Different methods trade off computational cost and acceptance rates; "slice" is generally used for dimensions other than 2 while "rejection" is used for 2-D to ensure correct distributional properties.
    
    Returns:
        numpy.ndarray: samples of shape (n_matrices, n_dim, n_dim) containing SPD matrices sampled from the Riemannian Gaussian centered at mean with dispersion controlled by sigma. Each slice samples[i] is a symmetric positive definite matrix obtained by transporting a centered sample to mean via congruence with mean's matrix square root. If generated matrices are numerically ill-conditioned (not sufficiently strictly positive definite), the function emits a warning indicating that some samples may not behave numerically as SPD matrices; in such a case, users should consider re-sampling, reducing matrix dimensionality, or adjusting sigma.
    
    Raises and failure modes:
        - ValueError is raised if mean does not have shape (n_dim, n_dim) consistent with a square matrix, or if sampling_method is not one of the allowed options.
        - Errors from underlying linear algebra routines (e.g., computing matrix square root) will propagate if mean is not SPD.
        - If parallel execution is requested via n_jobs and the environment does not support joblib-style parallelism, an appropriate error may be raised by the parallel backend.
        - The function issues a warning (not an exception) when some sampled matrices are very badly conditioned; such samples may fail downstream algorithms that assume strict SPD-ness.
    
    Notes on usage and significance:
        - This routine is useful for generating synthetic SPD covariance matrices for testing algorithms that operate on the manifold of SPD matrices (e.g., Riemannian classifiers, covariance estimators, simulation studies in BCI and remote sensing).
        - The returned samples are intended to be consumed by downstream pyRiemann functions (e.g., distance computations, tangent space mappings, or classifiers) that expect valid SPD inputs.
    """
    from pyriemann.datasets.sampling import sample_gaussian_spd
    return sample_gaussian_spd(
        n_matrices,
        mean,
        sigma,
        random_state,
        n_jobs,
        sampling_method
    )


################################################################################
# Source: pyriemann.datasets.simulated.make_masks
# File: pyriemann/datasets/simulated.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def pyriemann_datasets_simulated_make_masks(
    n_masks: int,
    n_dim0: int,
    n_dim1_min: int,
    rs: int = None
):
    """Generate a list of semi-orthogonal matrices ("masks") used to simulate linear mixing of multivariate signals.
    
    Args:
        n_masks (int): Number of masks to generate. In the simulated-datasets context (see README), each mask can act as a distinct linear mixing matrix for synthetic multichannel time series (for example, to simulate EEG/MEG channel mixing in BCI experiments or spatial mixing in remote sensing). If n_masks is zero or negative, the function returns an empty list (no masks); if it is not an integer a TypeError will be raised by Python's range().
        n_dim0 (int): First dimension of each mask (number of rows). Practically, this corresponds to the number of output channels or sensors in the simulated mixing (e.g., number of EEG channels). It must be an integer; values less than or equal to zero will lead to shapes that are not meaningful for downstream processing and may cause linear algebra functions to raise errors.
        n_dim1_min (int): Minimal allowed value for the second dimension (number of columns) of each mask. For each mask i the function samples an integer n_dim1_i such that n_dim1_min <= n_dim1_i < n_dim0 (the upper bound is exclusive, see behavior below). In applications, n_dim1_i is the number of latent sources or independent components mixed into n_dim0 sensors. If n_dim1_min is greater than or equal to n_dim0, the internal random sampling will raise a ValueError because a valid integer in the required range cannot be drawn.
        rs (int | RandomState instance | None, default=None): Random state for reproducible output across multiple function calls. If rs is None (default), a new non-deterministic RandomState is used so repeated calls produce different masks. If rs is an integer, it is interpreted as a seed to create a reproducible RandomState. If rs is an existing RandomState instance, it is used directly. Internally the function calls check_random_state(rs) to obtain a numpy RandomState-like object.
    
    Returns:
        list of ndarray: A list containing n_masks numpy arrays. Each array has shape (n_dim0, n_dim1_i) where n_dim1_i is independently drawn per mask from integers satisfying n_dim1_min <= n_dim1_i < n_dim0. Each returned array is semi-orthogonal: its columns are orthonormal (Q from a QR decomposition of a random Gaussian matrix), so mask.T @ mask equals the identity matrix of size n_dim1_i within numerical precision. The list length equals n_masks; if n_masks <= 0 the returned list is empty.
    
    Behavior and side effects:
        The function constructs each mask by sampling n_dim1_i with rs.randint(n_dim1_min, n_dim0) and then generating a random n_dim0-by-n_dim1_i matrix with rs.randn(...) from a standard normal distribution. It computes a QR decomposition (numpy.linalg.qr) and returns the Q factor as the semi-orthogonal mask. No global state besides the provided RandomState is modified. Providing a deterministic rs (seed or RandomState) guarantees reproducible masks across multiple calls.
    
    Failure modes:
        If n_dim1_min >= n_dim0 a ValueError will be raised by the underlying randint call because the half-open interval [n_dim1_min, n_dim0) is empty. If non-integer types are passed for n_masks, n_dim0, or n_dim1_min Python built-ins (e.g., range, ndarray shape construction) or numpy functions will raise TypeError or ValueError. If n_dim0 is not large enough relative to n_dim1_min the function cannot produce valid semi-orthogonal matrices and will raise an error as described above.
    """
    from pyriemann.datasets.simulated import make_masks
    return make_masks(n_masks, n_dim0, n_dim1_min, rs)


################################################################################
# Source: pyriemann.datasets.simulated.make_matrices
# File: pyriemann/datasets/simulated.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def pyriemann_datasets_simulated_make_matrices(
    n_matrices: int,
    n_dim: int,
    kind: str,
    rs: int = None,
    return_params: bool = False,
    evals_low: float = 0.5,
    evals_high: float = 2.0,
    eigvecs_same: bool = False,
    eigvecs_mean: float = 0.0,
    eigvecs_std: float = 1.0
):
    """Generate square matrices with controlled spectral properties for testing and simulation in pyRiemann algorithms (for example, to synthesize covariance-like SPD matrices used in biosignal processing and BCI pipelines). The function constructs real, symmetric, complex, Hermitian, positive-(semi)definite matrices or general complex matrices by drawing random eigenvectors and eigenvalues under specified distributions, optionally returning the eigen-parameters used to build the matrices for reproducibility and analysis.
    
    Args:
        n_matrices (int): Number of square matrices to generate. In the pyRiemann context this corresponds to the number of samples or trials (for example, number of epochs for EEG) for which one may want synthetic covariance or general matrices. Must be a non-negative integer; if zero, an empty array with shape (0, n_dim, n_dim) will be produced by NumPy behavior.
        n_dim (int): Dimension of each square matrix. In BCI/covariance applications this is the number of channels (spatial dimension). Must be a positive integer; it defines the second and third dimensions of the returned array of matrices.
        kind (str): Type of matrices to generate. Accepted kinds are those used in pyRiemann to represent data and model matrices: "real" (real-valued matrices), "sym" (real symmetric matrices), "spd" (symmetric positive-definite matrices), "spsd" (symmetric positive semi-definite matrices), "comp" (complex-valued matrices), "herm" (Hermitian matrices), "hpd" (Hermitian positive-definite matrices), and "hpsd" (Hermitian positive semi-definite matrices). The choice of kind determines whether the function returns general random matrices, symmetric/Hermitian matrices, or constructs matrices from sampled eigenvalues and orthonormal eigenvectors to guarantee (semi)definiteness. If an unsupported kind is provided, the function raises ValueError indicating the unsupported matrix kind.
        rs (int | RandomState instance | None): Random state for reproducible output across calls. Can be an integer seed, a numpy RandomState (or Generator/RandomState depending on check_random_state implementation), or None to use NumPy’s global RNG. The function calls check_random_state(rs) internally to obtain a RandomState-like object for all subsequent random draws; passing the same rs will reproduce the same matrices and eigen-parameters.
        return_params (bool): If True, also return the eigenvalues and eigenvectors used to construct matrices for the kinds that use spectral construction ("spd", "spsd", "hpd", "hpsd"). This is useful for unit tests, debugging, or when one needs to inspect or reuse the spectral decomposition that generated the matrices (for example, to feed into other algorithms or to verify condition numbers). Default is False, in which case only the generated matrices are returned.
        evals_low (float): Lowest value (inclusive) of the uniform distribution used to draw eigenvalues for spectral constructions. In positive-definite/Hermitian cases eigenvalues are drawn uniformly in [evals_low, evals_high). Must be strictly positive (the function raises ValueError if evals_low <= 0.0). Default is 0.5. This parameter controls the minimum scale of variance in synthetic covariance matrices.
        evals_high (float): Highest value (exclusive upper bound) of the uniform distribution used to draw eigenvalues for spectral constructions. Must be strictly greater than evals_low (the function raises ValueError if evals_high <= evals_low). Default is 2.0. Together with evals_low it sets the dynamic range of eigenvalues and thus condition numbers of generated (H)PD matrices.
        eigvecs_same (bool): If True, use the same set of sampled eigenvectors for all generated matrices; if False, sample independent eigenvector matrices for each sample. When True, the returned eigenvectors may have shape (n_dim, n_dim) rather than (n_matrices, n_dim, n_dim). This option is useful to simulate datasets with common spatial patterns across trials (a common scenario in BCI where the mixing matrix is shared across epochs).
        eigvecs_mean (float): Mean of the normal distribution used to draw raw eigenvector matrices before orthonormalization. The function draws entries ~ Normal(loc=eigvecs_mean, scale=eigvecs_std) to form candidate eigenvector matrices which are then orthonormalized via QR to obtain true eigenvectors. Default is 0.0. In practice this controls the center of the random matrix entries prior to orthonormalization.
        eigvecs_std (float): Standard deviation of the normal distribution used to draw raw eigenvector matrices before orthonormalization. Default is 1.0. Larger values increase variability of the initial random matrices and thus indirectly affect the distribution of orthonormal eigenvectors after QR.
    
    Behavior, defaults, side effects, and failure modes:
        The function first validates that kind is one of the supported kinds; otherwise it raises ValueError("Unsupported matrix kind: {kind}"). For "real" the function returns a real-valued array of shape (n_matrices, n_dim, n_dim) with independent Gaussian entries ~ N(eigvecs_mean, eigvecs_std^2). For "sym" it returns symmetric real matrices constructed as X + X^T where X is the raw random array.
        For kinds that include complex parts ("comp", "herm", "hpd", "hpsd") the function draws an additional independent real random array Y and forms complex matrices X + 1j * Y; for "herm" it returns Hermitian matrices formed as X + X^T + 1j * (Y - Y^T).
        For positive-(semi)definite kinds ("spd", "spsd", "hpd", "hpsd") the function performs a spectral construction: it samples eigenvalues uniformly in [evals_low, evals_high) for each matrix (array of shape (n_matrices, n_dim)). If kind is "spsd" or "hpsd" the function forces the last eigenvalue to a value close to zero (1e-10) to create a near-singular matrix representing a positive semi-definite matrix. The function checks and raises ValueError if evals_low <= 0.0 or if evals_high <= evals_low.
        Eigenvectors are sampled by drawing Gaussian matrices with mean eigvecs_mean and std eigvecs_std and then orthonormalizing via NumPy’s QR decomposition (np.linalg.qr). For NumPy versions prior to 1.22.0 the code falls back to orthonormalizing each matrix individually and stacking results (implementation detail that may affect performance). If eigvecs_same is True, the same orthonormal eigenvector matrix is used for all samples.
        Construction of matrices from eigen-decomposition follows standard conjugation: for real symmetric/spd/spsd this is V * diag(evals) * V^T; for complex Hermitian/Hermitian PD/PSD this is V * diag(evals) * V^H (conjugate transpose). The function uses a small epsilon for semi-definite kinds as noted above.
        The random draws use the RandomState obtained from check_random_state(rs) to provide reproducibility when rs is provided. No global state is modified beyond usage of the RNG in a standard manner.
        The function may return complex-valued arrays when kind is one of "comp", "herm", "hpd", or "hpsd". For "real", "sym", "spd", "spsd" the outputs are real-valued.
        Failure modes that raise exceptions: unsupported kind raises ValueError; non-positive evals_low raises ValueError; evals_high <= evals_low raises ValueError. Other runtime errors may propagate from NumPy (for example, if n_dim is negative or QR fails due to invalid shapes).
    
    Returns:
        mats (ndarray, shape (n_matrices, n_dim, n_dim)): Generated set of square matrices. For positive-definite/Hermitian constructions the matrices are guaranteed to be (H)PD when kind is "spd" or "hpd", and nearly (H)PSD when kind is "spsd" or "hpsd" (last eigenvalue set to 1e-10). The dtype may be complex for complex kinds.
        When return_params is True, returns a tuple (mats, evals, evecs) where:
            evals (ndarray, shape (n_matrices, n_dim)): Eigenvalues drawn and used for spectral constructions ("spd", "spsd", "hpd", "hpsd"). For semi-definite kinds the last column contains the small value used to approximate zero (1e-10).
            evecs (ndarray, shape (n_matrices, n_dim, n_dim) or (n_dim, n_dim)): Orthonormal eigenvector matrices used to construct the outputs. If eigvecs_same is True, evecs has shape (n_dim, n_dim) (shared across all matrices); otherwise it has shape (n_matrices, n_dim, n_dim). For complex Hermitian/Hermitian-PD(-PSD) kinds, evecs are complex; for real kinds they are real.
        If return_params is False, only mats is returned.
    
    Notes:
        This function is intended for simulation and testing within the pyRiemann framework (for instance to create synthetic covariance matrices or evaluate Riemannian classifiers/metrics). It uses QR-based orthonormalization to ensure eigenvector matrices are unitary/orthonormal, and constructs matrices via conjugation with the sampled eigenvalues. The semi-definite option uses a fixed small value (1e-10) for the smallest eigenvalue to emulate rank-deficiency without producing exact zeros that could cause numerical issues in some algorithms.
    """
    from pyriemann.datasets.simulated import make_matrices
    return make_matrices(
        n_matrices,
        n_dim,
        kind,
        rs,
        return_params,
        evals_low,
        evals_high,
        eigvecs_same,
        eigvecs_mean,
        eigvecs_std
    )


################################################################################
# Source: pyriemann.datasets.simulated.make_outliers
# File: pyriemann/datasets/simulated.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def pyriemann_datasets_simulated_make_outliers(
    n_matrices: int,
    mean: numpy.ndarray,
    sigma: float,
    outlier_coeff: float = 10,
    random_state: int = None
):
    """Generate outlier SPD matrices for a Riemannian Gaussian distribution with a fixed mean and dispersion.
    
    This function is used in the pyRiemann simulated datasets context to create covariance-like symmetric positive definite (SPD) matrices that lie far from a given Riemannian Gaussian center. Such outliers are useful to test robustness of Riemannian classifiers and processing pipelines (for example in biosignal/BCI or remote sensing applications where covariance matrices are processed on the SPD manifold). The generation procedure (used internally by pyRiemann) is: compute the matrix square root of the provided mean, draw random SPD matrices O_i, compute a scalar exponent epsilon from outlier_coeff, sigma and the Riemannian squared distance between O_i and the identity, then map O_i to the tangent-scaled outlier by conjugation outlier = mean_sqrt @ O_i**epsilon @ mean_sqrt. The resulting matrices are SPD and have shape (n_matrices, n_dim, n_dim).
    
    Args:
        n_matrices (int): Number of outlier matrices to generate. This determines the first dimension of the returned array. Must be a non-negative integer; passing zero returns an array with shape (0, n_dim, n_dim).
        mean (numpy.ndarray): Center of the Riemannian Gaussian distribution. Must be a square ndarray of shape (n_dim, n_dim) representing an SPD matrix (symmetric positive definite). The function uses mean to compute its matrix square root (mean_sqrt = sqrtm(mean)) and to conjugate generated matrices, so mean must be numerically symmetric and positive definite for meaningful results; otherwise a linear algebra error or incorrect output may occur.
        sigma (float): Dispersion (scale) parameter of the Riemannian Gaussian distribution. A non-negative scalar controlling how far outliers will be placed relative to the distribution spread. sigma participates linearly in the computation of the exponent epsilon; sigma <= 0 will produce degenerate scaling (epsilon may be zero), so provide a positive sigma for standard outlier behaviour.
        outlier_coeff (float, default=10): Coefficient that scales the definition of an outlier in units of sigma. Conceptually, this parameter controls "how many times the sigma parameter its distance to the mean should be" (as in the original implementation): larger values create more distant outliers. The default value 10 is the library default used historically in pyRiemann to produce clear outliers.
        random_state (int | RandomState instance | None, default=None): Pseudo-random number generator or seed used to draw intermediate SPD matrices. Pass an int for reproducible output across multiple calls, pass an instance of numpy.random.RandomState for a specific RNG state, or pass None to use the global RNG. The provided random_state is forwarded to the internal make_matrices call; therefore it controls reproducibility and also will be advanced (consumes randomness) as a side effect.
    
    Returns:
        outliers (numpy.ndarray): Array of generated outlier matrices with shape (n_matrices, n_dim, n_dim) and dtype float (standard NumPy ndarray). Each slice outliers[i] is an SPD matrix obtained as mean_sqrt @ powm(O_i, epsilon) @ mean_sqrt, where O_i is a random SPD matrix and epsilon is computed from outlier_coeff, sigma and the Riemannian squared distance between O_i and the identity. If n_matrices is zero, an empty array with shape (0, n_dim, n_dim) is returned.
    
    Behavior, side effects, defaults, and failure modes:
        - The function computes n_dim from mean.shape[1]; therefore mean must be two-dimensional and square (shape (n_dim, n_dim)). Supplying non-square mean will lead to an indexing or shape mismatch error.
        - mean is expected to be symmetric positive definite. If mean is not SPD, the matrix square root operation (sqrtm) or subsequent conjugation may fail or return non-SPD matrices.
        - sigma should be positive for meaningful outliers. sigma <= 0 can lead to epsilon == 0 and therefore all outliers equal to the mean, or to unexpected numerical behavior.
        - The computation of epsilon divides by the squared Riemannian distance distance_riemann(O_i, I, squared=True). If this denominator is zero (which can happen if the sampled O_i equals the identity matrix within numerical precision), a division by zero will occur raising a runtime or floating-point error; in practice sampling internals avoid exact identity but callers should be aware of this potential.
        - The function consumes randomness from the provided random_state (or the global RNG if None) as a side effect; repeated calls with the same seed produce reproducible results.
        - No in-place modification of the input mean is performed; the only outputs are the returned ndarray and the RNG state advancement.
        - The function was added to pyRiemann to support simulated dataset creation (version added in upstream history).
    """
    from pyriemann.datasets.simulated import make_outliers
    return make_outliers(n_matrices, mean, sigma, outlier_coeff, random_state)


################################################################################
# Source: pyriemann.stats.multiset_perm_number
# File: pyriemann/stats.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def pyriemann_stats_multiset_perm_number(y: numpy.ndarray):
    """pyriemann.stats.multiset_perm_number returns the number of unique permutations of the provided multiset array.
    This function is used in pyRiemann statistical utilities to quantify how many distinct orderings exist for a multiset of elements such as class labels or categorical epoch identifiers (for example in EEG/BCI permutation tests or combinatorial counts applied to covariance matrix labels).
    
    Args:
        y (numpy.ndarray): One-dimensional numpy array representing the multiset elements whose unique permutation count is required.
            Each entry is considered an element of the multiset (for example, labels for epochs in a BCI experiment). The algorithm
            computes the multiplicity of each distinct element (using numpy.unique and equality comparisons) and uses these multiplicities
            to compute the multinomial denominator. The function expects values that compare equal for identical elements; note that
            NaN values do not compare equal to themselves and therefore are not reliably grouped by this routine. The function uses
            len(y) and numpy.unique(y), so passing a multi-dimensional array will cause the length of the first axis to be used,
            which may produce unintended results. Provide a 1-D numpy.ndarray for correct semantic behavior.
    
    Returns:
        float: The number of unique permutations of the multiset represented by y, computed as factorial(len(y)) divided by the
        product of factorials of each element multiplicity:
            result = factorial(n) / prod(factorial(m_i))
        where n = len(y) and m_i are multiplicities of each distinct value in y. Mathematically this value is an integer, but the
        implementation returns a float because it performs true division. For small n the float will exactly represent the integer;
        for large n the float may lose precision. If an exact integer representation is required and the result is known to fit in Python's
        integer range, convert the returned value with int(result). No side effects occur; the function is pure. Be aware that for very
        large n the intermediate factorial computations can be computationally heavy and produce very large integers that may affect
        performance and floating-point precision.
    """
    from pyriemann.stats import multiset_perm_number
    return multiset_perm_number(y)


################################################################################
# Source: pyriemann.stats.unique_permutations
# File: pyriemann/stats.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def pyriemann_stats_unique_permutations(elements: list):
    """pyriemann.stats.unique_permutations returns a generator that yields all unique permutations of the input list elements as tuples. This function is a utility suitable for use in pyRiemann workflows where enumerating distinct orderings of items is needed (for example, generating distinct channel index orders, enumerating label permutations for nonparametric testing in BCI experiments, or exhaustive combinations in small-scale combinatorial search over feature orderings). It produces each permutation lazily (one at a time) and avoids emitting duplicate permutations when the input list contains repeated values.
    
    Args:
        elements (list): A Python list of items to permute. Each yielded permutation is a tuple containing the same items in a particular order and has length equal to len(elements). The function uses set(elements) internally to detect distinct values at each recursion level; therefore, duplicate values in elements produce fewer than n! permutations (only unique orderings). The parameter is not modified by the function: a shallow copy of the list is used when removing elements during recursion.
    
    Returns:
        generator: A generator that yields tuples. Each yielded value is a tuple representing a unique permutation of the input list elements. The generator yields permutations lazily (on-demand) rather than returning a concrete list of all permutations.
    
    Behavior and side effects:
        The implementation is recursive: the base case yields a single-element tuple when len(elements) == 1; otherwise, it iterates over the set of unique values present in elements to select a first element, removes one occurrence of that element from a copy of the list, and recursively yields permutations of the remaining items, prepending the selected first element. Because set(...) is used to enumerate candidate first elements, the order in which permutations are yielded is not guaranteed and may vary between Python versions or runs. The original input list is not mutated; copies are made for recursive calls.
    
    Failure modes and limits:
        If elements is an empty list, the function yields no values (no permutations are produced). If elements contains unhashable objects (for example, lists or dicts), calling set(elements) will raise a TypeError. The number of unique permutations grows combinatorially with len(elements); for modest n the total number may be large and iteration may be very slow or impractical. Deep recursion may raise a RecursionError for very large lists. The function expects a list as documented in the signature; passing a different type may change behavior or cause errors.
    """
    from pyriemann.stats import unique_permutations
    return unique_permutations(elements)


################################################################################
# Source: pyriemann.utils.ajd.ajd
# File: pyriemann/utils/ajd.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def pyriemann_utils_ajd_ajd(
    X: numpy.ndarray,
    method: str = "ajd_pham",
    init: numpy.ndarray = None,
    eps: float = 1e-06,
    n_iter_max: int = 100,
    **kwargs
):
    """pyriemann.utils.ajd.ajd computes an approximate joint diagonalization (AJD) of a set of symmetric matrices using a selected AJD algorithm implementation. In the pyRiemann workflow for multivariate biosignal processing (for example EEG/MEG covariance-based BCI pipelines), this function is used to estimate a joint diagonalizer matrix V that approximately diagonalizes a set of covariance (SPD) or symmetric matrices X; the diagonalizer V can be interpreted as an unmixing/spatial filter matrix and the returned quasi-diagonal matrices D contain the per-matrix component powers that are useful for downstream classification or feature extraction.
    
    Args:
        X (numpy.ndarray): Set of symmetric matrices to diagonalize. Expected shape is (n_matrices, n, n) where each X[i] is a real symmetric matrix (for example covariance matrices estimated from EEG epochs). The function assumes symmetry in each matrix; providing non-symmetric matrices may lead to incorrect results or errors in the underlying AJD implementations. This argument is the primary input on which the AJD operates and its values determine the joint diagonalizer V.
        method (str | callable): Method for performing AJD. Default is "ajd_pham". Accepted string values correspond to registered implementations such as "ajd_pham", "rjd", "uwedge"; alternatively a callable implementing the AJD interface may be passed directly. When a string is passed, the function looks up the corresponding algorithm implementation and dispatches to it; if the string is not recognized a ValueError will be raised. The choice of method controls the numerical algorithm, convergence behavior, and practical suitability for different data conditions (noise level, number of matrices, conditioning).
        init (numpy.ndarray | None): Optional initialization matrix for the diagonalizer. Expected shape is (n, n) where n is the matrix dimension in X (the second/third axes). If None (default), the AJD subfunction decides a default initialization (commonly the identity matrix or an algorithm-specific heuristic). Providing an explicit init allows users to seed the algorithm with a prior unmixing matrix (for example from a previous session) which can improve convergence or enforce continuity in transfer-learning workflows.
        eps (float): Tolerance for the stopping criterion of the underlying AJD algorithm. Default is 1e-6. This parameter controls when iterations are considered to have converged: smaller values demand closer-to-diagonal results and may increase iteration counts or numerical sensitivity; larger values can speed execution at the cost of less precise diagonalization. eps must be non-negative; negative values are not valid and will likely cause errors in the subfunction.
        n_iter_max (int): Maximum number of iterations allowed to reach convergence. Default is 100. If convergence (as defined by eps and the chosen method) is not achieved within n_iter_max iterations, the function returns the current estimate (i.e., the last iterate) and does not raise an exception. Users should check convergence via method-specific diagnostics if strict convergence guarantees are required.
        kwargs (dict): Additional keyword arguments forwarded directly to the chosen AJD implementation. These are implementation-specific options (for example step sizes, weighting, or verbose flags) and should match the signature expected by the selected method. If unexpected keywords are provided, the called subfunction may raise a TypeError.
    
    Returns:
        V (numpy.ndarray): The estimated diagonalizer, shape (n, n). In practical terms for pyRiemann applications, V acts as an unmixing or spatial filter matrix that, when applied to the original data or to the matrices in X, yields signals/components that are as decorrelated/diagonal as possible in the joint sense across the set. V is computed by the chosen AJD algorithm and its scale/ordering may be algorithm-dependent.
        D (numpy.ndarray): Set of quasi-diagonal matrices, shape (n_matrices, n, n). These matrices are the result of transforming each input matrix by the diagonalizer (for many algorithms D[i] ≈ V @ X[i] @ V.T). In BCI and covariance-processing contexts, the diagonals of D contain component variances (power) per matrix/epoch and are commonly used as features for classification.
    
    Behavior, side effects, and failure modes:
        - The function is a thin dispatcher: it resolves the requested method (string lookup or direct callable) and calls that AJD implementation with provided arguments. The specific numerical behavior, returned ordering/scaling of V, and any convergence diagnostics depend on the chosen implementation.
        - No in-place modification of the input X is performed by this function itself (subfunctions may or may not copy/modify internal buffers); callers should not rely on X being preserved unchanged if using third-party AJD implementations.
        - If the shape of X is inconsistent (not (n_matrices, n, n)) or if init has a mismatched shape, a ValueError or IndexError may be raised by the function or the underlying AJD implementation.
        - If method is a string that does not match a registered implementation, a ValueError will be raised. If method is a callable that does not accept the expected signature, a TypeError may be raised by Python at call time.
        - Convergence may not be reached within n_iter_max; in that case the function returns the last iterate (V, D) without raising an exception. Users requiring strict convergence should inspect method-specific outputs or increase n_iter_max and/or tighten eps.
        - Numerical issues (poor conditioning of X, near-singular matrices, or very small eps) can lead to instability or warnings from numerical linear algebra routines used by the AJD implementations.
    
    Notes:
        - This function is intended for use in pyRiemann pipelines for processing covariance or symmetric matrices derived from multivariate time-series (e.g., EEG epochs for BCI). The diagonalizer V and quasi-diagonal matrices D are commonly used to extract features, design spatial filters, or perform transfer learning between sessions/subjects.
        - Default values: method="ajd_pham", init=None, eps=1e-6, n_iter_max=100.
        - Additional method-specific parameters should be passed via kwargs and must be documented according to the specific AJD implementation used.
    """
    from pyriemann.utils.ajd import ajd
    return ajd(X, method, init, eps, n_iter_max, **kwargs)


################################################################################
# Source: pyriemann.utils.ajd.ajd_pham
# File: pyriemann/utils/ajd.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def pyriemann_utils_ajd_ajd_pham(
    X: numpy.ndarray,
    init: numpy.ndarray = None,
    eps: float = 1e-06,
    n_iter_max: int = 20,
    sample_weight: numpy.ndarray = None
):
    """pyriemann.utils.ajd.ajd_pham computes an approximate joint diagonalization (AJD) of a set of symmetric positive definite (SPD) or Hermitian positive definite (HPD) matrices using Pham's optimization of a log-likelihood criterion based on the Kullback-Leibler divergence. In the pyRiemann context, this function is used to find an invertible diagonalizer matrix V that transforms a batch of covariance matrices (e.g., covariance matrices estimated from multichannel biosignals such as EEG/MEG/EMG in brain–computer interface workflows, or covariance patches in hyperspectral/remote sensing applications) into approximately diagonal form D = V X V^H. The diagonalizer V and the transformed quasi-diagonal matrices D are commonly used downstream for Riemannian geometry processing, tangent-space embedding, or classifiers such as MDM.
    
    Args:
        X (numpy.ndarray): Input array of matrices with shape (n_matrices, n, n). Each entry X[i] is expected to be an SPD (if real) or HPD (if complex) matrix, typically covariance matrices estimated from multivariate time series. The function treats X as the set of matrices to be jointly diagonalized and uses its dtype to cast internal variables and the outputs.
        init (None | numpy.ndarray): Initialization for the diagonalizer with shape (n, n). If None, the identity matrix I_n is used as the initial diagonalizer. If provided, init is validated (shape must match n) and its dtype will be coerced to match X.dtype. The initialization influences convergence speed and the local minimum reached by the iterative AJD algorithm.
        eps (float): Tolerance for the stopping criterion (default 1e-6). Internally the algorithm uses an effective threshold equal to n*(n-1)*eps, where n is the matrix dimension. A smaller eps yields a stricter convergence test (potentially more iterations) and a larger eps relaxes convergence (fewer iterations). eps does not change the algebraic steps of the algorithm but controls when the iterative process is considered converged.
        n_iter_max (int): Maximum number of outer iterations to attempt (default 20). The algorithm performs up to n_iter_max full sweeps over matrix row/column pairs; it stops early if the convergence criterion is met. If the loop completes without meeting the criterion, a warning is emitted indicating convergence was not reached.
        sample_weight (None | numpy.ndarray): One-dimensional array of length n_matrices providing strictly positive weights for each matrix (shape (n_matrices,)). If None, equal weights are used. Weights are validated for positivity and normalized to sum to 1 before use (the function calls the package weight checker). The weights bias the algorithm toward better diagonalization of more heavily weighted matrices.
    
    Returns:
        tuple:
            V (numpy.ndarray): Array of shape (n, n) containing the learned diagonalizer matrix V (same dtype as X). V is returned as an invertible matrix such that multiplying the original matrices by V on the left and V^H on the right yields the quasi-diagonal matrices D. In practice V is used to project covariance matrices into an approximately independent coordinate system for downstream Riemannian processing or classification.
            D (numpy.ndarray): Array of shape (n_matrices, n, n) containing the quasi-diagonalized matrices D = V X V^H (same dtype as X). Each D[i] is the transformed version of X[i] under V and is approximately diagonal; these are the matrices one would feed to algorithms that assume diagonal or nearly diagonal covariance representations.
    
    Behavior, side effects, and failure modes:
        The algorithm is an iterative pairwise Jacobi-like procedure derived from Pham (2000). It performs complex arithmetic if X is complex; for real-valued inputs, special-case branches ensure the outputs remain real. The function normalizes sample_weight to sum to 1 and raises an error if weights are non-positive. If init is None the identity matrix is used; otherwise init is validated for shape and coerced to X.dtype. The stopping criterion is based on an accumulated "crit" value compared to the threshold n*(n-1)*eps; whenever crit < n*(n-1)*eps the algorithm stops early. If the maximum number of iterations n_iter_max is reached without satisfying the stopping criterion, the function emits a warnings.warn("Convergence not reached") and still returns the current V and D (caller should check this warning to detect possible lack of convergence). Inputs that are not SPD/HPD (for example non-symmetric or singular matrices) are outside the documented preconditions: the algorithm assumes SPD/HPD input and results may be meaningless or unstable if this assumption is violated. The outputs preserve the dtype of X.
    """
    from pyriemann.utils.ajd import ajd_pham
    return ajd_pham(X, init, eps, n_iter_max, sample_weight)


################################################################################
# Source: pyriemann.utils.ajd.rjd
# File: pyriemann/utils/ajd.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def pyriemann_utils_ajd_rjd(
    X: numpy.ndarray,
    init: numpy.ndarray = None,
    eps: float = 1e-08,
    n_iter_max: int = 100
):
    """pyriemann.utils.ajd.rjd: Orthogonal approximate joint diagonalization based on the JADE algorithm. This function implements the orthogonal AJD algorithm (JADE) using Jacobi rotations to find an orthogonal diagonalizer V that jointly (approximately) diagonalizes a set of symmetric matrices X. In the pyRiemann project this routine is used to diagonalize collections of symmetric positive definite covariance matrices (for example covariance estimates from multichannel biosignals such as EEG/MEG used in brain-computer interface applications) so that subsequent Riemannian processing, feature extraction, or transfer-learning operations can operate on quasi-diagonal representations.
    
    Args:
        X (numpy.ndarray): Set of symmetric matrices to diagonalize, with shape (n_matrices, n, n). Each slice X[i] is expected to be a real symmetric matrix (in pyRiemann typical inputs are covariance matrices estimated from multivariate time series). The algorithm concatenates and transposes X internally to build an intermediate array A; the input array X is not modified by the function. Supplying non-square arrays or arrays whose first dimension does not match the remaining two will cause shape errors.
        init (numpy.ndarray or None): Initialization for the diagonalizer, with shape (n, n). If None (the default), the identity matrix I_n is used as the initial orthogonal diagonalizer V. When provided, init is validated (via the internal check_init utility) for correct shape and compatibility with X; if init is incompatible the utility will raise an error. Providing a good initialization (for example an approximate eigenbasis) can reduce the number of Jacobi rotations and speed convergence.
        eps (float): Tolerance for the stopping criterion (default=1e-08). During iterations the algorithm computes Givens rotation angles; a rotation is applied only when its sine magnitude exceeds eps. The iterative process halts early when no rotation with |sine| > eps is found across all index pairs. Setting eps larger makes the algorithm more tolerant (fewer rotations, faster but less accurate diagonalization); setting eps smaller increases precision but may require more iterations and more compute.
        n_iter_max (int): The maximum number of outer iterations (default=100). Each outer iteration sweeps over all distinct index pairs (p, q) with p < q and may apply Jacobi rotations. If the algorithm does not reach the stopping criterion within n_iter_max iterations, the function exits the loop and emits a warning indicating that convergence was not reached. Increasing n_iter_max gives the algorithm more opportunity to converge at the cost of additional computation.
    
    Returns:
        V (numpy.ndarray): The computed diagonalizer, an orthogonal matrix of shape (n, n). Practically, V is the orthogonal change-of-basis matrix such that the returned quasi-diagonal matrices satisfy D[i] = V.T @ X[i] @ V (within numerical precision). In Riemannian applications this V can be used to align or normalize covariance matrices before downstream processing.
        D (numpy.ndarray): Array of quasi-diagonal matrices with shape (n_matrices, n, n). D contains the set of jointly transformed matrices D[i] = V.T @ X[i] @ V. Each D[i] is expected to be close to diagonal if joint diagonalization succeeded; off-diagonal values indicate residual coupling between components.
    
    Behavior, side effects, defaults, and failure modes:
        - The algorithm implements the Jacobi-angle JADE orthogonal AJD: it builds an internal 2D array A from X by concatenation and transposition, then iteratively applies Givens rotations (computed from pairs of rows/columns) to reduce off-diagonal energy.
        - The input X is not modified; intermediate arrays (A, V) are created and updated internally. The returned V and D are new numpy.ndarray objects.
        - The default initialization is the identity matrix. If a user-supplied init is passed, it is validated for shape and compatibility; invalid init leads to an error from the validation routine.
        - Convergence is declared when a full sweep applies no rotation with |s| > eps. If convergence is not reached within n_iter_max outer iterations, the function exits and issues a runtime warning ("Convergence not reached"); in that case the returned V and D correspond to the last computed iterate and may not achieve the desired diagonalization accuracy.
        - The function assumes real-valued symmetric matrices in X. Providing non-symmetric, complex, or otherwise incompatible matrices may lead to incorrect results, shape errors, or failures in numerical operations.
        - Numerical stability depends on eps, the conditioning of input matrices, and the number of iterations. For ill-conditioned covariance matrices or extremely small eps, round-off errors can affect results and may require preprocessing (regularization) of X before calling this function.
    
    Notes:
        - This routine is a translation of the original Matlab implementation of the JADE orthogonal AJD algorithm and follows the formulation described by Cardoso and Souloumiac (1996). It is commonly used in pyRiemann workflows to obtain quasi-diagonal representations of covariance matrices for BCI, EEG/MEG processing, and related multivariate signal analysis tasks.
    """
    from pyriemann.utils.ajd import rjd
    return rjd(X, init, eps, n_iter_max)


################################################################################
# Source: pyriemann.utils.ajd.uwedge
# File: pyriemann/utils/ajd.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def pyriemann_utils_ajd_uwedge(
    X: numpy.ndarray,
    init: numpy.ndarray = None,
    eps: float = 1e-07,
    n_iter_max: int = 100
):
    """pyriemann.utils.ajd.uwedge performs approximate joint diagonalization of a set of symmetric matrices using the U-WEDGE algorithm (Uniformly Weighted Exhaustive Diagonalization using Gauss iterations). This routine is used in pyRiemann for processing collections of symmetric (typically covariance) matrices arising from multichannel biosignals (e.g., EEG, MEG, EMG) or other applications where joint diagonalization aids source separation, dimensionality reduction, or Riemannian-based classification pipelines.
    
    Args:
        X (numpy.ndarray): Set of symmetric matrices to diagonalize with shape (n_matrices, n, n). Each entry X[k] is an n x n real (or complex) symmetric (or Hermitian) matrix; in BCI and remote sensing applications these are typically covariance matrices estimated from multichannel time series or image patches. The algorithm concatenates these matrices internally (see source), so providing the full collection in this shape is required.
        init (numpy.ndarray): Initialization matrix for the diagonalizer with shape (n, n). If None (the default), the function computes an initialization from the eigen-decomposition of the first block of the concatenated data as implemented in the original Matlab reference code: it uses eigenvectors and rescales them by the inverse square root of the absolute eigenvalues. If a numpy.ndarray is provided, it will be validated (via check_init in the source) and used as the starting diagonalizer V. Supplying a good initialization can speed convergence or improve stability when working with ill-conditioned matrices typical of real biosignal covariance estimates.
        eps (float): Stopping tolerance for the relative change of the AJD criterion. The algorithm iterates until the absolute difference between successive criterion values is below eps. Default is 1e-7. Use a smaller eps for stricter convergence (at the cost of more iterations) or a larger eps to stop earlier when an approximate diagonalization is acceptable for downstream tasks such as classification or feature extraction.
        n_iter_max (int): Maximum number of Gauss-iteration sweeps to perform (default 100). If the algorithm does not satisfy the eps stopping criterion within n_iter_max iterations, the function emits a runtime warning ("Convergence not reached") and returns the best-estimate diagonalizer found so far. Increase this value when signals require more iterations for convergence; be aware that larger values increase computation time.
    
    Returns:
        V (numpy.ndarray): The estimated diagonalizer with shape (n, n). V is the linear transform (applied on the left, and its transpose on the right) that approximately jointly diagonalizes the input set: for each k, D[k] = V @ X[k] @ V.T. In BCI and related workflows V can be interpreted as an unmixing or whitening-like transform that makes components approximately independent across the provided matrices. The routine rescales and updates V at each iteration (including a normalization step based on the diagonal of V @ X[0] @ V.T) and may be affected by ill-conditioned inputs or zero/near-zero diagonal values.
        D (numpy.ndarray): Array of quasi-diagonal matrices with shape (n_matrices, n, n) such that D[k] = V @ X[k] @ V.T for each k. These matrices are typically diagonally dominant but not exactly diagonal; they are returned in the original matrix ordering and are intended for downstream use (e.g., extracting diagonal features, computing Riemannian distances, or validating the quality of diagonalization).
    
    Behavior, side effects, defaults, and failure modes:
        The implementation follows the U-WEDGE AJD algorithm as translated from the authors' Matlab code and iteratively updates V by solving a linear system derived from weighted least-squares steps. Internally, the input X is concatenated and symmetrized per block before processing. If init is None, the function computes an eigen-based initialization as described above. At each iteration, V is renormalized using the inverse square root of the absolute diagonal of V @ X[0] @ V.T to avoid scale drift.
        If the convergence criterion |crit_new - crit| < eps is not met within n_iter_max iterations, the function issues a warnings.warn("Convergence not reached") and returns the current V and D; this is a normal failure mode indicating the algorithm did not reach the requested tolerance. Linear algebra operations (e.g., np.linalg.solve, eigen-decompositions, divisions by small diagonal values) can raise numpy.linalg.LinAlgError or produce inf/NaN values for ill-conditioned or degenerate inputs; callers should validate input conditioning (for example via regularization of covariance estimates) when using empirical biosignal covariances. The function does not modify X in-place; it constructs internal copies for computation and returns V and D without side effects on the provided array.
        This function is intended for use in pyRiemann pipelines where approximate joint diagonalization is needed (e.g., as a preprocessing or feature extraction step for Riemannian classifiers). See the original references (Tichavsky et al.) for algorithmic details and trade-offs.
    """
    from pyriemann.utils.ajd import uwedge
    return uwedge(X, init, eps, n_iter_max)


################################################################################
# Source: pyriemann.utils.base.ctranspose
# File: pyriemann/utils/base.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def pyriemann_utils_base_ctranspose(X: numpy.ndarray):
    """pyriemann.utils.base.ctranspose: Conjugate transpose operator for complex-valued arrays used in pyRiemann for handling Hermitian or real symmetric matrices.
    
    Args:
        X (ndarray): Matrices to be transposed and conjugated. Expected to be a NumPy ndarray with shape (..., n, m), i.e., at least a 2D array where the last two axes represent matrix rows (n) and columns (m). In the pyRiemann context, X can represent batches of multichannel covariance-like matrices computed from biosignals (EEG/MEG/EMG) or complex-valued spatial descriptors used in remote sensing (HPD matrices for SAR). The function applies element-wise complex conjugation and swaps the last two axes so that each matrix becomes its conjugate transpose; for purely real-valued arrays this reduces to the usual matrix transpose. The function does not accept non-array container types (e.g., lists) without first converting them to an ndarray.
    
    Returns:
        ndarray: Conjugate transpose of X with shape (..., m, n). The returned object contains elements that are the complex conjugates of the corresponding elements of X and with the last two axes swapped. Depending on NumPy internals, the result may be a view or a new array, but the operation does not modify X in-place from the caller's perspective.
    
    Raises:
        Exception: If X is not an ndarray with at least two dimensions, or if X does not implement element-wise conjugation, the underlying NumPy operations (ndarray.conj and numpy.swapaxes) will raise an error (for example, AttributeError, TypeError, or numpy.AxisError). Callers should ensure X is a properly-shaped ndarray before calling this function.
    
    Behavior and side effects:
        The function computes X.conj() and then swaps the last two axes via numpy.swapaxes(X.conj(), -2, -1). This yields the mathematical conjugate transpose (also called Hermitian transpose) used in linear algebra and in pyRiemann pipelines when converting between row/column orientations of complex-valued matrices or when constructing Hermitian positive-definite matrices. For real-valued input arrays the conjugation is a no-op and the result is identical to a plain transpose of the last two axes. There are no other side effects; original array X is not modified in-place by contract, though NumPy may return views in some circumstances. Note: this function was added in version 0.9 of pyRiemann.
    """
    from pyriemann.utils.base import ctranspose
    return ctranspose(X)


################################################################################
# Source: pyriemann.utils.base.ddexpm
# File: pyriemann/utils/base.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def pyriemann_utils_base_ddexpm(X: numpy.ndarray, Cref: numpy.ndarray):
    """pyriemann.utils.base.ddexpm: Compute the directional derivative of the matrix exponential at a reference SPD/HPD matrix.
    
    Computes the directional derivative of the matrix exponential at a reference symmetric positive definite
    (SPD) or Hermitian positive definite (HPD) matrix Cref in the direction(s) X. This function implements the
    formula used in Riemannian geometry of positive definite matrices (see Matrix Analysis) and is provided for
    algorithms that require the sensitivity of the matrix exponential with respect to perturbations of a reference
    covariance or scatter matrix. In the pyRiemann context, Cref typically represents a covariance matrix estimated
    from multichannel biosignal data (EEG/MEG/EMG) or remote sensing (hyperspectral/SAR) and X represents one or
    several perturbation directions (for example, differences between epoch covariances and the reference). The
    implementation obtains the eigen-decomposition of Cref, evaluates the first divided difference of the exponential
    on the eigenvalues, and reconstructs the directional derivative via conjugation by the eigenvectors.
    
    Args:
        X (ndarray): Array of one or more direction matrices with shape (..., n, n). Each trailing (n, n) matrix
            is interpreted as a symmetric (real) or Hermitian (complex) perturbation direction in the tangent space
            at the reference matrix. In practical use within pyRiemann, X often contains per-epoch covariance
            perturbations and may have leading batch dimensions for multiple directions computed at once.
        Cref (ndarray): Reference SPD/HPD matrix with shape (n, n). This square matrix is the point at which the
            directional derivative of the matrix exponential is evaluated. The function computes the eigenvalues and
            eigenvectors of Cref (via numpy.linalg.eigh) and uses them to form the action of the first divided
            difference of the exponential on the projected directions.
    
    Returns:
        ndarray: Array with shape (..., n, n) giving the directional derivative of the matrix exponential at Cref
        in the directions provided by X. The returned array has the same leading/batch dimensions as X and the same
        trailing (n, n) matrix shape. When X has shape (n, n) the result is a single (n, n) matrix. There are no
        in-place modifications of the inputs; the function returns a new array.
    
    Behavior, defaults, and failure modes:
        - The function performs an eigen-decomposition of Cref using numpy.linalg.eigh. The conjugate transpose of
          the eigenvector matrix is used when projecting X into the eigenbasis and when reconstructing the result.
        - The first divided difference of the exponential is evaluated on the eigenvalues of Cref and combined
          elementwise with the projected X to produce the derivative.
        - Broadcasting: X may contain leading batch dimensions; these are preserved in the output. Cref must be a
          single (n, n) matrix and is applied to all directional matrices in X.
        - Numerical considerations: if Cref has nearly equal eigenvalues, the computation of divided differences can
          be sensitive to numerical precision; results may be affected accordingly.
        - The function does not check definiteness beyond what numpy.linalg.eigh requires; if Cref is not Hermitian
          or symmetric numerical warnings or errors from numpy.linalg may occur.
        - Errors that can be raised include ValueError for incompatible array shapes between X and Cref, TypeError
          if inputs are not array-like, and numpy.linalg.LinAlgError if the eigen-decomposition fails.
    
    Notes:
        - This routine was added in version 0.8 of pyRiemann.
        - No side effects occur: inputs X and Cref are not modified. The function is intended for use in Riemannian
          computations on SPD/HPD matrices such as those arising from covariance estimation in BCI and remote sensing.
    """
    from pyriemann.utils.base import ddexpm
    return ddexpm(X, Cref)


################################################################################
# Source: pyriemann.utils.base.ddlogm
# File: pyriemann/utils/base.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def pyriemann_utils_base_ddlogm(X: numpy.ndarray, Cref: numpy.ndarray):
    """pyriemann.utils.base.ddlogm computes the directional derivative of the matrix logarithm at a reference SPD/HPD matrix.
    
    This function implements the directional derivative defined for symmetric (real SPD) or Hermitian (complex HPD) positive-definite matrices and used in pyRiemann when manipulating covariance or scatter matrices that arise in biosignal processing (EEG/MEG/EMG) and remote sensing workflows. Practically, ddlogm(X, Cref) returns the linearized change of logm(C) at C = Cref in the direction X, as given by Eq. (V.13) in R. Bhatia, "Matrix Analysis" (1997). The implementation performs an eigen-decomposition of the reference matrix Cref (Cref = V diag(d) V^H) and applies the first divided difference of the scalar logarithm to the eigenvalues; the result is V (fddlogm(diag(d)) ⊙ (V^H X V)) V^H. This quantity is commonly used in Riemannian geometry operations on SPD/HPD matrices such as tangent-space mappings, gradients, and differential corrections when classifying covariance matrices in brain-computer interface (BCI) pipelines or when processing covariance descriptors in hyperspectral/SAR remote-sensing applications.
    
    Args:
        X (numpy.ndarray): Direction matrix or batch of direction matrices with shape (..., n, n). Each trailing (n, n) matrix represents the direction in which the derivative of the matrix logarithm is evaluated. In typical use within pyRiemann, X is an SPD/HPD matrix (real symmetric positive-definite or complex Hermitian positive-definite) that matches the dimensionality n of Cref. Leading dimensions (...) allow batching multiple directions for the same Cref (for example, multiple epochs' covariance directions in EEG processing).
        Cref (numpy.ndarray): Reference SPD/HPD matrix with shape (n, n). This real symmetric or complex Hermitian positive-definite matrix is the point at which the matrix logarithm is linearized. The function computes the derivative at this matrix by eigen-decomposition of Cref; therefore Cref must be square and positive-definite in the application domain (e.g., a covariance matrix estimated from multichannel time series in BCI).
    
    Returns:
        ddlogm (numpy.ndarray): Array of shape (..., n, n) containing the directional derivative of the matrix logarithm evaluated at Cref in the direction(s) X. The trailing (n, n) matrices are the resulting Hermitian (or symmetric) matrices that represent the linearized change of logm(C) for small perturbations along X. The returned dtype and Hermitian/symmetric property follow from inputs (complex for HPD, real for SPD). The function does not modify X or Cref; it allocates and returns a new array.
    
    Behavior, side effects, and failure modes:
        The function computes an eigen-decomposition of Cref (np.linalg.eigh) and then applies a first divided difference of the scalar logarithm to the eigenvalues. Computational cost scales roughly as O(n^3) due to the eigen-decomposition, so large n or very large batches in X may be computationally expensive. No in-place modification of inputs is performed. Failure modes include: if Cref is not square, not Hermitian/symmetric within numerical tolerances, or not positive-definite, the eigen-decomposition may fail or produce invalid results; np.linalg.eigh may raise a numpy.linalg.LinAlgError on non-convergence, and downstream broadcasting or matrix-multiplication errors (ValueError) can occur if the trailing dimensions of X do not match (n, n). For HPD (complex) inputs the function uses conjugate transposes (V.conj().T) consistent with Hermitian algebra. The implementation relies on _first_divided_difference to handle eigenvalue coincidences; numerical stability for nearly repeated eigenvalues depends on that helper's behavior.
    
    Notes:
        This function was introduced in pyRiemann v0.8 and is intended for operations on covariance-like SPD/HPD matrices encountered in pyRiemann pipelines (e.g., covariance estimation, tangent-space mapping, and Riemannian classification).
    """
    from pyriemann.utils.base import ddlogm
    return ddlogm(X, Cref)


################################################################################
# Source: pyriemann.utils.base.expm
# File: pyriemann/utils/base.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def pyriemann_utils_base_expm(C: numpy.ndarray):
    """Exponential of SPD/HPD matrices.
    
    Compute the matrix exponential of one or several symmetric positive-definite (SPD)
    or Hermitian positive-definite (HPD) matrices. This function implements the
    symmetric/Hermitian matrix exponential commonly used in Riemannian-geometry-based
    processing of covariance matrices: given an input matrix C with eigendecomposition
    C = V Lambda V^H, the result is D = V exp(Lambda) V^H where exp() is applied
    elementwise to the eigenvalues. In the pyRiemann project this routine is used
    to map matrices from the tangent/log domain back to the manifold (for example,
    reconstructing covariance matrices after tangent-space operations) and is therefore
    directly relevant for biosignal applications (EEG/MEG/EMG) and remote-sensing
    workflows that operate on SPD/HPD covariance matrices.
    
    Args:
        C (ndarray, shape (..., n, n)): Input SPD/HPD matrices. A stacked array of
            one or more square matrices; the last two dimensions must define square
            matrices of size n x n and the array must be at least 2-D. Supported
            inputs are real symmetric positive-definite matrices (SPD) and complex
            Hermitian positive-definite matrices (HPD) as used throughout pyRiemann
            for covariance/second-order statistics. The function relies on an
            eigendecomposition of each matrix and applies the scalar exponential to
            each eigenvalue before recomposing the matrix. The dtype of the output
            will follow numpy's rules (e.g., complex dtype for complex-valued HPD
            inputs).
    
    Returns:
        D (ndarray, shape (..., n, n)): Matrix exponential of C. The returned array
        has the same stacking/batch shape as the input and contains for each input
        matrix the matrix exponential D = expm(C) computed via eigen-decomposition.
        In the context of pyRiemann, D represents the point on the manifold obtained
        by exponentiating the logarithmic/tangent representation or by applying
        elementwise exponential to eigenvalues when reconstructing covariance-like
        matrices.
    
    Behavior, side effects, and failure modes:
        This function is pure and has no side effects (it does not modify its input
        array). It preserves the input batch shape and generally preserves dtype
        according to numpy casting rules. The function requires the last two
        dimensions of C to be square (n x n) and at least 2-D; providing an array
        with fewer dimensions or non-square last two dimensions will raise an error
        (for example, a ValueError or a numpy.linalg-related exception raised during
        eigendecomposition). If the inputs are not Hermitian/symmetric or not
        positive-definite, the eigendecomposition may produce complex eigenvalues or
        fail numerically; while numpy will still compute an exponential of the
        eigenvalues, the mathematical guarantee that the result is SPD/HPD no longer
        holds and numerical instabilities can occur for ill-conditioned inputs. The
        implementation applies the standard exponential to eigenvalues and therefore
        may be sensitive to very large or very small eigenvalues due to floating
        point limits.
    """
    from pyriemann.utils.base import expm
    return expm(C)


################################################################################
# Source: pyriemann.utils.base.invsqrtm
# File: pyriemann/utils/base.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def pyriemann_utils_base_invsqrtm(C: numpy.ndarray):
    """pyriemann.utils.base.invsqrtm — Inverse square root of SPD/HPD matrices.
    
    Compute the matrix inverse square root of one or more symmetric positive definite (SPD)
    or Hermitian positive definite (HPD) matrices. This function is used throughout pyRiemann
    for operations on covariance matrices (for example whitening, normalization, and steps
    in Riemannian geometry-based classification pipelines for biosignals such as EEG/MEG/EMG
    in brain-computer interface applications). Given an input matrix C, the result D is
    defined by the eigen-decomposition of C: C = V Lambda V^H and D = V Lambda^{-1/2} V^H,
    where Lambda is the diagonal matrix of eigenvalues and V the matrix of corresponding
    eigenvectors. The elementwise operator applied to the eigenvalues is 1 / sqrt(lambda),
    implemented via a call to the internal matrix operator used by pyRiemann.
    
    Args:
        C (ndarray, shape (..., n, n)): SPD/HPD matrices to invert-square-root. Input must be
            at least a 2-D ndarray representing either a single n-by-n matrix or a stack/batch
            of matrices with leading dimensions indicated by "...". For real-valued matrices,
            C is expected to be symmetric positive definite (SPD); for complex-valued matrices,
            C is expected to be Hermitian positive definite (HPD). The function does not modify
            C in place; it reads C and returns a new ndarray with the same trailing (n, n)
            shape. Typical use in pyRiemann is to process covariance matrices estimated from
            multichannel time series (see package README and estimation modules).
    
    Returns:
        D (ndarray, shape (..., n, n)): The inverse square root of C, i.e. matrices D such that
            D @ D = C^{-1} (within numerical precision) and D is Hermitian when C is HPD.
            The returned array has the same batch shape as the input C. This value is typically
            used for whitening covariance matrices or as part of Riemannian metric computations.
    
    Behavior, side effects, and failure modes:
        - The computation is performed by diagonalizing each input matrix and applying the
          scalar function f(lambda) = 1 / sqrt(lambda) to the eigenvalues, then reconstructing
          the matrix with the original eigenvectors. This preserves the (Hermitian/symmetric)
          structure when C is HPD/SPD.
        - The function assumes the input matrices are positive definite. If an input matrix
          has non-positive eigenvalues (zero or negative), applying 1 / sqrt(lambda) will
          produce infinities, NaNs, or complex values for real inputs and the result will be
          numerically invalid. Callers should ensure matrices are positive definite, for
          example by adding a small diagonal regularization (ridge) before calling this
          function when necessary.
        - Very small eigenvalues lead to very large values in the output (amplification of
          numerical noise). Regularization or eigenvalue thresholding prior to calling this
          function is recommended in ill-conditioned cases encountered in covariance estimation.
        - No in-place mutation of the input array is performed; a new ndarray is returned.
        - The function relies on the underlying numpy/scipy eigen-decomposition routines for
          numerical behavior and may raise low-level linear algebra errors if decomposition
          fails for a given input matrix (for example, if the matrix is not well-formed or
          not square). Such errors propagate to the caller.
        - Data type and device semantics follow numpy behavior: the output dtype is determined
          by the input and the underlying eigen-computation.
    """
    from pyriemann.utils.base import invsqrtm
    return invsqrtm(C)


################################################################################
# Source: pyriemann.utils.base.logm
# File: pyriemann/utils/base.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def pyriemann_utils_base_logm(C: numpy.ndarray):
    """pyriemann.utils.base.logm: Compute the matrix logarithm of symmetric/Hermitian positive-definite (SPD/HPD) matrices.
    
    Computes the symmetric (for real-valued SPD) or Hermitian (for complex-valued HPD) matrix logarithm of the input matrix or batch of matrices using an eigendecomposition. The computation follows the spectral formula D = V log(Λ) V^H where Λ is the diagonal matrix of eigenvalues of C and V the corresponding eigenvectors. In the pyRiemann library this operation is used to map covariance matrices (SPD) or Hermitian positive-definite matrices (HPD) to their matrix-log domain as part of Riemannian-geometry-based processing pipelines (for example, TangentSpace mapping, distance computations, or preprocessing of EEG/MEG covariance matrices for BCI and remote sensing applications).
    
    Args:
        C (ndarray): SPD/HPD matrices with shape (..., n, n). C must be at least a 2-D NumPy array where the last two dimensions form square matrices. Leading dimensions (if any) are treated as batch dimensions and the operator is applied independently to each matrix in the batch. Each matrix is expected to be symmetric positive-definite (real-valued SPD) or Hermitian positive-definite (complex-valued HPD). The function does not modify C in-place.
    
    Returns:
        D (ndarray): Matrix logarithm of C with the same shape as C, i.e., (..., n, n). For real-valued SPD inputs the result is a real symmetric matrix; for complex-valued HPD inputs the result is a complex Hermitian matrix. The returned array contains the natural logarithm applied to the eigenvalues in the spectral decomposition and reconstructed via the corresponding eigenvectors.
    
    Behavior and practical notes:
        The implementation performs an eigendecomposition of each square matrix in C and applies the natural logarithm to the eigenvalues. Because the natural logarithm is applied to the spectrum, input matrices must be positive definite to guarantee purely real-valued logarithms for real SPD inputs. If eigenvalues are extremely small (close to zero) or negative due to numerical error, the logarithm may produce very large magnitude values or complex values. In such cases the caller should regularize the matrices before calling logm (for example by adding a small multiple of the identity to C) to ensure numerical stability and preserve the SPD/HPD property. If C is not at least 2-D or the last two dimensions are not square, the underlying linear algebra routines will raise an exception (for example from NumPy) indicating invalid shape or inability to diagonalize; these exceptions are propagated to the caller.
    
    Side effects:
        None. The function is pure and returns a new ndarray; it does not modify its input.
    
    Failure modes:
        Non-positive-definite or non-Hermitian inputs can produce complex outputs or trigger numerical errors during eigendecomposition. Very small eigenvalues may lead to large negative logarithms and numerical instability. For robust use within pyRiemann workflows (e.g., processing covariance matrices estimated from EEG/MEG/EMG), ensure matrices are well-conditioned (regularize if necessary) before calling logm.
    """
    from pyriemann.utils.base import logm
    return logm(C)


################################################################################
# Source: pyriemann.utils.base.nearest_sym_pos_def
# File: pyriemann/utils/base.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def pyriemann_utils_base_nearest_sym_pos_def(X: numpy.ndarray, reg: float = 1e-06):
    """Find the nearest symmetric positive definite (SPD) matrices to a batch of square matrices.
    
    This function is a NumPy port of John D'Errico's nearestSPD MATLAB code and implements a per-matrix projection to the nearest SPD matrix, following the approach credited to N. J. Higham. In the pyRiemann project, this routine is used to ensure that estimated covariance matrices (for example, covariance matrices computed from multichannel biosignals such as EEG, MEG, EMG in BCI applications, or covariance patches from hyperspectral / SAR imagery in remote sensing) are valid symmetric positive definite matrices so they can be processed safely with Riemannian-geometry-based methods (for example, distance computations, tangent-space mapping, or classifiers like MDM). The function processes each matrix in the input batch independently and returns a new array; the input array is not modified in-place.
    
    Args:
        X (numpy.ndarray): shape (n_matrices, n, n). Batch of square matrices to project to the nearest SPD matrices. Each entry X[i] is expected to be a real (or complex, if supported by the underlying helper) square matrix of size (n, n). In the pyRiemann workflow X typically contains covariance matrices estimated from multichannel time-series data (n is the number of channels). Passing an array that does not have three dimensions with square last two axes or whose individual matrices are not square will result in an error when the per-matrix routine is applied.
        reg (float): default=1e-6. Small non-negative regularization added on the diagonal during computation to enforce strict positive definiteness and numerical stability. This parameter controls the magnitude of the diagonal perturbation used when eigenvalues are too small or negative due to numerical issues. A typical value in pyRiemann pipelines is 1e-6; increasing reg increases the diagonal inflation and can stabilize ill-conditioned covariance matrices at the cost of slightly biasing eigenvalues. Using a negative reg is not recommended and may lead to non-SPD results or runtime errors.
    
    Returns:
        numpy.ndarray: shape (n_matrices, n, n). Array of nearest symmetric positive definite matrices, one per input matrix. The returned array is newly allocated (the input X is not modified in-place). Each returned matrix is symmetric (within numerical tolerance) and has strictly positive eigenvalues according to the applied regularization. If computation fails for any matrix (for example, due to invalid shape or non-finite values), an exception will be raised and no complete result will be returned.
    
    Notes:
        The algorithm is applied independently to each matrix in the first axis of X, so memory and runtime scale with the number of matrices. The per-matrix computation typically involves symmetric projection and eigenvalue adjustments, so runtime grows roughly with the cube of the matrix dimension n for each matrix. This function is commonly used in preprocessing steps in pyRiemann pipelines to make covariance matrices suitable for Riemannian operations and classifiers. References: John D'Errico's nearestSPD MATLAB file and N. J. Higham, "Computing a nearest symmetric positive semidefinite matrix" (Linear Algebra and its Applications, 1988).
    """
    from pyriemann.utils.base import nearest_sym_pos_def
    return nearest_sym_pos_def(X, reg)


################################################################################
# Source: pyriemann.utils.base.powm
# File: pyriemann/utils/base.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def pyriemann_utils_base_powm(C: numpy.ndarray, alpha: float):
    """Compute the matrix power of symmetric positive definite (SPD) or Hermitian positive definite (HPD) matrices.
    
    This function computes the matrix power D = V Lambda^alpha V^H for each input matrix C, where Lambda is the diagonal matrix of eigenvalues of C, V are the corresponding eigenvectors, and V^H denotes the conjugate transpose of V. In the pyRiemann project this operation is used on covariance matrices estimated from multichannel biosignals (EEG, MEG, EMG) and on HPD matrices from complex-valued data (e.g., SAR), for tasks such as whitening, computing matrix square roots or inverse square roots, normalization, and other Riemannian-geometry-based preprocessing steps used in brain-computer interface (BCI) pipelines and remote sensing workflows. The implementation applies the scalar power alpha to the eigenvalues and reconstructs the matrix; it supports batched inputs by treating leading dimensions as batch dimensions and preserving the final two dimensions as square matrices.
    
    Args:
        C (numpy.ndarray): SPD/HPD matrices with shape (..., n, n). This must be at least a 2-D NumPy array where the last two dimensions form square matrices. Matrices should be symmetric (real-valued) positive definite or Hermitian (complex-valued) positive definite as appropriate to the data modality (for example, covariance matrices from real EEG are symmetric SPD; covariance of complex-valued radar data are HPD). The function assumes the input matrices are numerically close to SPD/HPD; if they are not, results may be invalid or complex-valued.
        alpha (float): The scalar power to apply to the eigenvalues. Typical values in pyRiemann workflows include 0.5 (matrix square root), -0.5 (inverse square root), 1.0 (identity), 0.0 (matrix of identity scaled by 1), or other real exponents used for scaling or normalization. alpha may be negative or non-integer; negative or non-integer powers require all eigenvalues to be strictly positive to avoid infinities or complex results.
    
    Returns:
        numpy.ndarray: Matrix power of C with the same shape as C, i.e., (..., n, n). For real SPD inputs the output is real and symmetric; for complex HPD inputs the output is complex and Hermitian. The operation preserves batch/leading dimensions. If eigenvalues are non-positive due to non-SPD/HPD input or numerical errors, the function may produce complex results, NaNs, or raise linear algebra errors depending on the underlying eigen-decomposition routine.
    
    Raises and failure modes:
        ValueError: If C does not have at least two dimensions or the last two dimensions are not square, a ValueError is expected.
        numpy.linalg.LinAlgError (or backend equivalent): If eigen-decomposition fails for a given matrix (for example due to numerical instability), the underlying linear algebra routine may raise an error.
        Numerical issues: If input matrices are not strictly positive definite (e.g., contain zero or negative eigenvalues from noise or rank deficiency), applying non-positive or non-integer alpha can yield infinities, NaNs, or complex outputs. To mitigate this, pre-regularize matrices (for example by adding a small multiple of the identity) before calling this function when working with estimated covariance matrices.
    Side effects:
        None with respect to input arrays (the function returns a new array). The computation may allocate intermediate arrays proportional to the input size for eigen-decomposition and reconstruction.
    """
    from pyriemann.utils.base import powm
    return powm(C, alpha)


################################################################################
# Source: pyriemann.utils.base.sqrtm
# File: pyriemann/utils/base.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def pyriemann_utils_base_sqrtm(C: numpy.ndarray):
    """pyriemann.utils.base.sqrtm computes the symmetric (or Hermitian) matrix square root of one or multiple symmetric positive definite (SPD) or Hermitian positive definite (HPD) matrices using an eigen-decomposition-based formula D = V sqrt(Lambda) V^H. This function is used in pyRiemann workflows that manipulate covariance or scattering matrices derived from multivariate biosignals (EEG, MEG, EMG) or remote sensing images, for example when applying Riemannian geometry operations, whitening, or tangent-space mappings on SPD/HPD matrices.
    
    Args:
        C (numpy.ndarray): SPD/HPD matrices to process, provided as an ndarray with shape (..., n, n). The last two dimensions must form square matrices of size n x n; leading dimensions (the "..." part) are interpreted as a stack/batch of matrices and the operation is applied independently to each matrix in the stack. For real-valued covariance matrices (common in EEG/BCI applications) C is expected to be symmetric positive definite (SPD); for complex-valued matrices (e.g., some SAR or signal-processing contexts) C is expected to be Hermitian positive definite (HPD). This argument is the primary input to the function and represents the matrices whose principal square roots are required in downstream Riemannian and machine-learning computations.
    
    Returns:
        D (numpy.ndarray): Matrix square root of C, with the same batch shape (..., n, n). For each input matrix C, the returned matrix D satisfies D @ D^H = C within numerical precision, where D^H denotes the conjugate transpose (for real SPD inputs D is real symmetric and D^T @ D = C). The dtype of D will reflect the computation (real for real SPD input that yields a real root, complex if the input is complex or if numerical issues produce complex eigen-components). The returned array is newly allocated (the input C is not modified in-place).
    
    Behavior and practical details:
        The function computes an eigen-decomposition of each input matrix C = V Lambda V^H, takes the elementwise square root of the eigenvalues Lambda, and reconstructs D = V sqrt(Lambda) V^H. This approach yields the principal (symmetric/Hermitian) square root used in Riemannian treatments of SPD/HPD matrices.
    
    Failure modes, numerical considerations, and recommendations:
        - The input must be at least 2-D and its last two dimensions must be square (n x n). Supplying arrays with incompatible shapes will cause an error from the underlying shape checks or linear-algebra routines.
        - The mathematical definition requires positive eigenvalues. If C is not truly positive definite (zero or negative eigenvalues due to data or numerical noise), taking the square root of those eigenvalues can produce zeros, NaNs, or complex values; results will not satisfy the SPD/HPD assumptions. In practice, small negative eigenvalues arising from numerical imprecision should be handled before calling this function (for example by regularizing C with C + eps * I) to ensure stability in pipelines such as covariance estimation for BCI.
        - Computational cost scales roughly as O(n^3) per matrix because of the eigen-decomposition; consider this when applying sqrtm to large matrices or large batches.
        - For complex HPD inputs the conjugate-transpose is used in reconstruction; for real SPD inputs the result is real symmetric.
    
    No in-place modification is performed; the function returns the computed square-root matrices for use in subsequent Riemannian or machine-learning procedures (e.g., distance computations, whitening, tangent-space projection) commonly applied to covariance matrices in pyRiemann.
    """
    from pyriemann.utils.base import sqrtm
    return sqrtm(C)


################################################################################
# Source: pyriemann.utils.covariance.block_covariances
# File: pyriemann/utils/covariance.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def pyriemann_utils_covariance_block_covariances(
    X: numpy.ndarray,
    blocks: list,
    estimator: str = "cov",
    **kwds
):
    """Compute block diagonal covariance matrices by estimating covariances on channel subsets.
    
    This function is used in multivariate signal processing (for example EEG/MEG/EMG in brain-computer interface workflows or spatial covariance estimation in remote sensing) to compute a block-diagonal covariance matrix for each epoch/trial/observation in X. Each block of the returned matrix is computed as the covariance of a contiguous subset of channels (a group of sensors) whose sizes are given by blocks. The estimator argument selects the covariance estimator applied to each block; additional keyword arguments are forwarded to that estimator. The function validates that the sum of the block sizes equals the number of channels and that X has three dimensions (n_matrices, n_channels, n_times). If these conditions are not met, the function raises an error rather than returning a result.
    
    Args:
        X (numpy.ndarray): Multi-channel time-series array with shape (n_matrices, n_channels, n_times). Each entry X[i] is the time series for the i-th observation (epoch/trial) across n_channels sensors and n_times time samples. The function unpacks X.shape into three values; if X does not have exactly three dimensions, a ValueError or unpacking error will occur.
        blocks (list): List of integers specifying block sizes. Each integer indicates the number of consecutive channels that form one block. The sum of blocks must equal n_channels from X.shape. Blocks are applied in channel order (i.e., the first block uses channels 0..blocks[0]-1, the second block uses the next channels, etc.). If the sum of blocks does not equal n_channels, a ValueError is raised.
        estimator (str or callable): Covariance estimator to use for each block. By default "cov" is used. The estimator may be a string key referring to one of the estimators registered in pyriemann.utils.covariance.cov_est_functions (resolved via check_function), or a callable that accepts a 2D array of shape (n_channels_in_block, n_times) and returns a covariance matrix of shape (n_channels_in_block, n_channels_in_block). If an unrecognized string is provided, the underlying check_function will raise an error. The estimator controls how each block covariance is computed (e.g., sample covariance, shrunk covariance), which affects downstream Riemannian processing and classification.
        kwds (dict): Additional keyword arguments forwarded to the chosen covariance estimator. Typical keys depend on the estimator (for example regularization or shrinkage parameters). These are passed unchanged to the estimator for each block; incorrect or unsupported keywords will cause the estimator to raise an exception.
    
    Returns:
        numpy.ndarray: Array covmats of shape (n_matrices, n_channels, n_channels) where each covmats[i] is a block-diagonal covariance matrix for the i-th observation. Each diagonal block of covmats[i] is the covariance matrix computed from the corresponding contiguous subset of channels of X[i] using the specified estimator and kwds. There are no in-place side effects on X; if an error occurs (invalid estimator, mismatched block sizes, invalid X shape, or estimator failure), an exception is raised and no covariances are returned.
    """
    from pyriemann.utils.covariance import block_covariances
    return block_covariances(X, blocks, estimator, **kwds)


################################################################################
# Source: pyriemann.utils.covariance.coherence
# File: pyriemann/utils/covariance.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def pyriemann_utils_covariance_coherence(
    X: numpy.ndarray,
    window: int = 128,
    overlap: float = 0.75,
    fmin: float = None,
    fmax: float = None,
    fs: float = None,
    coh: str = "ordinary"
):
    """pyriemann.utils.covariance.coherence computes the squared coherence between all pairs of channels of a multichannel time series across frequency bins. This function is intended for spectral connectivity analysis of biosignals (for example EEG, MEG, EMG) commonly used in brain–computer interface (BCI) and related pipelines described in the pyRiemann README. It first estimates the cross-spectral density via an FFT-based windowed estimator (cross_spectrum), then builds a 3D array of squared coherence values for each channel pair and frequency. Different coherence variants ("ordinary", "instantaneous", "lagged", "imaginary") implement formulas used in connectivity analysis and correspond to the coherence types exposed by pyriemann.estimation.Coherences.
    
    Args:
        X (numpy.ndarray): Multi-channel real-valued time series with shape (n_channels, n_times). This is the input biosignal (for example an EEG epoch) from which frequency-domain cross-spectra are estimated. The function does not modify X in-place.
        window (int): Length of the FFT window (number of time samples) used for spectral estimation. Defaults to 128. A larger window gives finer frequency resolution but requires longer stationary segments of the signal.
        overlap (float): Fractional overlap between successive FFT windows, expressed between 0 and 1. Defaults to 0.75 (75% overlap). Higher overlap increases the number of averaged segments and reduces variance of spectral estimates at the cost of more computation.
        fmin (float | None): Minimal frequency (in Hz) to include in the returned result. Defaults to None, meaning no explicit lower frequency cutoff is applied; selection is delegated to cross_spectrum. When provided, fmin restricts the frequency bins used and the shape of the returned frequency axis.
        fmax (float | None): Maximal frequency (in Hz) to include in the returned result. Defaults to None, meaning no explicit upper frequency cutoff is applied; selection is delegated to cross_spectrum. When provided, fmax restricts the frequency bins used and the shape of the returned frequency axis.
        fs (float | None): Sampling frequency of the input time series in Hz. Defaults to None. If fs is provided, the returned frequency axis (freqs) will contain frequencies in Hz and Nyquist/DC handling for the "lagged" coherence mode will use fs to detect and exclude DC (0 Hz) and Nyquist (fs/2) bins. If fs is None, the underlying cross_spectrum may return freqs as None; in that case frequency-aware exclusions fall back to excluding the first and last FFT bins by index.
        coh (str): Coherence type to compute. Must be one of "ordinary", "instantaneous", "lagged", or "imaginary". Defaults to "ordinary".
            - "ordinary": squared magnitude coherence, computed as |S|^2 / (psd_i * psd_j), where S is the cross-spectral matrix and psd_i is the power spectral density of channel i. This is the standard magnitude-squared coherence used for general connectivity estimation.
            - "instantaneous": uses only the squared real part of S: (Re(S))^2 / (psd_i * psd_j), which emphasizes zero-phase (instantaneous) components.
            - "imaginary": uses the squared imaginary part of S: (Im(S))^2 / (psd_i * psd_j), often used to highlight phase-lagged interactions that are robust to volume conduction.
            - "lagged": isolates lagged (non-instantaneous) contributions by computing (Im(S))^2 / (psd_i*psd_j - (Re(S))^2). The diagonal real parts are forcibly set to zero for each frequency to prevent division by zero on the diagonal. Note that lagged coherence is not defined for DC and Nyquist bins; these bins are excluded (set to zero) and a warning is emitted. The coherence type meanings follow the conventions used in pyriemann.estimation.Coherences and are commonly applied in EEG/BCI connectivity studies.
    
    Behavior, defaults, and side effects:
        - The function calls cross_spectrum(X, window=window, overlap=overlap, fmin=fmin, fmax=fmax, fs=fs) to obtain the complex cross-spectral density S and the frequency axis freqs. S has shape (n_channels, n_channels, n_freqs) and freqs either is an ndarray of length n_freqs (if fs or frequency information is available) or may be None if cross_spectrum could not return frequencies.
        - The squared cross-spectral modulus S2 = |S|^2 is computed and used as a starting point to form the squared coherence C of shape (n_channels, n_channels, n_freqs).
        - For each frequency bin, the per-channel power spectral densities are computed as psd = sqrt(diag(S2[..., f])). The denominator for normalization is formed as the outer product psd_prod = outer(psd, psd).
        - For "lagged" coherence the function sets the real diagonal elements of S for the processed frequency to zero using np.fill_diagonal(S[..., f].real, 0.). This modifies the local cross-spectral array S used inside the function (it does not modify the input time series X). This in-place modification prevents division by zero on the diagonal for the lagged formula.
        - To preserve numerical stability in the "lagged" denominator denom = psd_prod - (Re(S))^2, values with absolute magnitude below 1e-10 are clamped to 1e-10 before division. This threshold is a built-in safeguard against extremely small denominators.
        - If a channel has (near) zero power at a frequency (psd equals zero), divisions in "ordinary", "instantaneous", and "imaginary" modes may produce infinities or NaNs; the function does not add automatic regularization to psd except for the denom clamp in "lagged" mode. Users should ensure spectral estimates are well-conditioned (for example via preprocessing or explicit regularization) when needed.
        - When coh == "lagged", DC and Nyquist bins are undefined because S is real at these frequencies; the function will either exclude those bins by index (if freqs is None) or by frequency comparison using fs and issue a warnings.warn message stating that DC and Nyquist bins are filled with zeros.
        - The function emits warnings.warn messages to notify about excluded bins when computing "lagged" coherence and does not raise an exception in that case.
    
    Failure modes and exceptions:
        - ValueError is raised if coh is not one of the supported strings "ordinary", "instantaneous", "lagged", or "imaginary".
        - Numerical issues (division by zero) may lead to inf or NaN values in the returned coherence matrices when one or more channels have zero spectral power at some frequencies; the function only clamps the lagged denominator as described above.
    
    Returns:
        C (numpy.ndarray): Squared coherence matrices with shape (n_channels, n_channels, n_freqs). For each frequency bin this array contains the pairwise squared coherence between channels according to the selected coherence type. Values are non-negative but may be infinite or NaN if spectral power is zero for some channels and no regularization is applied.
        freqs (numpy.ndarray | None): Array of length n_freqs containing the frequencies (in Hz) associated with the last axis of C, as returned by cross_spectrum. freqs may be None if cross_spectrum could not determine a frequency axis (for example when fs is None). When freqs is provided and coh == "lagged", DC (0 Hz) and Nyquist (fs/2) bins are excluded from the lagged calculation and corresponding entries in C are zero with a warning emitted.
    """
    from pyriemann.utils.covariance import coherence
    return coherence(X, window, overlap, fmin, fmax, fs, coh)


################################################################################
# Source: pyriemann.utils.covariance.cospectrum
# File: pyriemann/utils/covariance.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def pyriemann_utils_covariance_cospectrum(
    X: numpy.ndarray,
    window: int = 128,
    overlap: float = 0.75,
    fmin: float = None,
    fmax: float = None,
    fs: float = None
):
    """Compute co-spectral matrices (the real part of cross-spectra) from a multi-channel
    time-series. This function is intended for frequency-domain covariance estimation
    used in applications such as biosignal processing (EEG, MEG, EMG) and brain-computer
    interfaces (BCI). It segments the input signal into FFT windows, computes the
    cross-spectral density per frequency bin via cross_spectrum, and returns the real
    part of those cross-spectra (co-spectral matrices) together with the associated
    frequency vector. These co-spectral matrices provide frequency-specific estimates
    of linear dependencies between channels and can be used as features for
    classification, connectivity analysis, or as inputs to Riemannian geometry-based
    algorithms described in the project README.
    
    Args:
        X (numpy.ndarray): Multi-channel time-series, real-valued, with shape
            (n_channels, n_times). Each row corresponds to one channel and each
            column to a time sample. This array is the primary input from which
            frequency-domain covariance (co-spectra) are estimated.
        window (int): Length of the FFT window used for spectral estimation, in
            samples. This integer controls the time-frequency resolution trade-off:
            larger windows yield finer frequency resolution but coarser time
            localization. Default is 128.
        overlap (float): Fractional overlap between consecutive windows used for the
            FFT segmentation. For example, 0.75 corresponds to 75% overlap. Higher
            overlap increases the number of averaged segments and can reduce variance
            of the spectral estimates. Default is 0.75.
        fmin (float | None): Minimal frequency to be returned, in the same units as
            the frequencies produced by the function (see fs). If None, the lowest
            frequency produced by the internal spectral estimator is returned. Use
            this parameter to restrict the output to a band of interest (e.g., alpha
            band in EEG).
        fmax (float | None): Maximal frequency to be returned, in the same units as
            the frequencies produced by the function (see fs). If None, the highest
            frequency produced by the internal spectral estimator is returned. Use
            this to limit output to an upper frequency of interest.
        fs (float | None): Sampling frequency of the time-series in Hertz. If provided,
            the returned frequency vector is expressed in Hertz. If None, the
            frequency vector is returned using the same units produced by the
            underlying cross_spectrum implementation (commonly cycles per sample or
            normalized frequency), and downstream code should interpret frequencies
            accordingly.
    
    Returns:
        S (numpy.ndarray): Co-spectral matrices (real part of cross-spectra) with
            shape (n_channels, n_channels, n_freqs). For each frequency bin, S[..., k]
            is a real symmetric matrix representing the frequency-specific covariance
            (co-spectra) between channels. These matrices can be used as frequency
            domain covariance estimates in pipelines for feature extraction or
            classification in BCI and related domains.
        freqs (numpy.ndarray): Frequencies associated to the cospectra, shape
            (n_freqs,). The units are Hertz when fs is provided, otherwise as
            produced by the underlying spectral estimator.
    
    Notes:
        - This function delegates the heavy lifting to cross_spectrum and returns its
          real part; no in-place modification of the input X occurs.
        - X is expected to be a 2-D real-valued array with shape (n_channels, n_times).
          Passing arrays of other shapes or complex-valued arrays may lead to errors
          or undefined behavior raised by the underlying implementation.
        - The window parameter should be chosen with regard to the time-series length
          and the desired frequency resolution. Very large window values relative to
          n_times may result in few or no segments being available for averaging.
        - The function does not perform automatic detrending, filtering, or other
          preprocessing; apply such preprocessing before calling cospectrum if needed.
        - Underlying functions may raise exceptions (for example, due to invalid
          argument values or insufficient data length); callers should handle these
          exceptions as appropriate.
    """
    from pyriemann.utils.covariance import cospectrum
    return cospectrum(X, window, overlap, fmin, fmax, fs)


################################################################################
# Source: pyriemann.utils.covariance.covariance_mest
# File: pyriemann/utils/covariance.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def pyriemann_utils_covariance_covariance_mest(
    X: numpy.ndarray,
    m_estimator: str,
    init: numpy.ndarray = None,
    tol: float = 0.01,
    n_iter_max: int = 50,
    assume_centered: bool = False,
    q: float = 0.9,
    nu: int = 5,
    norm: str = "trace"
):
    """Compute a robust M-estimator of the covariance (scatter) matrix from a multichannel time series using a fixed-point algorithm.
    
    This function implements robust M-estimator covariance matrix estimation described in the literature on complex elliptically symmetric distributions and M-estimators (Huber, Student-t, Tyler). It is intended for estimating covariance (or scatter) matrices from multivariate time series such as biosignals (EEG, MEG, EMG) used in brain–computer interfaces or from spatial windows in remote sensing applications. The estimator downweights samples according to their squared Mahalanobis distance to provide robustness to outliers; the exact weighting depends on the chosen m_estimator ("hub", "stu", or "tyl"). The algorithm iterates a fixed-point update until the relative change in Frobenius norm meets tol or until n_iter_max iterations. Note that when assume_centered is False the input X is centered in place (modified).
    
    Args:
        X (numpy.ndarray): Input multichannel time-series with shape (n_channels, n_times). Each column is a multichannel sample. X may be real or complex-valued; n_channels and n_times are inferred from X.shape and determine the dimension of the returned covariance. Practical significance: this is the measured data from which a robust estimate of the covariance (scatter) matrix is computed for downstream Riemannian geometry-based processing and classification.
        m_estimator (str): Type of M-estimator to use; must be one of "hub", "stu", or "tyl". "hub" selects Huber's M-estimator which trades off between Tyler and the sample covariance via the q parameter; "stu" selects Student-t M-estimator which uses the nu degree-of-freedom parameter to control robustness; "tyl" selects Tyler's M-estimator which is scale-invariant and requires a normalization choice. A ValueError is raised if m_estimator is not one of these strings.
        init (numpy.ndarray): Optional initial covariance matrix used to start the fixed-point iterations, with shape (n_channels, n_channels). If None (default) the sample covariance matrix X @ X.conj().T / n_times is used as the initialization. Practical significance: supplying init can accelerate convergence or provide a regularized starting point when sample covariance is ill-conditioned.
        tol (float): Relative stopping tolerance for the fixed-point iterations (default=0.01). The algorithm stops early when the relative Frobenius-norm change norm(cov_new - cov, ord='fro') / norm(cov, ord='fro') is <= tol. A smaller tol forces stricter convergence (more iterations) and may be required for high-precision applications; a larger tol yields faster termination with potentially less accurate estimates.
        n_iter_max (int): Maximum number of fixed-point iterations (default=50). If convergence is not reached within n_iter_max iterations the function issues a warning ("Convergence not reached") and returns the last iterate; callers should check for this warning if strict convergence is required.
        assume_centered (bool): If False (default), the function subtracts the empirical mean across time from X before computing the estimator (centers X). If True, the function assumes X is already centered and does not modify the mean. Note: when assume_centered is False the centering operation modifies the input array X in place; if you must preserve X, pass a copy.
        q (float): Huber-specific parameter (default=0.9). For m_estimator="hub" q is the fraction in (0, 1] of samples considered uncorrupted (the remainder are treated as outliers under a Gaussian reference). The code enforces 0 < q <= 1 and raises ValueError otherwise. Practical significance: q closer to 1 produces behavior closer to the sample covariance; q smaller increases robustness toward outliers and moves the estimator toward Tyler-like behavior.
        nu (int): Student-t-specific degrees-of-freedom (default=5). For m_estimator="stu" nu must be strictly positive and controls the heaviness of tails in the Student-t weight; small nu increases robustness (approaching Tyler as nu -> 0), large nu approaches the sample covariance. A ValueError is raised if nu <= 0.
        norm (str): Tyler-specific normalization type (default="trace"). Applicable only when m_estimator="tyl". Accepted values are "trace" or "determinant". For "trace" the algorithm normalizes so that the trace of the resulting covariance equals n_channels (the implementation scales the normalized matrix by n_channels before returning); for "determinant" the covariance is normalized to have determinant equal to 1. Practical significance: the normalization enforces a scale convention for Tyler's scale-invariant scatter estimate used in applications that require a fixed matrix scale.
    
    Returns:
        numpy.ndarray: Robust M-estimator based covariance (scatter) matrix with shape (n_channels, n_channels). This matrix is the fixed-point solution approximated by the algorithm and is suitable for use with Riemannian geometry methods in pyRiemann (e.g., as input to tangent-space transforms or Riemannian classifiers). When m_estimator="tyl" the returned matrix is normalized according to the norm parameter (for "trace" the implementation ensures trace = n_channels by a final scaling). If the algorithm did not converge within n_iter_max, the returned matrix is the last iterate and a warning was emitted.
    
    Raises / Failure modes:
        ValueError: If m_estimator is not one of "hub", "stu", "tyl"; if m_estimator="hub" and q is not in (0, 1]; or if m_estimator="stu" and nu <= 0.
        Warning: If the fixed-point iterations reach n_iter_max without satisfying the relative tol criterion, a runtime warning "Convergence not reached" is emitted and the last iterate is returned. Users requiring guaranteed convergence should increase n_iter_max or relax tol.
        Side effects: If assume_centered is False the input array X is centered in place (X is modified). If the caller wishes to preserve the original X, pass a copy.
    
    Implementation notes and significance in domain:
        The routine computes squared Mahalanobis distances of each time sample to the current scatter estimate and applies a weight function specific to the chosen M-estimator: for Huber the weight clips distances using chi-square quantiles to downweight outliers, for Student-t the weight follows the t-distribution-derived form depending on nu, and for Tyler the weight is proportional to 1 / distance yielding a distribution-free scatter estimator. The fixed-point iterations update the weighted sample covariance until the relative Frobenius norm change falls below tol. This robust covariance estimator improves resilience to non-Gaussian noise and outliers in biosignal and remote sensing pipelines used throughout pyRiemann.
    """
    from pyriemann.utils.covariance import covariance_mest
    return covariance_mest(
        X,
        m_estimator,
        init,
        tol,
        n_iter_max,
        assume_centered,
        q,
        nu,
        norm
    )


################################################################################
# Source: pyriemann.utils.covariance.covariance_sch
# File: pyriemann/utils/covariance.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def pyriemann_utils_covariance_covariance_sch(X: numpy.ndarray):
    """Schaefer-Strimmer shrinkage covariance estimator for multichannel time series.
    
    This function computes a shrunk estimate of the sample covariance matrix using the
    Schaefer-Strimmer method: the shrinkage estimator is a weighted average between
    the sample covariance matrix (SCM) and a diagonal target matrix formed from the
    SCM diagonal. The optimal shrinkage intensity gamma is estimated using the
    authors' analytical estimator and then clipped to [0, 1]. The implementation
    centers the data (subtracts per-channel mean), uses the unbiased scaling
    factor n_times / (n_times - 1) on the SCM, and returns the final covariance
    matrix. This estimator is commonly used in domains supported by pyRiemann,
    including biosignal processing (EEG, MEG, EMG) for brain–computer interface
    (BCI) pipelines and in remote sensing for covariance estimation over spatial
    windows; the shrunk covariance improves conditioning and downstream
    Riemannian-geometry-based classification or signal processing.
    
    Args:
        X (numpy.ndarray): Multi-channel time-series data with shape
            (n_channels, n_times). Each row corresponds to one channel (sensor)
            and each column to a time sample. The array may contain real or
            complex values; complex-valued data are supported by the function.
            The function subtracts the per-channel mean (centering) and computes
            the sample covariance from the centered data. Practical significance:
            X is the raw multivariate signal from which a covariance matrix is
            estimated for use in Riemannian geometry based pipelines (for example,
            as input to pyRiemann estimators and classifiers). Requirements and
            failure modes: n_times must be greater than 1 (otherwise divisions by
            n_times - 1 occur); channels with zero variance may produce divisions
            by zero and yield NaNs/infs in intermediate computations; the caller
            should ensure input is finite and has sufficient time samples per
            channel.
    
    Returns:
        numpy.ndarray: Shrunk covariance matrix with shape (n_channels, n_channels).
        This matrix is the Schaefer-Strimmer estimator:
        C = (1 - gamma) * SCM_corrected + gamma * diag(SCM_corrected),
        where SCM_corrected is the sample covariance scaled by n_times / (n_times - 1)
        (unbiased correction) and diag(SCM_corrected) is the diagonal matrix formed
        from the SCM diagonal. The returned array is suitable for downstream use in
        Riemannian-geometry-based processing (e.g., TangentSpace, MDM) and for
        improving numerical conditioning of covariance matrices used in BCI or
        remote sensing applications. The shrinkage intensity gamma is estimated
        internally following Schafer and Strimmer (2005) and is constrained to the
        interval [0, 1] to guarantee a convex combination. No in-place modification
        of X is performed; the function returns a new numpy.ndarray.
    """
    from pyriemann.utils.covariance import covariance_sch
    return covariance_sch(X)


################################################################################
# Source: pyriemann.utils.covariance.covariance_scm
# File: pyriemann/utils/covariance.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def pyriemann_utils_covariance_covariance_scm(
    X: numpy.ndarray,
    assume_centered: bool = False
):
    """pyriemann.utils.covariance.covariance_scm estimates the sample (empirical) covariance matrix from a multichannel time series. It is used in pyRiemann to convert multichannel signals (for example EEG/MEG/EMG epochs in brain–computer interface workflows or spatial patches in remote sensing) into symmetric (real) or Hermitian (complex) positive semidefinite matrices that can be processed by Riemannian geometry-based algorithms (for example MDM, TangentSpace, or other covariance-based classifiers and pipelines).
    
    Args:
        X (numpy.ndarray): 2-D array of shape (n_channels, n_times) containing the multichannel time series. Each row corresponds to one channel/variable and each column corresponds to one time sample or observation. Values may be real or complex; when complex-valued data are provided, conjugate transposition is used where appropriate so that the returned covariance is Hermitian. This function does not accept higher- or lower-dimensional arrays: if X does not have exactly two dimensions, a ValueError will be raised by the shape unpacking performed at the start of the function. If X contains NaNs or Infs, the result will contain NaNs or Infs accordingly.
        assume_centered (bool): If True, the function assumes the per-channel mean is already zero and computes the (biased) sample covariance without centering using the matrix product X @ X.conj().T divided by n_times (the number of columns). This produces the maximum-likelihood / biased estimator normalized by n_times. If False (the default), the function will center the data by subtracting the mean of each row (channel) across time and compute the biased sample covariance using numpy.cov with bias=1, which also normalizes by n_times after centering. Use assume_centered=True when you know your signals are already mean-centered (for example after a preprocessing pipeline that removed the baseline), because skipping centering avoids an explicit subtraction and yields the uncentered second-moment matrix. If n_times == 0, dividing by n_times will raise an error or produce invalid values; ensure there is at least one time sample.
    
    Returns:
        cov (numpy.ndarray): Sample covariance matrix of shape (n_channels, n_channels). For real-valued inputs this is a symmetric matrix; for complex-valued inputs this is a Hermitian matrix (conjugate-symmetric). The matrix is the biased estimator (normalized by the number of time samples n_times). The returned array is a new numpy.ndarray; the function does not modify X in-place. This covariance is suitable as input to downstream Riemannian-processing functions in pyRiemann (e.g., covariance-based classifiers, tangent space mapping, or transfer-learning routines).
    
    Notes on behavior and failure modes:
        The function expects a 2-D numpy.ndarray with shape (n_channels, n_times). If X has a different dimensionality a ValueError will be raised during shape unpacking. If n_times is zero or extremely small, results may be invalid (division by zero or unstable estimates). If X contains NaN or infinite values, the output will propagate those values. The implementation uses two code paths: a fast linear-algebra path without centering (assume_centered=True) that computes X @ X.conj().T / n_times, and a centering path (assume_centered=False) that relies on numpy.cov(X, bias=1) to subtract per-channel means and normalize by n_times. The function preserves support for both real and complex data as required in biosignal and remote-sensing applications.
    """
    from pyriemann.utils.covariance import covariance_scm
    return covariance_scm(X, assume_centered)


################################################################################
# Source: pyriemann.utils.covariance.covariances
# File: pyriemann/utils/covariance.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def pyriemann_utils_covariance_covariances(X: numpy.ndarray, estimator: str = "cov", **kwds):
    """pyriemann.utils.covariance.covariances: Estimate covariance matrices from multichannel time-series for Riemannian-geometry based analysis.
    
    Estimates covariance (or correlation / robust / shrunk) matrices for each epoch/trial/patch in a 3D array of multichannel time-series. This function is commonly used in biosignal applications (EEG, MEG, EMG) and remote sensing (hyperspectral, SAR) to produce symmetric (or Hermitian for complex data) positive definite matrices that are the input to downstream pyRiemann algorithms such as TangentSpace, MDM, and other classifiers or manifolds-based pipelines. The function supports real and complex-valued data and delegates the per-epoch estimation to a selectable estimator implementation. The output array has dtype equal to X.dtype and contains one covariance matrix per input time-series.
    
    Args:
        X (ndarray): Multi-channel time-series input with shape (n_matrices, n_channels, n_times). Each entry X[i] is the time-series used to estimate one covariance matrix. X may be real or complex-valued; the function does not modify X in-place. A 3-dimensional array is required: if X does not have exactly three dimensions, the function will raise an error when attempting to unpack X.shape or when the chosen estimator validates its input.
        estimator (string | callable): Covariance estimator selector or a callable implementing an estimator. If a string, it selects one of the predefined estimators documented by the package (for example "cov" for NumPy covariance, "corr" for correlation coefficient matrix, "lwf" or "oas" for shrinkage estimators recommended for regularization, "hub", "mcd", "stu", "tyl" for robust estimators, "sch" for Schaefer-Strimmer, "scm" for sample covariance, etc.). If a callable, it must accept a 2D array of shape (n_channels, n_times) and return a 2D covariance matrix of shape (n_channels, n_channels). The estimator is resolved via the package's check_function mechanism; an invalid string or a callable that does not return the correct shape or type will cause an exception propagated from the resolution or from the estimator itself. For certain estimators ("lwf", "mcd", "oas", "sch") complex-valued covariance estimation follows the complex-extension behavior referenced in the code documentation.
        kwds (dict): Additional keyword arguments passed verbatim to the selected covariance estimator callable. Typical keywords depend on the estimator (for example, shrinkage parameters for Ledoit-Wolf or options for Huber/M-estimators). If kwds contains parameters not accepted by the estimator, the estimator call will raise a TypeError or ValueError as raised by that estimator implementation.
    
    Returns:
        ndarray: covmats, shape (n_matrices, n_channels, n_channels). A new array containing one estimated covariance matrix per entry in X. The returned array has the same dtype as X and is freshly allocated by the function; there are no in-place modifications of X. Failure modes include raising errors when X does not have three dimensions, when estimator resolution fails (invalid string), or when the estimator callable raises due to invalid input, incompatible kwds, or inability to produce a matrix of shape (n_channels, n_channels).
    """
    from pyriemann.utils.covariance import covariances
    return covariances(X, estimator, **kwds)


################################################################################
# Source: pyriemann.utils.covariance.covariances_EP
# File: pyriemann/utils/covariance.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def pyriemann_utils_covariance_covariances_EP(
    X: numpy.ndarray,
    P: numpy.ndarray,
    estimator: str = "cov",
    **kwds
):
    """Special covariance-matrix estimator that concatenates a time-aligned prototype P with each multichannel epoch in X and computes the covariance of the concatenated signals. This function is used in pyRiemann workflows (for example ERP / event-related potential processing in BCI pipelines) to augment each trial with a prototype/template signal and produce covariance matrices on the combined channel set for downstream Riemannian processing and classification.
    
    Args:
        X (numpy.ndarray): Multi-channel time-series array of shape (n_matrices, n_channels, n_times). Each entry X[i] is a single trial/epoch with n_channels spatial channels and n_times temporal samples. In BCI/EEG contexts, X represents recorded epochs (e.g., EEG trials) over which covariance features are estimated.
        P (numpy.ndarray): Prototype time-series of shape (n_channels_proto, n_times). This is a template or prototype signal (for example an averaged ERP or spatial template) that will be concatenated with each epoch along the channel axis to form an augmented multichannel signal before covariance estimation. n_times must equal the third dimension of X; otherwise a ValueError is raised.
        estimator (str | callable, optional): Covariance estimator to apply on the concatenated data. By default "cov" (the standard sample covariance) is used. Accepts either a string name corresponding to a registered estimator in pyriemann.utils.covariance.cov_est_functions (resolved via check_function) or a callable that takes a 2D array (n_channels_total, n_times) and returns its covariance matrix. If a string is given and it is not found among available estimators, check_function will raise an exception.
        kwds (dict, optional): Additional keyword arguments forwarded directly to the chosen covariance estimator. These parameters control estimator-specific behavior (for example shrinkage amount when using a shrinkage estimator) and are not inspected by covariances_EP itself.
    
    Returns:
        numpy.ndarray: covmats, array of covariance matrices with shape (n_matrices, n_channels + n_channels_proto, n_channels + n_channels_proto). covmats[i] is the covariance matrix computed on np.concatenate((P, X[i]), axis=0). The returned array uses the same dtype as X.dtype. No in-place modification of X or P occurs; the function allocates and returns a new array.
    
    Raises:
        ValueError: If the temporal dimension n_times of X and P differ.
        Exception: If estimator resolution fails (e.g., unknown estimator name) or if the estimator callable raises, the exception propagates to the caller.
    """
    from pyriemann.utils.covariance import covariances_EP
    return covariances_EP(X, P, estimator, **kwds)


################################################################################
# Source: pyriemann.utils.covariance.covariances_X
# File: pyriemann/utils/covariance.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def pyriemann_utils_covariance_covariances_X(
    X: numpy.ndarray,
    estimator: str = "cov",
    alpha: float = 0.2,
    **kwds
):
    """Special-form covariance matrix estimator that embeds a multichannel time-series into a
    block SPD matrix combining channel and time information for Riemannian processing.
    
    This function implements the "special form" covariance construction introduced for
    interpretation and visualization of multivariate time-series in the pyRiemann project
    (see README). It is intended for use with biosignals (e.g., EEG, MEG, EMG) or other
    multivariate time series where each epoch/trial is given as a matrix of channels by
    time samples. The resulting covariance matrices have shape (n_channels + n_times,
    n_channels + n_times) and are suitable for downstream Riemannian-geometry based
    algorithms (classification, averaging, tangent-space mapping) used in brain-computer
    interface (BCI) and remote sensing workflows.
    
    Args:
        X (numpy.ndarray): Input data array of shape (n_matrices, n_channels, n_times).
            Each entry X[i] is a multichannel time-series (n_channels rows, n_times
            columns) corresponding to one epoch/trial/observation. This argument is read
            (not modified in-place); the function will attempt to unpack X.shape and will
            raise a ValueError if X does not have exactly three dimensions or if the
            dimensions are inconsistent with the algorithm.
        estimator (str | callable): Covariance matrix estimator to use for each embedded
            matrix Y. By default this is the string "cov", which refers to the standard
            covariance estimator available in pyriemann.utils.covariance.covariances;
            alternatively a callable with signature est(Y, **kwds) returning an ndarray
            covariance matrix may be provided. The estimator is resolved with
            check_function(estimator, cov_est_functions) internally; if the string does
            not match a known estimator or the callable is incompatible, an exception
            from check_function or from the estimator call will be raised. Any estimator
            used must accept a 2D array Y and optional keyword parameters and return a 2D
            covariance matrix whose shape equals (n_channels + n_times, n_channels +
            n_times).
        alpha (float): Regularization parameter (strictly positive) used to build the
            block matrix embedding (see Eq(9) in the referenced paper). Default is 0.2.
            The function validates this value and raises ValueError if alpha <= 0. The
            final returned covariance matrices are normalized by dividing by (2 * alpha)
            (implements Eq(10)), so alpha controls both the off-diagonal regularization
            blocks and the overall scaling of the result.
        kwds (dict): Optional keyword parameters forwarded directly to the covariance
            estimator specified by estimator. Typical estimator-specific parameters (for
            example shrinkage intensity for a shrinkage estimator) should be provided
            here. These parameters are not inspected by covariances_X and any mismatch
            will surface as an error from the estimator call.
    
    Behavior and implementation details:
        The function first checks that alpha > 0 and resolves the estimator function.
        It computes centering matrices for channels and times:
        Hchannels = I_nchannels - (1/n_channels) 1 1^T and Htimes = I_ntimes - (1/n_times) 1 1^T.
        It applies double-centering to the input epochs as X_centered = Hchannels @ X @ Htimes
        (corresponding to Eq(8) in the reference), producing zero-mean rows and columns per
        epoch. For each epoch i a block matrix Y is constructed as:
        top-left block = X_centered[i] (n_channels x n_times),
        top-right block = alpha * I_nchannels,
        bottom-left block = alpha * I_ntimes,
        bottom-right block = X_centered[i].T (n_times x n_channels),
        resulting in Y of shape (n_channels + n_times, n_channels + n_times) (Eq(9)).
        The provided estimator is called as est(Y, **kwds) to compute a covariance matrix
        for Y. After all epochs are processed, the array of covariance matrices is divided
        by (2 * alpha) to apply the normalization indicated in the method (Eq(10)).
    
    Side effects, defaults, and failure modes:
        The function does not modify the caller's input array X in-place; it creates
        intermediate arrays for centering and for the block matrix Y. It allocates an
        output array covmats of shape (n_matrices, n_channels + n_times, n_channels +
        n_times). Errors that can be raised include ValueError when alpha <= 0, ValueError
        or IndexError when X does not have three dimensions or has inconsistent shapes,
        and errors propagated from check_function if estimator is an unrecognized string.
        If the estimator callable returns an array with an unexpected shape, a subsequent
        broadcasting or assignment error may be raised. Any estimator-specific errors
        triggered by parameters in kwds will propagate from the estimator.
    
    Returns:
        covmats (numpy.ndarray): Array of shape (n_matrices, n_channels + n_times,
            n_channels + n_times) containing the special-form covariance matrices for
            each epoch. Each returned matrix is a symmetric positive-definite (SPD) block
            matrix embedding channel and time information, normalized by (2 * alpha)
            as implemented in the algorithm. These matrices are intended for use with
            pyRiemann's Riemannian-geometry based processing and machine-learning
            algorithms (e.g., MDM classifier, tangent-space mapping) for BCI or remote
            sensing applications and related multivariate analysis tasks.
    """
    from pyriemann.utils.covariance import covariances_X
    return covariances_X(X, estimator, alpha, **kwds)


################################################################################
# Source: pyriemann.utils.covariance.cross_spectrum
# File: pyriemann/utils/covariance.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def pyriemann_utils_covariance_cross_spectrum(
    X: numpy.ndarray,
    window: int = 128,
    overlap: float = 0.75,
    fmin: float = None,
    fmax: float = None,
    fs: float = None
):
    """pyriemann.utils.covariance.cross_spectrum computes the complex cross-spectral matrices of a real multichannel time series using short-time Fourier transform windows and returns the frequency bins associated with those spectra. This function is used in pyRiemann to obtain frequency-domain covariance estimates from biosignals (for example EEG, MEG, EMG) for downstream Riemannian geometry processing and classification in brain-computer interface (BCI) and remote sensing workflows.
    
    Args:
        X (ndarray): Multi-channel time-series, real-valued, with shape (n_channels, n_times). Each row corresponds to one channel/sensor and columns correspond to time samples. The function requires real-valued input and will raise ValueError if X has a non-real dtype.
        window (int): Length of the FFT window used for spectral estimation. Default is 128. The value is cast to int; it must be a positive integer (>= 1) or a ValueError is raised. The window determines frequency resolution and the number of frequency bins n_freqs = int(window/2) + 1 for real-input half-spectrum.
        overlap (float): Fractional overlap between consecutive windows, expressed in (0, 1). Default is 0.75. A value outside the open interval (0, 1) raises a ValueError. The hop step is computed as int((1.0 - overlap) * window).
        fmin (float | None): Minimal frequency to include in the returned spectra. Default is None. This parameter is only applied when fs is provided; otherwise it is ignored and a runtime warning is emitted. If provided together with fs, fmin defaults to 0 when None.
        fmax (float | None): Maximal frequency to include in the returned spectra. Default is None. This parameter is only applied when fs is provided; otherwise it is ignored and a runtime warning is emitted. If provided together with fs, fmax defaults to fs/2 when None. If fmax <= fmin a ValueError is raised. If 2.0 * fmax > fs (violating Nyquist), a ValueError is raised.
        fs (float | None): Sampling frequency of the time-series. Default is None. When provided, the function builds the frequency axis in Hz and selects frequency bins between fmin and fmax. When fs is None, the function does not compute or return a frequency vector and ignores fmin/fmax (with warnings).
    
    Returns:
        S (ndarray): Cross-spectral matrices, complex-valued, with shape (n_channels, n_channels, n_freqs). For each retained frequency bin, S[..., k] is the Hermitian cross-spectral matrix computed as X_f.conj().T @ X_f over all time windows, normalized by the number of windows and the squared L2-norm of the Hanning window. The output respects Parseval's theorem normalization for the half-spectrum: spectral bins other than DC (and Nyquist when window is even) are multiplied by 2.
        freqs (ndarray | None): Frequencies associated to cross-spectra with shape (n_freqs,). This is a 1-D array in Hz when fs is provided. If fs is None, freqs is None and fmin/fmax are ignored (a runtime warning is issued).
    
    Raises:
        ValueError: If X is not real-valued, if window < 1, if overlap is not in (0, 1), if fmax <= fmin when fs is provided, or if fmax violates the Nyquist condition (2.0 * fmax > fs).
    
    Notes:
        The function performs the following processing steps and behaviors that are important for practical use in BCI and multivariate signal processing:
        - Uses a Hanning window of length window for each short-time FFT; the windowing and overlap determine time-frequency trade-offs relevant for frequency-domain covariance estimation.
        - Uses numpy.lib.stride_tricks.as_strided to form overlapping windowed views without copying the input when possible; this is memory efficient but relies on valid input shape and step calculation.
        - Computes the real-input FFT via numpy.fft.rfft and thus returns only the non-redundant half-spectrum (n_freqs = int(window / 2) + 1).
        - Normalizes the cross-spectral matrices by the number of windows and the squared norm of the Hanning window to respect energy conservation.
        - Applies a factor of 2 to non-DC (and non-Nyquist for even window) bins to account for the discarded negative-frequency half-spectrum for real signals.
        - When fs is provided, frequency selection between fmin and fmax is exact on the FFT frequency grid; fmin and fmax are inclusive.
        - When fs is None, no frequency axis is returned and fmin/fmax are ignored (warnings.warn is called).
        - Computational cost scales with n_channels, n_windows, and n_freqs; peak memory for S is O(n_channels^2 * n_freqs).
    """
    from pyriemann.utils.covariance import cross_spectrum
    return cross_spectrum(X, window, overlap, fmin, fmax, fs)


################################################################################
# Source: pyriemann.utils.covariance.eegtocov
# File: pyriemann/utils/covariance.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def pyriemann_utils_covariance_eegtocov(
    sig: numpy.ndarray,
    window: int = 128,
    overlapp: float = 0.5,
    padding: bool = True,
    estimator: str = "cov"
):
    """Convert an EEG continuous multichannel time series into a sequence of covariance matrices using a sliding window.
    
    This function is used in pyRiemann workflows for BCI and biosignal processing to estimate covariance matrices from EEG signals over time, enabling subsequent Riemannian-geometry-based processing and classification (for example, in motor imagery or ERP paradigms). It applies a sliding window of fixed length over the input signal, optionally pads the signal at both ends with zeros to allow windows centered on the original edges, resolves the string identifier of a covariance estimator via check_function against cov_est_functions, and calls that estimator on each window. Each window is passed to the estimator transposed as an array of shape (n_channels, window_length) so the estimator receives channels on the first axis and time samples on the second axis.
    
    Args:
        sig (numpy.ndarray): 2-D array representing a continuous EEG signal with time along the first axis and channels along the second axis (n_times x n_channels). This is the raw input from which covariance matrices are estimated. If sig does not have two dimensions, the function will raise an exception when accessing sig.shape or when concatenating padding.
        window (int): Number of time samples in each sliding window. The window determines the temporal extent used to compute each covariance matrix. Defaults to 128. The function uses int(window / 2) for padding when padding is True.
        overlapp (float): Fraction of the window used to compute the step (jump) between successive windows; the actual step in samples is computed as int(window * overlapp). For example, overlapp=0.5 gives a 50% overlap between consecutive windows and a step of window/2 samples. Defaults to 0.5. Note that if int(window * overlapp) evaluates to 0 (for example overlapp is 0.0 or very small relative to window), the internal loop will not advance and the function will not terminate.
        padding (bool): If True, the function pads the start and end of sig with zeros of length int(window / 2) samples (shape (int(window/2), n_channels)) before windowing. Padding allows windows to be centered over the original signal edges and increases the number of output windows accordingly. If False, no padding is applied and windows are taken only from the original signal. Defaults to True.
        estimator (str): Name of the covariance estimator to use, given as a string and resolved via check_function against the cov_est_functions registry in pyriemann.utils.covariance. The resolved estimator is called for each window with the window slice transposed (so the estimator receives an array of shape (n_channels, window)). The default "cov" typically corresponds to the sample covariance estimator in the registry. If the provided name is not found in the registry, check_function will raise an error.
    
    Returns:
        numpy.ndarray: Array containing the covariance matrices estimated for each sliding window. In the normal case this is a 3-D array with shape (n_windows, n_channels, n_channels), where n_windows is the number of windows extracted by the sliding procedure. If no windows satisfy the condition (for example if window is larger than the (possibly padded) signal length), an empty numpy.ndarray is returned (np.array(X) where X is an empty list). Each entry along the first axis is the matrix returned by the chosen estimator for the corresponding window.
    
    Behavior and failure modes:
    - The function resolves estimator by calling check_function(estimator, cov_est_functions); an invalid estimator name will raise an error coming from check_function.
    - The windowing loop computes jump = int(window * overlapp) and advances ix by jump; if jump == 0 the loop will not progress and the function will hang (infinite loop). Ensure overlapp and window produce a positive integer jump.
    - Padding uses int(window / 2) samples of zeros at each end; if window is odd the padding length is floored.
    - Each window passed to the estimator is sig[ix:ix + window, :].T (i.e., channels x time); estimators are expected to accept that layout.
    - If sig has incompatible dimensions or is not a numpy.ndarray with two dimensions, operations such as concatenation or shape indexing will raise an error.
    - The function makes no in-place modifications to the original sig object passed by the caller; it constructs a padded copy when padding is True.
    """
    from pyriemann.utils.covariance import eegtocov
    return eegtocov(sig, window, overlapp, padding, estimator)


################################################################################
# Source: pyriemann.utils.covariance.get_nondiag_weight
# File: pyriemann/utils/covariance.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def pyriemann_utils_covariance_get_nondiag_weight(X: numpy.ndarray):
    """Compute non-diagonality weights of a set of square matrices.
    
    This function computes a scalar weight for each square matrix provided in X that quantifies how non-diagonal the matrix is, following Eq(B.1) in [1]. In the context of pyRiemann and covariance-based multivariate analysis (for example EEG/MEG covariance matrices used in brain-computer interface pipelines and approximate joint diagonalization algorithms), these weights measure the relative energy of the off-diagonal elements compared to the diagonal elements and can be used to down-weight matrices that are nearly diagonal during joint-diagonalization or other aggregation steps.
    
    Args:
        X (ndarray, shape (..., n, n)): Set of square matrices. The last two axes of X must form square matrices of size n x n; any leading axes are treated as batch dimensions and are preserved in the output. Each matrix is treated elementwise (X**2 in the implementation), so X should be a numeric ndarray containing real or complex values as appropriate for the application (e.g., covariance/second-order statistics estimated from multichannel time series). The function expects at least a 2D ndarray; if X has additional leading dimensions they are interpreted as independent matrices.
    
    Returns:
        weights (ndarray, shape (...,)): Non-diagonality weights for each input matrix. For each matrix, the weight is computed as (1.0 / (n - 1)) * (sum_squared_off_diagonal / sum_squared_diagonal) where sum_squared_diagonal is the trace of X**2 and sum_squared_off_diagonal is the sum of all squared elements minus that trace. The returned array preserves the leading batch dimensions of X and has one value per matrix. Values are non-negative in normal cases; a value close to 0 indicates a nearly diagonal matrix, while larger values indicate relatively stronger off-diagonal energy.
    
    Behavior, side effects and failure modes:
        The function does not modify X in-place and returns a new ndarray of weights. It requires that the matrices are square: if is_square(X) is False a ValueError is raised with message "Matrices must be square". If the trace of X**2 (the denominator) is zero for some matrix (for example when all diagonal elements are exactly zero), the division yields infinities or NaNs according to NumPy semantics and a runtime warning may be emitted; such cases indicate that the non-diagonality metric is undefined for that matrix. Similarly, if n == 1 the factor (n - 1) equals zero and the computation results in division by zero (Inf/NaN) or a runtime warning; callers should avoid passing 1x1 matrices or must handle resulting infinite/NaN weights. The implementation uses NumPy operations (X**2, np.trace, np.sum) and preserves batch dimensions. The computation follows Eq(B.1) in [1] and is intended for use with covariance or second-order statistic matrices in pyRiemann workflows.
    
    References:
        The weight definition follows Eq(B.1) in [1], see: M. Congedo, C. Gouy-Pailler, C. Jutten, "On the blind source separation of human electroencephalogram by approximate joint diagonalization of second order statistics", Clinical Neurophysiology, 2008.
    """
    from pyriemann.utils.covariance import get_nondiag_weight
    return get_nondiag_weight(X)


################################################################################
# Source: pyriemann.utils.covariance.normalize
# File: pyriemann/utils/covariance.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def pyriemann_utils_covariance_normalize(X: numpy.ndarray, norm: str):
    """pyriemann.utils.covariance.normalize normalizes a batch of square matrices (covariance or similar) using one of three normalization schemes ("corr", "trace", "determinant"). This function is typically used in pyRiemann preprocessing pipelines to make covariance or Hermitian positive-definite matrices comparable across epochs, channels, sensors, or acquisitions before applying Riemannian geometry-based methods (for example, MDM classification, TangentSpace mapping, or other covariance-based BCI and remote sensing workflows described in the README). The normalization choices produce correlation matrices ("corr"), unit-trace matrices ("trace"), or matrices with determinant equal to +/-1 ("determinant"), which can help stabilize numerical processing and remove scale differences between matrices coming from different trials, sessions, or sensors.
    
    Args:
        X (numpy.ndarray): Set of square matrices with shape (..., n, n). This must be at least a 2-D NumPy array where the last two dimensions index square matrices. Each slice X[..., i, j] represents a matrix element. In the pyRiemann context, X typically contains estimated covariance or HPD/SPD matrices computed from multichannel biosignals (EEG/MEG/EMG) or spatial covariance estimates from remote sensing. The function does not modify the input array X in place; it returns a new array containing normalized matrices. Note that zero diagonal entries (for "corr") or zero traces (for "trace") or zero determinants (for "determinant") will lead to division by zero and produce inf or nan values in the output.
        norm (str): Normalization mode, one of {"corr", "trace", "determinant"}. The practical meaning of each mode in the pyRiemann workflow is:
            "corr": convert each matrix to a correlation matrix by dividing by the outer product of standard deviations computed from the diagonal. Resulting matrices have diagonal values equal to 1, off-diagonal values nominally in [-1, 1], and are suitable when only linear relationships (correlations) between channels are of interest.
            "trace": scale each matrix so its trace equals 1. This preserves relative variances across matrix entries while removing overall scale, useful when comparing matrices whose total power (trace) differs across trials or sensors.
            "determinant": scale each matrix so that its determinant equals +/-1. Implementation divides by |det(X)|^(1/n) (where n is matrix size), so the normalized determinant equals det(X)/|det(X)| (i.e., the sign of the original determinant). This is useful when relative volume (generalized variance) should be normalized while preserving determinant sign. Determinant normalization requires matrices to be invertible to avoid zero denominators; singular matrices will produce zeros in the denominator and lead to inf/nan in the result.
    
    Returns:
        numpy.ndarray: Xn, an array of the same shape as X containing the normalized matrices. The returned array is a new NumPy array (input X is not modified). For "corr", the function additionally applies clipping to force values into the closed interval [-1, 1] after normalization, ensuring numerical bounds for correlation coefficients while guaranteeing diagonal entries equal 1 (subject to floating-point rounding). For "trace" and "determinant", the returned matrices retain the original entry-wise signs/scales except for the global scaling applied; determinant normalization preserves the sign of the original determinant (output determinant is +1 or -1). Exceptions and failure modes: a ValueError is raised if X is not square along its last two dimensions, and a ValueError is raised if norm is not one of the supported modes. Division by zero can occur when diagonal entries, traces, or determinants are zero, producing inf or nan in Xn; determinant normalization in particular requires invertible matrices to avoid such issues.
    """
    from pyriemann.utils.covariance import normalize
    return normalize(X, norm)


################################################################################
# Source: pyriemann.utils.distance.distance
# File: pyriemann/utils/distance.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def pyriemann_utils_distance_distance(
    A: numpy.ndarray,
    B: numpy.ndarray,
    metric: str = "riemann",
    squared: bool = False
):
    """Distance between matrices according to a chosen metric, used for computing
    pairwise dissimilarities between symmetric/Hermitian positive definite (SPD/HPD)
    matrices such as covariance matrices estimated from multichannel biosignals
    (EEG/MEG/EMG) in brain-computer interface (BCI) workflows or covariance
    descriptors in remote sensing and hyperspectral imaging. This function accepts
    either a single pair of matrices A and B (both shape (n, n)) or a stack of
    matrices A with shape (n_matrices, n, n) and a single matrix B (n, n), and
    returns the scalar distance or a column vector of distances computed by the
    specified metric. The metric can be one of the predefined string identifiers
    implemented in the pyriemann distance module (for example "riemann", "euclid",
    "kullback", "wasserstein", etc.) or a user-provided callable implementing the
    same distance signature.
    
    Args:
        A (numpy.ndarray): First matrix or set of matrices. Must be either a 2-D
            array of shape (n, n) representing a single matrix, or a 3-D array of
            shape (n_matrices, n, n) representing a stack of n_matrices matrices.
            In typical pyRiemann use (BCI covariance-based pipelines), A contains
            symmetric (or Hermitian) positive definite covariance matrices estimated
            from multichannel time series. When A is a stack, the function computes
            the distance between each A[i] and B and returns an array of distances.
        B (numpy.ndarray): Second matrix. Must be a 2-D array of shape (n, n)
            compatible with A (same n). In standard applications B is typically a
            covariance or reference SPD matrix such as a class mean or a template.
        metric (str or callable): Metric for distance, default "riemann". Can be
            one of the implemented string identifiers: "chol", "euclid", "harmonic",
            "kullback", "kullback_right", "kullback_sym", "logchol", "logdet",
            "logeuclid", "riemann", "wasserstein", or a callable. If a callable is
            provided, it must accept arguments (A_sub, B, squared=bool) and return a
            scalar distance or a numpy array of shape (1,) consistent with the
            function's outputs. The default "riemann" corresponds to the Riemannian
            geodesic distance on the manifold of SPD matrices commonly used in BCI
            classification and covariance-based analysis.
        squared (bool): Return squared distance when True, default False. This
            behavior was added in version 0.5. When True, the function returns the
            squared value of the chosen metric; when False, it returns the usual
            (non-squared) distance. Use squared distances if downstream algorithms
            expect squared dissimilarities (for example some kernel or variance
            computations).
    
    Returns:
        d (float or numpy.ndarray): Distance between A and B. If A and B have the
        same 2-D shape (n, n), a single scalar float is returned representing the
        distance. If A has shape (n_matrices, n, n) and B has shape (n, n), an
        ndarray of shape (n_matrices, 1) is returned where each row i contains the
        distance between A[i] and B. The returned values follow the squared
        parameter: if squared is True the returned distance(s) are squared.
    
    Behavior and failure modes:
        The function resolves the metric parameter to a concrete distance function
        and then delegates computation to that function with signature
        distance_function(X, Y, squared=squared). If A and B are both 2-D and have
        identical shapes, the function computes a single distance. If A is 3-D and
        B is 2-D, the function computes distances for each matrix in A against B
        in a loop and returns a column vector of distances. If the input shapes are
        incompatible (for example A is 2-D and B is 3-D, or dimensions n do not
        match), the function raises ValueError with message "Inputs have incompatible
        dimensions." If the metric string is not one of the implemented identifiers
        or the callable does not follow the expected signature, the internal metric
        resolver will raise an error (typically ValueError or TypeError). No in-place
        modification of A or B is performed; the function allocates and returns new
        numpy arrays for batched outputs.
    
    Practical significance:
        In pyRiemann pipelines, this function is used to measure dissimilarities
        between SPD covariance matrices for tasks such as minimum distance to mean
        classification (MDM), tangent-space projections, transfer learning between
        sessions/subjects, and other Riemannian-geometry-based processing of
        biosignals and image-derived covariance descriptors. The choice of metric
        (for example "riemann" vs "euclid" vs "wasserstein") directly impacts
        downstream classification and clustering performance and should match the
        assumptions of the analysis and the properties of the input matrices (e.g.
        SPD requirement).
    """
    from pyriemann.utils.distance import distance
    return distance(A, B, metric, squared)


################################################################################
# Source: pyriemann.utils.distance.distance_chol
# File: pyriemann/utils/distance.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def pyriemann_utils_distance_distance_chol(
    A: numpy.ndarray,
    B: numpy.ndarray,
    squared: bool = False
):
    """Cholesky distance between two symmetric/Hermitian positive-definite (SPD/HPD) matrices.
    
    This function computes the Cholesky distance used in applications of Riemannian geometry to covariance matrices (for example, covariance matrices estimated from multichannel biosignals such as EEG, MEG or EMG in brain–computer interfaces, or local covariance estimates in remote sensing). The Cholesky distance between two SPD/HPD matrices A and B is defined as the Frobenius norm of the difference between their Cholesky factors: the function computes chol(A) and chol(B) using NumPy's Cholesky factorization and then returns the Frobenius norm (or its square) of their difference. Internally this function delegates to the Euclidean distance on the Cholesky factors (distance_euclid on np.linalg.cholesky(A) and np.linalg.cholesky(B)).
    
    Args:
        A (numpy.ndarray): First SPD/HPD matrix or batch of matrices. Must be at least 2-D with shape (..., n, n) where the last two dimensions form square, symmetric (real) or Hermitian (complex) positive-definite matrices. In the typical BCI/EEG workflow, A represents one estimated covariance matrix (or an array of covariance matrices across epochs or trials).
        B (numpy.ndarray): Second SPD/HPD matrix or batch of matrices, with the same shape as A (identical leading/batch dimensions and identical n for the last two dimensions). B represents the covariance matrix (or matrices) to compare with A in practical signal processing or classification pipelines.
        squared (bool): If False (default), return the Frobenius norm distance d = ||chol(A) - chol(B)||_F. If True, return the squared Frobenius norm distance d^2. Use squared=True when squared distances are required directly (for example, in some loss formulations) to avoid an unnecessary square-root.
    
    Returns:
        float or numpy.ndarray: The Cholesky distance(s) between A and B. If A and B are single matrices (shape (n, n)), a Python float is returned. If A and B contain batch dimensions (shape (..., n, n)), a numpy.ndarray of shape (...) is returned with one distance per pair of matrices along the batch dimensions. When squared is True, the returned value(s) are squared Frobenius norms.
    
    Behavior, side effects, and failure modes:
        The function computes NumPy Cholesky factorizations via np.linalg.cholesky(A) and np.linalg.cholesky(B) and then computes Euclidean distances between the resulting lower-triangular factors. No in-place modification of A or B occurs. The function requires that each matrix along the last two dimensions of A and B be symmetric (real) or Hermitian (complex) positive-definite; otherwise np.linalg.cholesky will raise numpy.linalg.LinAlgError. If A and B have incompatible shapes (different number of dimensions, different batch shapes, or different trailing matrix size n), a ValueError may be raised by NumPy operations or by the internal distance_euclid call. The function preserves dtype (real or complex) of inputs insofar as NumPy Cholesky and norm operations allow.
    """
    from pyriemann.utils.distance import distance_chol
    return distance_chol(A, B, squared)


################################################################################
# Source: pyriemann.utils.distance.distance_harmonic
# File: pyriemann/utils/distance.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def pyriemann_utils_distance_distance_harmonic(
    A: numpy.ndarray,
    B: numpy.ndarray,
    squared: bool = False
):
    """Harmonic distance between invertible matrices.
    
    Compute the harmonic distance d(A, B) = ||A^{-1} - B^{-1}||_F between two invertible matrices A and B. In the pyRiemann library this distance quantifies dissimilarity between precision matrices (inverses of covariance matrices) that are commonly estimated from multichannel biosignals (for example EEG/MEG in brain–computer interface applications) or from spatial patches in remote sensing. The function inverts each input matrix using numpy.linalg.inv and delegates the Frobenius-norm computation to the Euclidean-distance implementation (distance_euclid), so it is effectively the Euclidean distance applied to the inverses of the inputs.
    
    Args:
        A (numpy.ndarray): First invertible matrices. Must be at least a 2D square ndarray with shape (..., n, n). Each n-by-n matrix represents an invertible linear operator such as a covariance matrix estimated from multichannel time series; the function computes A^{-1} for every matrix in the leading batch dimensions. Passing non-square or singular matrices will cause numpy.linalg.inv to raise an error.
        B (numpy.ndarray): Second invertible matrices. Must have the same shape as A, i.e., (..., n, n), and each matrix must be invertible. B represents the matrices to compare against A (for example a template covariance or another batch of covariance estimates). Mismatched shapes between A and B will cause an error when the pairwise difference of inverses is computed.
        squared (bool): Whether to return the squared harmonic distance. Default False. If False, the function returns the Frobenius norm ||A^{-1} - B^{-1}||_F. If True, it returns the squared Frobenius norm (||A^{-1} - B^{-1}||_F)^2. Using the squared option avoids an extra square-root when algorithms require squared distances for efficiency or numerical reasons.
    
    Returns:
        float or numpy.ndarray: Harmonic distance(s) between A and B. If A and B are 2D arrays (single n-by-n matrices), a scalar float is returned. If A and B are arrays with leading batch dimensions (..., n, n), an ndarray of shape (...) is returned containing the distance for each corresponding pair of matrices. The returned value is the (optionally squared) Frobenius norm of the difference between the inverses and is used in pyRiemann pipelines as a dissimilarity measure between covariance/precision matrices for tasks such as classification or clustering of biosignal trials.
    
    Raises:
        numpy.linalg.LinAlgError: If any input matrix is singular or not invertible, numpy.linalg.inv will raise a LinAlgError.
        ValueError: If A and B do not have compatible shapes (they must match exactly in all dimensions) or are not at least two-dimensional square matrices, downstream shape checks or the Euclidean distance routine may raise a ValueError.
    
    Notes:
        Side effects and performance: This function computes explicit matrix inverses using numpy.linalg.inv, which is computationally expensive for large matrices and may be numerically unstable for poorly conditioned matrices. For covariance matrices commonly used in BCI and remote sensing, consider regularization (outside this function) to improve conditioning before computing the harmonic distance.
        Implementation detail: The function is implemented as distance_euclid(np.linalg.inv(A), np.linalg.inv(B), squared=squared), i.e., the Euclidean (Frobenius) distance applied to the matrix inverses.
    """
    from pyriemann.utils.distance import distance_harmonic
    return distance_harmonic(A, B, squared)


################################################################################
# Source: pyriemann.utils.distance.distance_kullback
# File: pyriemann/utils/distance.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def pyriemann_utils_distance_distance_kullback(
    A: numpy.ndarray,
    B: numpy.ndarray,
    squared: bool = False
):
    """pyriemann.utils.distance.distance_kullback computes the left Kullback-Leibler divergence between two symmetric/Hermitian positive definite (SPD/HPD) matrices.
    
    This function evaluates the information-theoretic left Kullback-Leibler divergence d(A,B) = 0.5*(tr(B^{-1}A) - n + log(det(B)/det(A))) for one or a batch of SPD/HPD matrices A and B. In the pyRiemann package this distance is commonly used to compare covariance matrices estimated from multichannel biosignals (e.g., EEG, MEG, EMG) for brain-computer interface (BCI) workflows and is also applicable to covariance-like matrices in remote sensing applications. The implementation computes the trace term via solving linear systems (effectively B^{-1}A), obtains signed log-determinants via numpy.linalg.slogdet, and returns the real, non-negative divergence values. The function accepts real-valued SPD and complex-valued Hermitian positive definite (HPD) matrices and supports broadcasting over leading batch dimensions to process multiple matrices at once.
    
    Args:
        A (numpy.ndarray): First input SPD/HPD matrix or stack of matrices with shape (..., n, n). In BCI and covariance-estimation contexts, A typically represents an estimated covariance matrix for one epoch, trial, or spatial window. The array must be at least 2-D, square on the last two axes, and contain SPD (for real data) or HPD (for complex data) matrices. The function will not modify A in-place.
        B (numpy.ndarray): Second input SPD/HPD matrix or stack of matrices with the same shape contractibility as A: (..., n, n). B commonly represents a reference covariance matrix (for example a class prototype or a session/subject reference) against which A is compared. B must be square on the last two axes and positive definite so that inversion is possible; the function uses linear solves with B rather than explicit matrix inversion for numerical stability.
        squared (bool): If False (default), return the Kullback-Leibler divergence d(A, B) as defined above. If True, return the squared divergence (d(A, B)**2). The default behavior (squared=False) preserves the standard divergence scale used in many classification and distance-based algorithms in pyRiemann; setting squared=True may be useful where a squared-distance form is required by downstream algorithms.
    
    Returns:
        float or numpy.ndarray: The left Kullback-Leibler divergence value(s) between A and B. If A and B describe single matrices (shape (n, n)), a scalar float is returned. If A and/or B contain a batch of matrices (shape (..., n, n)), an ndarray of shape (...) is returned containing one divergence value per corresponding matrix pair. Returned values are the real part of the computed expression and are non-negative in theory; numerical roundoff can produce values very close to zero.
    
    Behavior and side effects:
        The function validates inputs (shape and compatibility) and uses broadcasting semantics over leading axes to allow batch processing. It computes trace terms by solving linear systems with B and uses numpy.linalg.slogdet to compute log-determinants, then combines these terms according to the Kullback-Leibler formula. No in-place modification of inputs occurs.
    
    Failure modes and exceptions:
        The function will raise a ValueError if A and B do not have compatible shapes or are not at least 2-D square matrices on the last two axes. A numpy.linalg.LinAlgError (or other numpy linear-algebra exceptions) may be raised if B is singular or otherwise non-invertible, preventing the computation of B^{-1}A. A TypeError may occur if inputs are not numpy.ndarray instances. Users should ensure inputs are valid SPD/HPD matrices (for example, estimated covariance matrices should be regularized if necessary) before calling this function to avoid numerical or linear-algebra errors.
    """
    from pyriemann.utils.distance import distance_kullback
    return distance_kullback(A, B, squared)


################################################################################
# Source: pyriemann.utils.distance.distance_kullback_right
# File: pyriemann/utils/distance.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def pyriemann_utils_distance_distance_kullback_right(
    A: numpy.ndarray,
    B: numpy.ndarray,
    squared: bool = False
):
    """Summary:
    Compute the right Kullback-Leibler divergence between two covariance matrices. This function is a thin wrapper that implements the "right" version of the Kullback-Leibler divergence used in pyRiemann for comparing symmetric (or Hermitian) positive definite (SPD/HPD) covariance matrices commonly estimated from multichannel biosignals (e.g., EEG, MEG, EMG) or remote sensing data. Concretely, it calls the underlying distance_kullback function with its arguments swapped so that the divergence is evaluated as D_right(A, B) = distance_kullback(B, A, squared=squared). The returned value quantifies how much the distribution characterized by A diverges from that characterized by B in the context of Riemannian geometry on SPD/HPD matrices and can be used as a dissimilarity measure in classification, clustering, or transfer-learning pipelines described in the project README.
    
    Args:
        A (numpy.ndarray): Left-hand matrix argument of the right Kullback-Leibler divergence. Expected to be a square numpy.ndarray representing a covariance matrix (symmetric positive definite for real-valued data or Hermitian positive definite for complex-valued data). In typical pyRiemann usage this is an estimated covariance matrix of shape (n_channels, n_channels). The function treats A as the first argument of the right divergence D_right(A, B) which is implemented by calling distance_kullback(B, A, ...).
        B (numpy.ndarray): Right-hand matrix argument of the right Kullback-Leibler divergence. Expected to be a square numpy.ndarray with the same shape and type constraints as A (covariance / SPD / HPD matrix). In practical pipelines B often represents a reference or template covariance matrix (for example a class mean in Riemannian classification) against which A is compared.
        squared (bool): If False (default), return the standard Kullback-Leibler divergence value (a non-negative scalar). If True, return the squared form of the divergence if the underlying distance_kullback implementation supports a squared option. This flag is forwarded unchanged to the underlying distance_kullback call. Default behavior is squared=False.
    
    Behavior, side effects, defaults, and failure modes:
    This function performs no in-place modification of the inputs; it simply forwards B and A to distance_kullback and returns that result. It assumes A and B are valid covariance matrices: square, same shape, and positive definite (SPD/HPD). If the inputs do not satisfy these conditions, the underlying routines invoked by distance_kullback (for example matrix decompositions or inversions) may raise exceptions such as ValueError for shape mismatches or linear algebra errors (e.g., numpy.linalg.LinAlgError) if matrices are singular or not positive definite. The function preserves numeric dtype and will propagate any warnings or floating-point errors raised by numpy or scipy routines used internally. The returned divergence is non-negative in theory; numerical round-off may lead to very small negative values depending on implementation details of the underlying routines.
    
    Returns:
        float: A non-negative scalar representing the right Kullback-Leibler divergence between A and B as implemented by distance_kullback(B, A, squared=squared). The value quantifies the dissimilarity between the statistical descriptions encoded by the two covariance matrices and is suitable for use as a distance-like measure in Riemannian geometry-based machine learning workflows (classification, clustering, transfer learning) described in the pyRiemann README.
    """
    from pyriemann.utils.distance import distance_kullback_right
    return distance_kullback_right(A, B, squared)


################################################################################
# Source: pyriemann.utils.distance.distance_kullback_sym
# File: pyriemann/utils/distance.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def pyriemann_utils_distance_distance_kullback_sym(
    A: numpy.ndarray,
    B: numpy.ndarray,
    squared: bool = False
):
    """pyriemann.utils.distance.distance_kullback_sym computes the symmetrized Kullback-Leibler divergence (also known as the Jeffreys divergence) between two symmetric/Hermitian positive definite (SPD/HPD) matrices. This function is used in pyRiemann to quantify dissimilarity between covariance matrices arising from multivariate biosignals (e.g., EEG, MEG, EMG) and in other applications such as hyperspectral or SAR image processing where covariance/HPD matrices represent local statistics.
    
    Args:
        A (numpy.ndarray): First SPD/HPD matrix or batch of matrices with shape (..., n, n). Each last-two-dimensional block must be a square symmetric (real) or Hermitian (complex) positive definite matrix. In the pyRiemann context, A typically represents an estimated covariance matrix for one epoch, window, or spatial patch. The function requires at least a 2-D array; higher-dimensional arrays are treated as batches of matrices indexed by the leading dimensions. If A is not square or its trailing dimensions do not match B, a ValueError is raised.
        B (numpy.ndarray): Second SPD/HPD matrix or batch of matrices with the same shape as A, i.e., (..., n, n). B represents the comparison target (for instance, a reference covariance matrix, class mean, or other covariance estimate). A and B must have identical shapes; otherwise the function will raise an error. Both A and B must contain positive definite matrices; non-positive eigenvalues can cause numerical linear algebra errors (e.g., numpy.linalg.LinAlgError) from underlying computations.
        squared (bool): If False (default), return the symmetrized Kullback-Leibler divergence d between A and B. If True, return the squared divergence d**2. The squared option was added to provide compatibility with metric-based workflows that require squared distances (versionadded: 0.5). There are no side effects from toggling this flag; it only changes the returned numeric value.
    
    Returns:
        float or numpy.ndarray: The symmetrized Kullback-Leibler divergence (Jeffreys divergence) between A and B. For single matrix inputs (2-D arrays) a scalar float is returned. For batched inputs with shape (..., n, n) an array of shape (...) is returned, where each element corresponds to the divergence computed for the matching pair of matrices from A and B. If squared is True the returned value(s) are squared. The returned value measures dissimilarity on the manifold of SPD/HPD matrices and is commonly used in classification and distance-based algorithms in pyRiemann (for example, comparing covariance matrices between trials or subjects). Possible failure modes include ValueError for shape mismatches or non-square inputs, and numerical linear algebra errors if inputs are not positive definite or contain invalid values (NaN/Inf). There are no in-place modifications to A or B; the function is pure and has no side effects.
    """
    from pyriemann.utils.distance import distance_kullback_sym
    return distance_kullback_sym(A, B, squared)


################################################################################
# Source: pyriemann.utils.distance.distance_logchol
# File: pyriemann/utils/distance.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def pyriemann_utils_distance_distance_logchol(
    A: numpy.ndarray,
    B: numpy.ndarray,
    squared: bool = False
):
    """Log-Cholesky distance between two symmetric/Hermitian positive definite (SPD/HPD)
    matrices.
    
    This function computes the Log-Cholesky distance used in pyRiemann to compare SPD/HPD
    matrices such as covariance matrices estimated from multichannel biosignals (EEG,
    MEG, EMG) or from remote sensing data. The distance is computed by taking the
    Cholesky decomposition of each input matrix, extracting the strictly lower
    triangular entries and the diagonal entries, applying a natural logarithm to the
    diagonal entries, and then computing the Euclidean norm of the concatenated
    differences. Concretely, for matrices A and B, if L_A and L_B are their (lower)
    Cholesky factors, the squared distance equals the squared Frobenius norm of the
    difference of the strictly lower triangular parts plus the squared Frobenius
    norm of the difference of the elementwise logarithms of the diagonals. Returning
    the non-squared distance takes the square root of this sum. This representation
    is useful in Riemannian processing pipelines (e.g., covariance estimation and
    classification in BCI) because it provides a vectorized representation of SPD/HPD
    matrices that can be used with standard Euclidean methods.
    
    Args:
        A (numpy.ndarray): First input SPD/HPD matrix or batch of matrices with
            shape (..., n, n). Must be at least 2D. In pyRiemann typical usage is to
            pass covariance matrices estimated from multichannel time series (for
            example, shape n_epochs x n_channels x n_channels). The function treats
            the last two dimensions as the square matrix dimensions and computes the
            Cholesky decomposition along those dimensions.
        B (numpy.ndarray): Second input SPD/HPD matrix or batch of matrices with
            the same shape as A: (..., n, n). The function computes the pairwise
            Log-Cholesky distance between corresponding matrices in A and B along
            the leading dimensions. A and B must be compatible in shape; broadcasting
            is not performed by this function.
        squared (bool): Default False. If True, return the squared Log-Cholesky
            distance (the sum of squared differences of the strictly lower triangular
            parts and the log-diagonals) to avoid the square-root operation when a
            squared metric is desired for numerical or performance reasons. If False,
            return the non-squared distance (the square root of that sum).
    
    Returns:
        float or numpy.ndarray: If A and B are 2D (single matrices), returns a scalar
        float representing the Log-Cholesky distance between A and B. If A and B are
        arrays with leading batch dimensions, returns a numpy.ndarray of shape (...) —
        i.e., the same leading dimensions as the inputs — containing the distance for
        each corresponding pair of matrices. If squared is True, the returned values
        are the squared distances; otherwise they are the distances (non-negative).
        The return follows numpy's numeric type promotion rules (for complex HPD
        inputs the operation follows numpy complex arithmetic).
    
    Raises:
        ValueError: If A or B do not have at least two dimensions or if A and B do
            not have identical shapes (so pairwise distances cannot be computed).
        numpy.linalg.LinAlgError: If any matrix in A or B is not positive-definite,
            np.linalg.cholesky will fail (this includes the case of non-SPD/non-HPD
            matrices). This function does not alter the inputs; it calls numpy's
            Cholesky routine which enforces the positive-definite requirement.
        TypeError: If A or B are not array-like numpy.ndarrays or contain incompatible
            dtypes that prevent the required linear algebra operations.
    
    Notes:
        - The function internally validates inputs (via a helper used in pyRiemann)
          and then computes Cholesky decompositions with numpy.linalg.cholesky.
        - For typical pyRiemann workflows, A and B are covariance matrices estimated
          from biosignals or spatial covariance estimates from remote sensing; the
          Log-Cholesky distance provides a numerically stable and interpretable
          measure for comparing such SPD/HPD matrices in pipelines (e.g., for
          classification, clustering, or kernel computations).
        - The diagonal of the Cholesky factor is strictly positive for real SPD
          matrices, ensuring the logarithm is defined; for complex HPD matrices the
          computation follows numpy's complex log semantics.
    """
    from pyriemann.utils.distance import distance_logchol
    return distance_logchol(A, B, squared)


################################################################################
# Source: pyriemann.utils.distance.distance_logdet
# File: pyriemann/utils/distance.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def pyriemann_utils_distance_distance_logdet(
    A: numpy.ndarray,
    B: numpy.ndarray,
    squared: bool = False
):
    """Compute the log-det (Burg) distance between two symmetric/Hermitian positive-definite (SPD/HPD) matrices.
    
    This function implements the log-det distance d(A, B) = sqrt(log det((A + B) / 2) - 0.5 * log det(A B)) widely used in the Riemannian geometry of SPD/HPD matrices. In the pyRiemann library this distance is used to compare covariance matrices estimated from multivariate biosignals (e.g., EEG, MEG, EMG) in brain–computer interface and related applications, and forms a building block for classifiers and pipelines (for example, Minimum Distance to Mean (MDM) and other Riemannian methods). The implementation is numerically stable: it uses numpy.linalg.slogdet to compute logarithms of determinants and clips negative squared-distance values caused by floating-point rounding to zero before taking the square root.
    
    Args:
        A (numpy.ndarray): First SPD/HPD matrices. Expected to be at least 2D with shape (..., n, n), where the last two dimensions form a square positive-definite matrix (real symmetric for SPD or complex Hermitian for HPD). In the pyRiemann context A typically contains covariance matrices estimated from multichannel time series (e.g., shape n_epochs x n_channels x n_channels). The function calls _check_inputs(A, B) to validate shapes and basic compatibility; if the inputs are incompatible this preliminary check may raise an error.
        B (numpy.ndarray): Second SPD/HPD matrices. Must have the same shape as A: (..., n, n). B plays the role of the comparison matrix to A in distance computations used by Riemannian classifiers and pipelines. As with A, entries are expected to define positive-definite matrices; non-positive-definite inputs can lead to invalid results (see Failure modes).
        squared (bool): Whether to return the squared log-det distance. Default is False. If False (the default), the function returns the nonnegative distance d = sqrt(d2). If True, the function returns the nonnegative squared distance d2 = log det((A + B) / 2) - 0.5 * log det(A B). This option was added to allow callers to avoid an extra square-root operation when they need squared distances directly (versionadded: 0.5).
    
    Returns:
        float or numpy.ndarray: The computed log-det distance(s). If the inputs A and B represent a single pair of matrices (no leading batch dimensions), a scalar float is returned. If A and B contain batches of matrices with shape (..., n, n), an array with shape (...) is returned containing the distance for each corresponding pair. When squared=True the returned value(s) correspond to the squared distance d2; otherwise the square root of d2 is returned.
    
    Behavior, defaults, and failure modes:
        - The function first calls _check_inputs(A, B) to validate that A and B are compatible arrays (matching shapes and at least 2D). If that validation fails, the helper may raise an exception (for example ValueError or TypeError), and this function will not proceed.
        - Determinants are computed via numpy.linalg.slogdet for numerical stability: slogdet returns a sign and the natural logarithm of the absolute determinant; this code uses the logarithm value(s). For valid SPD/HPD inputs the determinants are positive and slogdet returns meaningful log-determinants.
        - Due to floating-point rounding, the intermediate squared-distance d2 can become a very small negative number; the implementation clamps d2 to be at least 0 using np.maximum(0, d2) before optionally taking the square root, ensuring real nonnegative outputs.
        - If A or B are not positive-definite (for example singular or indefinite), slogdet may produce -inf, nan, or unexpected values, and the returned distance may be nan or raise errors when downstream operations (e.g., taking the square root of a negative value) are attempted. Therefore, callers should ensure inputs are valid SPD/HPD matrices (for example covariance matrices estimated with proper regularization).
        - The function has no side effects: it does not modify A or B in-place and relies only on numpy operations.
        - The returned shapes preserve any leading batch dimensions present in the inputs, producing one distance value per matrix pair.
    """
    from pyriemann.utils.distance import distance_logdet
    return distance_logdet(A, B, squared)


################################################################################
# Source: pyriemann.utils.distance.distance_mahalanobis
# File: pyriemann/utils/distance.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def pyriemann_utils_distance_distance_mahalanobis(
    X: numpy.ndarray,
    cov: numpy.ndarray,
    mean: numpy.ndarray = None,
    squared: bool = False
):
    """pyriemann.utils.distance.distance_mahalanobis
    Compute the Mahalanobis distance between column vectors and a multivariate Gaussian distribution.
    
    This function computes the Mahalanobis distance between each column vector x in X and a multivariate Gaussian distribution N(mu, C) defined by mean vector mu and covariance matrix C. The computation follows d(x, N(mu, C)) = sqrt((x - mu)^H C^{-1} (x - mu)). It is used in pyRiemann for comparing feature vectors (for example, signal or covariance-derived features from EEG/MEG/EMG in BCI pipelines or patch vectors in remote sensing) to a Gaussian model, which is a common operation in classification, outlier detection, and distance-based algorithms that operate on covariance or multivariate data.
    
    Args:
        X (numpy.ndarray): Vectors provided as a 2-D array of shape (n, n_vectors), where n is the dimensionality of each vector and n_vectors is the number of column vectors to compare to the distribution. Columns of X are treated as the individual vectors x. The array may be real- or complex-valued as in the mathematical formulation in the codebase. Note: if mean is not None, X is modified in-place via X -= mean; pass a copy of X if you need to preserve the original data.
        cov (numpy.ndarray): Covariance matrix C of the multivariate Gaussian distribution, shaped (n, n). In the pyRiemann context this is typically an SPD (symmetric positive definite) matrix for real data or an HPD (Hermitian positive definite) matrix for complex data. cov must be invertible; if cov is singular, nearly singular, or not positive definite/Hermitian when expected, the internal matrix inverse square-root computation will fail or produce invalid results.
        mean (numpy.ndarray): Mean vector mu of the multivariate Gaussian distribution, shaped (n, 1). If provided, this mean is subtracted from each column of X before distance computation, implementing the (x - mu) term. If None (default), the distribution is considered centered at zero and X is used as-is. Providing a mean with an incorrect shape (not matching n) will raise a broadcasting/shape error at runtime.
        squared (bool): If False (default), return the Euclidean Mahalanobis distances d = sqrt((x-mu)^H C^{-1} (x-mu)). If True, return the squared Mahalanobis distances d^2 = (x-mu)^H C^{-1} (x-mu) without taking the square root. This option can avoid a square root when downstream code only needs squared distances. This parameter was introduced in version 0.5 of the package.
    
    Returns:
        d (numpy.ndarray): A 1-D numpy.ndarray of shape (n_vectors,) containing the Mahalanobis distances for each column vector in X. If squared is True, d contains squared Mahalanobis distances; otherwise d contains non-negative distances. The results are real-valued (the real part is returned) even when inputs are complex, consistent with the quadratic form; if numerical issues arise (e.g., non-invertible cov, NaNs or infinities in inputs), the returned array may contain NaNs or raise an exception during computation.
    
    Behavior, side effects, and failure modes:
        - The function subtracts mean from X in-place when mean is provided (X -= mean). To avoid mutating the caller's array, supply a copy of X.
        - The implementation computes Xw = invsqrtm(cov) @ X and then the squared norms via an einsum. Therefore cov must be suitable for inversion and matrix square-root inverse operations; non-positive-definite or singular covariance matrices typically cause linear-algebra errors from the internal invsqrtm routine.
        - Shape consistency is required: cov.shape[0] must equal X.shape[0], and mean, when provided, must have shape (n, 1) matching X's row dimension n. Mismatched shapes will raise numpy broadcasting or linear algebra exceptions.
        - Inputs containing non-finite values (NaN, inf) will propagate and may produce NaNs or exceptions.
        - This function is intended for numerical arrays consistent with pyRiemann conventions (covariance matrices from multichannel time series or spatial patches); it does not perform additional validation beyond relying on numpy/linear-algebra routines and will raise exceptions for invalid linear-algebra operations.
    """
    from pyriemann.utils.distance import distance_mahalanobis
    return distance_mahalanobis(X, cov, mean, squared)


################################################################################
# Source: pyriemann.utils.distance.distance_riemann
# File: pyriemann/utils/distance.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def pyriemann_utils_distance_distance_riemann(
    A: numpy.ndarray,
    B: numpy.ndarray,
    squared: bool = False
):
    """Affine-invariant Riemannian distance between two symmetric/Hermitian positive definite (SPD/HPD) matrices.
    
    This function computes the affine-invariant Riemannian metric used throughout pyRiemann for comparing covariance or similar SPD/HPD matrices arising in multivariate signal processing (for example EEG/MEG/EMG covariance matrices in brain–computer interface workflows, and covariance descriptors in remote sensing). Concretely, for two input matrices A and B the distance is computed as sqrt(sum_i (log(lambda_i))^2) where lambda_i are the joint (generalized) eigenvalues of the matrix pair (A, B). The value quantifies the geometric difference between A and B on the manifold of SPD/HPD matrices and is commonly used as a metric in classification (e.g., MDM), clustering, or embedding procedures in pyRiemann pipelines.
    
    Args:
        A (numpy.ndarray): First SPD/HPD matrices. Must be at least 2-D with shape (..., n, n). In typical use this is one or a batch of covariance matrices estimated from multichannel time series (e.g., EEG epochs). The function expects each trailing (n, n) block to be symmetric (or Hermitian) and positive definite. If these conditions are not met a validation error is raised by the internal input checker.
        B (numpy.ndarray): Second SPD/HPD matrices, same dimensions as A (shape (..., n, n)). B plays the role of the comparison target: the function measures the affine-invariant Riemannian distance from A to B elementwise across any leading batch dimensions. A and B must be aligned in shape; a mismatch in dimensions triggers a ValueError.
        squared (bool): Return squared distance when True (i.e., sum_i (log(lambda_i))^2) instead of the square root. Default is False (returns the standard Riemannian distance). Version added: 0.5. Using the squared distance can avoid an extra square-root operation when only squared distances are needed (for example when computing pairwise squared distances or optimizing least-squares objectives).
    
    Returns:
        float or numpy.ndarray: The affine-invariant Riemannian distance between A and B. If A and B represent single matrices (no leading batch dimensions) a scalar float is returned. If A and B contain batch dimensions the result is an ndarray with shape (...) corresponding to those leading dimensions. The returned value is non-negative and, when squared is False, equals sqrt(sum_i (log(lambda_i))^2). When squared is True the returned value is the non-negative squared distance (sum_i (log(lambda_i))^2).
    
    Behavior, side effects, and failure modes:
        - Inputs are validated by an internal checker (_check_inputs). Typical validation includes shape equality for A and B, at least 2-D inputs, and checks that each trailing (n, n) block is symmetric/Hermitian and positive definite. If validation fails a ValueError is raised.
        - The implementation computes generalized/joint eigenvalues (via a symmetric/Hermitian eigen solver). Numerical issues such as tiny negative eigenvalues due to floating-point errors may lead to invalid results when taking the logarithm; in such cases numpy may produce NaN or raise a numpy.linalg.LinAlgError if the eigen-decomposition fails. Users should ensure inputs are well-conditioned SPD/HPD (for example by regularization) when numerical stability is a concern.
        - No in-place modification of A or B is performed; the function is pure and has no side effects beyond returning the computed distances.
        - The function assumes real-valued symmetric or complex-valued Hermitian inputs consistent with pyRiemann usage for SPD/HPD matrices; it does not accept non-square matrices.
        - This distance is widely used in pyRiemann components (e.g., MDM classifier, tangent-space mapping and pairwise distance computations) where a Riemannian metric on covariance matrices is required.
    """
    from pyriemann.utils.distance import distance_riemann
    return distance_riemann(A, B, squared)


################################################################################
# Source: pyriemann.utils.distance.distance_wasserstein
# File: pyriemann/utils/distance.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def pyriemann_utils_distance_distance_wasserstein(
    A: numpy.ndarray,
    B: numpy.ndarray,
    squared: bool = False
):
    """pyriemann.utils.distance.distance_wasserstein computes the Wasserstein (Bures) distance between two symmetric/Hermitian positive semidefinite (SPSD/HPSD) matrices. In the pyRiemann context this function is used to measure dissimilarity between covariance or covariance-like matrices (for example, covariance matrices estimated from EEG/MEG/EMG epochs in brain–computer interface pipelines or covariance estimates in remote sensing applications) and can be employed in classification, clustering, or pipeline evaluation where Riemannian-aware distances are required.
    
    Args:
        A (ndarray): First SPSD/HPSD matrices, at least 2-D ndarray with shape (..., n, n). Each trailing pair of dimensions represents one n-by-n SPSD (real symmetric) or HPSD (complex Hermitian) matrix. In practical pyRiemann use this typically contains covariance matrices computed per epoch or spatial window; leading dimensions allow broadcasting and batch computation of distances across multiple matrices.
        B (ndarray): Second SPSD/HPSD matrices, same dtype and shape constraints as A and with identical trailing matrix dimensions (..., n, n). B plays the role of the second element in each pairwise comparison; when computing distances between corresponding matrices in A and B they must align on the leading shapes or be broadcastable according to numpy broadcasting rules that are accepted by the internal input checker.
        squared (bool): If False (default) the function returns the Wasserstein distance d(A,B) = sqrt(tr(A + B - 2 (B^{1/2} A B^{1/2})^{1/2})). If True the function returns the squared distance d^2 = tr(A + B - 2 (B^{1/2} A B^{1/2})^{1/2}) without taking the square root. The squared option was added in pyRiemann version 0.5 and can be useful to avoid an extra square-root operation when squared distances are needed for optimization or variance calculations.
    
    Returns:
        float or ndarray: Non-negative real scalar or ndarray of shape (...) containing the Wasserstein distance(s) between A and B. If both A and B are single matrices with shape (n, n) a scalar float is returned. If A and/or B contain leading batch dimensions the result has the corresponding leading shape and one value per matrix pair. If squared is True the returned values are the squared Wasserstein distances d^2; otherwise they are the non-squared distances d >= 0.
    
    Behavior, side effects, numerical details, and failure modes:
        The function validates inputs via an internal checker (_check_inputs) which enforces that A and B are at least 2-D, have compatible shapes for pairwise comparison, and are intended to be SPSD/HPSD matrices. If inputs do not meet these expectations the checker raises an error (for example ValueError or TypeError) and no distance is computed. The computation uses matrix square roots (sqrtm) and traces: it computes the trace of A + B - 2 * sqrtm(B^{1/2} A B^{1/2}). Because matrix square-root operations can introduce small numerical complex parts or tiny negative rounding errors for theoretically non-negative quantities, the implementation takes the real part of the traced result and clips negative values to zero (np.maximum(0, ...)) before optionally taking the square root. As a result the function returns real non-negative values; small imaginary numerical noise is discarded. If A or B are not SPSD/HPSD in practice the result may be meaningless or the internal checker may raise an error. No in-place modification of A or B occurs; all intermediate arrays are allocated internally.
    """
    from pyriemann.utils.distance import distance_wasserstein
    return distance_wasserstein(A, B, squared)


################################################################################
# Source: pyriemann.utils.distance.pairwise_distance
# File: pyriemann/utils/distance.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def pyriemann_utils_distance_pairwise_distance(
    X: numpy.ndarray,
    Y: numpy.ndarray = None,
    metric: str = "riemann",
    squared: bool = False
):
    """pyriemann.utils.distance.pairwise_distance computes the full pairwise distance matrix between two sets of square matrices (typically covariance matrices used in pyRiemann for biosignals such as EEG/MEG/EMG in BCI or remote-sensing applications). It supports the Riemannian metric by default ("riemann") and several other metrics implemented in pyriemann.utils.distance, and returns either the matrix of distances between all pairs in X (when Y is None) or between each element of X and each element of Y.
    
    Args:
        X (numpy.ndarray): Array of input matrices with shape (n_matrices_X, n, n). Each entry X[i] is expected to be a square matrix (for pyRiemann use, typically a symmetric positive definite (SPD) matrix for real-valued data or Hermitian positive definite (HPD) for complex-valued data) representing an estimated covariance or similar. This argument supplies the first set of matrices for which pairwise distances will be computed. The function assumes X is a 3-D NumPy array; malformed shapes (not 3-D or not square on the last two axes) will lead to errors from internal checks or downstream distance computations.
        Y (numpy.ndarray): Optional second array of matrices with shape (n_matrices_Y, n, n). Default is None, in which case Y is set to X to compute intra-set distances. When provided, Y must contain matrices with the same square dimension n as X. Passing None triggers symmetric optimization for metrics that are symmetric: when Y is None and the metric is symmetric (all supported metrics except some Kullback variants), the function computes only the upper-triangular block and mirrors it to produce a symmetric distance matrix. If Y is provided but has incompatible shape or dimension, a ValueError or an error from the underlying distance implementation will be raised.
        metric (str): Metric identifier selecting the distance to use. Default is "riemann". Supported metrics implemented or dispatched explicitly by this function include "riemann", "euclid", "harmonic", "logchol", "logeuclid", and the Kullback variants used by pyRiemann ("kullback", "kullback_right") as documented in pyriemann.utils.distance.distance. The chosen metric determines the mathematical formula used to compare two matrices (for example, Riemannian distance between SPD matrices is commonly used for covariance matrices in BCI). If an unknown or unsupported metric is passed, the underlying distance dispatcher will raise an error (typically ValueError).
        squared (bool): If False (default), return the actual distances. If True, return squared distances. This flag is passed to the underlying distance computations; note that for some metrics the squared/non-squared semantics correspond to returning the squared value of the metric defined by pyRiemann. This parameter was added in pyRiemann v0.5.
    
    Returns:
        numpy.ndarray: If Y is None, returns a square array of shape (n_matrices_X, n_matrices_X) containing pairwise distances between elements of X. If Y is provided, returns an array of shape (n_matrices_X, n_matrices_Y) containing distances between each element of X and each element of Y. When Y is None and the metric is symmetric, the implementation computes only the upper-triangular half of the matrix and mirrors it to produce a symmetric result for efficiency. The values are either distances or squared distances depending on the squared parameter.
    
    Behavior and failure modes:
        This function loops over matrix pairs and calls pyriemann.utils.distance.distance (or optimized specialized implementations) to compute each entry; computational complexity scales approximately as O(n_matrices_X * n_matrices_Y * cost(distance)), and when Y is None and the metric is symmetric it reduces redundant computations by exploiting symmetry. The function does not modify X or Y in place. Errors are raised if X or Y are not shaped as 3-D arrays of square matrices, if the inner matrix dimensions of X and Y do not match, or if an unsupported metric is requested (these errors originate from input validation or the underlying distance functions). The function is deterministic and has no side effects beyond computing and returning the distance array.
    """
    from pyriemann.utils.distance import pairwise_distance
    return pairwise_distance(X, Y, metric, squared)


################################################################################
# Source: pyriemann.utils.geodesic.geodesic
# File: pyriemann/utils/geodesic.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def pyriemann_utils_geodesic_geodesic(
    A: numpy.ndarray,
    B: numpy.ndarray,
    alpha: float,
    metric: str = "riemann"
):
    """pyriemann.utils.geodesic.geodesic: Compute the matrix located at a given position along the geodesic between two square matrices according to a chosen metric.
    
    Compute the matrix at position alpha on the geodesic connecting matrices A and B according to a specified metric. This function is used in pyRiemann for interpolating or extrapolating between matrices that represent multivariate descriptors such as covariance matrices arising in biosignal processing (EEG/MEG/EMG) and remote sensing applications. The common practical use is to interpolate between symmetric/Hermitian positive definite (SPD/HPD) matrices on a Riemannian manifold (for example, when working with covariance matrices in brain-computer interface pipelines), but other metrics are available and a user-provided callable can be used to implement custom geodesics.
    
    Args:
        A (numpy.ndarray): First matrices, with shape (..., n, n). Each trailing two dimensions must form a square matrix. In the pyRiemann context, these are typically covariance or HPD matrices estimated from multichannel time series; many metrics require these matrices to be symmetric/Hermitian positive definite. Supplying arrays with incompatible shapes (non-square trailing dimensions or mismatched batch shapes with B) will lead to an error when the chosen metric implementation is applied.
        B (numpy.ndarray): Second matrices, with shape (..., n, n). B must be conformable with A so that a geodesic between corresponding matrices can be computed; typically A and B have identical batch shapes and the same n. As with A, values are expected to be appropriate for the chosen metric (for example, SPD for the "riemann" metric).
        alpha (float): Position on the geodesic. Common usage is alpha in the interval [0, 1] where alpha == 0 returns A and alpha == 1 returns B; intermediate values give interpolated matrices. Values outside [0, 1] constitute extrapolation and may be supported or may produce invalid results depending on the chosen metric implementation.
        metric (string | callable, optional): Metric used to compute the geodesic, default "riemann". Accepted string keys (resolved via an internal dispatcher) include "euclid", "logchol", "logeuclid", "riemann", and "wasserstein". Alternatively, metric can be a callable implementing the geodesic operation; in that case the callable is expected to accept the same arguments (A, B, alpha) and return a numpy.ndarray of shape (..., n, n). If a string is provided, an internal check_function resolves it to the corresponding implementation; providing an unknown string will raise an error from that resolution step. Different metrics have different domain requirements and numerical behaviors: for example, Riemannian and log-based metrics assume SPD/HPD matrices and will raise or produce invalid outputs if inputs violate those assumptions.
    
    Returns:
        C (numpy.ndarray): Matrices on the geodesic, with shape (..., n, n). The returned array contains, for each corresponding pair of input matrices from A and B, the matrix located at position alpha along the geodesic defined by the selected metric. This function has no side effects: it does not modify A or B in place. Errors can occur if A and B have incompatible shapes, if the chosen metric implementation rejects the input arrays (for example because they are not SPD when required), or if metric resolution fails for an unrecognized string.
    """
    from pyriemann.utils.geodesic import geodesic
    return geodesic(A, B, alpha, metric)


################################################################################
# Source: pyriemann.utils.geodesic.geodesic_logchol
# File: pyriemann/utils/geodesic.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def pyriemann_utils_geodesic_geodesic_logchol(
    A: numpy.ndarray,
    B: numpy.ndarray,
    alpha: float = 0.5
):
    """pyriemann.utils.geodesic.geodesic_logchol computes the Log-Cholesky geodesic point between two symmetric/Hermitian positive definite matrices and reconstructs the corresponding SPD/HPD matrix at a given position alpha on that geodesic. This function implements the Log-Cholesky interpolation introduced in Z. Lin (2019) and is used in pyRiemann workflows to interpolate or extrapolate covariance/HPD matrices (for example, covariance matrices estimated from multichannel biosignals such as EEG/MEG in BCI applications) in a manner consistent with Riemannian geometry.
    
    Args:
        A (numpy.ndarray): First SPD/HPD matrix or batch of matrices with shape (..., n, n). In practical use within pyRiemann, A typically represents a covariance matrix estimated from multichannel time series (e.g., one epoch or one sensor configuration). The array must be symmetric (real) or Hermitian (complex) and positive definite so that a Cholesky factorization exists. The leading dimensions are treated as batch dimensions and are broadcasted consistently with B if needed; mismatched matrix shapes (different n) will raise an error.
        B (numpy.ndarray): Second SPD/HPD matrix or batch of matrices with shape (..., n, n). B is the other endpoint of the geodesic and plays the role of the target covariance/HPD matrix in interpolation or extrapolation tasks (e.g., transferring covariance structure between sessions or subjects). Like A, B must be symmetric/Hermitian and positive definite to allow Cholesky decomposition. The function computes the geodesic point between A and B for the same last-two dimensions.
        alpha (float): Position on the geodesic, default=0.5. alpha = 0 returns A, alpha = 1 returns B, and values in (0, 1) interpolate along the log-Cholesky geodesic. Values outside [0, 1] produce extrapolation along the same formula. In practice, alpha is used to obtain intermediate covariance matrices for interpolation, averaging, transfer learning, or path generation in Riemannian-based classification pipelines. The function does not enforce bounds on alpha; numerical behavior for extreme magnitudes follows the implemented algebraic formula.
    
    Returns:
        numpy.ndarray: C with shape (..., n, n). The SPD/HPD matrix or batch of matrices that lie at position alpha on the log-Cholesky geodesic between A and B. The returned matrix is reconstructed as L L* (conjugate-transpose) where L is the lower-triangular factor built by linearly interpolating strictly lower-triangular entries of the Cholesky factors of A and B and by taking elementwise power interpolation on the diagonal entries (geometric interpolation). In the context of pyRiemann, this returned matrix can be used as an intermediate covariance estimate for classification, visualization of Riemannian paths, or other geometry-aware processing steps.
    
    Behavior, side effects, defaults, and failure modes:
        The function computes np.linalg.cholesky(A) and np.linalg.cholesky(B) internally to obtain lower-triangular Cholesky factors. It then constructs an intermediate lower-triangular matrix by taking a weighted linear combination for strictly lower-triangular elements and a weighted power (A_diag^(1-alpha) * B_diag^alpha) for diagonal elements, and finally returns the Hermitian product geo @ geo.conj().swapaxes(-1, -2). There are no in-place modifications of the input arrays; the function allocates and returns a new array.
        If A or B is not positive definite (e.g., singular or indefinite), np.linalg.cholesky will raise numpy.linalg.LinAlgError. If the last-two dimensions of A and B are not equal (different n), broadcasting will fail or raise an error; ensure shapes match as (..., n, n). The function accepts complex-valued HPD matrices as used in SAR or other complex-domain applications supported by pyRiemann.
        Numerical round-off can produce very small deviations from exact symmetry/Hermiticity; downstream algorithms that require strict Hermitian structure may need a small symmetrization step if necessary. The default alpha is 0.5, which yields the midpoint on the log-Cholesky geodesic (commonly used as a symmetric interpolant between covariance matrices).
    """
    from pyriemann.utils.geodesic import geodesic_logchol
    return geodesic_logchol(A, B, alpha)


################################################################################
# Source: pyriemann.utils.geodesic.geodesic_riemann
# File: pyriemann/utils/geodesic.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def pyriemann_utils_geodesic_geodesic_riemann(
    A: numpy.ndarray,
    B: numpy.ndarray,
    alpha: float = 0.5
):
    """Affine-invariant Riemannian geodesic between symmetric/Hermitian positive-definite (SPD/HPD) matrices.
    
    Computes the matrix at position alpha on the affine-invariant Riemannian geodesic between two SPD/HPD matrices A and B using the formula C = A^{1/2} (A^{-1/2} B A^{-1/2})^alpha A^{1/2}. In the pyRiemann context, A and B typically represent covariance (SPD) or Hermitian positive-definite (HPD) matrices estimated from multichannel biosignals (EEG/MEG/EMG) or from spatial patches in remote sensing; this function is used when interpolating or moving along the Riemannian manifold of such covariance matrices (for example when computing midpoints for averaging, interpolation for data augmentation, or paths used in transfer learning and classification pipelines like MDM or tangent-space methods). The operation is affine-invariant: applying the same invertible linear transform to A and B yields the congruent transform on C.
    
    Args:
        A (numpy.ndarray): First SPD/HPD matrices with shape (..., n, n). Each trailing (n, n) block must be square, symmetric (real case) or Hermitian (complex case), and positive-definite. Leading dimensions (the "..." prefix) are supported for batch computation: the function broadcasts over matching leading dimensions when possible. In practice A is usually a covariance estimate per epoch or spatial window in BCI or remote-sensing workflows.
        B (numpy.ndarray): Second SPD/HPD matrices with shape (..., n, n). Must be compatible with A for broadcasting in the leading dimensions and have the same trailing matrix size n. B typically represents a target covariance/HPD matrix along the geodesic from A.
        alpha (float): Position on the geodesic. A scalar where alpha = 0 yields C equal to A and alpha = 1 yields C equal to B; the default is 0.5 producing the midpoint on the geodesic. Conceptually, alpha parametrizes interpolation (or extrapolation) along the affine-invariant Riemannian geodesic between A and B.
    
    Returns:
        C (numpy.ndarray): SPD/HPD matrices on the affine-invariant Riemannian geodesic, with shape (..., n, n). The returned array is newly allocated (no in-place modification of A or B occurs); each output (n, n) block is the result of C = A^{1/2} (A^{-1/2} B A^{-1/2})^alpha A^{1/2} applied to the corresponding inputs.
    
    Notes and failure modes:
        - The function relies on numerical routines that compute matrix square roots, inverse square roots, and fractional matrix powers. If A or B are not positive-definite, not square, or not symmetric/Hermitian, those underlying routines will typically raise errors (for example linear-algebra errors or ValueError) indicating invalid input. The caller is responsible for providing valid SPD/HPD matrices.
        - Leading-dimension broadcasting follows numpy semantics; mismatched leading shapes that cannot be broadcast will result in a runtime error from numpy operations.
        - Numerical stability depends on the conditioning of A and B; nearly singular matrices or matrices with very small eigenvalues can produce large numerical errors in the inverse-square-root and power operations.
        - There are no side effects: A and B are not modified in-place.
    """
    from pyriemann.utils.geodesic import geodesic_riemann
    return geodesic_riemann(A, B, alpha)


################################################################################
# Source: pyriemann.utils.geodesic.geodesic_wasserstein
# File: pyriemann/utils/geodesic.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def pyriemann_utils_geodesic_geodesic_wasserstein(
    A: numpy.ndarray,
    B: numpy.ndarray,
    alpha: float = 0.5
):
    """Wasserstein geodesic between two symmetric/Hermitian positive-definite (SPD/HPD) matrices.
    
    Compute the matrix C located at position alpha on the Wasserstein geodesic between two SPD/HPD matrices A and B using the closed-form expression implemented in this function. Concretely, for each pair of matching square matrices A and B of shape (..., n, n), the result C is computed as
    C = (1 - alpha)^2 * A + alpha^2 * B + alpha * (1 - alpha) * ( (A^(1/2) B A^(1/2))^(1/2) + its conjugate-transpose ),
    where A^(1/2) denotes the matrix square root of A and the inner square root is taken after conjugation by A^(1/2). This interpolation is the Wasserstein (optimal-transport) geodesic between Gaussian covariances and is commonly used in the pyRiemann workflow to interpolate or average covariance matrices estimated from multichannel biosignals (e.g., EEG, MEG, EMG) for BCI or remote-sensing applications. The returned matrices maintain SPD/HPD structure when inputs are SPD/HPD.
    
    Args:
        A (numpy.ndarray): First SPD/HPD matrices. Expected shape is (..., n, n), where the last two dimensions are square matrices. Each trailing (n, n) block should represent a symmetric positive-definite matrix (real-valued SPD) or a Hermitian positive-definite matrix (complex-valued HPD). In pyRiemann workflows, A typically represents covariance matrices estimated from multichannel time-series (e.g., EEG epochs).
        B (numpy.ndarray): Second SPD/HPD matrices. Must have the same leading/batch shape and same trailing (n, n) shape as A so that elementwise interpolation is well-defined. B plays the role of the endpoint covariance in the interpolation; common use cases include interpolating between session- or subject-specific covariance estimates for transfer learning or constructing geodesic midpoints for classifiers like MDM.
        alpha (float): Position on the geodesic. Default is 0.5 which yields the midpoint on the Wasserstein geodesic. alpha = 0 returns A and alpha = 1 returns B. The parameter controls interpolation (and extrapolation if values outside [0, 1] are supplied) between the two endpoints for each matrix pair. This function does not enforce bounds on alpha; behavior for values outside the unit interval corresponds to formal extrapolation of the quadratic expression.
    
    Returns:
        numpy.ndarray: SPD/HPD matrices on the Wasserstein geodesic. The returned array has the same leading/batch shape as A and B and trailing shape (n, n). For real-valued SPD inputs the output is real symmetric; for complex HPD inputs the output is Hermitian. The result is intended for downstream Riemannian processing in pyRiemann pipelines (e.g., classification, averaging, tangent-space mapping).
    
    Notes:
        - The implementation computes matrix square roots and inverse square roots (via sqrtm and invsqrtm-like routines) and performs matrix multiplications; computational complexity scales roughly as O(n^3) per (n, n) matrix due to these dense linear algebra operations.
        - Broadcasting over leading dimensions of A and B follows NumPy rules; A and B must be broadcast-compatible in their leading dimensions or have identical batch shapes.
        - The function is pure (no in-place modification of inputs) and returns a newly allocated array.
    
    Failure modes and exceptions:
        - If A or B are not positive-definite or contain numerical issues (e.g., non-finite entries), underlying matrix square-root or inverse-square-root routines may fail or raise linear algebra errors (for example numpy.linalg.LinAlgError) or produce complex/NaN results. Such exceptions are propagated from the underlying numerical routines.
        - Shape mismatches between A and B in the trailing (n, n) dimensions will raise broadcasting or shape errors from NumPy.
        - Numerical instabilities can occur for ill-conditioned matrices; in practice regularization (e.g., adding a small multiple of the identity) prior to calling this function can improve robustness in empirical pipelines.
    """
    from pyriemann.utils.geodesic import geodesic_wasserstein
    return geodesic_wasserstein(A, B, alpha)


################################################################################
# Source: pyriemann.utils.kernel.kernel
# File: pyriemann/utils/kernel.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def pyriemann_utils_kernel_kernel(
    X: numpy.ndarray,
    Y: numpy.ndarray = None,
    Cref: numpy.ndarray = None,
    metric: str = "riemann",
    reg: float = 1e-10
):
    """pyriemann.utils.kernel.kernel computes a kernel matrix of pairwise inner products between two sets of square matrices by projecting them to the tangent space at a reference matrix using a specified metric. This function is used in pyRiemann for building kernel matrices from symmetric/Hermitian positive definite (SPD/HPD) matrices (for example covariance matrices estimated from multichannel biosignals such as EEG, MEG or EMG) so they can be used with kernel methods (SVM, kernel ridge regression) or other algorithms that expect a kernel matrix.
    
    The function maps each input matrix to the tangent space at Cref according to metric and then computes inner products between the resulting tangent-space vectors to form the kernel matrix. If Y is None the kernel is computed between X and itself. If Cref is None a metric-specific reference (typically a mean on the manifold) is estimated from the input data according to the chosen metric. The regularization parameter reg is applied to mitigate numerical instabilities and to help ensure the resulting kernel matrix is positive-definite for downstream kernel algorithms commonly used in BCI and remote-sensing workflows.
    
    Args:
        X (ndarray): First set of matrices, shape (n_matrices_X, n, n). Each element is a square matrix (for example, an SPD covariance matrix estimated from one epoch of multichannel time series). In the BCI and remote-sensing domains this is typically an array of covariance matrices where n is the number of channels or spectral bands. Passing non-square arrays or arrays with inconsistent shapes will raise an error.
        Y (None | ndarray): Second set of matrices, shape (n_matrices_Y, n, n). If None (default) Y is set to X and the function returns a square kernel matrix between all pairs in X. When provided, Y must have the same n (matrix size) as X; mismatched matrix dimensions will raise a ValueError. Use Y to compute cross-kernels between two different sets of matrices (for example training and test covariance sets).
        Cref (None | ndarray): Reference matrix, shape (n, n). If provided, this square matrix is used as the tangent-space pole (reference point) where matrices are projected before inner-product computation. If None (default), a reference is estimated automatically using the metric (for example the Riemannian mean when metric="riemann"); this automatic estimation is the common choice in applications such as covariance-based classification in BCI. The reference must be compatible with the chosen metric and with the input matrix size n.
        metric (string | callable): Metric used for tangent-space mapping and for reference estimation when Cref is None. Default is "riemann". Accepted string values documented in pyRiemann include "euclid", "logeuclid", and "riemann", each selecting a specific geometry and tangent-space mapping strategy used for SPD/HPD matrices. A callable can be provided to supply a custom kernel implementation; the callable must be compatible with the internal dispatch used by pyRiemann (i.e., it should implement the kernel behavior expected by check_function/kernel_functions and accept the same inputs: X, Y, Cref, reg). Choosing different metrics changes how the tangent-space projection and reference estimation are performed and thus changes the resulting kernel values, which has practical impact on classification or regression performance in BCI and remote-sensing applications.
        reg (float): Regularization parameter, default 1e-10. Small non-negative value added to improve numerical stability during metric computations and kernel matrix formation (for example to avoid issues when estimating means or inverting matrices). This parameter helps produce a positive-definite kernel matrix required by many kernel algorithms; increasing reg increases stability but can slightly bias kernel values.
    
    Returns:
        ndarray: Kernel matrix K of shape (n_matrices_X, n_matrices_Y). When Y is None the returned K is square with shape (n_matrices_X, n_matrices_X). The entries K[i, j] are inner products between the tangent-space representations of X[i] and Y[j] at Cref according to metric. For identical inputs (Y is None) and symmetric metrics the matrix is symmetric; small numerical asymmetries may occur but reg helps enforce positive-definiteness for downstream kernel methods.
    
    Raises:
        ValueError: If input arrays X or Y are not three-dimensional arrays of square matrices with compatible sizes (n must match across X, Y and Cref), or if metric is a string not among the supported values and not a callable recognized by the internal dispatcher.
        TypeError: If X, Y or Cref are not numpy.ndarray objects where required by the metric implementations.
        RuntimeError: If the underlying metric computations (mean estimation, logarithm/exponential maps, or matrix decompositions) fail due to non-positive-definite inputs or numerical issues; increasing reg or correcting input matrices (e.g., ensuring symmetry and positive-definiteness) typically mitigates these failures.
    
    Notes:
        - This function was added in pyRiemann v0.3.
        - Typical usage in the pyRiemann workflow: estimate covariance matrices from multichannel epochs, call pyriemann.utils.kernel.kernel to obtain a kernel matrix, and feed that kernel matrix to a kernel classifier (e.g., SVM) to perform classification in BCI experiments.
        - The exact behavior when Cref is None depends on the chosen metric: for "riemann" the reference is usually the Riemannian mean, for "euclid" it is typically the arithmetic mean, and for "logeuclid" it follows the log-Euclidean framework.
    """
    from pyriemann.utils.kernel import kernel
    return kernel(X, Y, Cref, metric, reg)


################################################################################
# Source: pyriemann.utils.kernel.kernel_riemann
# File: pyriemann/utils/kernel.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def pyriemann_utils_kernel_kernel_riemann(
    X: numpy.ndarray,
    Y: numpy.ndarray = None,
    Cref: numpy.ndarray = None,
    reg: float = 1e-10
):
    """Affine-invariant Riemannian kernel between two sets of SPD matrices.
    
    Computes the affine-invariant Riemannian kernel matrix K between two collections of symmetric positive-definite (SPD) matrices by mapping each SPD matrix to the tangent space at a reference SPD matrix Cref and taking pairwise inner products there. Concretely, each matrix X_i (and Y_j) is congruence-transformed by Cref^{-1/2}, the matrix logarithm is applied, and kernel entries are computed as the trace of the product of the resulting log-mapped matrices:
    K_{i,j} = tr( log(Cref^{-1/2} X_i Cref^{-1/2}) · log(Cref^{-1/2} Y_j Cref^{-1/2}) ).
    This kernel is used in pyRiemann to compare covariance matrices estimated from multichannel biosignals (e.g., EEG/MEG/EMG) for applications such as brain–computer interfaces (BCI) and remote sensing; it provides an inner product in the tangent space at Cref that enables kernel methods (SVM, kernel PCA, etc.) to operate on SPD matrices using Riemannian geometry.
    
    Args:
        X (ndarray): First set of SPD matrices, stored with shape (n_matrices_X, n, n). Each entry X[i] must be a symmetric positive-definite matrix in R^{n x n}. This argument supplies the row objects for the returned kernel matrix. Supplying non-SPD matrices, mismatched inner dimensions, or arrays with incorrect shape will typically raise a linear algebra or shape-related error during processing.
        Y (None | ndarray): Second set of SPD matrices, stored with shape (n_matrices_Y, n, n). If None, Y is set to X and the function returns a Gram matrix (square, symmetric) of pairwise inner products among X. When provided, Y supplies the column objects for the returned kernel matrix and must have the same matrix dimension n as X (i.e., X.shape[1:]==Y.shape[1:]). Elements of Y must be SPD or numerical failures (NaN/Inf, exceptions) may occur.
        Cref (None | ndarray): Reference SPD matrix with shape (n, n) that defines the tangent space at which the kernel inner products are computed. If None, Cref is computed internally as the Riemannian mean of X using pyriemann.utils.mean.mean_riemann; this ensures the tangent-space mapping is centered on the dataset X (which is important in BCI pipelines to avoid bias). Providing an external Cref (for example, a mean computed on training data) is recommended to avoid data leakage in cross-validation. Cref must be SPD and conformable with the matrices in X and Y.
        reg (float): Regularization parameter, default 1e-10. This small positive scalar is passed to the internal kernel computation to mitigate numerical instabilities arising from ill-conditioned matrices, finite-precision errors in matrix inverse square roots and logarithms, and to stabilize the final kernel matrix estimation. Increasing reg can improve numerical robustness at the cost of slightly biasing kernel values; using the default 1e-10 is appropriate in typical applications. reg does not change the types or shapes of the inputs/outputs.
    
    Returns:
        K (ndarray): Affine-invariant Riemannian kernel matrix with shape (n_matrices_X, n_matrices_Y). Entry K[i, j] equals tr(log(Cref^{-1/2} X[i] Cref^{-1/2]) · log(Cref^{-1/2} Y[j] Cref^{-1/2])). When Y is None or Y is X, K is a square Gram matrix and is expected to be symmetric (within numerical tolerance). K is intended for use as a kernel matrix in downstream kernel methods applied to covariance/ SPD matrices.
    
    Notes:
        - The function does not modify X, Y, or Cref in-place. If Cref is None, a new reference matrix is computed and used internally; that computed Cref is not returned.
        - Internally, the routine relies on matrix inverse square root and matrix logarithm operations (invsqrtm, logm) and on _apply_matrix_kernel to assemble the final kernel; therefore ill-conditioned or non-SPD inputs can raise linear algebra errors (e.g., LinAlgError) or produce NaNs/Infs. Validate SPD-ness and array shapes before calling when possible.
        - Typical use cases come from pyRiemann workflows where X and Y are covariance matrices estimated from multichannel time series (EEG/MEG/EMG) for BCI or from local image windows for remote sensing; the kernel provides an inner product in the tangent space at Cref suitable for kernel-based classification or dimensionality reduction.
    """
    from pyriemann.utils.kernel import kernel_riemann
    return kernel_riemann(X, Y, Cref, reg)


################################################################################
# Source: pyriemann.utils.mean.maskedmean_riemann
# File: pyriemann/utils/mean.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def pyriemann_utils_mean_maskedmean_riemann(
    X: numpy.ndarray,
    masks: list,
    tol: float = 1e-08,
    maxiter: int = 100,
    init: numpy.ndarray = None,
    sample_weight: numpy.ndarray = None
):
    """Masked Riemannian mean of SPD/HPD matrices using semi-orthogonal projection masks and a gradient-descent solver tailored for partially observed covariance matrices in applications such as EEG/BCI and remote sensing.
    
    Args:
        X (numpy.ndarray): Array of shape (n_matrices, n, n) containing the set of symmetric positive-definite (SPD) or Hermitian positive-definite (HPD) matrices to be averaged. In the pyRiemann context this typically contains covariance matrices estimated from multichannel time series (e.g., EEG epochs) and each matrix must be a valid SPD/HPD matrix so that matrix logarithm/exponential operations used in the algorithm are defined and numerically stable.
        masks (list): List of length n_matrices, where each element is a numpy.ndarray of shape (n, n_i) representing a semi-orthogonal mask matrix (n_i <= n). Each mask defines a partial observation subspace for the corresponding matrix in X and is used to form the masked version of that matrix by projection (as implemented by the internal _apply_masks helper). Masks model partial observations of covariance structure (for example, selecting or projecting sensor subspaces in BCI or spatial windows in remote sensing).
        tol (float): Tolerance used as a stopping criterion for the gradient-descent iterations (default: 1e-08). The algorithm monitors the Frobenius norm of the Riemannian gradient (crit) and the current step-size nu; iteration stops when crit <= tol or nu <= tol. Tighter tolerances increase accuracy but may require more iterations and raise numerical sensitivity due to repeated matrix log/expm operations.
        maxiter (int): Maximum number of gradient-descent iterations (default: 100). If this number is reached before the tolerance criterion is met, the function issues a runtime warning ("Convergence not reached") and returns the current estimate. Use larger maxiter to attempt convergence for difficult problems at the cost of additional computation.
        init (numpy.ndarray): Optional initial SPD/HPD matrix of shape (n, n) used to initialize the mean (default: None). If None, the identity matrix I_n is used as the starting point. The provided init should be a valid SPD/HPD matrix conforming to the same ambient dimension n as the matrices in X; otherwise shape or definiteness checks performed internally (via check_init) may raise an error.
        sample_weight (numpy.ndarray): Optional 1-D array of length n_matrices providing non-negative weights for each matrix in X (default: None). If None, equal weights are used. Weights are validated internally (via check_weights) and are applied when aggregating the masked Riemannian gradients; they change the influence of individual observations on the final mean.
    
    Returns:
        numpy.ndarray: Array M of shape (n, n) containing the estimated masked Riemannian mean (an SPD/HPD matrix in the original ambient space). This matrix minimizes (via the implemented gradient descent) the sum of affine-invariant Riemannian distances between the masked observations and the masked mean, weighted by sample_weight. The returned matrix is the ambient-space (unmasked) estimate: masks are used only to construct the objective and gradient during optimization and are not applied to the final returned M.
    
    Behavior and side effects:
        The function computes masked versions of the input matrices using the provided masks and then runs a gradient-descent loop to minimize the sum of affine-invariant Riemannian distances between those masked matrices and a masked mean. At each iteration it computes Riemannian gradient contributions using matrix square roots, inverses, matrix logarithms and exponentials (sqrtm, invsqrtm, logm, expm), aggregates them with sample_weight, and updates the ambient mean using a multiplicative update M <- sqrt(M) * exp(invsqrt(M) * (nu * J) * invsqrt(M)) * sqrt(M). The adaptive step-size nu is decreased when the surrogate step measure h does not improve, and convergence is declared when the gradient norm or nu fall below tol or when maxiter iterations are performed. If convergence is not reached within maxiter, a runtime warning is emitted but the last iterate is returned.
    
    Failure modes and numerical considerations:
        The algorithm requires that inputs in X and the init (if provided) are SPD/HPD to ensure matrix logarithms/exponentials and square roots are well-defined and numerically stable; non-SPD inputs may produce exceptions or NaNs. Masks must have compatible shapes with X (each mask of shape (n, n_i) with n_i <= n and list length equal to n_matrices); mismatched shapes will raise an error when applying masks or when assembling the gradient. Very small tolerances, ill-conditioned matrices, or extreme weighting in sample_weight can lead to slow convergence, numerical instability, or premature reduction of the internal step-size nu; in such cases increase tol, regularize inputs, or adjust sample_weight. The function does not modify X or masks in-place; it may issue a warnings.warn("Convergence not reached") if the maximum number of iterations is hit without meeting tol.
    
    Notes:
        This implementation follows the approach described in "Geodesically-convex optimization for averaging partially observed covariance matrices" (Yger et al., ACML 2020) and was added to pyRiemann to support averaging of partially observed covariance matrices arising in BCI (EEG/MEG) and remote-sensing applications.
    """
    from pyriemann.utils.mean import maskedmean_riemann
    return maskedmean_riemann(X, masks, tol, maxiter, init, sample_weight)


################################################################################
# Source: pyriemann.utils.mean.mean_ale
# File: pyriemann/utils/mean.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def pyriemann_utils_mean_mean_ale(
    X: numpy.ndarray,
    tol: float = 1e-06,
    maxiter: int = 50,
    sample_weight: numpy.ndarray = None,
    init: numpy.ndarray = None
):
    """AJD-based log-Euclidean (ALE) mean of SPD/HPD matrices.
    
    Compute the ALE mean of a set of symmetric (resp. Hermitian) positive definite
    matrices using approximate joint diagonalization (AJD) followed by a
    log-Euclidean update. This function is used in pyRiemann for aggregating
    covariance (SPD) or covariance-like (HPD) matrices estimated from multichannel
    biosignals (e.g., EEG, MEG, EMG) or spatial patches in remote sensing, where a
    robust and computationally efficient surrogate of the Riemannian geometric
    mean is required. The algorithm performs an AJD (via ajd_pham) to obtain an
    initial diagonalizing transform (unless an explicit initializer is provided),
    iteratively updates a diagonal correction in the log-domain using matrix
    log/exponential (logm/expm), and reconstructs the mean by conjugation with the
    inverse AJD transform.
    
    Args:
        X (numpy.ndarray): Array of input SPD/HPD matrices with shape
            (n_matrices, n, n). Each X[i] is an n-by-n symmetric (real) or
            Hermitian (complex) positive definite matrix representing a single
            covariance or similar second-order statistic. The function assumes
            matrices are positive definite; providing singular or non-positive
            definite matrices will likely lead to failures in logm/expm or inversion.
        tol (float): Tolerance for the convergence criterion of the iterative
            procedure. The iteration stops when the Riemannian distance between the
            identity and the diagonal corrective matrix falls below tol. Default is
            1e-06. Smaller values request tighter convergence and may increase
            iterations and runtime; larger values speed up computation but may
            produce a less accurate approximation of the geometric mean.
        maxiter (int): Maximum number of iterations for the gradient-descent-like
            update on the diagonal correction. Default is 50. If the algorithm
            reaches maxiter without meeting tol the function emits a UserWarning
            ("Convergence not reached") and returns the current estimate.
        sample_weight (numpy.ndarray): Optional 1-D array of length n_matrices
            containing non-negative weights for each input matrix. If None (default)
            equal weights are used. Weights influence the AJD-weighted average in
            the log-domain and therefore control the contribution of each matrix to
            the returned mean. The argument must be a numpy.ndarray with shape
            (n_matrices,); incompatible shapes will cause a ValueError from internal
            weight checks.
        init (numpy.ndarray): Optional initializer matrix with shape (n, n). If
            provided, it must be an SPD/HPD matrix used to build the initial
            diagonalizing transform for the iterative scheme (checked by
            check_init). If None (default), the joint diagonalizer returned by
            ajd_pham(X) is used as the initialization. Providing an explicit init
            allows warm-starting when an informed prior is available.
    
    Returns:
        numpy.ndarray: An n-by-n SPD/HPD matrix M containing the ALE mean of the
        input matrices X. The returned matrix is the reconstructed mean obtained by
        conjugating the exponential of the (weighted) log-domain diagonal correction
        with the inverse of the joint diagonalizer. This matrix can be used as a
        representative covariance for downstream tasks such as Riemannian
        classification (e.g., MDM), tangent-space projection, or as a template in
        transfer learning across sessions or subjects.
    
    Behavior, side effects, and failure modes:
        The function does not modify the input array X in-place. Internally it
        calls check_weights, ajd_pham, logm, expm, distance_riemann, and
        np.linalg.inv; behavior and limitations of those routines apply (for
        example, logm/expm expect matrices with suitable numerical properties).
        If the input shapes are inconsistent (e.g., X is not 3-D with equal
        square matrices, or sample_weight has incorrect length), an exception such
        as ValueError will be raised by the internal checks. If matrix inversion
        fails (e.g., due to a numerically singular initializer), numpy.linalg.LinAlgError
        (or a similar exception) may be raised. If the iterative procedure does not
        converge within maxiter iterations, a UserWarning ("Convergence not
        reached") is emitted and the current estimate is returned. The algorithm
        assumes inputs are positive definite; providing matrices that are not SPD/HPD
        can produce undefined behavior, numeric errors in matrix logarithms/exponentials,
        or raised exceptions.
    """
    from pyriemann.utils.mean import mean_ale
    return mean_ale(X, tol, maxiter, sample_weight, init)


################################################################################
# Source: pyriemann.utils.mean.mean_alm
# File: pyriemann/utils/mean.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def pyriemann_utils_mean_mean_alm(
    X: numpy.ndarray,
    tol: float = 1e-14,
    maxiter: int = 100,
    sample_weight: numpy.ndarray = None
):
    """Ando-Li-Mathias (ALM) mean of SPD/HPD matrices.
    
    Computes the recursive geometric mean of a set of symmetric (real) positive
    definite (SPD) or Hermitian positive definite (HPD) matrices using the
    Ando-Li-Mathias (ALM) algorithm [Ando et al., 2004]. In the pyRiemann
    ecosystem this function is intended to average covariance matrices estimated
    from multichannel time series (for example EEG/MEG/EMG in brain-computer
    interface applications or covariance patches in remote sensing). The ALM mean
    is computed recursively and reduces to a geodesic interpolation for two
    matrices. The implementation is intentionally exact but computationally
    expensive: it requires many recursive calls and thus is extremely slow for
    large numbers of matrices.
    
    Args:
        X (numpy.ndarray): Array of SPD/HPD matrices with shape (n_matrices, n, n).
            Each X[i] is an n-by-n symmetric (or Hermitian) positive definite
            matrix representing, for example, a covariance estimate for an epoch
            or a spatial window. The function assumes matrices are SPD/HPD; if
            non-positive-definite matrices are supplied, underlying linear
            algebra routines (matrix square roots, inverses, geodesic computations)
            may raise exceptions.
        tol (float): Tolerance for the stopping criterion, default=1e-14.
            The algorithm monitors the relative change of the first matrix in the
            current iterate using the spectral norm (2-norm). Convergence is
            declared when norm(M_iter[0] - M[0], 2) / norm(M[0], 2) < tol. Use a
            smaller tol to request tighter convergence at the cost of more
            iterations and CPU time. The default is chosen to be very small to
            obtain a precise mean but may be unnecessary in noisy applications.
        maxiter (int): Maximum number of outer iterations, default=100.
            The algorithm performs at most maxiter iterations of the recursive
            update loop. If convergence is not reached within maxiter, the
            function issues a runtime warning ("Convergence not reached") and
            returns the current estimate. Increasing maxiter may improve
            convergence for difficult problems but increases runtime and memory
            usage.
        sample_weight (numpy.ndarray): Optional one-dimensional array of length
            n_matrices containing a nonnegative weight for each input matrix.
            If None (the default) equal weights are assumed. The provided weights
            are used to influence the recursive two-matrix geodesic step and the
            effective contribution of each matrix to the final mean. The weights
            are passed to the internal weight-checking logic (check_weights) so
            they must be compatible with that helper (for example they should have
            length n_matrices). If incompatible weights are provided, an error
            from the weight checking routine may be raised.
    
    Returns:
        numpy.ndarray: ALM mean matrix of shape (n, n). This output is an n-by-n
        SPD/HPD matrix representing the geometric center of the input set under
        the ALM definition. Special cases:
        - If n_matrices == 1, the single input matrix X[0] is returned directly.
        - If n_matrices == 2, the function returns the geodesic interpolation
          between X[0] and X[1] using geodesic_riemann with an internal alpha
          computed from the two sample weights (alpha = sample_weight[1] /
          sample_weight[0] / 2).
        For n_matrices > 2 the algorithm recursively calls itself on subsets of
        matrices to build the ALM mean, iterates until the relative spectral-norm
        change on the first iterate is below tol or maxiter is reached, and finally
        returns the arithmetic mean across the final recursive iterates
        (M_iter.mean(axis=0)). If convergence is not reached within maxiter, a
        warning is emitted and the last iterate's mean is returned.
    
    Behavior, side effects, and failure modes:
        - The ALM mean is computed via a recursive formulation and therefore can
          be extremely slow and memory intensive for large n_matrices. For many
          practical applications (large datasets, real-time pipelines) consider
          using faster alternatives provided in pyRiemann (for example other
          covariance mean implementations referenced in the library).
        - The stopping check uses the spectral norm (np.linalg.norm(..., 2)).
        - If inputs are not SPD/HPD, underlying operations such as matrix square
          roots or geodesic computations may raise linear algebra exceptions.
        - If sample_weight is incompatible with the number of matrices, the
          internal weight checking routine may raise an error.
        - If maxiter is reached without satisfying tol, the function issues a
          warnings.warn("Convergence not reached") and returns the current
          estimate; callers should check for this warning if strict convergence is
          required.
        - The function performs deepcopy of intermediate iterates; memory usage
          grows with n_matrices and matrix size.
    
    Practical significance in pyRiemann:
        This function provides a mathematically principled geometric mean used in
        Riemannian pipelines for classification and averaging of covariance-like
        descriptors in EEG/BCI and remote sensing workflows. The ALM mean is
        useful when an exact recursive geometric average is required and when
        computational cost is secondary to mathematical fidelity.
    
    References:
        T. Ando, C.-K. Li, and R. Mathias. "Geometric Means." Linear Algebra and
        its Applications, Volume 385, July 2004, Pages 305-334.
    """
    from pyriemann.utils.mean import mean_alm
    return mean_alm(X, tol, maxiter, sample_weight)


################################################################################
# Source: pyriemann.utils.mean.mean_covariance
# File: pyriemann/utils/mean.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def pyriemann_utils_mean_mean_covariance(
    X: numpy.ndarray,
    *args,
    metric: str = "riemann",
    sample_weight: numpy.ndarray = None,
    **kwargs
):
    """Mean of a set of square matrices according to a chosen metric or divergence and return their mean matrix computed with the metric-specific mean estimator. This function is a high-level dispatcher used throughout pyRiemann for summarizing sets of symmetric (or Hermitian) positive definite matrices such as covariance matrices estimated from multichannel biosignals (EEG, MEG, EMG) in brain-computer interface (BCI) workflows or covariance patches in remote sensing. It selects the appropriate mean implementation for the requested metric, handles legacy metric names via internal deprecation handling, forwards additional positional and keyword parameters to the selected mean routine, and supports weighted means via sample_weight.
    
    Args:
        X (numpy.ndarray): Set of matrices with shape (n_matrices, n, n). Each entry X[i] is a square matrix (n x n) representing, for example, an estimated covariance matrix for one epoch/trial or one spatial window. The function expects a 3-D numpy array where the first dimension indexes matrices. If X does not have shape (n_matrices, n, n) a ValueError will be raised by this function or by the downstream mean implementation. X is not modified in-place by mean_covariance; the array is forwarded to the metric-specific mean routine.
        args (tuple): Additional positional arguments forwarded to the metric-specific mean function. These can include optional method-specific parameters documented for the individual metric implementations. Historically, some metrics accepted an exponent argument labeled "power" or "poweuclid"; legacy argument names are handled by the internal _deprecate call. Do not rely on positional arguments for parameters that are available as keyword arguments in the metric implementations; they are forwarded in the same order they are passed here.
        metric (str | callable): Metric identifier or a callable implementing a mean estimator. Default is "riemann". If a string, it must match one of the implemented metrics: "ale", "alm", "euclid", "harmonic", "identity", "kullback_sym", "logchol", "logdet", "logeuclid", "riemann", "wasserstein". If a callable is provided, it must implement the same calling convention used by pyriemann mean functions (it will be called as mean_function(X, *args, sample_weight=sample_weight, **kwargs) and must return a numpy.ndarray of shape (n, n)). The function performs a deprecation/compatibility check on the metric and positional args via an internal _deprecate call before resolving the final mean function.
        sample_weight (None | numpy.ndarray): Weights for each matrix with shape (n_matrices,). If None (default), uniform weights are used so the mean is an unweighted average according to the metric. If provided, sample_weight must be a 1-D numpy array whose length equals n_matrices; otherwise a ValueError may be raised by this function or by the selected mean implementation. The weights influence the computed mean according to the mathematical definition of the metric-specific weighted mean (for example, weighted Riemannian mean).
        kwargs (dict): Additional keyword arguments forwarded to the metric-specific mean function. These allow configuring method-specific options (for example, convergence tolerance, maximum iterations, or numerical stabilization parameters) documented by the individual mean implementations. Unexpected or unsupported keywords may raise a TypeError in the resolved mean function.
    
    Returns:
        M (numpy.ndarray): Mean matrix with shape (n, n). This is the metric-dependent central estimate of the input matrices (for example, the Riemannian geometric mean if metric="riemann"). The returned matrix type and numerical properties (e.g., positive-definiteness) depend on the metric-specific implementation; for SPD-consistent metrics the result is expected to be SPD. Errors raised can include ValueError for shape/weight mismatches, KeyError or ValueError if metric is an unknown string, or exceptions propagated from the metric-specific implementation (for example, if the iterative solver fails to converge).
    """
    from pyriemann.utils.mean import mean_covariance
    return mean_covariance(X, *args, metric, sample_weight, **kwargs)


################################################################################
# Source: pyriemann.utils.mean.mean_harmonic
# File: pyriemann/utils/mean.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def pyriemann_utils_mean_mean_harmonic(
    X: numpy.ndarray,
    sample_weight: numpy.ndarray = None
):
    """pyriemann.utils.mean.mean_harmonic computes the harmonic mean of a collection of invertible square matrices.
    
    Computes the weighted harmonic mean defined by the matrix identity M = (sum_i w_i X_i^{-1})^{-1}. This function is intended for use in pyRiemann pipelines where sets of symmetric positive definite (SPD) covariance matrices (for example, covariance estimates from multichannel biosignals such as EEG, MEG, or EMG used in brain-computer interface workflows) need to be averaged using the harmonic operator. The implementation inverts each input matrix, computes the (weighted) Euclidean mean of these inverses via mean_euclid, then inverts the result to produce the harmonic mean. The operation returns a new numpy.ndarray and does not modify the input X in-place.
    
    Args:
        X (numpy.ndarray): Set of invertible matrices with shape (n_matrices, n, n). Each X[i] must be a square, invertible matrix (for covariance applications these are typically SPD matrices). The first dimension indexes matrices to be averaged. The function will attempt to compute the inverse of each X[i]; if any matrix is singular or nearly singular, a numpy.linalg.LinAlgError or numerical instability may occur.
        sample_weight (numpy.ndarray): Weights for each matrix, array of shape (n_matrices,). If None (the default), equal weights are used (i.e., all weights are 1/n_matrices). When provided, sample_weight[i] is multiplied with X[i]^{-1} in the summation. The length of sample_weight must match the number of matrices (first dimension of X). Negative or zero-sum weights are not explicitly checked here and may lead to unexpected results or exceptions from downstream linear algebra operations.
    
    Returns:
        M (numpy.ndarray): Harmonic mean matrix with shape (n, n). Concretely, M = (sum_i w_i X_i^{-1})^{-1}, where the sum is over the first axis of X and w_i are the provided weights (or equal weights if sample_weight is None). The returned matrix is a new array and represents the harmonic average used in pyRiemann for aggregating covariance/HPD matrices in applications such as BCI feature preparation.
    
    Raises:
        ValueError: If the shape of sample_weight does not match the number of matrices in X or if X does not have three dimensions with square matrices.
        numpy.linalg.LinAlgError: If one or more matrices in X are singular (non-invertible) or if inversion fails due to numerical issues.
    """
    from pyriemann.utils.mean import mean_harmonic
    return mean_harmonic(X, sample_weight)


################################################################################
# Source: pyriemann.utils.mean.mean_kullback_sym
# File: pyriemann/utils/mean.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def pyriemann_utils_mean_mean_kullback_sym(
    X: numpy.ndarray,
    sample_weight: numpy.ndarray = None
):
    """Mean of SPD/HPD matrices according to the symmetrized Kullback-Leibler divergence.
    
    This function computes the symmetrized Kullback-Leibler mean (also called the Kullback symmetrized mean) of a set of symmetric positive-definite (SPD) or Hermitian positive-definite (HPD) matrices. The symmetrized Kullback-Leibler mean is implemented as the geometric midpoint (Riemannian geodesic at t=0.5) between the Euclidean mean and the harmonic mean: it first calls mean_euclid(X, sample_weight) and mean_harmonic(X, sample_weight) and then computes their Riemannian geodesic midpoint via geodesic_riemann(..., 0.5). In pyRiemann this is used to aggregate covariance or scatter matrices (for example EEG/MEG covariance matrices in brain–computer interface pipelines or windowed covariance matrices in remote sensing) into a single representative SPD/HPD matrix that is meaningful under information-geometric criteria. The returned matrix is suitable as a central estimator on the SPD/HPD manifold and can be used downstream in Riemannian algorithms such as MDM classification or tangent-space projection.
    
    Args:
        X (ndarray, shape (n_matrices, n, n)): Set of input SPD/HPD matrices to average. Each entry X[i] is an n-by-n symmetric (or Hermitian) positive-definite matrix representing, for instance, a covariance matrix estimated from one epoch/trial (in BCI applications) or from a local spatial window (in remote sensing). The first dimension n_matrices is the number of matrices to aggregate. All matrices must be square and of identical dimension; mismatch in shapes will lead to an error in the underlying mean computations. The matrices are expected to be positive-definite; providing matrices that are not SPD/HPD (for example singular matrices or matrices with non-positive eigenvalues) may cause numerical errors or exceptions in the underlying linear algebra operations.
    
        sample_weight (None | ndarray, shape (n_matrices,), default=None): Optional nonnegative weights for each input matrix. If None, equal weights are used so that each matrix contributes uniformly to both the Euclidean and harmonic means before forming the geodesic midpoint. When provided, sample_weight must have the same length as the first dimension of X; mismatched lengths will raise an error in the underlying mean functions. Negative weights or NaNs are not supported and can produce invalid results or exceptions. The weights affect the influence of each input matrix on both intermediate means (euclidean and harmonic) and therefore on the final symmetrized Kullback-Leibler mean.
    
    Returns:
        M (ndarray, shape (n, n)): Symmetrized Kullback-Leibler mean of the input matrices. M is an n-by-n SPD (or HPD) matrix of the same numeric type as the inputs (real or complex), computed as the Riemannian geodesic midpoint between the weighted Euclidean mean and the weighted harmonic mean. This matrix serves as a central representative on the manifold of positive-definite matrices and can be used in downstream Riemannian processing (for example as a reference mean in classifiers or for tangent-space mapping). If inputs are invalid (non-square, mismatched shapes, non-positive-definite, or invalid weights), the function may raise exceptions or return numerically unstable results from the underlying linear algebra routines.
    """
    from pyriemann.utils.mean import mean_kullback_sym
    return mean_kullback_sym(X, sample_weight)


################################################################################
# Source: pyriemann.utils.mean.mean_logchol
# File: pyriemann/utils/mean.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def pyriemann_utils_mean_mean_logchol(X: numpy.ndarray, sample_weight: numpy.ndarray = None):
    """Mean of SPD/HPD matrices according to the log-Cholesky metric.
    
    This function computes the log-Cholesky mean M of a set of symmetric (real) positive
    definite (SPD) or Hermitian (complex) positive definite (HPD) matrices X. The
    log-Cholesky mean is useful in pyRiemann for averaging covariance matrices
    estimated from multichannel biosignals (for example EEG, MEG or EMG) in
    brain–computer interface (BCI) and remote sensing workflows. The returned mean
    matrix M has the same matrix size as each input matrix and preserves the SPD/HPD
    structure by construction (M = L @ L.conj().T where L is a lower-triangular
    matrix whose diagonal is computed in the log-domain and whose off-diagonal
    entries are averaged linearly).
    
    Args:
        X (numpy.ndarray): Set of SPD/HPD matrices with shape (n_matrices, n, n).
            Each X[i] is expected to be an n-by-n complex- or real-valued matrix
            that is symmetric/Hermitian and positive definite so that a Cholesky
            decomposition exists. In the pyRiemann domain, X typically contains
            covariance matrices estimated from n channels across n_matrices epochs
            or samples. The function calls numpy.linalg.cholesky on each matrix,
            so inputs that are not positive definite will cause a numpy.linalg.LinAlgError.
        sample_weight (numpy.ndarray | None): Weights for each matrix with shape
            (n_matrices,), or None to use equal weights. The provided weights are
            passed to an internal check_weights routine which validates the length
            and returns normalized weights (summing to 1). If sample_weight is None,
            equal weighting is used. If the length does not match n_matrices or the
            weights are otherwise invalid, check_weights raises a ValueError. Weights
            affect the linear averaging of the lower-triangular (off-diagonal)
            Cholesky entries and the weighted log-average of the diagonal entries,
            which are exponentiated to preserve positivity of the resulting diagonal.
    
    Returns:
        numpy.ndarray: The log-Cholesky mean matrix M with shape (n, n). The result
        is computed as M = L @ L.conj().T where L is built by (1) averaging the
        strictly lower-triangular entries of the Cholesky factors of X with the
        provided weights, and (2) computing the weighted average of the logarithm
        of the diagonal entries of the Cholesky factors and exponentiating the
        result to obtain positive diagonal entries. The returned matrix is real
        symmetric when inputs are real SPD and complex Hermitian when inputs are
        complex HPD.
    
    Behavior and failure modes:
        The function calls numpy.linalg.cholesky on each matrix X[i]; if any input
        matrix is not positive definite (for example singular or indefinite),
        numpy.linalg.LinAlgError will be raised. The diagonal averaging is performed
        in the log-domain: taking the logarithm of strictly nonpositive diagonal
        entries (which should not occur for valid Cholesky outputs) will raise a
        FloatingPointError or produce invalid values; therefore inputs must be
        strictly positive definite. Extremely ill-conditioned matrices may induce
        numerical instability in the Cholesky, logarithm, or exponential steps.
        No in-place modification of X is performed; the function returns a new
        numpy.ndarray. This implementation was added in pyRiemann version 0.7 and
        follows the formulation in Z. Lin, "Riemannian geometry of symmetric
        positive definite matrices via Cholesky decomposition" (SIAM J Matrix Anal
        Appl, 2019).
    """
    from pyriemann.utils.mean import mean_logchol
    return mean_logchol(X, sample_weight)


################################################################################
# Source: pyriemann.utils.mean.mean_logdet
# File: pyriemann/utils/mean.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def pyriemann_utils_mean_mean_logdet(
    X: numpy.ndarray,
    tol: float = 0.0001,
    maxiter: int = 50,
    init: numpy.ndarray = None,
    sample_weight: numpy.ndarray = None
):
    """pyriemann.utils.mean.mean_logdet computes the log-det (Log-Determinant) mean of a set of symmetric positive definite (SPD) or Hermitian positive definite (HPD) matrices using the iterative log-det metric. This mean is commonly used in pyRiemann when averaging covariance matrices arising from multivariate biosignals (e.g., EEG, MEG, EMG) for brain–computer interface (BCI) pipelines and remote sensing applications where Riemannian geometry of SPD/HPD matrices is required.
    
    Args:
        X (numpy.ndarray): Array of SPD/HPD matrices with shape (n_matrices, n, n). Each X[i] is a square SPD/HPD matrix, for example a covariance matrix estimated from one epoch or one spatial window. The function assumes these matrices are intended to be positive definite; passing singular or non-positive-definite matrices may cause a linear algebra error during matrix inversion.
        tol (float): Tolerance to stop the iterative update, expressed as the stopping threshold on the Frobenius norm of the update difference ||M_new - M||_F. Default=0.0001. Iteration terminates early when the criterion drops to or below this value.
        maxiter (int): Maximum number of iterations allowed for the fixed-point / gradient-descent-like procedure. Default=50. If this number is reached without meeting the tolerance, the function issues a convergence warning and returns the last iterate.
        init (numpy.ndarray): Initial SPD/HPD matrix for the iterative procedure, with shape (n, n). If None (default), the function initializes M with the weighted Euclidean mean of X (computed by mean_euclid), which provides a practical starting point for covariance matrices in BCI or remote sensing contexts.
        sample_weight (numpy.ndarray): Optional 1-D array of shape (n_matrices,) containing nonnegative weights for each matrix in X. If None (default), equal weights are used. The weights are validated/normalized by the internal check_weights utility before they are applied.
    
    Returns:
        numpy.ndarray: The log-det mean M with shape (n, n). Concretely, the algorithm iteratively updates M according to the fixed-point rule M <- (sum_i w_i * (0.5 * M + 0.5 * X_i)^{-1})^{-1} until the Frobenius norm of the change is <= tol or until maxiter is reached. The returned matrix is intended to be SPD/HPD and represents the mean under the log-det metric, which is suitable for downstream Riemannian-based classification or distance computations in pyRiemann.
    
    Behavior, side effects, and failure modes:
        The function performs repeated matrix inversions and uses numpy.linalg.inv; for each iteration it computes inv(0.5 * X_i + 0.5 * M) for all i, forms the weighted sum, then inverts that sum to obtain the new iterate. Because of these inversions, input matrices that are not positive definite or are numerically singular can raise numpy.linalg.LinAlgError. The implementation relies on helper utilities: check_weights to validate and normalize sample_weight, mean_euclid to compute the Euclidean mean when init is None, and check_init to validate a provided init. If the iterative loop completes maxiter iterations without meeting tol, a warning is emitted ("Convergence not reached") and the last computed M is returned. The algorithm has cubic complexity in matrix dimension per inversion and scales linearly with the number of matrices, so large n or many matrices may incur significant computational cost; if inputs are ill-conditioned, external regularization (e.g., shrinkage) should be applied before calling this function.
    """
    from pyriemann.utils.mean import mean_logdet
    return mean_logdet(X, tol, maxiter, init, sample_weight)


################################################################################
# Source: pyriemann.utils.mean.mean_power
# File: pyriemann/utils/mean.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def pyriemann_utils_mean_mean_power(
    X: numpy.ndarray,
    p: float,
    sample_weight: numpy.ndarray = None,
    zeta: float = 1e-09,
    maxiter: int = 100,
    init: numpy.ndarray = None
):
    """pyriemann.utils.mean.mean_power computes the power mean (fixed-point power mean) of a set of symmetric/Hermitian positive definite (SPD/HPD) matrices.
    
    This function implements the order-p matrix power mean described in the pyRiemann README and the referenced literature: it returns the unique SPD/HPD matrix M that satisfies the fixed-point relation M = sum_i w_i (M #_p X_i), where A #_p B denotes the geodesic (matrix geometric interpolation) between A and B. In the pyRiemann context this routine is used to average covariance matrices estimated from multichannel biosignals (EEG/MEG/EMG) or spatial covariance descriptors in remote sensing: the output M is a single SPD/HPD covariance matrix summarizing the input set according to the chosen exponent p. The implementation uses a fixed-point iteration that reduces to standard means for special values of p (Euclidean when p == 1, Riemannian/Karcher when p == 0, harmonic when p == -1).
    
    Args:
        X (ndarray): Set of SPD/HPD matrices with shape (n_matrices, n, n). Each entry X[i] is expected to be a symmetric (real) positive definite or Hermitian (complex) positive definite matrix representing a covariance-like matrix for one epoch / window / spatial patch. The function does not modify X.
        p (float): Exponent defining the order of the power mean. Must be a scalar in the closed interval [-1, +1]. Practical significance: p controls interpolation between means: p == 1 returns the Euclidean mean (elementwise average of matrices), p == 0 returns the Riemannian (Karcher) mean, and p == -1 returns the harmonic mean. Values between these extremes produce intermediate power means used in BCI and remote sensing covariance aggregation.
        sample_weight (None | ndarray): Optional one-dimensional array of length n_matrices containing non-negative weights for each matrix; default is None which is interpreted as uniform equal weighting across matrices. Internally the weights are normalized by check_weights to sum to one; an incorrect length or invalid weight values will raise an error from the weight-checking utility.
        zeta (float): Stopping tolerance for the fixed-point iteration; default 1e-09. Convergence is tested with the criterion crit = ||H - I||_F / sqrt(n) (Frobenius norm normalized by sqrt(n)), where H is the current averaged iterate in the transform space. Iteration stops when crit <= zeta. Smaller values of zeta demand stricter convergence and may increase iteration count.
        maxiter (int): Maximum number of fixed-point iterations to attempt; default 100. If convergence is not reached within maxiter iterations the function issues a warning ("Convergence not reached") and returns the last estimate. The caller should inspect this warning if strict convergence is required.
        init (None | ndarray): Optional initial SPD/HPD matrix of shape (n, n) used to initialize the iterative solver. If None (default) the method initializes using the weighted power Euclidean mean computed as powm(sum_i w_i powm(X_i, p), 1/p). If provided, the value is validated (shape and SPD/HPD property) by the internal check_init routine and a validation error will be raised for incompatible inputs.
    
    Returns:
        M (ndarray, shape (n, n)): The computed power mean matrix (SPD/HPD) of the input set X according to exponent p and provided sample_weight. In the BCI / covariance-processing domain this matrix is the aggregate covariance representative returned by the algorithm: for p == 1 it is the Euclidean mean, for p == 0 it is the Riemannian (Karcher) mean (delegated to mean_riemann), and for p == -1 it is the harmonic mean (delegated to mean_harmonic). The returned matrix can be used directly in downstream classification or signal-processing pipelines (for example as input to MDM or tangent-space embeddings).
    
    Raises / Failure modes:
        ValueError: if p is not a scalar numeric type or if p is outside the allowed interval [-1, +1]; if sample_weight has incorrect length it will be rejected by the internal weight checker; if init is provided with wrong shape or is not SPD/HPD it will be rejected by the internal initializer check.
        Warning: if the iterative solver does not meet the convergence tolerance within maxiter iterations a runtime warning is emitted and the last iterate is returned.
    """
    from pyriemann.utils.mean import mean_power
    return mean_power(X, p, sample_weight, zeta, maxiter, init)


################################################################################
# Source: pyriemann.utils.mean.mean_riemann
# File: pyriemann/utils/mean.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def pyriemann_utils_mean_mean_riemann(
    X: numpy.ndarray,
    tol: float = 1e-08,
    maxiter: int = 50,
    init: numpy.ndarray = None,
    sample_weight: numpy.ndarray = None
):
    """Mean of SPD/HPD matrices according to the affine-invariant Riemannian metric.
    
    This function computes the affine-invariant Riemannian mean (also called Karcher mean)
    of a set of symmetric (or Hermitian) positive-definite (SPD/HPD) matrices. In the
    context of pyRiemann and the README use-cases, X typically contains covariance
    matrices estimated from multichannel biosignals (EEG/MEG/EMG) or covariance-like
    matrices from remote sensing; the returned matrix M is the Riemannian average
    used by downstream algorithms (for example MDM classification, tangent-space
    projection, or transfer learning). The implementation performs an iterative
    gradient-descent-like update on the manifold using matrix square-roots,
    inverse-square-roots, matrix logarithm and exponential to minimize the sum of
    squared affine-invariant Riemannian distances to the input matrices. The
    stopping criterion implemented is the one described in Congedo et al. (PLOS
    ONE, 2015).
    
    Args:
        X (numpy.ndarray): Array of SPD/HPD matrices with shape (n_matrices, n, n).
            Each entry X[i] is expected to be an n-by-n symmetric (or Hermitian)
            positive-definite matrix such as a covariance matrix estimated from an
            epoch of multichannel time series. The function assumes these matrices
            are valid SPD/HPD; if they are not (non-square, non-symmetric Hermitian,
            or not positive-definite), the internal matrix square-root, inverse or
            logarithm operations may fail or produce invalid values (NaN/Inf),
            causing exceptions or non-convergence.
        tol (float): Tolerance used as the stopping criterion for the iterative
            algorithm. The iterations stop when the Frobenius norm of the Riemannian
            gradient (see J in the source) is less than or equal to tol, or when the
            adaptive step-size (nu) becomes less than or equal to tol. Smaller values
            make the algorithm stricter about convergence and may increase runtime.
            Default: 1e-08.
        maxiter (int): Maximum number of iterations allowed for the fixed-point /
            gradient-descent procedure. If convergence is not reached within
            maxiter iterations, the function exits the loop and emits a warning
            ("Convergence not reached"). Increasing maxiter can allow harder-to-
            converge problems to reach the tolerance but increases computation time.
            Default: 50.
        init (numpy.ndarray): Optional initialization matrix of shape (n, n) used
            to start the iterative procedure. This must be an SPD/HPD matrix. If
            init is None (the default), the routine initializes M with the weighted
            Euclidean mean of X (via mean_euclid) which is a common practical
            choice to accelerate convergence in covariance-based pipelines (e.g.,
            BCI preprocessing). If provided, init is validated for shape and
            properties; invalid initializations can lead to errors or non-convergence.
            Default: None.
        sample_weight (numpy.ndarray): Optional one-dimensional array of length
            n_matrices containing non-negative weights for each matrix in X. When
            provided, these weights influence the minimization (the algorithm
            minimizes the weighted sum of squared Riemannian distances). If
            sample_weight is None, equal weights are used. The input weights are
            validated and (when appropriate) normalized internally so they sum to 1;
            invalid shapes or negative weights will be rejected by the internal
            weight-checking routine.
    
    Returns:
        numpy.ndarray: M, shape (n, n). The affine-invariant Riemannian mean of the
        input matrices X. This matrix is SPD/HPD and represents the minimizer of
        the weighted sum of squared affine-invariant Riemannian distances to the
        matrices in X. In practical pipelines (e.g., EEG covariance classification
        or remote sensing covariance processing), M is used as a reference or
        template for Riemannian-based classifiers, tangent-space mapping, or
        transfer-learning procedures.
    
    Behavior and side effects:
        The function performs an iterative update M <- M^{1/2} exp(nu * J) M^{1/2},
        where J is the weighted average of log-mapped matrices in the tangent space
        at M. The step-size nu is adapted during iterations to help convergence.
        The routine uses matrix square-root, inverse-square-root, logarithm and
        exponential; these operations are numerically sensitive and assume input
        SPD/HPD matrices. If convergence is not reached within maxiter iterations,
        the function issues a Python warning "Convergence not reached". If the
        inputs are invalid (wrong shapes, non-positive-definite matrices, NaNs or
        Infs), numerical linear-algebra routines may raise exceptions or cause the
        algorithm to fail to converge.
    
    Failure modes and numerical considerations:
        - Non-SPD/HPD inputs or near-singular matrices may cause the matrix square
          root, inverse, log or exp routines to fail or produce unstable results.
        - Mismatched shapes (not 3D array of shape (n_matrices, n, n)) will cause
          shape-related errors.
        - Very tight tol values or ill-conditioned inputs can dramatically increase
          runtime or prevent convergence within maxiter.
        - The algorithm has cubic complexity in n for dense matrices due to the
          spectral or decomposition operations used internally; large n can be
          computationally expensive.
    
    Practical significance in pyRiemann workflows:
        This routine is a core primitive for Riemannian geometry workflows in
        pyRiemann: it computes the central SPD/HPD template (Riemannian mean) used
        by classifiers such as MDM, by tangent-space embeddings, and in transfer
        learning between subjects/sessions. For covariance matrices derived from
        biosignals (EEG/MEG/EMG) or spatial covariance estimates from hyperspectral
        / SAR images, computing a robust Riemannian mean improves downstream
        classification and domain-adaptation performance.
    """
    from pyriemann.utils.mean import mean_riemann
    return mean_riemann(X, tol, maxiter, init, sample_weight)


################################################################################
# Source: pyriemann.utils.mean.mean_wasserstein
# File: pyriemann/utils/mean.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def pyriemann_utils_mean_mean_wasserstein(
    X: numpy.ndarray,
    tol: float = 1e-08,
    maxiter: int = 50,
    init: numpy.ndarray = None,
    sample_weight: numpy.ndarray = None
):
    """Mean of SPD/HPD matrices according to the Wasserstein metric.
    
    Computes the Bures-Wasserstein barycenter (Wasserstein mean) of a set of symmetric positive definite
    (SPD) or Hermitian positive definite (HPD) matrices using the inductive mean algorithm.
    This function is intended for use on covariance or structured positive-definite matrices that
    arise in multivariate signal processing and machine learning workflows described in the pyRiemann
    README, for example covariance matrices estimated from multichannel biosignals (EEG, MEG, EMG)
    in brain–computer interface (BCI) applications or covariance descriptors in remote sensing and
    hyperspectral imaging. The implementation follows the inductive mean approach and uses the same
    convergence criterion convention as the Riemannian mean implementation in this package. The
    algorithm iterates using Wasserstein log and exponential maps (log_map_wasserstein and
    exp_map_wasserstein) and supports optional sample weighting.
    
    Args:
        X (numpy.ndarray): Array of input SPD/HPD matrices with shape (n_matrices, n, n).
            Each slice X[i] is an n-by-n SPD/HPD matrix (for example, a covariance matrix estimated
            from one epoch or one spatial window). These matrices are the elements whose Wasserstein
            barycenter is estimated. The function does not modify X in-place; it reads X to compute
            tangent updates via log_map_wasserstein.
        tol (float): Tolerance threshold for stopping the iterative procedure. The iterative loop
            computes a gradient-like update J and stops when the Euclidean norm of J is less than or
            equal to tol. Smaller tol typically yields a more accurate barycenter but may require
            more iterations. The default value is 1e-08.
        maxiter (int): Maximum number of iterations allowed for the inductive mean algorithm. If the
            algorithm does not satisfy the tolerance criterion within maxiter iterations, the function
            exits the loop and emits a RuntimeWarning ("Convergence not reached") while returning the
            last iterate. Default is 50.
        init (None | numpy.ndarray): Initial SPD/HPD matrix used to start the iterative procedure.
            If provided, init must have shape (n, n) and should be SPD/HPD to be meaningful in the
            Wasserstein geometry. If init is None (the default), the Euclidean mean of X (computed by
            mean_euclid with the provided sample_weight) is used as the starting point. The initial
            value determines the starting tangent around which log and exp maps are computed.
        sample_weight (None | numpy.ndarray): Optional one-dimensional array of length n_matrices
            containing nonnegative weights for each input matrix. When None (the default), equal
            weights are used. Weights are normalized internally by helper utilities (check_weights)
            before being applied to the inductive update; supplying sample_weight allows emphasizing
            particular samples when estimating the barycenter.
    
    Returns:
        M (numpy.ndarray): Array of shape (n, n) containing the estimated Wasserstein mean (the
            Bures-Wasserstein barycenter) of the input SPD/HPD matrices. The returned matrix is the
            final iterate of the inductive algorithm; if convergence (norm of the update <= tol) was
            not reached within maxiter iterations, the last iterate is returned and a RuntimeWarning is
            emitted. The returned matrix is intended to serve as a representative covariance/HPD
            descriptor in downstream tasks (for example, as a mean covariance in classification or
            transfer-learning pipelines).
    
    Notes:
        - The algorithm internally calls check_weights and check_init to validate and prepare weights
          and the initialization, and uses log_map_wasserstein and exp_map_wasserstein to perform
          tangent-space updates. Those helper functions perform shape and validity checks and may
          raise exceptions (for example, on incompatible shapes).
        - No in-place modification of the input array X is performed; the function produces and
          returns a separate matrix M.
        - If the input matrices are not SPD/HPD or have incompatible shapes, helper validation
          functions will raise appropriate exceptions. If the iterative procedure fails to converge
          within maxiter, a warning is issued but the last computed estimate is still returned.
    """
    from pyriemann.utils.mean import mean_wasserstein
    return mean_wasserstein(X, tol, maxiter, init, sample_weight)


################################################################################
# Source: pyriemann.utils.mean.nanmean_riemann
# File: pyriemann/utils/mean.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def pyriemann_utils_mean_nanmean_riemann(
    X: numpy.ndarray,
    tol: float = 1e-08,
    maxiter: int = 100,
    init: numpy.ndarray = None,
    sample_weight: numpy.ndarray = None
):
    """Compute the Riemannian NaN-mean of SPD/HPD matrices, i.e., the masked Riemannian mean
    applied to a set of symmetric (or Hermitian) positive definite matrices that may contain
    symmetric NaN entries. This function is used in pyRiemann to average covariance (SPD)
    or Hermitian positive definite (HPD) matrices estimated from multichannel biosignals
    (e.g., EEG/MEG/EMG) or remote-sensing windows when some entries are missing or masked
    by NaN values, producing a single representative SPD/HPD matrix on the manifold while
    respecting the geometry of SPD/HPD matrices.
    
    Args:
        X (numpy.ndarray): Input array of shape (n_matrices, n, n). Each slice X[i] is a
            symmetric (for real-valued) or Hermitian (for complex-valued) n-by-n matrix
            representing an SPD/HPD matrix estimate (for example, a covariance matrix for
            one epoch or spatial window). Some entries in these matrices may be NaN to
            indicate missing or unobserved values; NaN positions are expected to be
            symmetric across the diagonal (i.e., if X[i, p, q] is NaN then X[i, q, p]
            should also be NaN). This function will raise a ValueError if X does not have
            three dimensions (n_matrices, n, n) or if the second and third dimensions are
            not equal. Behavior is undefined for inputs that are not intended to represent
            SPD/HPD matrices.
        tol (float): Tolerance used to stop the iterative Riemannian gradient descent.
            Iteration stops when the change in the current estimate falls below tol.
            Default is 1e-08. Choosing a smaller tol yields a more accurate mean at the
            cost of more iterations and compute time; choosing a larger tol may stop early
            and return a less accurate estimate.
        maxiter (int): Maximum number of iterations allowed for the Riemannian iterative
            solver (gradient descent on the manifold). Default is 100. If the solver does
            not converge within maxiter iterations (given tol), the last iterate is
            returned; thus lack of convergence is not raised as an exception by this
            function but may produce a less accurate mean.
        init (numpy.ndarray): Optional initial SPD/HPD matrix of shape (n, n) used to
            initialize the Riemannian gradient descent. If provided, it must be a valid
            SPD/HPD matrix (and will be validated via the internal check_init utility).
            Supplying a good init (close to the true mean) can reduce iterations. If None
            (default), the function initializes the solver with a regularized Euclidean
            NaN-mean computed as np.nanmean(X, axis=0) + 1e-6 * I_n where I_n is the n-by-n
            identity; this regularization helps ensure a valid positive definite starting
            point even when NaNs or rank-deficiencies are present.
        sample_weight (numpy.ndarray): Optional 1D array of shape (n_matrices,) containing
            a non-negative weight for each matrix in X. If None (default), equal weights
            are used (uniform averaging). Weights influence the contribution of each
            matrix to the Riemannian mean; providing weights is useful in applications
            where some covariance estimates are deemed more reliable (for example, longer
            epochs or higher SNR). If provided, its length must equal n_matrices.
    
    Returns:
        numpy.ndarray: M, array of shape (n, n). The computed Riemannian NaN-mean matrix
        (SPD/HPD). This matrix is the output of the masked mean algorithm: the function
        internally builds binary masks from NaN locations across X, replaces NaNs with
        zeros for numerical linear-algebra operations (using np.nan_to_num) to avoid
        contamination during matrix multiplications, and then delegates to the masked
        Riemannian mean solver (maskedmean_riemann) with the provided tol, maxiter, init,
        and sample_weight. No in-place modification of X is performed; the function
        returns a new array containing the mean.
    
    Notes:
        - This routine implements the algorithm described in "Geodesically-convex
          optimization for averaging partially observed covariance matrices" (Yger et al.,
          ACML 2020) and was added to pyRiemann for robust averaging when entries are
          partially observed (symmetric NaNs).
        - The function assumes symmetry/Hermitian structure and symmetric NaN masks. If
          NaNs are not symmetrically placed, the behavior may be incorrect because the
          underlying masked-mean routines expect symmetric masks and matrices.
        - If X contains NaNs such that no valid observed entry contributes to some
          positions across all matrices, the masked mean solver may be unable to estimate
          those components reliably; the returned M will reflect the algorithm's handling
          of these missing entries (which may include reliance on the initialization and
          regularization).
        - Input validation (e.g., shape and SPD/HPD checks for init) is delegated to
          internal utilities (such as check_init); those utilities may raise exceptions
          (ValueError, TypeError) if inputs do not conform to expected shapes or properties.
    """
    from pyriemann.utils.mean import nanmean_riemann
    return nanmean_riemann(X, tol, maxiter, init, sample_weight)


################################################################################
# Source: pyriemann.utils.median.median_riemann
# File: pyriemann/utils/median.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def pyriemann_utils_median_median_riemann(
    X: numpy.ndarray,
    tol: float = 1e-05,
    maxiter: int = 50,
    init: numpy.ndarray = None,
    weights: numpy.ndarray = None,
    step_size: float = 1
):
    """Compute the affine-invariant Riemannian geometric median of a set of
    SPD/HPD matrices.
    
    This function implements the iterative estimator described in the
    literature for the Riemannian geometric median on the manifold of
    symmetric (or Hermitian) positive definite matrices. In the pyRiemann
    package this is used as a robust central tendency estimator for
    covariance or scatter matrices that commonly arise from multichannel
    biosignals (EEG/MEG/EMG) or from local covariance descriptors in
    remote sensing and hyperspectral imaging. The median returned is
    affine-invariant (it respects congruent transformations) and is more
    robust to outliers than the Riemannian mean.
    
    Args:
        X (ndarray): Array of SPD/HPD matrices with shape (n_matrices, n, n).
            This is the input sample for which the geometric median is
            computed. In BCI and biosignal processing workflows, X typically
            contains covariance matrices estimated from individual epochs or
            time windows (n_matrices = number of epochs). All matrices in X
            are expected to be symmetric (or Hermitian) and positive definite;
            behavior is undefined for non-SPD/HPD inputs because the
            underlying matrix logarithm/exponential and square-root
            operations assume positive definiteness.
        tol (float): Stopping tolerance for the iterative gradient-descent
            procedure. The algorithm computes the Frobenius norm of the
            update tangent vector J at each iteration and stops when
            norm(J, 'fro') <= tol. Default is 10e-6 (1e-5). A smaller tol
            yields a more accurate median at the cost of more iterations.
        maxiter (int): Maximum number of iterations to attempt before
            terminating. If convergence (as defined by tol) is not reached
            within maxiter iterations, the function issues a runtime
            warning and returns the last iterate. Default is 50. Increase
            this if the dataset is large or the median converges slowly.
        init (None | ndarray): Optional initial SPD/HPD matrix with shape
            (n, n) used to initialize the descent. If None, the function
            uses the (weighted) Euclidean mean of X (via mean_euclid) as the
            initial point. Providing a good initialization (for example a
            prior estimate or the Riemannian mean) can reduce iterations
            and numerical error.
        weights (None | ndarray): Optional 1-D array of length n_matrices
            containing non-negative weights for each matrix in X. If None,
            equal weights are used for all matrices. Internally the weights
            are processed by pyriemann.utils.check_weights to produce a
            valid weight vector; during each iteration the algorithm scales
            contributions by 1/distance and renormalizes the active weights
            (see Notes). Use weights to emphasize certain covariance samples
            (e.g., more reliable trials or sensors).
        step_size (float): Multiplicative step size applied to the tangent
            update inside the Riemannian exponential map. The update step
            M <- sqrt(M) @ expm(step_size * J) @ sqrt(M) controls how far
            the iterate moves along the manifold in response to J. Valid
            values are in the interval (0, 2]. The code raises ValueError if
            step_size is not in (0, 2]. Default is 1.
    
    Returns:
        M (ndarray): The estimated affine-invariant Riemannian geometric
            median, returned as an SPD/HPD matrix with shape (n, n). This
            matrix represents a robust central covariance (or scatter)
            estimate for the input set X and can be used downstream in
            Riemannian classifiers (e.g., MDM) or as a reference point for
            tangent-space projections.
    
    Notes:
        - Algorithm: at each iteration the implementation computes affine-
          invariant Riemannian distances from the current iterate M to each
          X_i, forms tangent vectors via the matrix logarithm in the
          tangent space at M (after applying M^{-1/2} congruence), weights
          them by w_i / d_R(M, X_i) (ignoring zero distances), averages
          to obtain J, and updates M using the matrix exponential of
          step_size * J mapped back to the manifold by conjugation with
          sqrt(M). The stopping criterion is the Frobenius norm of J.
        - Weight handling: matrices whose distance to the current iterate
          is exactly zero are omitted from the weighted average to avoid
          division by zero; the remaining weights are renormalized by the
          algorithm before forming J.
        - Input validation: the function validates step_size and will raise
          a ValueError when step_size <= 0 or step_size > 2. The function
          does not internally verify positive definiteness of X or init;
          passing non-SPD/HPD matrices will typically lead to exceptions or
          invalid numerical results from sqrtm/logm/expm operations.
        - Convergence: if the maximum number of iterations maxiter is
          reached without meeting the tolerance tol, the function emits a
          warning ("Convergence not reached") and returns the last estimate.
        - Practical significance: this median is useful in BCI pipelines
          as a robust template for classifiers or as a central estimator for
          transfer-learning/alignment tasks, providing resilience to
          outlier covariance matrices compared to the Riemannian mean.
    """
    from pyriemann.utils.median import median_riemann
    return median_riemann(X, tol, maxiter, init, weights, step_size)


################################################################################
# Source: pyriemann.utils.tangentspace.exp_map
# File: pyriemann/utils/tangentspace.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def pyriemann_utils_tangentspace_exp_map(
    X: numpy.ndarray,
    Cref: numpy.ndarray,
    metric: str = "riemann"
):
    """Project matrices in a tangent space back onto the symmetric positive-definite (SPD) manifold
    using the exponential map at a given reference matrix. This function is part of pyRiemann's
    tangent-space utilities and is used to convert tangent-space representations (for example,
    tangent vectors obtained from covariance matrices of biosignals such as EEG, MEG or EMG)
    back to manifold-valued matrices prior to downstream processing (classification, visualization,
    or saving). The selected metric determines the definition of the exponential map and thus how
    tangent displacements are re-applied to the reference matrix Cref.
    
    Args:
        X (numpy.ndarray): Matrices in the tangent space with shape (..., n, n). The trailing
            two dimensions must form square matrices compatible with Cref (same n). The leading
            dimensions (represented by ...) are treated as batch dimensions and are preserved
            in the output. In practical BCI or remote-sensing workflows, X represents tangent
            deviations from a reference covariance matrix (Cref) computed from multichannel time
            series or image patches.
        Cref (numpy.ndarray): Reference matrix with shape (n, n). Cref is the base point on the
            SPD manifold at which the exponential map is centered; in typical use this is a
            covariance matrix estimated from multichannel data (e.g., an average covariance for
            a subject/session). For metrics that require a positive-definite reference (for
            example "riemann" or "wasserstein"), Cref must be symmetric (or Hermitian for
            complex-valued data) and positive-definite; otherwise the underlying exponential
            implementation may raise a linear algebra error.
        metric (str | callable): Metric used to compute the exponential map. Default is "riemann".
            Accepts the string values "euclid", "logchol", "logeuclid", "riemann", "wasserstein",
            which select the corresponding built-in exponential-map implementation, or a
            callable implementing the same behavior. When metric is a callable, it must accept
            the same (X, Cref) inputs and return an array with shape (..., n, n) mapping tangent
            matrices to manifold matrices. The function dispatches to an internal implementation
            (via check_function) and will raise an error if metric is neither one of the accepted
            strings nor a valid callable.
    
    Returns:
        numpy.ndarray: Matrices on the manifold with shape (..., n, n). The output preserves the
        leading batch dimensions of X and contains manifold-valued matrices obtained by applying
        the exponential map at Cref to each tangent matrix in X. In the typical domain use-cases
        (BCI, hyperspectral or SAR image processing) the returned matrices are intended to be
        valid covariance (SPD) matrices ready for classifiers or further Riemannian operations.
    
    Behavior, defaults and failure modes:
        - Default metric is "riemann", which computes the Riemannian exponential map at Cref.
        - The function performs a dispatch using the provided metric and calls the selected
          implementation; no in-place modification of X or Cref is performed (pure functional).
        - Input shape requirements: X must have trailing dimensions (n, n) matching Cref. If
          shapes are incompatible a ValueError is raised by the underlying implementation.
        - For metrics that require Cref to be positive-definite, providing a non-SPD Cref can
          raise numpy.linalg.LinAlgError or a similar numerical error from the chosen routine.
        - If metric is an invalid string, or a callable that does not conform to the required
          signature or behavior, a ValueError (or the callable's own exception) will be raised by
          the dispatch step.
        - Passing NaN or Inf values in X or Cref may produce non-finite outputs or trigger
          numerical errors in matrix factorizations; inputs should be finite, well-conditioned
          covariance-like matrices when used in BCI/remote-sensing contexts.
        - No side effects occur: inputs are not modified in-place and the function returns a new
          array with the manifold matrices.
    """
    from pyriemann.utils.tangentspace import exp_map
    return exp_map(X, Cref, metric)


################################################################################
# Source: pyriemann.utils.tangentspace.exp_map_logchol
# File: pyriemann/utils/tangentspace.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def pyriemann_utils_tangentspace_exp_map_logchol(X: numpy.ndarray, Cref: numpy.ndarray):
    """Project matrices back to SPD/HPD manifold using the log-Cholesky exponential map.
    
    This function implements the log-Cholesky exponential map described in Table 2 of Lin (2019) to move a batch of matrices X from the tangent space at a reference SPD/HPD matrix Cref back onto the SPD/HPD manifold. In the context of pyRiemann, this operation is used when working with covariance matrices (real symmetric positive definite, SPD) or Hermitian positive definite (HPD) matrices arising from multivariate biosignal (EEG/MEG/EMG) processing and BCI pipelines: for example, after projecting covariance matrices to a tangent space for classification or transfer learning, exp_map_logchol reconstructs manifold-valued covariance matrices from their tangent representations. The implementation performs a Cholesky-based reconstruction: it computes the Cholesky factor of Cref, forms a normalized tangent increment, enforces the upper-triangular/log-Cholesky structure, exponentiates diagonal log-parameters, builds a modified Cholesky factor, and returns the reconstructed SPD/HPD matrices as L @ L^H.
    
    Args:
        X (ndarray, shape (..., n, n)): Matrices expressed in the tangent space at Cref. X may contain a batch of matrices in its leading dimensions (denoted by ...). Each trailing (n, n) block represents the tangent-space coordinates that will be mapped back to the manifold using the log-Cholesky exponential map. In practical pyRiemann usage, X typically contains tangent vectors obtained from a TangentSpace transform applied to covariance matrices for machine-learning pipelines (e.g., feature extraction before a classifier).
        Cref (ndarray, shape (n, n)): Reference SPD (real) or HPD (complex) matrix that defines the base point of the tangent space. Cref is the matrix at which the exponential map is centered (for example, a Riemannian mean covariance estimated from training data). Cref must be positive definite so that a Cholesky decomposition exists; its Cholesky factor is used to normalize tangent increments and to reconstruct manifold-valued matrices.
    
    Returns:
        ndarray, shape (..., n, n): Reconstructed matrices on the SPD/HPD manifold. For each input tangent matrix X[i,...], the function returns a positive definite matrix A = L @ L^H where L is the modified upper-triangular Cholesky-like factor computed by adding the normalized tangent increment to the Cholesky factor of Cref and exponentiating diagonal log-parameters. The returned array preserves the leading batch dimensions of X.
    
    Behavior, side effects, and failure modes:
        The function performs the following numerical steps internally: (1) compute Cref_chol = cholesky(Cref) and its inverse; (2) compute diff_bracket = Cref_invchol @ X @ Cref_invchol.conj().T to express the tangent increment in the Cholesky-normalized coordinates; (3) zero out strictly lower-triangular entries of diff_bracket and halve its diagonal entries (as required by the log-Cholesky parametrization); (4) compute a diff term in the original scale via multiplication by Cref_chol; (5) form an upper-triangular factor exp_map by adding diff to the upper-triangular part of Cref_chol and exponentiating the diagonal contributions using exp(diff_bracket_diag) * Cref_chol_diag; (6) return exp_map @ exp_map.conj().swapaxes(-1, -2), which yields Hermitian positive definite (or symmetric positive definite) matrices. There are no in-place modifications of the inputs; the function returns a newly allocated array.
    
        Failure modes include:
        - numpy.linalg.LinAlgError if Cref is not positive definite and its Cholesky decomposition fails.
        - ValueError or broadcasting errors if the shapes of X and Cref are incompatible (the trailing two dimensions of X must be (n, n) matching Cref).
        - Numerical overflow or invalid values if diagonal entries of diff_bracket are very large, since exponential is applied to diagonal terms.
    
        This function was added in pyRiemann version 0.7 and is intended for reconstructing manifold covariance/HPD matrices after tangent-space processing in BCI, hyperspectral, or remote sensing workflows where Riemannian geometry of SPD/HPD matrices is used. References: Lin, "Riemannian geometry of symmetric positive definite matrices via Cholesky decomposition" (SIAM J Matrix Anal Appl, 2019).
    """
    from pyriemann.utils.tangentspace import exp_map_logchol
    return exp_map_logchol(X, Cref)


################################################################################
# Source: pyriemann.utils.tangentspace.exp_map_riemann
# File: pyriemann/utils/tangentspace.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def pyriemann_utils_tangentspace_exp_map_riemann(
    X: numpy.ndarray,
    Cref: numpy.ndarray,
    Cm12: bool = False
):
    """pyriemann.utils.tangentspace.exp_map_riemann projects matrices from a tangent space back onto the manifold of symmetric (SPD) or Hermitian (HPD) positive-definite matrices using the Riemannian exponential map; it is used in pyRiemann for converting tangent-space representations (e.g., produced by TangentSpace transformer) back to covariance-like SPD/HPD matrices for downstream algorithms (MDM, visualization, or re-injection into pipelines).
    
    Args:
        X (numpy.ndarray): Matrices in tangent space with shape (..., n, n). Each matrix is expected to be a real symmetric matrix (for SPD manifold) or a complex Hermitian matrix (for HPD manifold) representing a tangent vector at the reference point Cref. In typical pyRiemann workflows X is the output of a log map / tangent-space embedding and its entries represent elements of the matrix logarithm domain. If X does not satisfy the expected symmetry/Hermiticity, numerical results are undefined and the result may not lie on the SPD/HPD manifold.
        Cref (numpy.ndarray): Reference SPD/HPD matrix with shape (n, n). This matrix is the reference point on the manifold at which the tangent space is defined (for example, a geometric mean or an epoch-specific covariance). Cref must be symmetric positive-definite (or Hermitian positive-definite for complex data); if Cref is not positive-definite the internal matrix square root and inverse-square-root computations will fail or produce invalid results.
        Cm12 (bool, optional): If False (default), the function applies the exponential map using the formulation X_original = Cref^{1/2} exp(X) Cref^{1/2}. If True, the function applies the full Riemannian exponential map with congruence by the inverse square root of Cref: X_original = Cref^{1/2} exp(Cref^{-1/2} X Cref^{-1/2}) Cref^{1/2}. Use Cm12=True when X is expressed in the canonical Euclidean tangent coordinates that require congruence by Cref^{-1/2} before exponentiation (see Pennec et al., Section 3.4); use Cm12=False when X has already been transported/normalized relative to Cref so that direct exponentiation is appropriate.
    
    Returns:
        numpy.ndarray: Matrices mapped back to the SPD/HPD manifold, with shape (..., n, n). Under the assumption that Cref is positive-definite and X is symmetric/Hermitian, the returned arrays are symmetric (or Hermitian) positive-definite matrices. Numerical round-off may introduce tiny asymmetries or small imaginary components in practice; these are not corrected by this function. Failure modes: providing non-square inputs, mismatched dimensions between X and Cref, or a non-positive-definite Cref will raise errors during matrix square-root, inverse-square-root, or matrix exponential computations. The function has no side effects on its inputs.
    """
    from pyriemann.utils.tangentspace import exp_map_riemann
    return exp_map_riemann(X, Cref, Cm12)


################################################################################
# Source: pyriemann.utils.tangentspace.exp_map_wasserstein
# File: pyriemann/utils/tangentspace.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def pyriemann_utils_tangentspace_exp_map_wasserstein(X: numpy.ndarray, Cref: numpy.ndarray):
    """Project matrices back to SPD/HPD manifold using the Wasserstein exponential map.
    
    This function implements the Wasserstein Riemannian exponential map that takes a perturbation in the tangent space at a reference symmetric (or Hermitian) positive definite (SPD/HPD) matrix and returns the corresponding matrix on the SPD/HPD manifold. In pyRiemann pipelines for biosignal processing and brain–computer interfaces (BCI), this is used to map tangent-space representations (for example, results of a TangentSpace transform or a gradient update) back to covariance matrices (SPD) or Hermitian positive definite matrices (HPD) so they can be interpreted or further processed as covariance estimates for multichannel time series. The implementation follows Eq.(36) in the referenced Wasserstein geometry paper and uses the eigen-decomposition of the reference matrix Cref to compute the map.
    
    Args:
        X (numpy.ndarray): Matrices in tangent space with shape (..., n, n). Each leading index corresponds to a tangent-space perturbation associated with the same reference matrix Cref. In practice X contains symmetric (or Hermitian) perturbation matrices computed from covariance estimators or operations performed in the tangent space. The last two dimensions must form square matrices of size n and must be consistent with Cref.
        Cref (numpy.ndarray): Reference SPD/HPD matrix with shape (n, n). This matrix defines the base point on the manifold where the exponential map is applied. Cref must be symmetric (real) positive definite or Hermitian positive definite (complex) with strictly positive eigenvalues; it is typically a covariance matrix estimated from multichannel biosignal data (EEG/MEG/EMG) or spatial covariance in remote sensing applications.
    
    Returns:
        X_original (numpy.ndarray): Matrices in SPD/HPD manifold with shape (..., n, n). The returned arrays are the result of applying the Wasserstein exponential map at Cref to each tangent perturbation in X. The mapping implemented is:
            1) compute eigen-decomposition Cref = V diag(d) V* (V* is conjugate transpose),
            2) form the pairwise reciprocal sum matrix C_ij = 1 / (d_i + d_j),
            3) rotate X into the eigenbasis, apply the elementwise scaling by C_ij, perform the quadratic update with diag(d), rotate back,
            4) return Cref + X + X_tmp (the mapped SPD/HPD matrices).
        The function returns a new numpy.ndarray and does not modify the input arrays in place.
    
    Behavior and practical significance:
        - The function projects tangent-space matrices back to the manifold so they can be used as valid covariance/HPD matrices in downstream algorithms (e.g., classification with MDM, pipeline steps that require SPD inputs).
        - It is appropriate when the Wasserstein (optimal-transport-induced) Riemannian metric is the chosen geometry for processing Gaussian/covariance objects, as described in the referenced paper.
        - This exponential map differs from affine-invariant exponential maps and is specifically designed for the Wasserstein geometry of Gaussian densities.
    
    Side effects and defaults:
        - No side effects: inputs X and Cref are not modified; a new array is allocated and returned.
        - There are no optional parameters or defaults; both X and Cref must be provided.
    
    Failure modes and numerical considerations:
        - Cref must be SPD/HPD with strictly positive eigenvalues. If Cref has nonpositive or very small eigenvalues, the computation of 1/(d_i + d_j) can overflow or produce very large values, leading to numerical instability and results that are not meaningful on the manifold.
        - np.linalg.eigh is used to compute the eigen-decomposition of Cref. If Cref is not Hermitian (for complex arrays) or not symmetric (for real arrays), np.linalg.eigh may raise an error or produce undefined results. Ensure Cref is Hermitian/symmetric before calling this function.
        - X must have last two dimensions of size n matching Cref. Shape mismatches will raise broadcasting or linear algebra errors during matrix multiplications.
        - The operation assumes that X represents a valid tangent vector at Cref in the Wasserstein geometry (typically symmetric/Hermitian structure). Supplying arbitrary matrices in X may produce outputs that are not meaningful as manifold points.
    
    Notes:
        - versionadded:: 0.8
    
    References:
        - L. Malagò, L. Montrucchio, G. Pistone, "Wasserstein Riemannian geometry of Gaussian densities", Information Geometry, 2018, Eq.(36).
    """
    from pyriemann.utils.tangentspace import exp_map_wasserstein
    return exp_map_wasserstein(X, Cref)


################################################################################
# Source: pyriemann.utils.tangentspace.log_map
# File: pyriemann/utils/tangentspace.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def pyriemann_utils_tangentspace_log_map(
    X: numpy.ndarray,
    Cref: numpy.ndarray,
    metric: str = "riemann"
):
    """Project SPD/HPD matrices into the tangent space at a reference matrix using a selectable logarithmic map.
    
    This function is part of pyriemann.utils.tangentspace and is used to convert a collection of symmetric (or Hermitian) positive definite matrices—commonly covariance matrices estimated from multichannel biosignals such as EEG/MEG/EMG in brain-computer interface (BCI) workflows—into their representation in the tangent space at a provided reference matrix. The tangent-space representation produced by this logarithmic map is typically used as a Euclidean feature space for downstream machine learning algorithms (for example, TangentSpace transformer or MDM classifier described in the pyRiemann README and examples). The choice of metric controls the geometry used for projection and therefore affects distances, averaging, and the performance of classifiers or regressors operating on the transformed data.
    
    Args:
        X (ndarray, shape (..., n, n)): Matrices to project into the tangent space. This is a batch of square matrices where the trailing two dimensions are the matrix rows and columns (n x n). In the typical pyRiemann use case these are symmetric (or Hermitian) positive definite (SPD/HPD) covariance matrices estimated from multichannel time series; providing non-square or improperly-shaped arrays will raise an error in the underlying linear-algebra routines. The function does not modify X in place; it returns a new array with the projected values.
        Cref (ndarray, shape (n, n)): Reference matrix that defines the tangent space origin. Cref must be a square matrix with the same dimensionality n as the matrices in X (i.e., X[..., :, :] must have trailing dimensions n x n). In pyRiemann typical usage Cref is an SPD/HPD matrix such as the geometric mean or a class mean; if Cref is not compatible (wrong shape, not positive definite when required by the chosen metric) the computation may fail or produce invalid results.
        metric (string | callable, default="riemann"): Metric used to select the logarithmic map implementation. If a string, it must match one of the built-in implementations: "euclid", "logchol", "logeuclid", "riemann", or "wasserstein"; the string selects the corresponding predefined log-map function used to project X onto the tangent space at Cref. If a callable, it is expected to implement an equivalent logarithmic-map interface and will be invoked as log_map_function(X, Cref) to produce the result. The default "riemann" selects the Riemannian logarithmic map commonly used for SPD matrices in pyRiemann. If metric is not recognized, the internal dispatch (via check_function) will raise an error.
    
    Returns:
        X_new (ndarray, shape (..., n, n)): New array of matrices representing X projected into the tangent space at Cref. The returned array has the same batch shape as the input X and the same trailing matrix shape (n, n). The values are the result of applying the selected logarithmic map and are intended to be used as Euclidean features for machine learning algorithms; no in-place modification of the inputs occurs.
    
    Raises:
        ValueError: If metric is a string that does not match a supported implementation, the internal function dispatch will raise a ValueError.
        ValueError or numpy.linalg.LinAlgError: If the shapes of X and Cref are incompatible (mismatched n or non-square matrices), or if numerical linear-algebra operations (e.g., matrix logarithm, Cholesky, or eigen-decomposition) required by the chosen metric fail because inputs are not positive definite or are numerically ill-conditioned, an exception will be raised by the underlying implementation.
    
    Notes:
        - This function is a thin dispatcher: it selects a concrete log-map implementation via an internal registry (check_function) and then calls that implementation with (X, Cref). The semantics and numerical details of each named metric are implemented in their respective functions (log_map_euclid, log_map_logchol, log_map_logeuclid, log_map_riemann, log_map_wasserstein).
        - Users should ensure matrices are SPD/HPD as required by the chosen metric to avoid complex-valued results or decomposition failures; when working with estimated covariance matrices from biosignals, typical preprocessing includes regularization or shrinkage to enforce positive definiteness.
    """
    from pyriemann.utils.tangentspace import log_map
    return log_map(X, Cref, metric)


################################################################################
# Source: pyriemann.utils.tangentspace.log_map_logchol
# File: pyriemann/utils/tangentspace.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def pyriemann_utils_tangentspace_log_map_logchol(X: numpy.ndarray, Cref: numpy.ndarray):
    """pyriemann.utils.tangentspace.log_map_logchol projects SPD/HPD matrices to the tangent space at a reference matrix using the log-Cholesky logarithmic map (see Lin 2019, Table 2). This implementation is intended for use with covariance or Hermitian positive definite matrices encountered in multivariate biosignal processing and remote sensing workflows (for example EEG/MEG covariance matrices used in BCI pipelines) where one needs a Euclidean representation (tangent vector) of SPD/HPD matrices relative to a chosen reference.
    
    Args:
        X (numpy.ndarray): Matrices on the SPD/HPD manifold to project. Expected shape is (..., n, n), i.e., one or more square matrices where the last two dimensions are n-by-n. Leading dimensions are treated as independent samples/batch dimensions (for example, epochs of covariance matrices estimated from multichannel time series). X must contain real symmetric positive definite matrices or complex Hermitian positive definite matrices (SPD/HPD), since the algorithm relies on Cholesky decomposition.
        Cref (numpy.ndarray): Reference SPD/HPD matrix that defines the tangent space origin. Expected shape is (n, n) and must match the last two dimensions of X. Cref is used as the base point at which the logarithmic map is computed (typical use: the geometric mean or class reference covariance in Riemannian BCI pipelines).
    
    Returns:
        numpy.ndarray: X_new, matrices projected in the tangent space at Cref. Returned shape is (..., n, n), matching the leading/batch dimensions of X and the (n, n) spatial dimensions. Each returned matrix is the tangent-space representation (a Hermitian/symmetric matrix) corresponding to the input matrix relative to Cref, obtained by: (1) computing Cholesky factors of X and Cref, (2) forming a lower-triangular residual with diagonal entries equal to Cref_chol * log(X_chol / Cref_chol) and strictly lower entries equal to the difference X_chol - Cref_chol, and (3) mapping back with X_new = Cref_chol @ res.conj().swapaxes(-1, -2) + res @ Cref_chol.conj().swapaxes(-1, -2). The returned array is a new allocation; inputs are not modified in-place.
    
    Notes:
        - This function implements the log-Cholesky logarithmic map as described in Z. Lin, "Riemannian geometry of symmetric positive definite matrices via Cholesky decomposition" (SIAM J. Matrix Anal. Appl., 2019), Table 2. It is useful when converting SPD/HPD matrices (e.g., covariance matrices used in BCI classification or hyperspectral image processing) to Euclidean tangent vectors that can be processed with classical machine learning methods.
        - The input X may contain a batch of matrices; broadcasting of batch dimensions follows NumPy broadcasting rules only for leading dimensions. The spatial dimensions (last two) must correspond to square matrices of the same size n.
        - The function supports both real-valued SPD and complex-valued HPD matrices, preserving complex conjugation where appropriate via .conj() calls.
    
    Raises:
        numpy.linalg.LinAlgError: If any matrix in X or Cref is not positive definite (Cholesky decomposition fails). This occurs when inputs are not valid SPD/HPD matrices.
        ValueError: If the spatial dimensions of X do not match the shape of Cref (i.e., if X.shape[-2:] != (n, n) for the n of Cref), or if input arrays are not at least two-dimensional square matrices.
    
    Version:
        Added in pyRiemann 0.7.
    
    References:
        Z. Lin. "Riemannian geometry of symmetric positive definite matrices via Cholesky decomposition." SIAM J. Matrix Anal. Appl., 2019, 40(4), pp. 1353-1370.
    """
    from pyriemann.utils.tangentspace import log_map_logchol
    return log_map_logchol(X, Cref)


################################################################################
# Source: pyriemann.utils.tangentspace.log_map_riemann
# File: pyriemann/utils/tangentspace.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def pyriemann_utils_tangentspace_log_map_riemann(
    X: numpy.ndarray,
    Cref: numpy.ndarray,
    C12: bool = False
):
    """Project SPD/HPD matrices to the tangent space at a reference matrix using the Riemannian logarithmic map.
    
    This function implements the Riemannian logarithmic map used in pyRiemann to embed symmetric positive definite (SPD) or Hermitian positive definite (HPD) matrices into the tangent space associated with a reference SPD/HPD matrix. The mapping is commonly used in biosignal and remote-sensing workflows provided by pyRiemann (for example, to project covariance matrices estimated from multichannel EEG/MEG/EMG recordings into a vector space suitable for classical machine learning classifiers). Mathematically, for each matrix X in the input batch the default output is
    X_new = log(Cref^{-1/2} X Cref^{-1/2}),
    where log denotes the matrix logarithm and Cref^{-1/2} the inverse square root of the reference matrix. If C12 is True the function returns the full Riemannian logarithmic map
    X_new = Cref^{1/2} log(Cref^{-1/2} X Cref^{-1/2}) Cref^{1/2},
    which yields a matrix expressed in the original ambient space but carrying the tangent-space displacement relative to Cref. The function is pure (no in-place modification of inputs) and returns a new array.
    
    Args:
        X (numpy.ndarray): Input matrices to project, with shape (..., n, n). Each trailing (n, n) matrix is expected to lie on the SPD (real symmetric positive definite) or HPD (complex Hermitian positive definite) manifold depending on data modality. In pyRiemann workflows, X typically contains covariance matrices estimated from multichannel time series (e.g., EEG epochs). The leading dimensions allow batching of multiple matrices.
        Cref (numpy.ndarray): Reference SPD/HPD matrix with shape (n, n). This matrix defines the tangent space (the point on the manifold where the tangent space is attached). In practical use, Cref is often the geometric mean or a class mean covariance used to center matrices before applying tangent-space methods.
        C12 (bool): If False (default), return the tangent vector expressed in the reference tangent space using the canonical mapping log(Cref^{-1/2} X Cref^{-1/2}). If True, return the full mapped matrix Cref^{1/2} log(Cref^{-1/2} X Cref^{-1/2}) Cref^{1/2}, which relocates the tangent vector back into the ambient matrix space while preserving the Riemannian displacement relative to Cref. Use C12=True when the downstream algorithm expects outputs in the original matrix coordinate system; use False when working with tangent-space vectors (e.g., for vectorization and classical classifiers).
    
    Returns:
        X_new (numpy.ndarray): Projected matrices with shape (..., n, n). When C12 is False, each output matrix is the matrix logarithm of the congruence transform Cref^{-1/2} X Cref^{-1/2} and thus lives in the tangent space at the identity (conjugated by Cref). When C12 is True, each output matrix is conjugated back by Cref^{1/2} and therefore lives in the ambient matrix space but encodes the tangent displacement relative to Cref. The returned dtype may be real or complex depending on the inputs and numerical results of the matrix functions.
    
    Behavior, defaults, and failure modes:
        The function calls internal helpers to check input dimensionality and then computes matrix inverse square root, matrix logarithm, and (optionally) matrix square root. It does not modify X or Cref in-place. By default C12 is False to produce tangent vectors convenient for vectorization and machine learning pipelines (this matches common usage in pyRiemann's TangentSpace transformer). If input shapes are incompatible (the trailing dimensions of X do not match the shape of Cref, or matrices are not square) the internal dimension check will fail. If Cref is not a valid SPD/HPD matrix (e.g., not positive definite) the inverse square root or logarithm may fail, produce warnings, or yield NaN/inf/complex numerical results depending on the numerical routines; similarly, if X contains matrices outside the SPD/HPD manifold numerical issues can occur. Users should ensure inputs are valid covariance or HPD matrices (for example, by using pyriemann.estimation.Covariances to estimate well-conditioned covariance matrices) before calling this function.
    
    References and provenance:
        The implementation follows the Riemannian framework used in pyRiemann and described in the literature on SPD/HPD geometry (see Pennec et al., 2006). The function was added to pyRiemann to support tangent-space embeddings for classification and regression on covariance-like features.
    """
    from pyriemann.utils.tangentspace import log_map_riemann
    return log_map_riemann(X, Cref, C12)


################################################################################
# Source: pyriemann.utils.tangentspace.log_map_wasserstein
# File: pyriemann/utils/tangentspace.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def pyriemann_utils_tangentspace_log_map_wasserstein(X: numpy.ndarray, Cref: numpy.ndarray):
    """pyriemann.utils.tangentspace.log_map_wasserstein: Project SPD/HPD matrices to the tangent space using the Wasserstein logarithmic map.
    
    Args:
        X (numpy.ndarray): Matrices in the SPD/HPD manifold to project. Expected shape (..., n, n), where the last two dimensions form square matrices. In the pyRiemann context these are typically covariance matrices estimated from multichannel biosignals (e.g., EEG, MEG, EMG) or spatial covariance estimates for remote sensing; each matrix represents a point on the manifold to be mapped to the tangent space at Cref. Elements may be real (SPD) or complex (HPD) depending on the application.
        Cref (numpy.ndarray): Reference SPD/HPD matrix at which the tangent space is defined. Expected shape (n, n) and must have the same n as the last two dimensions of X. In practice Cref is often a reference covariance matrix (for example a Riemannian mean or session/subject-specific reference) used to linearize the manifold around that point for downstream Euclidean processing (vectorization, classifiers, transfer learning).
    
    Returns:
        numpy.ndarray: Array X_new of shape (..., n, n) containing matrices projected in the tangent space at Cref using the Wasserstein logarithmic map. Each output matrix corresponds to the expression (X Cref)^{1/2} + (Cref X)^{1/2} - 2 Cref and is symmetrized/Hermitianized by construction (the implementation returns tmp + tmp.conj().swapaxes(-2, -1) - 2 * Cref). The returned matrices are suitable for Euclidean operations in pipelines such as TangentSpace or classifiers that expect tangent-space representations.
    
    Detailed behavior, defaults, and failure modes:
        This function validates input dimensions via an internal check and then computes the Wasserstein log map according to Proposition 9 of Malagò et al. (2018). Internally it computes a matrix square root of Cref, its inverse square root, and the square root of the bracket P12 @ X @ P12, then constructs the tangent vector and symmetrizes it. The function does not modify X or Cref in-place; it returns a new array.
        Failure modes include:
        - Dimension mismatch: if Cref.shape != (n, n) does not match X.shape[-2:], the internal dimension check will raise an error.
        - Non-(H)PD inputs: if Cref is not positive definite (or positive definite in the complex Hermitian sense for HPD) the inverse square root cannot be computed reliably and numerical linear algebra routines may raise errors (for example due to zero or negative eigenvalues). Similarly, X must be an SPD/HPD matrix for the mathematical interpretation to hold; non-(H)PD X may produce complex or invalid results from matrix square roots.
        - Numerical instability: when matrices are near-singular or ill-conditioned, matrix square roots and inverse square roots can be numerically unstable and produce large rounding errors or complex components; handle such cases by regularizing covariance estimates (e.g., adding a small multiple of the identity) before calling this function.
        This function was added in pyRiemann version 0.8 and is intended to be used in pipelines that linearize manifold-valued covariance data for classical machine learning algorithms.
    """
    from pyriemann.utils.tangentspace import log_map_wasserstein
    return log_map_wasserstein(X, Cref)


################################################################################
# Source: pyriemann.utils.tangentspace.tangent_space
# File: pyriemann/utils/tangentspace.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def pyriemann_utils_tangentspace_tangent_space(
    X: numpy.ndarray,
    Cref: numpy.ndarray,
    metric: str = "riemann"
):
    """pyriemann.utils.tangentspace.tangent_space: Transform symmetric/Hermitian matrices (typically covariance matrices) into tangent-space vectors using a logarithmic map around a reference matrix.
    
    Transforms input matrices X into vectors in the tangent space at a reference matrix Cref by applying a matrix logarithmic map (log_map) with respect to Cref and then vectorizing the upper-triangular part (upper). This mapping is commonly used in pyRiemann pipelines for brain-computer interface (BCI) and hyperspectral/remote-sensing workflows to convert symmetric positive definite (SPD) covariance or Hermitian positive definite (HPD) matrices into fixed-length feature vectors for classifiers (for example, SVM in TangentSpace).
    
    Args:
        X (ndarray): Matrices defined on the manifold to be mapped. Expected shape is (..., n, n), where the last two dimensions form square matrices. In practice X contains estimated covariance matrices per epoch or spatial window (e.g., EEG epochs with shape n_channels x n_times converted to covariance matrices). The function will pass X to log_map and then to upper; therefore X must be compatible with those functions (square matrices, matching dtype). Supplying arrays with incompatible shapes (non-square last two dims) or wrong dimensionality will raise an error.
        Cref (ndarray): Reference matrix for the logarithmic map, expected shape (n, n) where n matches the matrix dimension of X. In common use Cref is the Riemannian mean or another central SPD/HPD matrix around which tangent vectors are computed. Cref must be appropriate for the chosen metric: for metrics that require positive definiteness (for example the default "riemann" or "wasserstein") providing a non-SPD matrix may lead to numerical errors or linear algebra exceptions.
        metric (string | callable): Metric specifying which logarithmic map to apply. Default is "riemann". Acceptable string values (as supported by log_map) include "euclid", "logchol", "logeuclid", "riemann", and "wasserstein". Alternatively, a callable can be provided: the callable should implement the same mathematical role as log_map and accept (X, Cref, metric=...) or at least return an ndarray of the same shape as X representing mapped matrices in the tangent-space numerator before vectorization. The chosen metric determines the geometry used for the map; some metrics require SPD/HPD inputs and will fail or produce invalid values if inputs are not positive definite. If metric is an unsupported string, a runtime error will be raised.
    
    Returns:
        ndarray, shape (..., n * (n + 1) / 2): Tangent vectors obtained by first applying the logarithmic map around Cref to each matrix in X and then extracting and stacking the upper-triangular elements (including the diagonal) into a vector. Each returned row/vector corresponds to one input matrix from X (preserving the leading batch dimensions). These vectors are intended for downstream machine learning algorithms (for example, feeding into linear classifiers or pipelines such as Covariances -> TangentSpace -> SVM). No in-place modifications of X or Cref occur; the function returns a new array. Errors are raised for shape mismatches, unsupported metric strings, or numerical failures arising from non-positive-definite inputs for metrics that require SPD/HPD matrices.
    """
    from pyriemann.utils.tangentspace import tangent_space
    return tangent_space(X, Cref, metric)


################################################################################
# Source: pyriemann.utils.tangentspace.transport
# File: pyriemann/utils/tangentspace.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def pyriemann_utils_tangentspace_transport(
    X: numpy.ndarray,
    A: numpy.ndarray,
    B: numpy.ndarray
):
    """pyriemann.utils.tangentspace.transport: Parallel transport of tangent-space matrices between two positive-definite base points under the affine-invariant metric.
    
    This function implements the parallel transport of symmetric (real) or Hermitian (complex) matrices X that live in the tangent space of the manifold of symmetric/Hermitian positive definite (SPD/HPD) matrices. It moves tangent vectors defined at a source base point A to the tangent space at a target base point B along the geodesic given by the affine-invariant metric and Levi-Civita connection. In the pyRiemann workflow this is used when comparing or combining tangent-space representations of covariance/HPD matrices computed from multichannel biosignals (e.g., EEG, MEG, EMG) or remote sensing data across different sessions or subjects, where covariance matrices have been mapped to tangent space by a matrix logarithm and must be transported to a common reference base point before classification or transfer learning.
    
    Args:
        X (numpy.ndarray): Symmetric (real) or Hermitian (complex) matrices in tangent space with shape (..., n, n). Each entry is a tangent vector (for example, the matrix logarithm of an SPD/HPD covariance matrix). The leading axes (...) allow transporting a batch of tangent matrices in one call. The function expects these inputs to already be in tangent space (i.e., obtained via a logarithmic map from SPD/HPD matrices); passing raw SPD/HPD matrices will produce incorrect results.
        A (numpy.ndarray): Initial base SPD/HPD matrix of shape (n, n). This is the manifold point where the input tangent matrices X are currently based (for example, a reference covariance estimated from one subject/session). A must be symmetric/Hermitian positive definite and well-conditioned so that matrix inverse and square-root operations are numerically stable.
        B (numpy.ndarray): Final base SPD/HPD matrix of shape (n, n). This is the target manifold point to which tangent matrices X will be transported (for example, a global reference covariance for alignment). B must be symmetric/Hermitian positive definite and compatible in dimension with A and the trailing dimensions of X.
    
    Returns:
        X_new (numpy.ndarray): Transported tangent matrices with the same shape as X, i.e., (..., n, n). Each returned matrix is given by X_new = E @ X @ E^H where E = (B A^{-1})^{1/2}. In the implementation E is computed using the stable identity E = A^{1/2} (A^{-1/2} B A^{-1/2})^{1/2} A^{-1/2} to improve numerical stability. The returned array is a new object; the inputs X, A and B are not modified.
    
    Behavior, side effects, and failure modes:
        This function performs linear algebra operations (matrix inverse and matrix square-root) and therefore has cubic time complexity in n (roughly O(n^3) per matrix) and requires sufficient numerical precision. If A or B is not positive definite (e.g., singular, indefinite, or numerically rank-deficient), the underlying square-root or inverse computations may fail or return NaNs/Infs; in such cases the function will propagate the error from the linear-algebra routines or produce invalid outputs. Shape mismatches between A, B, and the trailing dimensions of X will result in broadcasting or multiplication errors from NumPy. For complex-valued inputs the conjugate-transpose (Hermitian transpose) is used in the final formula to preserve Hermitian structure. To ensure correct usage in typical pyRiemann pipelines, first compute tangent-space matrices with the logarithmic map (e.g., logm of SPD/HPD covariances) at base A, then call this function to transport them to base B before downstream tasks such as TangentSpace feature extraction, classifier training, or transfer learning across subjects/sessions.
    """
    from pyriemann.utils.tangentspace import transport
    return transport(X, A, B)


################################################################################
# Source: pyriemann.utils.tangentspace.untangent_space
# File: pyriemann/utils/tangentspace.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def pyriemann_utils_tangentspace_untangent_space(
    T: numpy.ndarray,
    Cref: numpy.ndarray,
    metric: str = "riemann"
):
    """Transform tangent vectors back to matrices using an exponential map referenced to a base matrix.
    
    This function is the inverse operation of the tangent-space mapping used in pyRiemann's TangentSpace pipeline: it takes vectorized tangent representations (typically obtained from covariance matrices in biosignal/BCI processing) and reconstructs matrices on the manifold by first rebuilding a symmetric matrix from the upper-triangular vector and then applying an exponential map centered at the provided reference matrix Cref. It is commonly used to convert features used by Euclidean classifiers (vectors in tangent space) back to symmetric positive-definite (SPD) matrices for interpretation, visualization, or further manifold-valued processing in applications such as EEG/MEG/BCI or hyperspectral/remote-sensing covariance analysis.
    
    Args:
        T (numpy.ndarray): Tangent vectors to transform back to matrices. Expected shape is (..., n * (n + 1) / 2), i.e., the last axis is the flattened upper-triangular part (including diagonal) of an n x n symmetric matrix. Each entry typically represents the vectorized form produced by a TangentSpace transform applied to SPD covariance matrices; values should be finite real numbers. The function internally calls unupper(T) to reconstruct full symmetric matrices from this representation.
        Cref (numpy.ndarray): Reference matrix used as the base point for the exponential map. Expected shape is (n, n). In pyRiemann workflows this is typically an SPD (symmetric positive-definite) covariance matrix representing the manifold reference (for example, the Riemannian mean of training covariances). Cref must be square and compatible with the dimension implied by the last axis of T (i.e., if T has last axis length m = n*(n+1)/2, Cref must be n x n with that same n). Non-square Cref or mismatched dimensions will raise an error when reconstructing or applying the exponential map.
        metric (str): Metric used to select the exponential map applied after reconstructing the symmetric matrix. Default is "riemann". Accepted string values are "euclid", "logchol", "logeuclid", "riemann", and "wasserstein", or metric can be a callable implementing the appropriate exponential-map behavior expected by pyRiemann's exp_map. The metric determines the geometric interpretation of the exponential map (e.g., "riemann" for the Riemannian exponential on SPD manifold) and thus controls how tangent vectors are mapped back to manifold matrices. If a callable is provided, it must conform to the exp_map callable contract used within pyRiemann; otherwise a runtime error may occur.
    
    Returns:
        numpy.ndarray: Matrices on the manifold reconstructed from the tangent vectors. Shape is (..., n, n), i.e., the last two axes produce square matrices of size n x n corresponding to the reference matrix dimension. Returned matrices are the result of unupper(T) followed by exp_map(..., Cref, metric=metric). In typical BCI/covariance workflows these are SPD matrices suitable for downstream manifold-aware processing or visualization.
    
    Raises:
        ValueError: If the last dimension of T does not equal n * (n + 1) / 2 where n is derived from Cref.shape, or if Cref is not square; such mismatches indicate incompatible input shapes.
        TypeError: If metric is provided as a callable that does not conform to the expected exp_map signature or behavior, a TypeError or other exception may be raised when exp_map is invoked.
        RuntimeError: Under certain metrics, exp_map may require Cref to be positive-definite; if Cref is not valid for the chosen metric (for example not SPD when required), exp_map may raise an error indicating failure to compute the exponential map.
    
    Notes:
        - The function has no side effects: it returns a new array and does not modify T or Cref in place.
        - This routine is intended for use in pipelines that convert between covariance matrices and their vectorized tangent representations (e.g., for classifiers or visualization in EEG/BCI and remote-sensing applications).
        - Numerical stability and validity of the output depend on the numeric properties of T and Cref and the chosen metric; ensure inputs are preprocessed appropriately (finite values, correct conditioning) before calling this function.
    """
    from pyriemann.utils.tangentspace import untangent_space
    return untangent_space(T, Cref, metric)


################################################################################
# Source: pyriemann.utils.tangentspace.unupper
# File: pyriemann/utils/tangentspace.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def pyriemann_utils_tangentspace_unupper(T: numpy.ndarray):
    """Inverse upper triangular unpacking for symmetric/Hermitian matrices.
    
    This function reconstructs symmetric (for real-valued data) or Hermitian
    (for complex-valued data) matrices from their weighted upper triangular
    vectorized representations. It is the inverse operation of pyriemann.utils.tangentspace.upper
    and is used in pyRiemann workflows that convert between matrix and vector
    representations (for example, the TangentSpace transformer and any pipeline
    that vectorizes SPD/HPD matrices for standard machine learning estimators).
    The input T is expected to contain the upper-triangular entries where
    off-diagonal elements were previously multiplied by sqrt(2) (the symmetric
    weighting convention used by upper). This function divides those off-diagonal
    entries by sqrt(2) and fills the lower triangle as the conjugate transpose
    of the upper triangle to restore symmetry/Hermiticity.
    
    Args:
        T (ndarray): Weighted upper triangular parts of symmetric/Hermitian matrices.
            Expected shape is (..., n * (n + 1) / 2), where the last axis contains
            the flattened upper-triangular entries (including the diagonal) for
            matrices of size n x n. The array may have any leading batch shape
            (denoted by ...). The dtype of T determines the dtype of the output;
            complex dtypes produce Hermitian matrices, real dtypes produce symmetric
            matrices.
    
    Returns:
        X (ndarray): Symmetric/Hermitian matrices reconstructed from T.
            The returned array has shape (..., n, n), matching the batch shape of T
            with the last axis expanded into square matrices. Off-diagonal entries
            are divided by sqrt(2) to undo the weighting applied by upper, and the
            lower-triangle is filled as the conjugate of the corresponding
            upper-triangle entries, so X[..., i, j] == conj(X[..., j, i]) holds.
    
    Behavior and side effects:
        - The function computes n from the last dimension m of T using the
          quadratic relation m = n * (n + 1) / 2 and the formula
          n = int((sqrt(1 + 8 * m) - 1) / 2). The integer n determines the
          reconstructed square matrix size.
        - A new array X is allocated (np.empty) and populated; the input array T
          is not modified (no in-place changes to T).
        - Data type is preserved: the dtype of X equals the dtype of T.
        - Off-diagonal positions in the stored upper-triangle are assumed to have
          been scaled by sqrt(2); this function divides those positions by sqrt(2)
          to restore the true matrix values.
    
    Failure modes and constraints:
        - The last dimension of T must equal n * (n + 1) / 2 for some integer n.
          If the last axis length is not a triangular number, the computed n will
          be incorrect and the reconstruction will be invalid; this can lead to
          incorrect output shapes, IndexError, or silently wrong matrices.
        - No explicit input validation or error is raised by this function; callers
          should ensure the shape constraint is satisfied before calling.
        - The function assumes the ordering of values in the last axis matches the
          ordering produced by the corresponding upper function; mismatched ordering
          will produce incorrect matrices.
    
    Notes:
        - This function was added in version 0.4 of pyRiemann and follows the same
          weighted upper-triangular convention used throughout the tangentspace
          utilities in the library.
    """
    from pyriemann.utils.tangentspace import unupper
    return unupper(T)


################################################################################
# Source: pyriemann.utils.tangentspace.upper
# File: pyriemann/utils/tangentspace.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def pyriemann_utils_tangentspace_upper(X: numpy.ndarray):
    """pyriemann.utils.tangentspace.upper returns the weighted, vectorized upper-triangular part of square symmetric/Hermitian matrices to produce the minimal tangent-space representation used throughout pyRiemann for processing covariance/HPD matrices (for example, covariance matrices estimated from multichannel biosignals such as EEG/MEG/EMG in BCI pipelines).
    
    This function extracts the upper triangular entries of each input matrix (including the diagonal), applies a weight of 1.0 to diagonal elements and sqrt(2) to off-diagonal elements, and flattens those weighted entries into a 1-D vector per matrix. The weighting (unity on the diagonal, sqrt(2) on off-diagonals) yields the minimal representation commonly used in tangent-space/vectorization routines so that inner products and norms are preserved when mapping symmetric/Hermitian matrices to Euclidean vectors. The function supports broadcasting over any leading dimensions, so it accepts batches of matrices with shape (..., n, n) and returns a batch of vectors with shape (..., n * (n + 1) / 2). The routine does not validate that the numerical values of X are symmetric/Hermitian; it only requires square matrices and will operate on the provided upper-triangular values. It performs no in-place modification of the input and returns a new ndarray.
    
    Args:
        X (ndarray): Symmetric/Hermitian matrices with shape (..., n, n). In the pyRiemann workflow these are typically covariance (SPD) or Hermitian positive-definite (HPD) matrices estimated from multichannel time-series (e.g., EEG epochs). The trailing two dimensions must be square (n x n). Leading dimensions, if any, are treated as batch dimensions and are preserved in the output. The function assumes the matrix entries are arranged so that taking the upper-triangular part is meaningful; it does not check numerical symmetry beyond the shape requirement.
    
    Returns:
        T (ndarray): Weighted upper triangular parts of the input matrices, vectorized into shape (..., n * (n + 1) / 2). Each output vector contains the diagonal entries (weighted by 1) and the strictly upper-triangular entries (weighted by sqrt(2)), ordered according to numpy.triu indexing. This vector is the minimal tangent-space representation commonly used downstream in pyRiemann for feature extraction and classification (for example, before applying Euclidean classifiers such as SVM in a TangentSpace pipeline).
    
    Raises:
        ValueError: If the last two dimensions of X are not equal (i.e., matrices are not square), a ValueError is raised with message "Matrices must be square". This is the only explicit check performed by the function; other invalid inputs (wrong dtype, non-numeric entries) will raise errors from numpy operations.
    
    Notes:
        - The function has no side effects on its input array; it returns a new array containing the weighted, vectorized upper-triangular entries.
        - Computational cost is O(n^2) per matrix due to extraction of all upper-triangular entries.
        - Introduced in pyRiemann v0.4 as part of utilities supporting tangent-space representations for machine learning on covariance/HPD matrices.
    """
    from pyriemann.utils.tangentspace import upper
    return upper(X)


################################################################################
# Source: pyriemann.utils.test.is_hankel
# File: pyriemann/utils/test.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def pyriemann_utils_test_is_hankel(X: numpy.ndarray):
    """pyriemann.utils.test.is_hankel checks whether a given square numpy.ndarray is a Hankel matrix.
    
    A Hankel matrix is a square matrix that is constant along its anti-diagonals (elements with equal i+j indices are equal). In the context of pyRiemann, which processes multichannel time-series and covariance-like matrices for biosignals (EEG/MEG/EMG) and remote sensing, detecting a Hankel structure can be used to validate time-delay embeddings, structured covariance estimates, or to gate algorithms that assume anti-diagonal constancy. This function performs an element-wise exact comparison to verify that property without modifying the input.
    
    Args:
        X (numpy.ndarray): 2-D square array with shape (n, n). This argument is the matrix to test for the Hankel property. The matrix typically represents a covariance-like or time-delay embedded matrix in pyRiemann workflows; the function expects a two-dimensional, square numpy.ndarray and will return False for inputs that are not square or not two-dimensional.
    
    Returns:
        bool: True if and only if X is a Hankel matrix as defined by constant values on each anti-diagonal. The function returns False in the following cases: X is not a square matrix, X.ndim != 2, or any pair of elements that should be equal on an anti-diagonal differ. Note that comparisons are exact (using !=) so floating-point rounding or tiny numerical differences may cause the function to return False even for matrices that are approximately Hankel; callers requiring tolerance should pre-process X (for example with rounding) before calling this function. The function has no side effects and does not modify X. Computational complexity is O(n^2) for an n-by-n input.
    """
    from pyriemann.utils.test import is_hankel
    return is_hankel(X)


################################################################################
# Source: pyriemann.utils.test.is_herm_pos_def
# File: pyriemann/utils/test.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def pyriemann_utils_test_is_herm_pos_def(X: numpy.ndarray, tol: float = 0.0):
    """Check if all matrices in X are Hermitian positive-definite (HPD).
    
    This utility is used in pyRiemann to validate matrices (for example covariance or cross-spectral
    matrices computed from multivariate biosignals such as EEG, MEG, EMG or from remote sensing data)
    before applying algorithms that assume Hermitian (conjugate-symmetric) and positive-definite
    structure. The function returns True only when every square matrix in X is both Hermitian and
    positive-definite according to a numerical threshold; it composes the results of is_hermitian(X)
    and is_pos_def(X, tol=tol).
    
    Args:
        X (numpy.ndarray): The set of square matrices to test. Must be an ndarray with shape (..., n, n)
            (at least 2D) where the last two dimensions index square matrices of size n x n. Each
            matrix is checked for Hermitian symmetry (A == A.conj().T) and for positive-definiteness
            (all eigenvalues > tol). In pyRiemann this commonly represents estimated covariance or
            Hermitian covariance-like matrices used by Riemannian geometry methods for classification
            and signal processing.
        tol (float): Threshold below which eigenvalues are considered zero. Default 0.0. A value of
            tol = 0.0 enforces strict positive-definiteness (all eigenvalues must be strictly > 0).
            A small positive tol can be used to tolerate numerical round-off or near-singular matrices
            by treating eigenvalues <= tol as non-positive. This parameter is forwarded to the
            underlying is_pos_def check.
    
    Returns:
        bool: True if and only if every matrix in X is Hermitian and positive-definite with eigenvalues
        exceeding tol. Returns False if any matrix fails the Hermitian check or the positive-definiteness
        check. There are no side effects. Note that for inputs with NaN or infinite values, or for inputs
        that do not conform to the required shape (..., n, n), the behavior depends on the underlying
        helpers (is_hermitian / is_pos_def) and may result in False or in an exception from those routines.
        Also note that the checks may be computationally expensive for large batches of large matrices
        because they typically involve eigenvalue computations.
    """
    from pyriemann.utils.test import is_herm_pos_def
    return is_herm_pos_def(X, tol)


################################################################################
# Source: pyriemann.utils.test.is_herm_pos_semi_def
# File: pyriemann/utils/test.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def pyriemann_utils_test_is_herm_pos_semi_def(X: numpy.ndarray):
    """Check whether every matrix in a collection is Hermitian positive semi-definite (HPSD).
    
    Args:
        X (numpy.ndarray): The set of square matrices to test. This must be at least a 2-D numpy.ndarray with shape (..., n, n) where the last two dimensions index each n-by-n matrix. In the pyRiemann context, X typically contains covariance or cross-spectral matrices estimated from multichannel biosignals (EEG/MEG/EMG) or from spatial windows in remote sensing; these matrices may be real-valued symmetric (for SPD/real-valued covariance) or complex-valued Hermitian (for HPD/complex-valued covariance). The function verifies properties across the whole collection provided in X.
    
    Returns:
        bool: True if and only if every n-by-n matrix contained in X is both Hermitian (equal to its own conjugate transpose) and positive semi-definite (all eigenvalues are non-negative). This function computes the logical conjunction of two lower-level checks (is_hermitian and is_pos_semi_def) and returns False if either check fails for any matrix.
    
    Behavior, side effects, defaults, and failure modes:
        This function performs a pure logical test and has no side effects (it does not modify X). It delegates to the helper functions is_hermitian(X) and is_pos_semi_def(X): the result is True only when both helpers return True. The expected input type and layout is required: if X is not at least 2-D or the last two dimensions are not square (i.e., not shape (..., n, n)), the underlying checks will typically fail and the function will return False or propagate an error raised by those helper functions. Inputs containing NaNs, Infs, or invalid dtypes may cause the helper checks to return False or raise exceptions depending on numpy operations used internally. Use this function before passing matrices to pyRiemann algorithms (for example, covariance-based estimators, MDM classifier, or tangent space mappings) which require Hermitian positive semi-definite inputs.
    """
    from pyriemann.utils.test import is_herm_pos_semi_def
    return is_herm_pos_semi_def(X)


################################################################################
# Source: pyriemann.utils.test.is_hermitian
# File: pyriemann/utils/test.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def pyriemann_utils_test_is_hermitian(X: numpy.ndarray):
    """Check whether every square matrix in X is Hermitian.
    
    In the pyRiemann library this function is used to validate complex-valued square matrices (for example complex covariance estimates or kernel matrices encountered in biosignal processing, BCI, or remote sensing workflows) before treating them as Hermitian positive-definite (HPD) objects for Riemannian geometry operations. A matrix is considered Hermitian here when its real part is symmetric and its imaginary part is skew-symmetric. The implementation performs this check by testing symmetry of X.real and skew-symmetry of X.imag using the library helpers is_sym and is_skew_sym.
    
    Args:
        X (numpy.ndarray): The set of square matrices to test, with shape (..., n, n). Must be at least 2D: the trailing two dimensions correspond to n x n matrices. Typically this array contains real or complex-valued matrices arising from multichannel time-series analysis (e.g., covariance matrices estimated from EEG/MEG/EMG epochs) where Hermitian structure is required for downstream HPD/HPD-based algorithms.
    
    Returns:
        bool: True if and only if every n x n matrix in X satisfies the Hermitian condition checked by this function (real part symmetric AND imaginary part skew-symmetric). A False return means at least one matrix fails the Hermitian checks. There are no side effects; the function does not modify X.
    
    Notes on behavior and failure modes:
        The function assumes X has shape (..., n, n). If X is not at least 2D or its trailing dimensions are not square matrices, behavior is undefined: the underlying helper functions (is_sym and is_skew_sym) may return False or raise an exception depending on the provided input shape and contents. Numerical comparisons are delegated to the underlying helpers; precision and tolerance characteristics follow their implementations. The function is deterministic and has no external dependencies beyond evaluating X.real and X.imag with the helper checks.
    """
    from pyriemann.utils.test import is_hermitian
    return is_hermitian(X)


################################################################################
# Source: pyriemann.utils.test.is_pos_def
# File: pyriemann/utils/test.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def pyriemann_utils_test_is_pos_def(
    X: numpy.ndarray,
    tol: float = 0.0,
    fast_mode: bool = False
):
    """Check whether all matrices contained in X are positive definite (PD).
    
    This utility is used within pyRiemann to validate symmetric/Hermitian positive definite (SPD/HPD) matrices such as covariance matrices estimated from multichannel biosignals (EEG/MEG/EMG) or spatial covariance blocks in remote sensing. A return value of True indicates that every matrix in the input satisfies the positive definiteness criterion required by downstream Riemannian-geometry-based algorithms (e.g., covariance-based classification, MDM, TangentSpace). The function accepts a single square matrix of shape (n, n) or a batch of matrices with shape (..., n, n).
    
    Args:
        X (numpy.ndarray): The set of square matrices to test. Expected shape is (..., n, n) (at least a 2-D ndarray). Elements may be real (SPDs) or complex (HPDs) as used across pyRiemann workflows. The last two dimensions must form square matrices; otherwise the checks will fail and the function will return False (fast_mode) or perform an explicit square check (non-fast mode).
        tol (float): Threshold below which eigenvalues are considered non-positive. Default is 0.0. In the full (non-fast) mode, each matrix is declared positive definite only if all of its eigenvalues are strictly greater than tol (i.e., eigenvalue > tol). This parameter controls numerical tolerance for near-singular matrices when using eigenvalue-based verification. Note: tol is ignored when fast_mode is True.
        fast_mode (bool): Use a Cholesky decomposition-based check when True (default False). In fast mode the function attempts np.linalg.cholesky(X) on the input arrays: if the decomposition succeeds for all matrices, they are considered positive definite and the function returns True; if np.linalg.cholesky raises numpy.linalg.LinAlgError (for non-PD or non-square last two dims) the function catches it and returns False. Fast mode avoids the more expensive eigen decomposition but may misclassify matrices that are theoretically PD but numerically unstable under Cholesky due to floating-point errors.
    
    Returns:
        bool: True if and only if all matrices in X are positive definite according to the selected verification method. In non-fast mode this means is_square(X) is True and all eigenvalues computed by the internal eigenvalue routine are strictly greater than tol. In fast mode this means the Cholesky decomposition succeeded for the provided matrices. This boolean is intended to be used as a precondition check before passing matrices to pyRiemann estimators and classifiers that require SPD/HPD inputs.
    
    Behavior, performance, and failure modes:
        - No in-place modification of X occurs; the function has no side effects beyond computation and returning a boolean.
        - Full eigenvalue-based verification (fast_mode=False) computes all eigenvalues for each matrix (computational complexity ~O(n^3) per matrix) and applies the tol threshold; it is robust for detecting near-singular matrices according to the supplied tolerance.
        - Fast Cholesky-based verification (fast_mode=True) is typically faster and avoids explicit eigenvalue computation, but relies on numerical stability of the Cholesky algorithm and ignores tol. It will return False if the input is not square along the last two dimensions or if Cholesky fails due to non-PD or numerical issues; np.linalg.LinAlgError is caught and results in False.
        - Other exceptions (e.g., invalid input types) may propagate from NumPy routines; the caller should ensure X is a numpy.ndarray with appropriate shape.
    """
    from pyriemann.utils.test import is_pos_def
    return is_pos_def(X, tol, fast_mode)


################################################################################
# Source: pyriemann.utils.test.is_pos_semi_def
# File: pyriemann/utils/test.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def pyriemann_utils_test_is_pos_semi_def(X: numpy.ndarray):
    """pyriemann.utils.test.is_pos_semi_def checks whether every matrix contained in the input array X is positive semi-definite (PSD). This is a lightweight validator used in pyRiemann to verify covariance matrices or other square matrix collections before algorithms that assume non-negative eigenvalues (for example, Riemannian geometry routines that require SPD/PSD inputs).
    
    Args:
        X (numpy.ndarray): The collection of square matrices to test. X must be a NumPy ndarray with shape (..., n, n), i.e., the last two axes represent n-by-n matrices and any leading axes index multiple matrices. In the pyRiemann context, X commonly holds estimated covariance matrices from multichannel biosignals (e.g., EEG epochs n_epochs x n_channels x n_channels) or spatial covariance blocks for remote sensing. The function first checks squareness via is_square(X); if X is not at least 2-D with square trailing dimensions, the function will return False. Elements are expected to be numeric (typically real-valued floats) because the check relies on eigenvalue computation. Note that floating-point round-off can produce small negative eigenvalues for matrices that are theoretically PSD; such numerical artifacts will cause this function to return False because the implementation tests eigenvalues with the comparison >= 0.0. The implementation delegates eigenvalue extraction to _get_eigenvals and does not modify X.
    
    Returns:
        bool: True if and only if X is square (last two axes are n x n) and every n-by-n matrix in X has all eigenvalues greater than or equal to 0.0. Returns False if any matrix has a negative eigenvalue, if X is not square, or if X has insufficient dimensions. This boolean result is intended as a quick validity check for covariance or similarity matrices before using pyRiemann routines that require PSD/SPD inputs.
    """
    from pyriemann.utils.test import is_pos_semi_def
    return is_pos_semi_def(X)


################################################################################
# Source: pyriemann.utils.test.is_real
# File: pyriemann/utils/test.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def pyriemann_utils_test_is_real(X: numpy.ndarray):
    """Check whether every matrix in a collection has no significant imaginary component.
    
    Args:
        X (numpy.ndarray): Array of matrices to test, with shape (..., n, m). In the pyRiemann context this is typically a collection of multichannel covariance or cross-spectral matrices estimated from biosignals (e.g., EEG, MEG, EMG) or image patches (remote sensing). The function expects a NumPy ndarray as provided by upstream estimation routines; the dtype may be real or complex. The check is performed on the imaginary part of X (X.imag), so passing a non-ndarray or an object without an imaginary attribute may raise an exception.
    
    Returns:
        bool: True if all matrices are considered strictly real, False otherwise. "Strictly real" here means that the imaginary parts of all entries are numerically zero within NumPy's default allclose tolerances (np.allclose with default rtol and atol), i.e., np.allclose(X.imag, np.zeros_like(X.imag)) is True. This provides more robust handling of tiny numerical imprecisions (round-off errors) than a literal check such as np.all(np.isreal(X)). There are no side effects.
    
    Behavior and failure modes:
        The function inspects X.imag and compares it to an array of zeros of the same shape using np.allclose. It returns True when the imaginary components are negligible within NumPy's default relative and absolute tolerances; otherwise it returns False. If X is not a numpy.ndarray or lacks an .imag attribute, calling this function may raise an AttributeError or TypeError. The function does not modify X and has no other side effects. Use this check in preprocessing pipelines (for example, before applying Riemannian geometry operations on SPD/HPD matrices) to ensure that matrices are effectively real-valued despite numerical noise.
    """
    from pyriemann.utils.test import is_real
    return is_real(X)


################################################################################
# Source: pyriemann.utils.test.is_real_type
# File: pyriemann/utils/test.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def pyriemann_utils_test_is_real_type(X: numpy.ndarray):
    """Check if a collection of matrices is of a real numeric type.
    
    This function is used in pyRiemann preprocessing and validation steps where algorithms
    assume real-valued matrices (for example, covariance estimation and Riemannian
    operations on symmetric positive definite (SPD) matrices). It inspects the numeric
    dtype of the input matrix array to determine whether the data are stored as real
    valued numbers (no imaginary component). This is a lightweight, dtype-level test:
    it does not check matrix properties such as symmetry, positive-definiteness, or
    shape validity beyond the array's dtype.
    
    Args:
        X (ndarray): The set of matrices to test, expected to be a NumPy ndarray
            with shape (..., n, m). The leading dimensions (represented by ...)
            allow batching (e.g., multiple covariance matrices stacked as
            n_matrices x n_channels x n_channels). The function expects numeric
            entries and uses the array's dtype to decide whether values are real.
            Behavior is defined for ndarray inputs; passing non-ndarray objects is
            not guaranteed by this docstring and may produce results consistent
            with NumPy's isrealobj but is otherwise outside the stated contract.
    
    Returns:
        ret (bool): True if the input array X has a real numeric dtype (i.e., no
            complex/imaginary component in the array's declared dtype) and therefore
            is suitable for pyRiemann code paths that require real-valued matrices.
            Returns False if the array has a complex dtype or otherwise indicates
            complex-valued entries. This function has no side effects.
    
    Behavior, defaults, and failure modes:
        This function delegates to NumPy's dtype inspection (np.isrealobj). It runs
        in time proportional to the cost of checking the array object (constant time
        with respect to element inspection because it reads dtype information,
        not elementwise values). It does not validate matrix mathematical properties
        (symmetry, positive-definiteness) nor does it inspect individual elements'
        imaginary parts beyond what the dtype indicates. If X has an object dtype
        or a nonstandard dtype, the result follows NumPy's isrealobj semantics. The
        caller should ensure X is a correctly shaped ndarray of numeric type when
        relying on this function for data validation in pyRiemann workflows.
    
    Notes:
        This function was added in pyRiemann version 0.6 and is intended for use in
        data validation steps before applying algorithms that assume real-valued
        SPD matrices (common in biosignal processing and BCI pipelines).
    """
    from pyriemann.utils.test import is_real_type
    return is_real_type(X)


################################################################################
# Source: pyriemann.utils.test.is_skew_sym
# File: pyriemann/utils/test.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def pyriemann_utils_test_is_skew_sym(X: numpy.ndarray):
    """Check if all matrices in X are skew-symmetric (X^T = -X) across the last two axes.
    
    This utility is part of pyriemann.utils.test and is intended for use in unit tests, validation checks, and preprocessing/diagnostics inside the pyRiemann library. In the pyRiemann context (multivariate biosignal processing and Riemannian geometry of matrices), it helps verify that a candidate array of square matrices represents antisymmetric operators (for example, elements of the Lie algebra of SO(n) or antisymmetric residuals produced during intermediate computations). The check is performed element-wise with a numerical tolerance (see behavior below) and does not modify the input array.
    
    Args:
        X (numpy.ndarray): The input array containing one or more square matrices to test. Expected shape is (..., n, n), i.e. at least a 2-D array where the last two dimensions form n-by-n matrices. Each n-by-n matrix is tested independently for skew-symmetry. The array must be a NumPy ndarray; if another array-like object is supplied, NumPy operations inside the function may raise an exception.
    
    Returns:
        bool: True if every n-by-n matrix contained in X (along the last two axes) is skew-symmetric within numerical tolerance, False otherwise. Concretely, the function returns True when X is square (the last two dimensions are equal) and np.allclose(X, -np.swapaxes(X, -2, -1)) is True. If X is not at least 2-D or the last two dimensions are not equal (i.e., the matrices are not square), the function returns False. The comparison uses numpy.allclose with its default relative and absolute tolerances, so small floating-point rounding errors are tolerated. There are no side effects; the input X is not modified.
    """
    from pyriemann.utils.test import is_skew_sym
    return is_skew_sym(X)


################################################################################
# Source: pyriemann.utils.test.is_square
# File: pyriemann/utils/test.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def pyriemann_utils_test_is_square(X: numpy.ndarray):
    """pyriemann.utils.test.is_square checks whether the last two dimensions of a numpy array represent square matrices.
    
    This function is typically used in pyRiemann preprocessing and validation code to ensure that inputs that are expected to be batches of square matrices (for example, covariance matrices estimated from multichannel biosignals such as EEG, which are symmetric positive definite and therefore square) have the correct shape before further Riemannian operations. It performs a lightweight shape check only and does not inspect matrix values or properties such as symmetry or positive definiteness.
    
    Args:
        X (numpy.ndarray): Array expected to contain one or more matrices arranged in its last two axes, with shape (..., n, n). The function requires X to be at least 2-dimensional (ndim >= 2). In typical pyRiemann usage, X would be an array of covariance matrices with shape (n_epochs, n_channels, n_channels) or a single matrix with shape (n_channels, n_channels).
    
    Returns:
        bool: True if X has at least two dimensions and the size of its last-but-one axis equals the size of its last axis (X.ndim >= 2 and X.shape[-2] == X.shape[-1]), indicating the entries along the last two axes form square matrices. Returns False otherwise. Note that this check only verifies shape equality; it treats zero-sized last dimensions (e.g., shape (..., 0, 0)) as square. There are no side effects.
    
    Failure modes and additional details:
        - The function assumes X provides numpy.ndarray attributes ndim and shape. If a non-array object lacking these attributes is passed, an AttributeError may be raised by attempting to access X.ndim or X.shape.
        - The function does not validate data type, matrix symmetry, positive definiteness, or numerical content; those validations must be performed separately if required by downstream Riemannian algorithms.
        - The operation is constant-time and incurs negligible overhead (simple attribute and tuple comparisons).
    """
    from pyriemann.utils.test import is_square
    return is_square(X)


################################################################################
# Source: pyriemann.utils.test.is_sym
# File: pyriemann/utils/test.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def pyriemann_utils_test_is_sym(X: numpy.ndarray):
    """Check if all matrices in X are symmetric.
    
    This function is used in pyRiemann to validate that a collection of square matrices (for example covariance matrices estimated from multichannel biosignals such as EEG/MEG used in BCI applications) satisfy the symmetry property required by symmetric positive definite (SPD) or Hermitian positive definite (HPD) matrix processing. It operates on a single numpy.ndarray that may contain a batch of matrices in its leading dimensions and checks symmetry on the last two axes.
    
    Args:
        X (numpy.ndarray): The input array containing one or more square matrices, with shape (..., n, n). Each element along the last two axes is treated as an n-by-n matrix. The array must be at least 2-D. Typical usage in the pyRiemann domain is to pass covariance estimates shaped (n_epochs, n_channels, n_channels) or single matrices shaped (n_channels, n_channels).
    
    Returns:
        bool: True if and only if X contains only square matrices (the check performed by the helper is_square) and every matrix is symmetric within numerical tolerance. Symmetry is tested by comparing X to X with its last two axes swapped using numpy.allclose with numpy's default relative and absolute tolerances (so small floating-point rounding errors will be treated as symmetric). Returns False if X is not square, not at least 2-D, or if any matrix differs from its transpose by more than the allclose tolerances.
    
    Behavior, side effects, defaults, and failure modes:
        - Behavior: The function performs a structural check (is_square) followed by an elementwise numerical comparison between each matrix and its transpose (implemented as np.swapaxes(X, -2, -1)) using numpy.allclose. This makes the function suitable for verifying matrices prior to Riemannian geometry operations that assume symmetry (for example, methods that require SPD/HPD inputs).
        - Side effects: None. The function does not modify X and has no external side effects.
        - Defaults: The numerical tolerance for the symmetry check is governed by numpy.allclose default parameters (numpy.set_printoptions does not affect this); no tolerances are exposed or configurable via this function's API.
        - Failure modes: The function expects a numpy.ndarray of numeric dtype. If X is not a numpy.ndarray (for example an object that lacks a shape attribute) or contains non-numeric/object dtype entries, numpy operations used internally may raise exceptions or produce unexpected results. If X has mismatched or non-square trailing dimensions, the function will return False rather than attempting to coerce shapes. The caller should ensure that inputs conform to the expected shape and dtype when used in pyRiemann preprocessing pipelines.
    """
    from pyriemann.utils.test import is_sym
    return is_sym(X)


################################################################################
# Source: pyriemann.utils.test.is_sym_pos_def
# File: pyriemann/utils/test.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def pyriemann_utils_test_is_sym_pos_def(X: numpy.ndarray, tol: float = 0.0):
    """pyriemann.utils.test.is_sym_pos_def checks whether every matrix in a provided array X is symmetric positive-definite (SPD). This is typically used in pyRiemann to validate covariance or Hermitian positive-definite matrices estimated from multichannel biosignals (e.g., EEG, MEG, EMG) before applying Riemannian-geometry-based processing and classification algorithms that require SPD inputs.
    
    Args:
        X (numpy.ndarray): The set of square matrices to check, provided as a numpy ndarray with shape (..., n, n). The trailing two dimensions must form square matrices; the leading dimensions (if any) index multiple matrices. In the pyRiemann context, X commonly contains covariance matrices estimated from multichannel time series (n channels), and ensuring they are SPD is necessary for downstream algorithms (e.g., TangentSpace, MDM).
        tol (float): Threshold (default 0.0) used when assessing positive-definiteness. Eigenvalues strictly below this threshold are treated as zero for the purpose of the positive-definiteness test. Practically, this allows numerical tolerance when small negative or near-zero eigenvalues arise from estimation or floating-point errors; increasing tol makes the check more conservative (matrices with small positive eigenvalues less than tol will be considered singular/non-positive-definite).
    
    Behavior and practical details:
        The function returns True only if all matrices in X are both symmetric (within the tolerance used by the internal symmetry check) and positive-definite according to the tol threshold. Internally it delegates to is_sym and is_pos_def checks, so symmetry is checked first and positive-definiteness is checked next. The function performs these checks elementwise over the stack of matrices represented by X; there are no side effects (it does not modify X). Performance cost depends on the size and number of matrices (e.g., eigenvalue computations for positive-definiteness checks), so use with large numbers of large matrices may be computationally expensive.
        The function expects X to be at least a 2-D numpy.ndarray where the last two dimensions are square. Supplying arrays with incompatible shapes, non-numeric dtypes, or non-ndarray types may cause the underlying checks to return False or to raise an error depending on those checks' validations. Use this function as a precondition check before applying algorithms in the pyRiemann pipeline that require SPD inputs.
    
    Returns:
        bool: True if and only if every matrix in X is symmetric and considered positive-definite given the tol threshold. False otherwise.
    """
    from pyriemann.utils.test import is_sym_pos_def
    return is_sym_pos_def(X, tol)


################################################################################
# Source: pyriemann.utils.test.is_sym_pos_semi_def
# File: pyriemann/utils/test.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def pyriemann_utils_test_is_sym_pos_semi_def(X: numpy.ndarray):
    """Check whether every square matrix in X is symmetric and positive semi-definite (SPSD).
    
    This utility is used in pyRiemann to validate matrices that are expected to represent covariance or kernel matrices (for example, covariance matrices estimated from multichannel EEG/MEG/EMG trials in brain–computer interface pipelines). The function returns True only when every matrix in the input is both symmetric and positive semi-definite according to the underlying checks is_sym and is_pos_semi_def; otherwise it returns False. The function does not modify the input array.
    
    Args:
        X (numpy.ndarray): The set of square matrices to test, with shape (..., n, n). The array must be at least 2-D and the last two dimensions are interpreted as the row and column indices of n-by-n matrices. In the pyRiemann domain, X typically contains covariance matrices estimated from multivariate time-series; the check ensures these matrices meet the mathematical prerequisites for Riemannian-geometry-based processing (e.g., MDM, tangent-space mapping). The function delegates to the helper functions is_sym (which verifies symmetry numerically) and is_pos_semi_def (which verifies positive semi-definiteness) and returns the logical AND of their results. The function does not change X in place. If X contains NaNs or infinities, or if it is not shaped as (..., n, n), the helper checks will typically fail and the function will return False (or may raise an error raised by those helper functions if they do not accept the input type).
    
    Returns:
        bool: True if and only if every n-by-n matrix represented in X (along the last two axes) is symmetric and positive semi-definite. A True result indicates the matrices are valid covariance-like inputs for downstream pyRiemann algorithms; a False result indicates at least one matrix fails symmetry or positive semi-definiteness checks, and downstream algorithms that require SPSD/SPD matrices may produce incorrect results or raise errors.
    """
    from pyriemann.utils.test import is_sym_pos_semi_def
    return is_sym_pos_semi_def(X)


################################################################################
# Source: pyriemann.utils.utils.check_function
# File: pyriemann/utils/utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def pyriemann_utils_utils_check_function(fun: str, functions: dict):
    """Check which function implementation to use given a user input name or a callable.
    
    This utility is used across the pyRiemann package to resolve a requested function into a concrete callable implementation used by the API (for example to select implementations for covariance estimation, tangent-space transforms, metric computations or other processing steps in BCI and remote sensing pipelines described in the project README). The function accepts either a string naming one of the available API functions or a callable provided by the user, and returns a callable that can then be invoked by downstream code to process SPD/HPD matrices or multichannel biosignal-derived objects. The resolution does not perform signature compatibility checks on user-provided callables; it only maps names to implementations or verifies that the provided object is callable.
    
    Args:
        fun (string | callable): Function identifier or callable to check. If a string, it must exactly match one of the keys in the functions mapping and the corresponding callable implementation from functions will be returned. If a callable, it is assumed to be a function defined in the pyRiemann API or supplied by the user; in this latter case the caller is responsible for ensuring that the callable's signature and behavior match the expectations of the component that will use it (for example, the same parameters and return semantics as the API implementations listed in functions). The function will not inspect or validate the callable's signature; it only verifies that it is callable.
        functions (dict): Mapping of available API function names to their callable implementations. This mapping is used only when fun is provided as a string: the string must be a key in this dictionary, and the corresponding value (a callable implementation) will be returned. The mapping is not modified by this function; it is treated as a lookup table of available implementations provided by the pyRiemann API or by the caller.
    
    Returns:
        callable: The callable implementation to use. If fun was a string, this is the callable retrieved from functions[fun]. If fun was already a callable, the same object is returned. The returned callable is intended to be invoked by pyRiemann components (for example, to estimate covariance matrices, compute Riemannian metrics, or transform data in BCI-style pipelines). No additional wrapping or validation of the callable is performed.
    
    Raises:
        ValueError: If fun is a string but not present among functions.keys(), a ValueError is raised with a message listing the available keys. If fun is neither a string nor a callable, a ValueError is raised indicating that the argument must be a string or a callable and reporting the received type.
    
    Version:
        Added in pyRiemann 0.6.
    
    Behavioral notes and side effects:
        This function performs a pure resolution/validation operation and has no side effects: it does not modify the functions mapping or the callable object. It is intentionally conservative: when given a callable it trusts the caller that the callable's interface is compatible with the rest of the pyRiemann API and does not perform runtime signature checks. Use this utility to centralize name-to-callable resolution in pipelines and configuration parsers where users may specify either built-in function names or their own implementations.
    """
    from pyriemann.utils.utils import check_function
    return check_function(fun, functions)


################################################################################
# Source: pyriemann.utils.utils.check_init
# File: pyriemann/utils/utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def pyriemann_utils_utils_check_init(init: numpy.ndarray, n: int):
    """Check the initial matrix used to initialize algorithms in pyriemann.utils.utils.check_init. This function verifies that the provided initial matrix has the exact square shape (n, n) required by downstream algorithms that operate on n-by-n matrices (for example, covariance matrices and other SPD/HPD matrices used in pyRiemann for biosignal processing and remote sensing). The function also converts the input to a numpy.ndarray using np.asarray but does not perform checks on symmetry or positive-definiteness.
    
    Args:
        init (numpy.ndarray): A square matrix intended to initialize an algorithm. This parameter is converted to a numpy.ndarray via np.asarray so inputs provided as lists or array-like objects become numpy arrays. The matrix is expected to represent an n-by-n quantity such as a covariance matrix used by Riemannian geometry based methods in pyRiemann; however, this function only enforces the shape and does not validate matrix properties like symmetry or positive-definiteness.
        n (int): The expected dimension of the matrix. This integer specifies the required number of rows and columns for init. Typical callers provide the number of channels or features (for example, the number of EEG sensors or spatial bands used to build an n-by-n covariance matrix).
    
    Returns:
        numpy.ndarray: The checked square matrix used to initialize the algorithm. This is the value of init after conversion with np.asarray and is guaranteed to have shape (n, n) when returned. The returned array may be a view of the original input or a new array depending on the input type and numpy conversion rules.
    
    Raises:
        ValueError: If the array-converted init does not have shape (n, n). The raised ValueError uses the message "Init matrix does not have the good shape. Should be ({n},{n}) but got {init.shape}." where {n} and {init.shape} are replaced by the expected dimension and the actual shape respectively.
    
    Notes:
        Version added: 0.8. This function is lightweight and intentionally only checks shape and performs array conversion; callers that require validation of symmetry, positive-definiteness, or numerical conditioning must perform those checks separately before passing the matrix to algorithms that assume SPD/HPD inputs.
    """
    from pyriemann.utils.utils import check_init
    return check_init(init, n)


################################################################################
# Source: pyriemann.utils.utils.check_metric
# File: pyriemann/utils/utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def pyriemann_utils_utils_check_metric(
    metric: dict,
    expected_keys: list = ['mean', 'distance']
):
    """Check and normalize a metric argument for pyriemann algorithms.
    
    This function validates and normalizes the "metric" argument used throughout pyRiemann algorithms that operate on symmetric/Hermitian positive definite (SPD/HPD) covariance matrices (for example in EEG/MEG/BCI pipelines and remote sensing applications described in the project README). The metric argument can be provided as a single metric name string to be applied to all algorithm steps, or as a dictionary mapping specific algorithm steps (for example "mean" and "distance") to metric name strings. A common practical use is to pass a faster metric (e.g., "logeuclid") for the "mean" computation to speed up covariance averaging, while using a more sensitive metric (e.g., "riemann") for the "distance" computation to preserve classification performance in BCI pipelines.
    
    Args:
        metric (string | dict): Metric specification for algorithm steps. If a string is provided, that single metric name is duplicated and returned for every entry in expected_keys; this is convenient when the same metric should be used for all steps. If a dict is provided, it must map each name in expected_keys to a metric name string (for example {"mean": "logeuclid", "distance": "riemann"}). The metric names returned are intended to be consumed by downstream pyRiemann components that compute means or distances on covariance matrices. Supplying a dict allows fine-grained control of computational trade-offs (speed vs. sensitivity) per algorithm step.
        expected_keys (list of str, default=["mean", "distance"]): Ordered list of step names for which a metric is required. The function returns a list of metrics whose ordering matches this list. By default the typical steps are "mean" (covariance averaging step used in estimators/classifiers) and "distance" (pairwise or classifier distance used for comparisons). Pass a different list to adapt to algorithms that use other named steps; the function will enforce that a provided dict contains all these keys.
    
    Returns:
        list of str: A list of metric name strings, one per element in expected_keys, in the same order as expected_keys. If metric was a single string, the returned list contains that string repeated len(expected_keys) times. If metric was a dict, the returned list contains the dict values corresponding to each key in expected_keys in order. The returned metric names are intended for immediate use by pyRiemann routines that perform mean estimation or distance computation on SPD/HPD covariance matrices.
    
    Raises:
        KeyError: If metric is a dict but does not contain all keys listed in expected_keys. The raised KeyError is produced with a message indicating the required expected_keys and the keys actually found in the provided dict (the code uses f"metric must contain {expected_keys}, but got {metric.keys()}").
        TypeError: If metric is neither a string nor a dict. In this case the function raises a TypeError with the message "metric must be str or dict, but got {type(metric)}" as emitted by the current implementation.
    
    Behavior and side effects:
        This function performs only validation and normalization and does not modify the provided metric dict or expected_keys list. It has no side effects beyond returning the normalized list or raising an exception on invalid input. It is commonly used in pyRiemann estimators and classifiers to ensure consistent handling of metric configuration across different algorithmic steps (for example to tune performance and classification sensitivity in BCI workflows that estimate and compare covariance matrices).
    """
    from pyriemann.utils.utils import check_metric
    return check_metric(metric, expected_keys)


################################################################################
# Source: pyriemann.utils.utils.check_version
# File: pyriemann/utils/utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def pyriemann_utils_utils_check_version(library: str, min_version: str, strip: bool = True):
    """Check minimum library version required for pyRiemann runtime or optional dependency checks.
    
    This function is used in pyRiemann (a package for processing multivariate biosignal data such as EEG/MEG for BCI applications) to verify that an external Python library dependency is present and meets a minimum semantic version requirement before enabling features that rely on that library. It attempts to import the named library, reads its __version__ attribute, optionally strips PEP440 development markers (for backward compatibility with LooseVersion-like behavior), and compares the resulting version string against the supplied minimum. The function returns a boolean indicating whether the dependency is available and sufficiently recent. Behavior and edge cases follow the implementation in pyriemann.utils.utils and are adapted from MNE-Python.
    
    Args:
        library (str): The import name of the external Python library to check (for example, "mne"). This string is passed to the built-in import mechanism (i.e., __import__), so it must be a valid importable module name. The imported module is expected to expose a __version__ attribute; if it does not, attribute access will raise an exception that is propagated to the caller. Importing the module is a side effect: module-level code may run and the module will be added to sys.modules.
        min_version (str): The minimum acceptable version string for the library (for example, "1.1"). The implementation expects a conventional version pattern composed of digits, lowercase letters, and dots (the original code refers to components matching '(\d+ | [a-z]+ | \.)'). If min_version is falsy or equal to the literal string "0.0", the function will skip the version comparison step and consider the import alone sufficient. This parameter is used to decide whether the installed library version is new enough for pyRiemann features that depend on it.
        strip (bool): If True (default), strip PEP440 development markers such as ".devN" from the library's __version__ before comparison. This makes prerelease/dev versions like "1.1.dev0" compare as if they were "1.1", preserving backward-compatible behavior similar to LooseVersion. If False, development markers are left intact and the comparison follows the module's internal parsing semantics, which may diverge from the stripped behavior. Changing this flag only affects how the version string is normalized prior to comparison.
    
    Returns:
        bool: True if and only if the library can be imported and (when a non-falsy min_version other than "0.0" is provided) the library's version is greater than or equal to min_version according to the function's internal comparison. Returns False if the import fails (ImportError) or if the imported library's version is present but is strictly less than the required min_version. Note that other exceptions (for example AttributeError if __version__ is missing, or parsing/comparison errors from the internal routines) are not converted to False and will propagate to the caller.
    """
    from pyriemann.utils.utils import check_version
    return check_version(library, min_version, strip)


################################################################################
# Source: pyriemann.utils.utils.check_weights
# File: pyriemann/utils/utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def pyriemann_utils_utils_check_weights(
    weights: numpy.ndarray,
    n_weights: int,
    check_positivity: bool = False
):
    """Check weights.
    
    Validate, normalize and (optionally) enforce strict positivity of a 1-D weight vector. This utility is used throughout pyRiemann to prepare weighting coefficients that combine objects such as covariance matrices (for example when computing weighted Riemannian means or weighted sums of SPD matrices in BCI and remote sensing pipelines). The function ensures a deterministic, normalized set of weights summing to 1 and raises clear errors for malformed inputs.
    
    Args:
        weights (None | ndarray, shape (n_weights,), default=None): Input weights provided by the caller. If None, the function produces equal weights of length n_weights (each initially 1 and then normalized). If an ndarray is provided, it must have exact shape (n_weights,) otherwise a ValueError is raised. The array is converted with numpy.asarray, so a mutable numpy array passed in may be modified in-place by the normalization step.
        n_weights (int): Number of weights expected or to generate when weights is None. This integer determines the required length of the weights vector and is used to create equal weights when weights is None.
        check_positivity (bool, default=False): When True, enforce strict positivity of every weight. If any entry is less than or equal to zero, a ValueError is raised. When False, non-positive entries are accepted (but may lead to invalid normalized results such as NaNs if the sum is zero).
    
    Returns:
        weights (ndarray, shape (n_weights,)): The validated and normalized weights vector. The returned array sums to 1 (weights /= np.sum(weights) is applied). The dtype may be promoted to a floating-point type due to normalization. This array is intended to be used as weighting coefficients (e.g., to form weighted averages of covariance matrices in EEG/MEG/EMG processing or of spatial covariance estimates in hyperspectral/SAR imaging).
    
    Behavior and side effects:
        If weights is None, an array of ones of length n_weights is created and then normalized to equal weights summing to 1. If a numpy array is provided, it is converted with numpy.asarray and then normalized in-place with the /= operator, so the original array object passed by the caller may be modified. No additional copies are guaranteed.
        If check_positivity is True and any(weights <= 0), a ValueError is raised with message "Weights must be strictly positive." If weights.shape != (n_weights,), a ValueError is raised indicating the expected and actual shapes. If the sum of weights is zero and check_positivity is False, the normalization will perform a division by zero, which will produce NaNs/Infs and may emit a RuntimeWarning; callers should avoid passing a zero-sum vector unless this behavior is acceptable.
    
    Failure modes:
        ValueError: Raised when the provided weights array does not have shape (n_weights,).
        ValueError: Raised when check_positivity is True and any weight is <= 0.
        Division by zero / invalid values: If weights sum to zero (and positivity is not enforced), normalization leads to NaNs/Infs.
    
    Version:
        Added in pyRiemann 0.4.
    """
    from pyriemann.utils.utils import check_weights
    return check_weights(weights, n_weights, check_positivity)


################################################################################
# Source: pyriemann.utils.viz.plot_bihist
# File: pyriemann/utils/viz.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def pyriemann_utils_viz_plot_bihist(
    X: numpy.ndarray,
    y: numpy.ndarray,
    n_bins: int = 10,
    title: str = "Histogram"
):
    """pyriemann.utils.viz.plot_bihist plots a bi-class histogram of predictions, distances or probabilities produced by binary classifiers or similarity scoring methods used in pyRiemann workflows (for example in BCI or remote sensing pipelines where per-epoch or per-window scores are produced and represented as two-element vectors).
    
    This function expects a 2-column array of scores (one column per class) for a set of examples and a matching 1-D array of labels. It rescales each row of X to sum to 1 (interpreting rows as unnormalized scores or probabilities), selects the scores corresponding to each of the two unique labels in y, computes a complementary score for the second class (1 - score), and draws two overlapping histograms with a vertical reference line at 0.5. The histogram bins are computed with a helper that forces the bin edges to include 0.5 so that the decision threshold is aligned with bin boundaries. The function produces a matplotlib figure (side effect) sized to (6, 5), configures axis labels and title, draws a legend titled "Classes" in the upper-left, and returns the figure object for further manipulation or saving.
    
    Args:
        X (numpy.ndarray): ndarray, shape (n_matrices, 2). Predictions, distances or probabilities for each example (row). Each row represents scores for the two possible classes produced by a classifier or a scoring function in pyRiemann pipelines (e.g., per-epoch covariance-based classifier outputs used in BCI). The function will normalize each row by its row sum (X = X / np.sum(X, axis=1, keepdims=True)) so rows are treated as relative scores/probabilities. If X.ndim != 2, the function raises ValueError("Input X has not 2 dimensions"). If X.shape[1] != 2, the function raises ValueError("Input X has not 2 classes").
        y (numpy.ndarray): ndarray, shape (n_matrices,). 1-D array of labels corresponding to each row of X. Must contain exactly two unique labels (bi-class). The function identifies classes = np.unique(y) and will raise ValueError("Input y has not 2 labels") if the number of unique labels is not 2. The labels are used to select the subset of scores for each class when plotting the two histograms.
        n_bins (int): default=10. Number of bins used to compute the histograms. Internally passed to numpy.histogram_bin_edges to compute bin edges; a helper routine then adjusts the returned bin edges so that the target value 0.5 is included exactly as one of the bin edges (this guarantees the decision threshold lies on a bin boundary). Typical use is to set a moderate integer (default 10) to visualize score distributions; non-positive or non-integer values are not validated beyond what numpy.histogram_bin_edges accepts and may raise errors from numpy.
        title (str): default="Histogram". Title string to set on the figure (ax.set(title=title)). This describes the plotted histogram in downstream reports or figures (for example "Histogram" by default, but can be set to a descriptive title such as "MDM outputs" in BCI analyses).
    
    Returns:
        fig (matplotlib figure): Figure of histogram. A matplotlib figure object (created via plt.subplots(figsize=(6, 5))) containing two overlaid histograms (alpha=0.5), a vertical dotted reference line at x=0.5, symmetric x-axis limits around 0.5 adjusted so both tails are visible, x-axis label "Rescaled predictions", y-axis label "Frequency", and a legend titled "Classes" placed at the upper left. The returned figure is the primary side effect; callers can save or further modify it (e.g., fig.savefig or ax.set_...) as required.
    
    Raises:
        ValueError: If X.ndim != 2, with message "Input X has not 2 dimensions".
        ValueError: If X.shape[1] != 2, with message "Input X has not 2 classes".
        ValueError: If np.unique(y).shape[0] != 2, with message "Input y has not 2 labels".
    
    Notes:
        Version added in pyRiemann 0.6. This utility is intended for quick visualization of binary-class score distributions produced in typical pyRiemann use cases (EEG/MEG motor imagery or ERP classification and also remote sensing covariance-based analyses) to inspect separability and calibration of the two-class outputs.
    """
    from pyriemann.utils.viz import plot_bihist
    return plot_bihist(X, y, n_bins, title)


################################################################################
# Source: pyriemann.utils.viz.plot_biscatter
# File: pyriemann/utils/viz.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def pyriemann_utils_viz_plot_biscatter(X: numpy.ndarray, y: numpy.ndarray):
    """Plot scatter of bi-class predictions.
    
    This function visualizes two-dimensional predictions, distances, or probabilities for a binary
    classification problem as a scatter plot. It is intended for use in pyRiemann workflows
    where classifiers or estimators produce two values per sample (for example, scores derived
    from covariance-matrix-based BCI pipelines such as MDM or tangent-space classifiers). The
    plot helps inspect separation between the two classes by grouping points according to the
    labels provided in y, drawing a diagonal reference line, and enforcing equal axis limits so
    that distances from the diagonal are comparable on both axes.
    
    Args:
        X (ndarray, shape (n_matrices, 2)): 2D array with one row per sample (n_matrices).
            Each row contains two numeric values produced by a classifier, distance metric,
            or probability estimator. In BCI and covariance-matrix analysis contexts (see
            the package README), X typically contains scores computed from SPD/HPD matrix
            representations of multichannel biosignals. The function requires X to be a
            two-dimensional array with exactly two columns; otherwise a ValueError is raised.
        y (ndarray, shape (n_matrices,)): Array of labels with one entry per row of X.
            Labels indicate class membership for each sample and are used to split X into
            two point sets to be plotted with different transparencies. The function expects
            exactly two unique labels in y (binary classification). Labels can be numeric
            or string-like as supported by numpy.unique. If the number of unique labels is
            not exactly two, a ValueError is raised. y must have the same first-dimension
            length as X (n_matrices); mismatch will cause indexing errors.
    
    Returns:
        fig (matplotlib figure): Matplotlib Figure object containing the scatter plot.
            Side effects: this function creates a new matplotlib Figure and Axes via
            plt.subplots(figsize=(7, 7)), plots the two classes with different alpha
            translucencies (first class alpha=1, second class alpha=0.5), adds a legend
            titled "Classes" at the upper-left, draws a dotted black diagonal reference
            line, and forces identical x and y limits so the diagonal is at 45 degrees.
            The returned Figure can be further modified by the caller or saved using
            matplotlib's savefig. The function raises ValueError for malformed inputs:
            when X is not 2-D, when X does not have exactly two columns, or when y does
            not contain exactly two unique labels. Version added: 0.6.
    """
    from pyriemann.utils.viz import plot_biscatter
    return plot_biscatter(X, y)


################################################################################
# Source: pyriemann.utils.viz.plot_cospectra
# File: pyriemann/utils/viz.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def pyriemann_utils_viz_plot_cospectra(
    X: numpy.ndarray,
    freqs: numpy.ndarray,
    ylabels: list = None,
    title: str = "Cospectra"
):
    """Plot cospectral matrices.
    
    This function visualizes a sequence of cospectral matrices (frequency-resolved cross-spectral/covariance estimates)
    as a grid of heatmap subplots. In the pyRiemann workflow for biosignals (for example EEG/MEG processing in BCI applications),
    cospectral matrices represent multichannel frequency-domain covariance structure estimated per frequency; this
    function helps inspect those matrices across frequencies to diagnose spectral structure, channel interactions,
    or preprocessing issues prior to downstream Riemannian processing or classification.
    
    Args:
        X (numpy.ndarray): Cospectral matrices with shape (n_freqs, n_channels, n_channels). Each entry X[f]
            is a square matrix for frequency index f representing the cospectral (cross-spectral / covariance)
            relationships between channels at the corresponding frequency. The function requires a 3-dimensional
            ndarray and will raise a ValueError if X.ndim != 3. These matrices are the primary input used to
            generate the heatmap grid and are commonly produced by frequency-domain estimation procedures in
            biosignal processing pipelines.
        freqs (numpy.ndarray): 1D array with shape (n_freqs,) giving the frequency values (in Hz) associated
            with the cospectra in X. The length of freqs must exactly match the first dimension of X; otherwise
            a ValueError is raised. Each element freqs[f] is used to title the corresponding subplot as
            "<freq> Hz" so the user can associate each heatmap with its frequency.
        ylabels (list of str or None): y-axis labels for channels, default=None. If provided, this should be a
            list of channel names (length n_channels). For readability the function attempts to display alternating
            labels on two reference subplots: it sets even-indexed labels on the first subplot (f == 0) and
            odd-indexed labels on the ninth subplot (f == 8) so that long channel name lists remain legible.
            When ylabels is provided the tick label font size is reduced (labelsize 7) for those subplots.
            If None (the default), y-axis tick labels are suppressed on all subplots. Note: the current layout
            only places ylabels on those two subplots to avoid clutter; if you need labels on every subplot,
            post-process the returned figure/axes or call this function separately per subplot.
        title (str): Figure title, default "Cospectra". The title is set as the figure supratitle (fig.suptitle)
            and identifies the plot in a larger report or notebook. Passing a custom title is useful to indicate
            subject/session/processing parameters in BCI or remote-sensing analyses.
    
    Behavior and side effects:
        The function creates a matplotlib.figure.Figure via plt.figure(figsize=(12, 7)) and lays out subplots
        in a grid with up to 8 columns; the number of rows is computed as ((n_freqs - 1) // 8) + 1 so that all
        frequency slices are plotted row-wise. For each frequency f it creates a subplot at position f+1 using
        plt.subplot and displays the matrix X[f] with plt.imshow using the "Reds" colormap. X-axis ticks are
        hidden for all subplots. If ylabels are provided the function sets y-ticks on the first (f == 0) and
        ninth (f == 8) subplot only, showing even and odd channel labels respectively, and reduces their font size.
        Each subplot receives a title of the form "<freq> Hz". The function mutates the current matplotlib state
        (creating figures and axes) and returns the created Figure object for further manipulation or saving.
    
    Failure modes and validation:
        The function validates inputs and will raise ValueError in two cases:
        1) If X.ndim != 3, a ValueError("Input X has not 3 dimensions") is raised because a sequence of square
           matrices is expected.
        2) If freqs.shape != (n_freqs,), a ValueError is raised indicating the number of provided frequencies does
           not match the number of cospectral matrices in X.
        Other runtime errors may occur if matplotlib is not available or if the provided arrays contain non-finite
        values; these are not explicitly checked by the function.
    
    Returns:
        fig (matplotlib.figure.Figure): The matplotlib Figure instance created by the function containing the grid
        of cospectra heatmaps. The returned figure can be used to further adjust layout, save to file (e.g.,
        fig.savefig), or embed in a report. The plotting side effect (creation/modification of matplotlib state)
        still occurs even if the returned Figure is not captured.
    """
    from pyriemann.utils.viz import plot_cospectra
    return plot_cospectra(X, freqs, ylabels, title)


################################################################################
# Source: pyriemann.utils.viz.plot_embedding
# File: pyriemann/utils/viz.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def pyriemann_utils_viz_plot_embedding(
    X: numpy.ndarray,
    y: numpy.ndarray = None,
    embd_type: str = "Spectral",
    metric: str = "riemann",
    title: str = "Embedding of SPD matrices",
    normalize: bool = True,
    max_iter: int = 50
):
    """Plot embedding of SPD matrices using manifold embedding algorithms and display the result with matplotlib.
    
    This function is used in the pyRiemann library to visualize low-dimensional embeddings of symmetric positive definite (SPD) matrices (for example covariance matrices estimated from multichannel biosignals such as EEG, MEG or EMG, or from spatial windows in hyperspectral/remote sensing applications). It supports three embedding algorithms implemented in the codebase: SpectralEmbedding, LocallyLinearEmbedding and TSNE. The function (1) fits the selected embedding estimator to the provided SPD matrices X, (2) transforms X to the low-dimensional embedding, and (3) creates and returns a matplotlib Figure containing a scatter plot of the embedding with per-class markers and a legend. The visualization is intended to help inspect class separability and structure of SPD matrix datasets used in brain-computer interface (BCI) and related pipelines described in the project README.
    
    Args:
        X (numpy.ndarray): Set of SPD matrices to embed, with shape (n_matrices, n_channels, n_channels). Each entry is an SPD matrix typically obtained by covariance estimation from multichannel time series (n_matrices is the number of trials/observations, n_channels is the number of sensors or spatial bands). The shape is used to configure certain embedding estimators (for example LocallyLinearEmbedding uses n_neighbors = X.shape[1] as the number of neighbors).
        y (None | numpy.ndarray): Labels for each matrix, with shape (n_matrices,). When provided, points belonging to the same label are plotted with the same marker and the unique labels are added to the legend. If None (default), all samples are assigned to a single class (an array of ones) so the plot shows all points as a single class.
        embd_type (str): Type of the embedding algorithm to use. Must be one of "Spectral", "LocallyLinear", or "TSNE". The function constructs the corresponding estimator as follows:
            - "Spectral": SpectralEmbedding with n_components=2 and the specified metric.
            - "LocallyLinear": LocallyLinearEmbedding with n_components=2, n_neighbors set to X.shape[1] (number of channels), and the specified metric.
            - "TSNE": TSNE with n_components=2, the specified metric, and max_iter controlling gradient descent iterations.
        metric (str): Metric name passed to the embedding estimator to measure dissimilarity between SPD matrices. Valid metric values accepted by the implemented estimators (as used in the source) are:
            - For SpectralEmbedding: "riemann", "logeuclid", "euclid", "logdet", "kullback", "kullback_right", "kullback_sym".
            - For LocallyLinearEmbedding: "riemann", "logeuclid", "euclid".
            - For TSNE: "riemann", "logeuclid", "euclid".
            The metric determines how distances between SPD matrices are computed and therefore affects the resulting embedding geometry.
        title (str): Title suffix used when setting the matplotlib axes title. The full title is formed as "{embd_type} {title}". Default is "Embedding of SPD matrices". This string is used for display only and does not affect computation.
        normalize (bool): If True (default), axis ticks for plotted axes are set to the normalized range [-1.0, -0.5, 0.0, 0.5, 1.0] for the X and Y axes and, in the TSNE branch, for the Z axis as well. This affects only the displayed tick marks and does not rescale the underlying embedding coordinates.
        max_iter (int): Maximum number of iterations used for the gradient descent of the TSNE estimator. Default is 50. This parameter is forwarded to TSNE when embd_type == "TSNE" and controls the convergence behavior and runtime of the TSNE optimizer.
    
    Returns:
        matplotlib.figure.Figure: A matplotlib Figure object containing the embedding scatter plot. Side effects: the function creates a new figure and axes (2D axes for "Spectral" and "LocallyLinear", a 3D axes for "TSNE"), sets axis labels to φ1 and φ2 (and φ3 label in the TSNE branch), sets the title to "{embd_type} {title}", disables the grid, and adds a legend listing unique class labels. The returned Figure can be shown with matplotlib.pyplot.show() or saved to file using Figure.savefig().
    
    Behavior, side effects, defaults, and failure modes:
        - The function selects the embedding estimator based on embd_type and constructs it as shown above. If embd_type is not one of "Spectral", "LocallyLinear", or "TSNE", a ValueError is raised describing the valid types.
        - After fitting and transforming X via the estimator.fit_transform(X), the function expects embd to be indexable for plotting. For "Spectral" and "LocallyLinear" the function expects embd to have at least two columns and uses embd[:, 0] and embd[:, 1] as 2D coordinates. For the "TSNE" branch the function creates a 3D axes and accesses embd entries as embd[:, 0, 0], embd[:, 0, 1], and embd[:, 1, 1] to form X, Y, Z coordinates respectively (this indexing follows the implementation in the source). If the fitted embedding array embd does not have the indexing structure expected by the selected branch, an IndexError or similar indexing error may be raised at plotting time.
        - For LocallyLinearEmbedding the estimator is created with n_neighbors = X.shape[1]. If X.shape[1] (n_channels) is not appropriate for the dataset or for the embedding algorithm (e.g., too small or larger than allowed by the estimator), the underlying estimator may raise an error; the function does not validate or modify n_neighbors beyond using X.shape[1].
        - If y is provided, its length must match the number of rows in X (n_matrices). A mismatch in sizes may lead to boolean indexing errors.
        - The function relies on matplotlib being available and will raise the usual matplotlib import/runtime errors if plotting backends are not configured.
        - The function was added in version 0.2.6 and is intended for exploratory visualization in workflows that compute covariance/SPD matrices as described in the pyRiemann README (for BCI, remote sensing, and related applications).
    """
    from pyriemann.utils.viz import plot_embedding
    return plot_embedding(X, y, embd_type, metric, title, normalize, max_iter)


################################################################################
# Source: pyriemann.utils.viz.plot_waveforms
# File: pyriemann/utils/viz.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def pyriemann_utils_viz_plot_waveforms(
    X: numpy.ndarray,
    display: str,
    times: numpy.ndarray = None,
    color: str = "gray",
    alpha: float = 0.5,
    linewidth: float = 1.5,
    color_mean: str = "k",
    color_std: str = "gray",
    n_bins: int = 50,
    cmap: str = None
):
    """Plot repetitions of a multichannel waveform for inspection and visualization in biosignal/BCI workflows.
    
    This function is used to visualize repeated multichannel time-series (for example EEG/MEG/EMG epochs used in brain-computer interface research as described in the package README). It creates a matplotlib figure with one subplot per channel and supports four display modes: plotting every repetition, plotting the mean waveform, plotting the mean with a shaded standard-deviation band, or plotting a 2D histogram of all repetitions. The figure it returns can be embedded in analysis reports, used for manual quality control of epoch preprocessing, or for illustrative figures in publications.
    
    Args:
        X (ndarray, shape (n_reps, n_channels, n_times)): Repetitions of the multichannel waveform. This 3-D array must contain n_reps repeated trials/epochs, each with n_channels spatial channels and n_times time samples. The function requires X.ndim == 3 and will raise a ValueError if this is not satisfied.
        display ({"all", "mean", "mean+/-std", "hist"}): Type of display to produce. "all" plots every repetition as a separate line (useful to inspect trial-to-trial variability). "mean" plots only the across-repetition mean waveform per channel (useful to summarize a condition). "mean+/-std" plots the mean and shades the area mean +/- standard deviation per channel (useful to visualize variability). "hist" renders a 2D histogram of all repetition samples over time (useful to view distribution/density of values across repetitions).
        times (None | ndarray, shape (n_times,), default=None): Values to display on the x-axis (time points). If None (default), the function uses np.arange(n_times) where n_times is taken from X.shape[2]. If provided, times must have shape (n_times,) matching X; otherwise a ValueError is raised. This parameter controls the horizontal axis coordinates for all plotting modes and is required to align plotted waveforms to real timestamps.
        color (matplotlib color, optional): Color used for individual repetition lines when display == "all". Default is "gray". This parameter accepts any matplotlib-compatible color specification and affects the visual emphasis of single-trial traces.
        alpha (float, optional): Alpha (transparency) used when plotting repeated lines in display == "all". Default is 0.5. Lower values increase transparency so that overlapping repetitions visually accumulate density; values outside [0.0, 1.0] will be passed to matplotlib and may lead to rendering behavior defined by matplotlib.
        linewidth (float, optional): Line width in points for the mean line when display == "mean" or "mean+/-std". Default is 1.5. Controls the visual thickness of the mean trace.
        color_mean (matplotlib color, optional): Color of the mean line when display == "mean" or "mean+/-std". Default is "k" (black). Accepts any matplotlib color specifier and determines how the aggregated signal is highlighted.
        color_std (matplotlib color, optional): Color used to fill the standard-deviation area when display == "mean+/-std". Default is "gray". This color fills the region between mean - std and mean + std per channel.
        n_bins (int, optional): Number of vertical bins for the 2D histogram when display == "hist". Default is 50. Internally the histogram is computed with bins=(n_times, n_bins) so n_bins controls the resolution along the value axis (amplitude distribution) while horizontal resolution is fixed to the number of time samples.
        cmap (Colormap or str, optional): Colormap used for the histogram when display == "hist". Default is None, which lets matplotlib choose the default colormap. This accepts either a matplotlib Colormap instance or a colormap name string and controls the color mapping of histogram density.
    
    Behavior, side effects, defaults, and failure modes:
        The function validates inputs and raises ValueError in the following cases:
            - If X.ndim != 3, a ValueError is raised with message "Input X has not 3 dimensions".
            - If times is provided but times.shape != (n_times,), a ValueError is raised with message "Parameter times has not the same number of values as X".
            - If display is not one of the accepted strings, a ValueError is raised with message "Unknown parameter display {display}" where {display} is the passed value.
        On success the function creates a matplotlib figure and one subplot per channel via plt.subplots(nrows=n_channels, ncols=1). If n_channels == 1 the single axis is wrapped into a list for uniform iteration. For display == "all", every repetition is plotted using ax.plot(times, X[i_rep, channel], c=color, alpha=alpha). For display in {"mean", "mean+/-std"}, the across-repetition mean is computed with np.mean(X, axis=0) and plotted; when "mean+/-std" is selected, the standard deviation np.std(X, axis=0) is computed and drawn as a filled area via ax.fill_between(times, mean - std, mean + std, color=color_std). For display == "hist", the function flattens repeated time and amplitude values and draws a 2D histogram via ax.hist2d(times_rep.ravel(), X[:, channel, :].ravel(), bins=(n_times, n_bins), cmap=cmap) where times_rep is created by repeating the times row n_reps times.
        Additional side effect: when n_channels > 1 the function removes x-axis tick labels for all subplots except the last (axes[:-1].set_xticklabels([])) to produce a cleaner stacked plot layout. The function relies on matplotlib.pyplot (imported as plt) and numpy (imported as np) for plotting and numeric operations.
        Defaults are as documented above. The function does not modify X or times in-place; it only reads these inputs and renders a figure.
    
    Returns:
        fig (matplotlib figure): Figure of waveform (one subplot per channel). The returned figure contains the created axes and plotted content and can be further modified, saved, or displayed by the caller (for example with plt.show() or fig.savefig()).
    """
    from pyriemann.utils.viz import plot_waveforms
    return plot_waveforms(
        X,
        display,
        times,
        color,
        alpha,
        linewidth,
        color_mean,
        color_std,
        n_bins,
        cmap
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
