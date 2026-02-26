"""
Regenerated Google-style docstrings for module 'batchgenerators'.
README source: others/readme/batchgenerators/Readme.md
Generated at: 2025-12-02T00:47:01.997443Z

Total functions: 26
"""


import numpy

################################################################################
# Source: batchgenerators.augmentations.color_augmentations.augment_brightness_additive
# File: batchgenerators/augmentations/color_augmentations.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def batchgenerators_augmentations_color_augmentations_augment_brightness_additive(
    data_sample: numpy.ndarray,
    mu: float,
    sigma: float,
    per_channel: bool = True,
    p_per_channel: float = 1.0
):
    """batchgenerators.augmentations.color_augmentations.augment_brightness_additive adds an additive brightness offset sampled from a Gaussian distribution to each channel of a single sample image tensor. This function implements the "brightness (additive)" color augmentation used in the batchgenerators data augmentation pipeline (see README). It is intended to simulate global or per-channel illumination/brightness shifts for 2D or 3D image data and is applied on a per-sample basis (not per-batch) using numpy's random number generators.
    
    Args:
        data_sample (numpy.ndarray): A single-sample image array with channel-first ordering. Expected shape is (c, x, y) for 2D or (c, x, y, z) for 3D, where c is the number of channels. This array is mutated in-place: the additive brightness offsets are added directly to data_sample. The function returns the same numpy.ndarray object after modification. For correct behavior the array should contain numeric data; to avoid loss of precision use a floating-point dtype.
        mu (float): Mean (loc) parameter of the Gaussian distribution used to sample additive brightness offsets. Each sampled offset is drawn with numpy.random.normal(loc=mu, scale=sigma). Typical usage: set mu to 0.0 to sample zero-mean brightness perturbations.
        sigma (float): Standard deviation (scale) parameter of the Gaussian distribution used to sample offsets. Must be non-negative; numpy.random.normal will raise an error for invalid scale values (negative sigma).
        per_channel (bool): If True (default), a separate Gaussian random offset is sampled for each channel that is selected for augmentation. If False, a single Gaussian random offset is sampled once and, when selected, added to all channels (i.e., identical offset across channels). This flag controls whether brightness perturbations vary across channels or are shared.
        p_per_channel (float): Probability in [0, 1] (default 1.0) that a given channel will be augmented. For each channel the code draws np.random.uniform() and compares it to p_per_channel; if the draw is <= p_per_channel the channel receives additive brightness. With the default p_per_channel=1.0 every channel is augmented (because numpy.random.uniform() < 1.0 for standard numpy behavior). Reproducibility of these random decisions can be controlled externally via numpy.random.seed.
    
    Returns:
        numpy.ndarray: The same numpy.ndarray object passed as data_sample after applying additive brightness offsets. The array shape and channel order are preserved; values of selected channels have been incremented by Gaussian samples. This function performs the augmentation in-place and also returns the mutated array to allow chaining.
    
    Behavior and side effects:
        The function uses numpy.random.normal to sample additive offsets and numpy.random.uniform to decide per-channel application. When per_channel is False a single offset is sampled once and added to all channels that pass the p_per_channel check. When per_channel is True an independent offset is sampled for each channel that passes the p_per_channel check. Because the operation is in-place, any references to the original data_sample will observe the modified values after this call.
    
    Failure modes and notes:
        If data_sample is not a numpy.ndarray or does not have a channel-first shape as expected, indexing by data_sample.shape[0] or the in-place additions will raise an exception. If sigma is negative, numpy.random.normal will raise a ValueError. If data_sample has an integer dtype, additions of floating-point offsets will follow numpy's casting rules and may result in unexpected truncation; to preserve augmentation fidelity, provide a floating-point array. Randomness and reproducibility depend on numpy's global random state (numpy.random.seed).
    """
    from batchgenerators.augmentations.color_augmentations import augment_brightness_additive
    return augment_brightness_additive(
        data_sample,
        mu,
        sigma,
        per_channel,
        p_per_channel
    )


################################################################################
# Source: batchgenerators.augmentations.crop_and_pad_augmentations.crop
# File: batchgenerators/augmentations/crop_and_pad_augmentations.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def batchgenerators_augmentations_crop_and_pad_augmentations_crop(
    data: numpy.ndarray,
    seg: numpy.ndarray = None,
    crop_size: int = 128,
    margins: tuple = (0, 0, 0),
    crop_type: str = "center",
    pad_mode: str = "constant",
    pad_kwargs: dict = {'constant_values': 0},
    pad_mode_seg: str = "constant",
    pad_kwargs_seg: dict = {'constant_values': 0}
):
    """Crops a batch of images (and optional segmentation maps) to a target spatial size and pads when the crop exceeds image bounds. This function is used by the batchgenerators data augmentation pipeline for 2D and 3D medical images to produce fixed-size spatial patches (for example to feed into a neural network). It performs either a center crop or a random crop (with an enforceable margin from image borders), preserves the data and segmentation dtypes, and pads using numpy.pad when the requested crop extends beyond the image extent. The function accepts either a numpy array shaped as (b, c, x, y) or (b, c, x, y, z) or a list/tuple of per-sample arrays where each sample has shape (c, x, y(, z)). The segmentation input, if provided, must have matching spatial dimensions and will be transformed identically to the image data so that spatial correspondence is preserved.
    
    Args:
        data (numpy.ndarray): Input image batch. Accepted inputs are a numpy.ndarray of shape (b, c, x, y) for 2D or (b, c, x, y, z) for 3D, or a list/tuple of per-sample numpy arrays where each element has shape (c, x, y(, z)). b is the batch size and c is the number of channels. This argument is required. The function checks that data is either a numpy array or a list/tuple and will raise TypeError otherwise. The dtype of the returned data patches will match data[0].dtype.
        seg (numpy.ndarray): Optional segmentation batch that will be cropped/padded in the same way as data to keep spatial alignment. Must be either None or a numpy.ndarray or list/tuple with the same spatial dimensions as data (same x,y(,z) per sample). If provided, seg[0].dtype is preserved for seg_return. If seg is not None but not a numpy array or list/tuple, a TypeError is raised. If spatial dimensions do not match those of data an AssertionError is raised.
        crop_size (int): Target size of the crop in each spatial dimension. Default is 128. If a single integer is provided it is replicated for all spatial dimensions. The implementation also accepts a tuple, list or numpy.ndarray of integers (one per spatial dimension); in that case its length must match the spatial dimensionality of the input (2 or 3) or an AssertionError is raised. The output spatial dimensions will equal this crop_size (per dimension).
        margins (tuple): Distance from each image border that random crops must respect. Default is (0, 0, 0). If a scalar is provided it is replicated for each spatial dimension. Margins are only respected when crop_type is "random". Margins can be negative, which causes the function to pad the input first (so negative margins effectively expand the field of view) and then perform cropping with margin treated as zero for those axes.
        crop_type (str): Crop strategy. Supported values are "center" (default) to extract a center crop for each sample, and "random" to extract a random crop whose lower bounds are computed with respect to margins. Any other value raises NotImplementedError. "center" uses an internal helper to compute lower bounds for a centered extraction; "random" uses an internal helper to compute randomized lower bounds subject to margins.
        pad_mode (str): Argument forwarded to numpy.pad for image data when padding is necessary (see numpy.pad pad_width and mode semantics). Default is 'constant'. This controls how image borders are filled when crop_size extends beyond the available image region.
        pad_kwargs (dict): Keyword arguments forwarded to numpy.pad for image data (for example {'constant_values': 0} by default). These are passed exactly to numpy.pad and control the padding values/behavior together with pad_mode.
        pad_mode_seg (str): Same as pad_mode but used for the segmentation array when seg is provided. Default is 'constant'. Use a segmentation-appropriate padding mode and kwargs to avoid introducing spurious labels.
        pad_kwargs_seg (dict): Same as pad_kwargs but used for the segmentation array when seg is provided. Default is {'constant_values': 0}. These kwargs are passed to numpy.pad for the segmentation padding.
    
    Returns:
        data_return (numpy.ndarray): A new numpy array of shape (b, c, ...) where the spatial dimensions equal crop_size (the replicated/intended per-dimension sizes). The dtype matches data[0].dtype. This array contains either the cropped region for each sample or the cropped+padded region (if the crop exceeded the image bounds). The function allocates this array internally (filled with zeros initially) and returns it; inputs are not modified in-place.
        seg_return (numpy.ndarray or None): If seg was provided, a numpy array shaped like data_return (b, c, ...) and with dtype matching seg[0].dtype containing the segmentation patches transformed identically to the images. If seg was None, seg_return is None.
    
    Behavioral details, side effects and failure modes:
        The function iterates over batch samples and computes per-sample lower bounds (lbs) for the crop using either a center or random strategy. It then computes upper bounds (ubs) and determines required padding per axis when lbs is negative or ubs exceeds the available image extent. Cropping is performed first (to minimize I/O for memory-mapped arrays and reduce RAM usage) and padding is applied only when necessary via numpy.pad using pad_mode/pad_kwargs for image data and pad_mode_seg/pad_kwargs_seg for segmentation. If any input type checks fail (data or seg not being numpy.ndarray/list/tuple) a TypeError is raised. If provided crop_size has the wrong length for the data dimensionality or seg and data spatial dimensions mismatch, an AssertionError is raised. If crop_type is not "center" or "random", a NotImplementedError is raised. The function allocates memory for the returned arrays proportional to batch size, channels and crop_size; extremely large crop_size or batch sizes may therefore raise MemoryError from the allocator. The function preserves input dtypes and returns new arrays (no in-place modification of the original inputs). This function is intended for use in preprocessing/augmentation pipelines (for example within batchgenerators' MultiThreadedAugmenter) to generate fixed-size patches for downstream training or inference.
    """
    from batchgenerators.augmentations.crop_and_pad_augmentations import crop
    return crop(
        data,
        seg,
        crop_size,
        margins,
        crop_type,
        pad_mode,
        pad_kwargs,
        pad_mode_seg,
        pad_kwargs_seg
    )


################################################################################
# Source: batchgenerators.augmentations.crop_and_pad_augmentations.get_lbs_for_center_crop
# File: batchgenerators/augmentations/crop_and_pad_augmentations.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def batchgenerators_augmentations_crop_and_pad_augmentations_get_lbs_for_center_crop(
    crop_size: tuple,
    data_shape: tuple
):
    """batchgenerators.augmentations.crop_and_pad_augmentations.get_lbs_for_center_crop computes the per-axis lower-bound indices for performing a center crop on spatial data; this is used by the crop-and-pad augmentations in batchgenerators (useful for 2D and 3D medical image augmentation pipelines where data arrays have batch and channel leading dimensions).
    
    Args:
        crop_size (tuple): Tuple of integers specifying the size of the desired center crop along each spatial axis (e.g., (x_crop, y_crop) for 2D or (x_crop, y_crop, z_crop) for 3D). In the context of the README and the crop/pad augmentations, this tuple represents how many voxels/pixels should be kept along each spatial dimension when extracting a centered sub-volume from the input. The function expects the order of elements in crop_size to match the spatial axes order in data_shape starting after the batch and channel dimensions.
        data_shape (tuple): Tuple describing the full shape of the array to be cropped, including batch and channel dimensions as the first two entries (for example, (b, c, x, y) for 2D or (b, c, x, y, z) for 3D, as described in the README). This value is used to determine the spatial extent of the input along each axis and therefore to compute the center-based lower-bound index for cropping for each spatial axis.
    
    Returns:
        list: A list of integers (one per spatial axis, i.e., length = len(data_shape) - 2) providing the starting indices (lower bounds) for a centered crop along each spatial axis. Each entry is computed as floor((data_shape[axis] - crop_size[axis_offset]) / 2) using integer floor division (the implementation uses Python's // operator). These indices are the positions in the input array from which a crop of size crop_size should start to be centered.
    
    Behavior and details:
        The function iterates over the spatial axes implied by data_shape (all dimensions after the first two: batch and channel) and for spatial axis i computes (data_shape[i+2] - crop_size[i]) // 2. This yields the lower-bound (start) index for a symmetric, center-aligned crop along that axis and is intended for use by center-crop or crop-and-pad augmentations in the batchgenerators augmentation pipeline.
        The function performs no in-place modifications and has no side effects; it only returns the computed list of lower bounds.
        The function does not perform extensive validation: it assumes data_shape includes batch and channel dimensions and that crop_size is indexed in the same order as the spatial dimensions of data_shape. If crop_size has fewer elements than the number of spatial axes (len(data_shape) - 2), an IndexError will be raised when the function attempts to access crop_size for a missing axis. If crop_size has more elements than the number of spatial axes, the extra elements are ignored by the loop. If any crop_size element is larger than the corresponding spatial extent in data_shape, the computed lower bound will be non-positive or negative according to Python's integer floor division behavior; downstream cropping code must handle negative starts (which commonly indicate a need to pad before cropping).
        The returned indices are integers suited for indexing numpy arrays in subsequent crop operations within data augmentation code paths (for example, cropping a sample with shape (b, c, x, y(, z)) as described in the README).
    """
    from batchgenerators.augmentations.crop_and_pad_augmentations import get_lbs_for_center_crop
    return get_lbs_for_center_crop(crop_size, data_shape)


################################################################################
# Source: batchgenerators.augmentations.crop_and_pad_augmentations.get_lbs_for_random_crop
# File: batchgenerators/augmentations/crop_and_pad_augmentations.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def batchgenerators_augmentations_crop_and_pad_augmentations_get_lbs_for_random_crop(
    crop_size: list,
    data_shape: tuple,
    margins: list
):
    """get_lbs_for_random_crop computes integer lower-bound indices for a random spatial crop.
    It is used by the crop-and-pad augmentations in batchgenerators (a medical image
    augmentation framework developed at DKFZ) to determine where to place a random crop
    inside an image (or volume) given desired crop size and safety margins. For each
    spatial dimension (x, y, and optionally z) the function returns a lower bound
    index that can be used to slice the input data array. The returned indices are
    suitable for indexing numpy arrays with shapes that follow the batchgenerators
    convention (data_shape = (b, c, x, y[, z])).
    
    Args:
        crop_size (list): A list of integers specifying the desired extent of the crop
            along each spatial dimension, in the same order as the spatial dimensions
            in data_shape (i.e., [crop_x, crop_y] for 2D or [crop_x, crop_y, crop_z] for 3D).
            In the batchgenerators context this represents the target patch size that
            downstream augmentations or model inputs expect. The function assumes one
            element per spatial axis and uses these values without type conversion.
        data_shape (tuple): The full shape of the data array that will be cropped,
            following batchgenerators convention: (b, c, x, y) for 2D or (b, c, x, y, z) for 3D.
            The spatial dimensions begin at index 2. The lengths of crop_size and margins
            are expected to match len(data_shape) - 2. This parameter provides the valid
            spatial extent within which the crop lower bounds are computed.
        margins (list): A list of non-negative integers, one per spatial dimension,
            specifying the minimum allowed distance from each image border to the crop.
            Margins are used to avoid placing crops too close to image edges (for example
            to prevent cropping out relevant anatomy). Values are interpreted in the same
            coordinate units as data_shape and crop_size.
    
    Returns:
        list: A list of integers (one per spatial dimension) representing the lower-bound
        index (start coordinate) for the random crop in each spatial axis. These indices
        are computed as follows for each spatial axis i:
        - If there is strictly more room than twice the margin (data_shape[i+2] - crop_size[i] - margins[i] > margins[i]),
          a random integer is drawn uniformly from the inclusive margin lower bound to the exclusive upper bound
          returned by numpy.random.randint(margins[i], data_shape[i+2] - crop_size[i] - margins[i]).
          Note: numpy.random.randint uses a half-open interval [low, high), so the upper bound is exclusive.
        - Otherwise (not enough room to satisfy both margins), the crop is centered along that axis by returning
          the integer division result (data_shape[i+2] - crop_size[i]) // 2.
        Each list element is intended to be used as the start index for slicing the corresponding spatial axis
        of the input numpy array.
    
    Behavior, side effects, defaults, and failure modes:
        - The function does not modify its inputs or global state except that it consumes
          numpy's random number generator when selecting random lower bounds. For reproducible
          behavior across runs, set numpy's RNG seed (e.g., numpy.random.seed(...)) before calling.
        - The function assumes len(crop_size) == len(margins) == (len(data_shape) - 2). If these
          lengths do not match, the function will attempt to index beyond the provided lists/tuple
          and will raise a standard IndexError or produce incorrect results; callers should validate
          lengths beforehand.
        - If crop_size[i] is larger than the corresponding spatial dimension data_shape[i+2],
          the centered computation (data_shape[i+2] - crop_size[i]) // 2 may yield a negative value.
          The function does not perform automatic padding or clipping; callers in the medical image
          augmentation pipeline should ensure they pad the image to at least crop_size or otherwise
          handle negative indices before using the returned lower bounds.
        - Margins should be non-negative integers. If margins[i] is large enough that
          data_shape[i+2] - crop_size[i] - margins[i] <= margins[i], the function will fall back
          to centering the crop for that axis (see above).
        - This function is lightweight and intended to be called per sample when computing
          per-sample random crops in 2D/3D augmentation pipelines used by batchgenerators.
    """
    from batchgenerators.augmentations.crop_and_pad_augmentations import get_lbs_for_random_crop
    return get_lbs_for_random_crop(crop_size, data_shape, margins)


################################################################################
# Source: batchgenerators.augmentations.crop_and_pad_augmentations.pad_nd_image_and_seg
# File: batchgenerators/augmentations/crop_and_pad_augmentations.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def batchgenerators_augmentations_crop_and_pad_augmentations_pad_nd_image_and_seg(
    data: numpy.ndarray,
    seg: numpy.ndarray,
    new_shape: list = None,
    must_be_divisible_by: list = None,
    pad_mode_data: str = "constant",
    np_pad_kwargs_data: dict = None,
    pad_mode_seg: str = "constant",
    np_pad_kwargs_seg: dict = None
):
    """Pads a data array and its corresponding segmentation array to a target minimum spatial size and/or to sizes that are divisible by given factors. This function is used in the batchgenerators data augmentation pipeline (medical image augmentation for 2D and 3D data) to ensure that images and optional segmentation maps meet minimum size requirements (new_shape) and architectural constraints (must_be_divisible_by, e.g., for UNet downsampling stages). Padding is performed per spatial dimension while preserving batch and channel dimensions and returns new arrays without modifying the inputs in place.
    
    Args:
        data (numpy.ndarray): Input image data to be padded. In the batchgenerators context this is expected to follow the internal data layout (for example 2D: (b, c, x, y); 3D: (b, c, x, y, z)). Padding is applied only to spatial dimensions (x, y, z). This array is not modified in place; a padded copy is returned. The function forwards padding behavior to numpy.pad via pad_mode_data and np_pad_kwargs_data.
        seg (numpy.ndarray): Optional segmentation array corresponding to data. Shape conventions match data (batch and channel first). If seg is None, no segmentation padding is performed and the returned segmentation value will be None. If provided, padding is applied in the same spatial layout as for data so that spatial alignment between data and seg is preserved.
        new_shape (list = None): Minimum target shape for the spatial dimensions. Interpreted as a per-spatial-dimension minimum size (min_shape). If any spatial dimension of data or seg is smaller than the corresponding entry in new_shape, that dimension is increased by padding. If data/seg is already larger along a dimension, that dimension is left unchanged. If new_shape is None, no minimum size is enforced and only must_be_divisible_by constraints (if any) are applied. When provided, new_shape length should match the number of spatial dimensions.
        must_be_divisible_by (list = None): If provided, the resulting spatial shape (after applying new_shape) will be increased if necessary so that each spatial dimension is divisible by the corresponding integer in this list. This is used for network architectures that require inputs to be divisible by a factor (for example, UNet with multiple downsampling operations). must_be_divisible_by should be a list of int with the same length as new_shape (or the number of spatial dimensions when new_shape is None). Values are treated as divisibility factors and dimensions are increased (never decreased) to satisfy divisibility.
        pad_mode_data (str = "constant"): Padding mode name passed to numpy.pad for the data array. Behaves like the mode argument of numpy.pad (for example 'constant', 'edge', 'reflect', etc.). Choose a mode appropriate for image data; default 'constant' pads with a constant value (see np.pad documentation for exact semantics).
        np_pad_kwargs_data (dict = None): Additional keyword arguments forwarded to numpy.pad when padding data (for example {'constant_values': 0}). If None, default numpy.pad behavior for the chosen pad_mode_data is used. These kwargs allow control over exact padding values and behavior.
        pad_mode_seg (str = "constant"): Padding mode name passed to numpy.pad for the segmentation array. Often different choices (for example 'edge' or 'constant') are appropriate for segmentations to avoid introducing artificial labels; the mode must be one accepted by numpy.pad. Default is 'constant'.
        np_pad_kwargs_seg (dict = None): Additional keyword arguments forwarded to numpy.pad when padding seg. If None, default numpy.pad behavior for pad_mode_seg is used. Use this to set constant_values or other numpy.pad options specific to segmentation padding.
    
    Returns:
        tuple:
            A tuple (sample_data, sample_seg) where sample_data is a numpy.ndarray of the padded data and sample_seg is the padded segmentation numpy.ndarray or None if seg was None. Batch and channel dimensions are preserved; only spatial dimensions may be increased. If must_be_divisible_by was provided, the returned spatial sizes will be at least new_shape (if given) and additionally adjusted upwards so they are divisible by the specified factors. The function returns new arrays and does not modify the input data or seg in place.
    
    Raises:
        ValueError: If new_shape or must_be_divisible_by lengths are incompatible with the number of spatial dimensions of data/seg, or if entries in must_be_divisible_by are not positive integers. Underlying numpy.pad or the internal pad_nd_image helper may raise their own exceptions for invalid pad modes or kwargs.
    """
    from batchgenerators.augmentations.crop_and_pad_augmentations import pad_nd_image_and_seg
    return pad_nd_image_and_seg(
        data,
        seg,
        new_shape,
        must_be_divisible_by,
        pad_mode_data,
        np_pad_kwargs_data,
        pad_mode_seg,
        np_pad_kwargs_seg
    )


################################################################################
# Source: batchgenerators.augmentations.resample_augmentations.augment_linear_downsampling_scipy
# File: batchgenerators/augmentations/resample_augmentations.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def batchgenerators_augmentations_resample_augmentations_augment_linear_downsampling_scipy(
    data_sample: numpy.ndarray,
    zoom_range: list = (0.5, 1),
    per_channel: bool = True,
    p_per_channel: float = 1,
    channels: list = None,
    order_downsample: int = 1,
    order_upsample: int = 0,
    ignore_axes: tuple = None
):
    """augment_linear_downsampling_scipy(data_sample, zoom_range=(0.5, 1), per_channel=True, p_per_channel=1,
                                      channels=None, order_downsample=1, order_upsample=0, ignore_axes=None)
    
    Short summary:
    Per-sample spatial downsampling and upsampling augmentation that simulates lower-resolution acquisitions and subsequent nearest/linear resampling artifacts commonly encountered in medical image preprocessing. This function is used in the batchgenerators augmentation pipeline to degrade each channel of a single sample by a randomly chosen isotropic or per-axis zoom factor and then restore the original voxel grid; it is intended for channel-first image arrays (single sample) used by batchgenerators (for example, shape (C, X, Y) for 2D or (C, X, Y, Z) for 3D). The operation helps train models to be robust against changes in image resolution and partial loss of high-frequency detail.
    
    Args:
        data_sample (numpy.ndarray): Channel-first numpy array representing a single sample to be augmented. Expected shape is (C, X, Y) for 2D or (C, X, Y, Z) for 3D in the batchgenerators data convention (note: batch dimension is not present here). The function modifies data_sample in place by replacing each augmented channel slice and also returns the modified array. Practical significance: this is the image volume that will be downsampled/upsampled per channel to simulate lower-resolution acquisitions in medical imaging augmentation pipelines.
        zoom_range (list): Sampling interval(s) for the zoom (scaling) factor. Although typed as list in the signature, the implementation accepts list, tuple or numpy.ndarray. Two valid forms are supported: (1) a two-element sequence [low, high] that defines a single interval from which a scalar zoom factor is uniformly sampled and applied isotropically to all spatial axes (or to each channel if per_channel=True); zoom values < 1 will reduce spatial resolution (downsampling) and values > 1 will increase it (upsampling); (2) a sequence of per-axis two-element sequences (e.g., [(low_x, high_x), (low_y, high_y), ...]) to sample one zoom factor per spatial axis (requires the number of inner tuples to match the spatial dimensionality). The default (0.5, 1) produces zoom factors in [0.5, 1.0], biasing toward downsampling. If a per-axis form is provided, an AssertionError will be raised when its length does not equal the number of spatial dimensions.
        per_channel (bool): If True, a new zoom factor (or per-axis zooms) is sampled independently for each channel (practical for simulating modality-specific resolution differences). If False, a single zoom (or per-axis zooms) is sampled once and reused for all channels in this sample. Default True. Behavior: when False and a per-axis zoom_range is provided, the code asserts that zoom_range length equals the spatial dimension and draws one zoom per axis; when True this sampling is repeated per channel.
        p_per_channel (float): Probability in [0,1] that a given channel is augmented (downsampled then upsampled). For each channel index in channels, a uniform random draw is compared to p_per_channel; channels for which the draw is >= p_per_channel are left unchanged. Default 1 (always augment selected channels). Practical significance: allows stochastic per-channel augmentation, increasing variability across modalities/channels.
        channels (list): List of integer channel indices that are eligible for augmentation. If None (default), all channels in data_sample (range(data_sample.shape[0])) are eligible. Practical significance: restrict augmentation to a subset of modalities (for example, only T1-weighted images in a multi-contrast MRI sample). Note that p_per_channel still gates augmentation per entry in this list.
        order_downsample (int): Interpolation order used when resampling to the smaller (downsampled) target grid. This integer is passed to the underlying resize call (e.g., 0=nearest, 1=linear). Default 1 (linear interpolation for downsampling) which preserves more structure than nearest-neighbor while remaining computationally efficient. Practical significance: choice affects aliasing and smoothing during downsampling.
        order_upsample (int): Interpolation order used when resampling back to the original grid. Default 0 (nearest-neighbor), which is useful to mimic the blockiness / label-preserving behavior that can occur when low-resolution images are resampled to higher resolution. Practical significance: using 0 preserves piecewise-constant artifacts while higher orders introduce smoother interpolations.
        ignore_axes (tuple): Tuple of spatial axis indices (0-based relative to the spatial dimensions of data_sample) that should be exempt from resampling. For any axis index in ignore_axes the target size is set equal to the original size, effectively preventing scaling along that axis. If None (default), all spatial axes are subject to the sampled zoom. Practical significance: useful to avoid resampling along certain axes (for example, slices or time) where resampling would be inappropriate. If ignore_axes contains invalid indices, the behavior depends on numpy indexing and may raise an IndexError.
    
    Returns:
        numpy.ndarray: The same numpy.ndarray object passed in as data_sample but with selected channel slices replaced by their downsampled-and-then-upsampled versions. The function returns this array for convenience, but it also performs in-place modification: callers should assume data_sample will be altered. The returned array has the same shape as the input (same channel and spatial dimensions).
    
    Behavior, side effects, defaults, and failure modes:
        - Sampling: zoom factors are sampled uniformly from the intervals specified in zoom_range. If zoom_range is a single two-element interval, a scalar zoom is sampled; if zoom_range is a per-axis sequence, one zoom per spatial axis is sampled. If per_channel is True this sampling is repeated independently for every channel that is chosen to be augmented.
        - Target shape computation: target_shape is computed as round(original_shape * zoom). If ignore_axes is provided, those axes keep their original sizes. If target_shape contains zeros (e.g., due to very small zoom values), the underlying resize routine may raise an error; users should choose zoom_range to avoid zero target sizes.
        - Resampling: implementation performs a two-step operation per augmented channel: first resize to target_shape using order_downsample and mode='edge' with anti_aliasing=False, then resize back to the original shape using order_upsample and the same mode/anti_aliasing settings. This simulates information loss at lower resolution followed by regridding to the original voxel grid.
        - In-place modification: the input data_sample is modified in place (channel slices overwritten). The function also returns the modified array.
        - Channel selection and probability: if channels is None all channel indices are eligible; each eligible channel is independently augmented with probability p_per_channel. If channels contains indices outside the valid channel range, an IndexError or unexpected behavior may occur.
        - Dimensionality and assertions: when zoom_range is provided as per-axis intervals, the code asserts that the number of intervals equals the number of spatial dimensions computed from data_sample.shape[1:]; mismatches raise AssertionError. The function expects channel-first single-sample input; supplying a batched array (with batch dimension) will lead to incorrect behavior.
        - Numeric types: the code converts channel data to float prior to resizing; dtype/precision considerations apply when converting back or using the returned array downstream.
        - Determinism: randomness comes from uniform sampling and numpy.random.uniform; to obtain deterministic behavior control the random seed in the calling context (e.g., numpy.random.seed).
    """
    from batchgenerators.augmentations.resample_augmentations import augment_linear_downsampling_scipy
    return augment_linear_downsampling_scipy(
        data_sample,
        zoom_range,
        per_channel,
        p_per_channel,
        channels,
        order_downsample,
        order_upsample,
        ignore_axes
    )


################################################################################
# Source: batchgenerators.augmentations.spatial_transformations.augment_resize
# File: batchgenerators/augmentations/spatial_transformations.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def batchgenerators_augmentations_spatial_transformations_augment_resize(
    sample_data: numpy.ndarray,
    sample_seg: numpy.ndarray,
    target_size: list,
    order: int = 3,
    order_seg: int = 1
):
    """augment_resize resizes a single-sample image (and optional corresponding segmentation) to a given spatial target size.
    
    This function is used in the spatial augmentation / resampling stage of the batchgenerators pipeline to reshape a single example's image data and, if present, its segmentation maps to a new spatial resolution. It computes the spatial dimensionality from the provided sample (by taking len(sample_data.shape) - 1, treating the first axis as the channel axis) and then uses the package's resize utilities (resize_multichannel_image and resize_segmentation) to perform interpolation. This routine returns newly allocated arrays and does not modify the input arrays in-place. It is intended for per-sample usage inside augmentation/data-loading workflows that follow the batchgenerators data conventions (per-sample image shape (c, x, y) for 2D or (c, x, y, z) for 3D, where c is the channel count). Note that segmentation resizing in this package uses the resize_segmentation helper (which, per release notes, operates with 'edge' border handling instead of constant cval borders).
    
    Args:
        sample_data (numpy.ndarray): Per-sample image data to be resized. Must be a NumPy array with shape (c, x, y) for 2D or (c, x, y, z) for 3D, where c is the number of channels and the remaining axes are spatial dimensions. The function determines spatial dimensionality from len(sample_data.shape) - 1. This argument is central to the augmentation pipeline because images are interpolated to target spatial sizes before being passed to models or further augmentations. Passing an object without a .shape attribute will raise an AttributeError. The function does not accept batched arrays that include a leading batch dimension; pass single-sample arrays.
        sample_seg (numpy.ndarray or None): Optional per-sample segmentation maps corresponding to sample_data. If not None, this must be a NumPy array with the same channel-first layout (c, x, y(, z)) as sample_data. If provided, each channel in sample_seg will be resized independently via the package's resize_segmentation function and the resulting segmentation array (with shape [c] + target_size) will be returned as the second element of the return tuple. If None, no segmentation resizing is performed and the function returns None for the segmentation. Providing a segmentation with mismatched spatial dimensionality relative to sample_data or with incompatible shape will lead to errors (e.g., index/shape errors) or incorrect results.
        target_size (int or list/tuple of int): Desired spatial size for the output image (and segmentation, if provided). If a single int is provided, it is broadcast to all spatial dimensions (e.g., for 3D sample_data, target_size=128 becomes [128, 128, 128]). If a list or tuple is provided, its length must equal the spatial dimensionality computed from sample_data (that is, len(target_size) must equal len(sample_data.shape) - 1). If the lengths do not match, an AssertionError is raised with a message instructing to match dimensionality. The values in target_size specify the output size along each spatial axis.
        order (int): Interpolation order used for resizing the image data sample_data. This integer is passed to the underlying skimage.transform.resize call via resize_multichannel_image and controls the spline interpolation order used for image intensities. The choice of order affects smoothness and aliasing of the resampled image; higher orders produce smoother results but are computationally more expensive. Defaults to 3.
        order_seg (int): Interpolation order used for resizing segmentation maps (sample_seg). This integer is passed to resize_segmentation (which in this package uses edge border handling). The interpolation order affects whether label maps remain discrete or become interpolated; choose an order appropriate for label preservation (for example, nearest-neighbor/0 is commonly used to preserve integer labels, while linear/1 may produce intermediate values that require post-processing). Defaults to 1.
    
    Returns:
        tuple: A pair (resized_data, resized_seg) where:
            resized_data (numpy.ndarray): The resized image data with channel-first layout and spatial dimensions equal to target_size (i.e., shape (c, ...) where ... is target_size). This is a newly allocated array returned by resize_multichannel_image.
            resized_seg (numpy.ndarray or None): The resized segmentation array with shape [c] + target_size if sample_seg was provided, otherwise None. Each channel is produced by resize_segmentation. Note that resized_seg may contain interpolated values depending on order_seg and may require post-processing (e.g., thresholding or argmax) to convert back to discrete labels.
    
    Raises / Failure modes:
        AssertionError: If a list/tuple is provided for target_size and its length does not match the spatial dimensionality of sample_data.
        AttributeError / TypeError: If sample_data does not have a .shape attribute or is not a NumPy-like array, operations that access shape or indexing will fail.
        ValueError / other errors from underlying resize utilities: resize_multichannel_image and resize_segmentation may raise errors for invalid target sizes, invalid interpolation orders, or incompatible input shapes; these propagate to the caller.
    
    Side effects and notes:
        This function does not modify inputs in-place; it returns newly created arrays. It relies on the module-level helper functions resize_multichannel_image and resize_segmentation to perform actual interpolation and border handling. In the batchgenerators context this function is intended to be used per sample (not on batched arrays that include a batch axis).
    """
    from batchgenerators.augmentations.spatial_transformations import augment_resize
    return augment_resize(sample_data, sample_seg, target_size, order, order_seg)


################################################################################
# Source: batchgenerators.augmentations.spatial_transformations.augment_rot90
# File: batchgenerators/augmentations/spatial_transformations.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def batchgenerators_augmentations_spatial_transformations_augment_rot90(
    sample_data: numpy.ndarray,
    sample_seg: numpy.ndarray,
    num_rot: tuple = (1, 2, 3),
    axes: tuple = (0, 1, 2)
):
    """Augment a data sample and its segmentation by rotating the spatial axes by a multiple of 90 degrees.
    
    This function is used in the spatial augmentation pipeline of batchgenerators (a medical image data augmentation library) to increase training variability for 2D and 3D image data. It selects a rotation amount (an integer count of 90-degree steps) randomly from num_rot and selects two spatial axes randomly from axes to define the rotation plane. The chosen rotation is applied to sample_data and, if present, to sample_seg using numpy.rot90 so that both image data and corresponding segmentation remain spatially aligned. This transform is appropriate for per-sample spatial augmentation of batches that follow the batchgenerators data layout (see README: 'data' should have shape (b, c, x, y) for 2D or (b, c, x, y, z) for 3D).
    
    Args:
        sample_data (numpy.ndarray): The image data array to rotate. In the batchgenerators context this is expected to follow the internal data layout: batch as first dimension and channel as second (shape (b, c, x, y) for 2D or (b, c, x, y, z) for 3D). The spatial axes referenced by the axes parameter correspond to the x, y, (z) spatial dimensions and the function adds an internal offset of +1 to these axis indices to account for the batch and channel dimensions before calling numpy.rot90. The function does not modify the input array in-place; it returns a rotated array.
        sample_seg (numpy.ndarray): The segmentation array corresponding to sample_data that should receive the identical spatial rotation. This must use the same layout as sample_data (batch and channel as first two dimensions). If sample_seg is None, no segmentation rotation is performed and None is returned in its place. If provided, the same rotation parameters (k and axes) are applied so labels remain properly aligned with the image.
        num_rot (tuple = (1, 2, 3)): Tuple of integers indicating how many times to rotate by 90 degrees. At runtime one element k is chosen uniformly at random from this tuple and numpy.rot90 is called with k (i.e., image is rotated by k * 90 degrees counter-clockwise in the chosen plane). Use a single-element tuple (e.g., (1,)) to enforce a deterministic rotation amount; control randomness via numpy's random seed if reproducibility is required.
        axes (tuple = (0, 1, 2)): Tuple of integers indexing spatial axes in the order (x, y, z) relative to the spatial dimensions (0->x, 1->y, 2->z). Two distinct values are chosen at random (without replacement) from this tuple to define the plane of rotation. Internally these indices are incremented by 1 to map to the full array dimensions (to skip the batch and channel axes) before calling numpy.rot90. The user must ensure the provided axes are compatible with the spatial dimensionality of sample_data (e.g., for strictly 2D data only axes drawn from (0, 1) are valid); otherwise numpy.rot90 will raise an IndexError.
    
    Returns:
        tuple: A tuple (rotated_data, rotated_seg) where rotated_data is a numpy.ndarray containing the rotated image data and rotated_seg is a numpy.ndarray containing the rotated segmentation or None if sample_seg was None. Both outputs preserve the original batch and channel axes ordering. No in-place modification of the original inputs is performed; the function returns newly rotated arrays (or None for the segmentation if not provided).
    
    Behavioral notes, side effects and failure modes:
        - Random choices: num_rot and the pair of axes are sampled using numpy.random.choice. If deterministic behavior is needed, set numpy's random seed or provide single-element tuples.
        - Axis indexing: the function assumes the first two array dimensions are batch and channel. It adds +1 to the chosen spatial axis indices to obtain numpy-compatible axes for numpy.rot90; this is important to ensure rotations operate on spatial dimensions rather than batch/channel axes.
        - Segmentation handling: when sample_seg is not None it receives the exact same rotation so spatial correspondence is preserved.
        - Possible exceptions: ValueError or IndexError can arise if num_rot or axes are empty tuples, if axes contain indices incompatible with the spatial dimensionality of sample_data, or if inputs are not numpy.ndarray. TypeErrors may result from invalid tuple contents.
        - Intended use: designed for data augmentation of medical imaging tasks (classification, segmentation) where preserving alignment between image and segmentation is required.
    """
    from batchgenerators.augmentations.spatial_transformations import augment_rot90
    return augment_rot90(sample_data, sample_seg, num_rot, axes)


################################################################################
# Source: batchgenerators.augmentations.spatial_transformations.augment_spatial_2
# File: batchgenerators/augmentations/spatial_transformations.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def batchgenerators_augmentations_spatial_transformations_augment_spatial_2(
    data: numpy.ndarray,
    seg: numpy.ndarray,
    patch_size: tuple,
    patch_center_dist_from_border: int = 30,
    do_elastic_deform: bool = True,
    deformation_scale: tuple = (0, 0.25),
    do_rotation: bool = True,
    angle_x: tuple = (0, 6.283185307179586),
    angle_y: tuple = (0, 6.283185307179586),
    angle_z: tuple = (0, 6.283185307179586),
    do_scale: bool = True,
    scale: tuple = (0.75, 1.25),
    border_mode_data: str = "nearest",
    border_cval_data: float = 0,
    order_data: int = 3,
    border_mode_seg: str = "constant",
    border_cval_seg: int = 0,
    order_seg: int = 0,
    random_crop: bool = True,
    p_el_per_sample: float = 1,
    p_scale_per_sample: float = 1,
    p_rot_per_sample: float = 1,
    independent_scale_for_each_axis: bool = False,
    p_rot_per_axis: float = 1,
    p_independent_scale_per_axis: float = 1
):
    """augment_spatial_2 applies combined spatial augmentations (elastic deformation, rotation, scaling, and cropping)
    to a batch of images and optionally their corresponding segmentation maps. This function is used in the
    batchgenerators library (developed for medical image augmentation) to produce augmented training samples from
    2D or 3D image batches. It expects inputs that follow the batchgenerators internal data structure: data arrays
    with shape (b, c, x, y) for 2D or (b, c, x, y, z) for 3D, and an optional seg array with matching spatial shape.
    The function returns a new batch cropped to patch_size for model training or further processing. The returned
    arrays are newly allocated; inputs are not modified in-place.
    
    Args:
        data (numpy.ndarray): Input image batch to augment. Must follow batchgenerators convention:
            shape (b, c, x, y) for 2D or (b, c, x, y, z) for 3D, where b is batch size and c is channel count.
            Each sample in the batch will be transformed independently according to the probabilistic parameters.
            This parameter is the primary image data that downstream models (classification, segmentation)
            will consume after augmentation.
        seg (numpy.ndarray): Optional segmentation batch paired with data. If provided, the same spatial
            transformations applied to data are applied to seg so labels remain aligned. Must have the same
            batch and spatial dimensions as data (e.g., (b, c, x, y) or (b, c, x, y, z)). If None, no segmentation
            output is produced. This parameter is used when training models that require label consistency
            (for example supervision with segmentation maps).
        patch_size (tuple): Spatial size of the output patch per sample. For 2D this is (x, y), for 3D (x, y, z).
            The function returns images and segmentations cropped/resampled to this spatial size. This defines
            the target spatial extent that downstream networks will receive.
        patch_center_dist_from_border (int): Default 30. Distance in pixels from the image border used when
            sampling random patch centers during cropping. In code this may also be provided as a sequence
            (list/tuple/numpy.ndarray) with one value per spatial dimension; when a single integer is given
            it is broadcasted to all spatial dimensions. Higher values constrain sampled patch centers further
            away from the image border. Used only for random cropping; ignored for deterministic center crop.
        do_elastic_deform (bool): If True, per-sample elastic deformations (smooth non-linear spatial warps)
            may be applied. Elastic deformation simulates soft-tissue and anatomical variability in medical
            images and increases robustness to local shape changes.
        deformation_scale (tuple): Range (min, max) sampled uniformly to determine the deformation scale
            as a fraction of the patch_size. The chosen scale is multiplied by the patch spatial size per-dimension
            to obtain the sigma used for the elastic field generation. Typical default (0, 0.25) allows no deformation
            up to deformations spanning 25% of the patch size.
        do_rotation (bool): If True, random rotations may be applied to each sample. Rotation is useful to
            simulate orientation variability of anatomical structures or acquisition setups.
        angle_x (tuple): Range (min, max) in radians used to sample rotation around the X axis when rotation is
            applied and when a rotation around this axis is enabled for the current sample. For 2D inputs only
            angle_x is used (rotation in the image plane). The default (0, 2*pi) corresponds to a full circle.
        angle_y (tuple): Range (min, max) in radians used to sample rotation around the Y axis for 3D inputs.
            Only used for 3D data when rotation is enabled for that axis.
        angle_z (tuple): Range (min, max) in radians used to sample rotation around the Z axis for 3D inputs.
            Only used for 3D data when rotation is enabled for that axis.
        do_scale (bool): If True, random scaling (isotropic or per-axis depending on flags) may be applied.
            Scaling simulates zoom and size variability of structures between acquisitions.
        scale (tuple): Range (min, max) sampled to determine scaling factor(s). If per-axis independent scaling is
            not selected, a single scale is sampled from this range; otherwise, independent per-axis samples may
            be drawn (see independent_scale_for_each_axis and p_independent_scale_per_axis).
        border_mode_data (str): Border handling mode for image interpolation when transformed coordinates
            sample outside the image domain (used by interpolate_img). Typical values include 'nearest' (default),
            'reflect', etc.; behavior follows the interpolation routine used internally. This controls how image
            intensity values are extrapolated beyond original boundaries.
        border_cval_data (float): Constant value used when border_mode_data selects a constant fill value.
            This is applied during interpolation for data channels when sampling outside the original image.
        order_data (int): Interpolation order (degree of the spline) used for resampling data channels. Higher
            values yield smoother interpolation; lower values preserve sharper edges. The integer controls the
            polynomial degree used by the underlying interpolation routine.
        border_mode_seg (str): Border handling mode used for segmentation interpolation. Default is 'constant'.
            Because segmentations are discrete labels, the choice of border mode affects how labels are extended
            outside the original domain.
        border_cval_seg (int): Constant value used for segmentation interpolation when border_mode_seg is 'constant'.
            Typical default 0 fills outside areas with background label.
        order_seg (int): Interpolation order for segmentations. Default 0 (nearest neighbor) is typical and preserves
            discrete label values; setting higher orders may produce non-integer values and is generally not desired
            for segmentation labels.
        random_crop (bool): If True, when no spatial modification has been applied (no elastic/rotation/scale
            sampled for a given sample) a random crop centered within patch_center_dist_from_border is performed.
            If False and no modifications are applied, a deterministic center crop (center of image) is returned.
        p_el_per_sample (float): Probability in [0, 1] of applying elastic deformation to an individual sample.
            The decision is made independently for each sample in the batch. This enables per-sample augmentation
            variability rather than whole-batch application.
        p_scale_per_sample (float): Probability in [0, 1] of applying scaling to an individual sample.
        p_rot_per_sample (float): Probability in [0, 1] of applying rotation to an individual sample.
        independent_scale_for_each_axis (bool): If True and scaling is applied, the function may sample an independent
            scaling factor per spatial axis (subject to p_independent_scale_per_axis). This simulates anisotropic
            zooming (different scale along different axes) which may be desirable for some 3D augmentations.
        p_rot_per_axis (float): Per-axis probability in [0, 1] that a rotation angle is sampled for each rotation axis.
            When less than 1.0, some axes may remain unrotated (angle = 0) even if a rotation is applied to the sample.
        p_independent_scale_per_axis (float): Probability in [0, 1] that, when do_scale is chosen, the scaling will
            be independent for each axis (if independent_scale_for_each_axis is True). This allows mixing isotropic and
            anisotropic scaling behaviors across samples.
    
    Returns:
        tuple:
            data_result (numpy.ndarray): Augmented image batch with shape (b, c, ...) where the spatial dimensions
                are equal to patch_size (i.e., (b, c, x, y) or (b, c, x, y, z) depending on patch_size). This array
                is newly allocated and contains the transformed and resampled image data ready for model input.
            seg_result (numpy.ndarray or None): Augmented segmentation batch aligned with data_result if seg was
                provided. If seg was None on input, seg_result will be None. When present, seg_result has the same
                batch and channel dimensions as seg and spatial dimensions equal to patch_size.
    
    Behavior and side effects:
        - The function does not modify the input data or seg arrays in-place; it constructs and returns new arrays.
        - For each sample, augmentations (elastic deformation, rotation, scaling) are sampled independently according
          to the p_* probabilities. If no spatial modification is applied to a sample, either a random crop (if
          random_crop=True) or a center crop will be returned to reach patch_size.
        - When elastic deformation is applied, a single deformation scale is sampled per sample (from deformation_scale)
          and converted into per-dimension sigmas by multiplying with the patch size; per-dimension magnitudes are
          sampled in a range derived from those sigmas. This produces anatomically plausible smooth deformations.
        - Interpolation for image data and segmentation maps uses order_data and order_seg respectively; using
          order_seg > 0 may produce non-integer segmentation values and is generally not recommended for discrete labels.
        - The function supports both 2D and 3D inputs; the dimensionality is inferred from patch_size length and
          the data spatial dimensions. Rotation parameters angle_y and angle_z are ignored for 2D inputs.
        - If patch_center_dist_from_border is provided as a sequence, it is used per-dimension; if a single int is
          provided it is broadcasted to all spatial dimensions.
    
    Failure modes and validation notes:
        - Mismatched shapes between data and seg (different batch size, channel count, or spatial dims) will lead
          to incorrect alignment or runtime errors. Ensure seg is None or has matching batch and spatial layout.
        - patch_size must have the same number of spatial dimensions as the input data (2D vs 3D). Supplying a
          patch_size of incorrect dimensionality will result in unexpected behavior or errors.
        - Probabilities p_* and p_rot_per_axis and p_independent_scale_per_axis are expected in the range [0, 1].
          Values outside this range will still be processed by the underlying uniform random draws but are semantically
          incorrect and may produce unexpected augmentation frequencies.
        - Interpolation orders and border modes are passed to the internal interpolate routine; incompatible or
          unsupported values will raise errors originating from that routine.
        - All randomness is drawn using numpy.random; callers who require reproducible augmentations should set
          numpy.random seed externally before calling this function.
    """
    from batchgenerators.augmentations.spatial_transformations import augment_spatial_2
    return augment_spatial_2(
        data,
        seg,
        patch_size,
        patch_center_dist_from_border,
        do_elastic_deform,
        deformation_scale,
        do_rotation,
        angle_x,
        angle_y,
        angle_z,
        do_scale,
        scale,
        border_mode_data,
        border_cval_data,
        order_data,
        border_mode_seg,
        border_cval_seg,
        order_seg,
        random_crop,
        p_el_per_sample,
        p_scale_per_sample,
        p_rot_per_sample,
        independent_scale_for_each_axis,
        p_rot_per_axis,
        p_independent_scale_per_axis
    )


################################################################################
# Source: batchgenerators.augmentations.spatial_transformations.augment_transpose_axes
# File: batchgenerators/augmentations/spatial_transformations.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def batchgenerators_augmentations_spatial_transformations_augment_transpose_axes(
    data_sample: numpy.ndarray,
    seg_sample: numpy.ndarray,
    axes: tuple = (0, 1, 2)
):
    """Augment by randomly permuting the specified spatial axes of a single sample array while preserving the channel axis and keeping segmentation aligned. This function is used in the batchgenerators spatial augmentation pipeline (medical image computing/data augmentation) to change image orientation by transposing spatial dimensions (e.g., x,y,(z)) for 2D and 3D samples, increasing variability during training. It operates on single samples where the first axis is the channel axis (shape c,x,y or c,x,y,z as commonly used inside batchgenerators per-sample augmentations).
    
    Args:
        data_sample (numpy.ndarray): Input image sample with channel-first layout and no batch dimension. Expected shape is (c, x, y) for 2D or (c, x, y, z) for 3D, where c is the number of channels (modalities). The function will permute only spatial axes (x, y, z) according to the shuffled axes argument and will return a transposed array that preserves the channel axis at index 0. This parameter is the primary data to be augmented and must be a NumPy array.
        seg_sample (numpy.ndarray): Segmentation sample that corresponds to data_sample and has the same layout (c, x, y(, z)). If provided (not None), the same transpose permutation that is applied to data_sample is also applied to seg_sample to keep image-segmentation alignment. If seg_sample is None, no segmentation is transformed and None is returned in its place. Although annotated as numpy.ndarray in the signature, the code accepts None to indicate absence of a segmentation.
        axes (tuple): Tuple (or list-like) of integer indices selecting which spatial axes to permute. These indices are expressed relative to the spatial dimensions only, starting from 0 for the first spatial axis (x). Internally, the function adds 1 to each provided index to account for the leading channel axis in the array, so an input axes=(0, 1, 2) refers to spatial axes x, y, z and maps to actual array axes (1, 2, 3). The default (0, 1, 2) is appropriate for 3D spatial data; for 2D use (0, 1). The order of axes is shuffled randomly with numpy.random.shuffle, so results are nondeterministic unless NumPy's random seed is fixed for reproducibility.
    
    Returns:
        tuple: A tuple (data_sample, seg_sample) where data_sample is the transposed numpy.ndarray with the same dtype as the input and seg_sample is the correspondingly transposed numpy.ndarray if a segmentation was provided, otherwise None. The function does not modify the original objects in-place in a documented way (NumPy.transpose typically returns a view), but callers should treat the returned arrays as the augmented outputs to be used downstream in the batchgenerators pipeline.
    
    Behavior and side effects:
        The function constructs a permutation of the full array axes that leaves the channel axis (index 0) in place and replaces the positions of the specified spatial axes with a randomized ordering of those axes. It then calls ndarray.transpose with that permutation and returns the transposed arrays. Because numpy.random.shuffle is used, the permutation is random; set NumPy's RNG state for deterministic behavior across runs. If seg_sample is provided, it is transposed with the identical permutation to maintain spatial correspondence with data_sample.
    
    Failure modes and validation:
        The function asserts that the adjusted axis indices are within the valid range for the input array. Concretely, after adding 1 to the supplied axes (to account for the channel axis), the maximum of these adjusted indices must be less than or equal to len(data_sample.shape), otherwise an AssertionError is raised with the message "axes must only contain valid axis ids". Supplying axes that refer to non-existent spatial dimensions (for example passing a z index for 2D data) will trigger this assertion. Also ensure data_sample follows the expected channel-first layout (c, x, y(, z)); otherwise axis interpretation will be incorrect.
    """
    from batchgenerators.augmentations.spatial_transformations import augment_transpose_axes
    return augment_transpose_axes(data_sample, seg_sample, axes)


################################################################################
# Source: batchgenerators.augmentations.spatial_transformations.augment_zoom
# File: batchgenerators/augmentations/spatial_transformations.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def batchgenerators_augmentations_spatial_transformations_augment_zoom(
    sample_data: numpy.ndarray,
    sample_seg: numpy.ndarray,
    zoom_factors: int,
    order: int = 3,
    order_seg: int = 1
):
    """batchgenerators.augmentations.spatial_transformations.augment_zoom
    Zoom (resample) a single multi-channel image and its optional segmentation by integer scaling factors.
    This function is part of the batchgenerators spatial augmentation tools used in medical image computing workflows (see repository README). It computes a target spatial size by multiplying the per-channel spatial shape of sample_data by zoom_factors, resamples the image channels with a continuous interpolation order suitable for image intensity data, and resamples each segmentation channel with a discrete interpolation order suitable for label maps. The function is intended to be used on a single sample (not a batch) with shape conventions used across the project: per-sample image arrays are expected to have shape (c, x, y) for 2D or (c, x, y, z) for 3D, where c is the number of channels (modalities). It relies on the repository helpers resize_multichannel_image and resize_segmentation (which in this code base follow the skimage.transform.resize semantics and, per project release notes, use 'edge' mode for segmentation resizing).
    
    Args:
        sample_data (numpy.ndarray): Input image to be zoomed. Must be a numpy.ndarray representing a single sample with channel-first spatial layout: (c, x, y) for 2D or (c, x, y, z) for 3D. Each channel is treated as an intensity image and will be resampled with interpolation order given by order. In the batchgenerators pipeline this corresponds to the per-sample 'data' entry (one sample, multiple modalities/channels) used for spatial augmentations; the function computes dimensionality as len(sample_data.shape) - 1 and derives the spatial shape from sample_data.shape[1:].
        sample_seg (numpy.ndarray): Optional segmentation corresponding to sample_data. If not None, it must have the same layout and spatial dimensionality as sample_data: (c, x, y) or (c, x, y, z), where c may be 1 or more segmentation channels. Each segmentation channel is resampled separately using a discrete interpolation order appropriate for label maps (order_seg). If sample_seg is None, no segmentation output is produced and the function returns None in the segmentation slot.
        zoom_factors (int): Integer scaling factor to multiply each spatial axis by, or an iterable (list/tuple) of ints with length equal to the spatial dimensionality (2 for 2D, 3 for 3D). If a single int is provided, the same factor is applied to all spatial axes. The function computes the target spatial size as np.round(original_spatial_shape * zoom_factors_here).astype(int). zoom_factors therefore controls upsampling (factor > 1) and downsampling (factor < 1 if provided as float by caller, but the implementation expects integer factors; providing non-positive or non-integer values is not supported by the documented usage and may lead to unexpected results or errors from the underlying resize routines).
        order (int, optional): Interpolation order passed to the image resizer for sample_data (follows skimage.transform.resize semantics). Default is 3. Use higher orders for smoother intensity interpolation; in medical imaging augmentations this is typically set to cubic (3) for image modalities. Be aware that higher orders increase computation and can introduce ringing artifacts at sharp boundaries.
        order_seg (int, optional): Interpolation order passed to the segmentation resizer for sample_seg (follows skimage.transform.resize semantics). Default is 1. Use order_seg=0 or 1 for label/segmentation maps to avoid creating non-integer labels; the repository default and recommended setting for segmentation resizing is low-order interpolation (order_seg=1) and the helper resize_segmentation enforces appropriate handling of label maps (see project notes about using 'edge' mode and not using a cval for segmentation).
    
    Returns:
        tuple: A tuple (resized_data, resized_seg) where:
            resized_data (numpy.ndarray): The resampled image array with shape (c, ...) where the spatial dimensions are the computed target sizes (rounded integers). This is a newly produced array resulting from resize_multichannel_image; original sample_data is not modified in-place by this function.
            resized_seg (numpy.ndarray or None): If sample_seg was provided, this is the resampled segmentation array with shape (c, ...) matching resized_data spatial dimensions and produced by calling resize_segmentation for each segmentation channel; otherwise None. Segmentation resizing uses discrete-friendly interpolation (order_seg) and the project-recommended handling for segmentation borders (see resize_segmentation behavior in repository).
    
    Behavior, side effects, defaults, and failure modes:
        - Dimensionality is inferred as len(sample_data.shape) - 1. The function expects channel-first single-sample arrays (c, spatial...). Supplying arrays with different layout will produce incorrect results.
        - If zoom_factors is a scalar int, the same integer factor is applied to every spatial axis. If zoom_factors is a list/tuple, its length must equal the spatial dimensionality; otherwise an AssertionError is raised with the message requiring matching dimensionality.
        - Target sizes are computed with np.round(shape * zoom_factors_here).astype(int). This rounding may cause loss of exact proportionality with non-integer intermediate results; very small targets (zero or negative sizes) will cause downstream errors from the resize helpers.
        - The function delegates interpolation to resize_multichannel_image (for intensity images) and resize_segmentation (for label maps). Errors, exceptions, or warnings raised by these helpers (for example due to invalid target shapes, non-finite values, unsupported dtypes, or incompatible input layout) will propagate to the caller.
        - The function returns newly allocated arrays; it does not guarantee in-place modification of the inputs. Caller code should treat returned arrays as the canonical resized sample and segmentation for further augmentation or model input.
        - The implementation assumes zoom_factors are intended as multiplication factors (integers recommended). Supplying float factors, negative factors, or non-numeric types is outside the documented usage and may raise TypeError/ValueError from numpy or the resize helpers.
        - For segmentation correctness in medical image pipelines, use low-order interpolation (order_seg) and be aware that resizing can alter small structures; prefer appropriate factor choices and, if required, consider post-processing to enforce label consistency.
    """
    from batchgenerators.augmentations.spatial_transformations import augment_zoom
    return augment_zoom(sample_data, sample_seg, zoom_factors, order, order_seg)


################################################################################
# Source: batchgenerators.augmentations.utils.convert_seg_image_to_one_hot_encoding
# File: batchgenerators/augmentations/utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def batchgenerators_augmentations_utils_convert_seg_image_to_one_hot_encoding(
    image: numpy.ndarray,
    classes: numpy.ndarray = None
):
    """Convert a segmentation label map to a one-hot encoded array with the class/channel axis first.
    
    This function is used in the batchgenerators data-augmentation and preprocessing pipeline to convert a single-sample segmentation label map (a spatial label image) into a one-hot representation where each output channel corresponds to a semantic class. In the context of the README and the medical image augmentation use cases, this is typically applied to a per-sample segmentation (for example a 2D label map of shape (x, y) or a 3D label map of shape (x, y, z)) prior to spatial or intensity augmentations or before feeding the labels into a model that expects channel-first one-hot targets. If classes is None, the set and order of channels is determined from the unique values present in image using numpy.unique. The function performs elementwise equality comparisons (image == c) to generate each class mask; therefore exact value matching semantics apply (this is important for floating-point label maps).
    
    Args:
        image (numpy.ndarray): Input segmentation label map. This is an N-dimensional numpy array representing spatial labels for a single sample (typical shapes in this repository are 2D: (x, y) or 3D: (x, y, z)). The array stores discrete label values for each spatial location. The function treats every axis of image as spatial (no implicit batch or channel axis is handled); when using batched data or data with an explicit channel axis, call this function per sample or reshape accordingly. The data type of image is preserved in the output array and is used for the one-hot values (0 and 1 stored with image.dtype).
        classes (numpy.ndarray): Array of label values that define the class channels and their order in the output. If provided, classes must be a numpy.ndarray (or an array-like that can be interpreted as such) containing the distinct label values to encode. The output will contain len(classes) channels in the same order as classes. If classes is None (default), the function computes classes = numpy.unique(image) and uses that sorted set of unique labels from the input; in that case the channel order follows numpy.unique's output. If classes contains values that do not appear in image, the corresponding output channels will be all zeros.
    
    Returns:
        numpy.ndarray: One-hot encoded array with shape (n_classes, ...) where n_classes == len(classes) and ... corresponds to the spatial shape of the input image (image.shape). The returned array uses the same dtype as the input image (dtype == image.dtype) and contains values 0 and 1 produced by equality comparison (image == c) for each class c. No in-place modification of the input image occurs; the function allocates and returns a new numpy array. Failure modes include high memory usage for large images or many classes (may raise MemoryError), and incorrect results if the label values are floating-point and exact equality is inappropriate for the labeling scheme.
    """
    from batchgenerators.augmentations.utils import convert_seg_image_to_one_hot_encoding
    return convert_seg_image_to_one_hot_encoding(image, classes)


################################################################################
# Source: batchgenerators.augmentations.utils.convert_seg_image_to_one_hot_encoding_batched
# File: batchgenerators/augmentations/utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def batchgenerators_augmentations_utils_convert_seg_image_to_one_hot_encoding_batched(
    image: numpy.ndarray,
    classes: list = None
):
    """Convert a batched segmentation label image to a one-hot encoded batch along a new class/channel axis.
    
    This function is used in the batchgenerators augmentation pipeline to transform batched segmentation maps (for example the 'seg' entry in the data dictionary used by DataLoaderBase and MultithreadedAugmenter) into a one-hot representation required by many downstream tasks such as multi-class loss computation, network training, or metric calculation. It expects a batch of single-channel label maps and produces an output where each class is represented by a separate channel. If classes is None, the set of classes is inferred from the labels present in the provided batch.
    
    Args:
        image (numpy.ndarray): A batched segmentation array containing integer (or otherwise comparable) class labels for each spatial location. Expected shapes are (b, x, y) for 2D batches or (b, x, y, z) for 3D batches where b is the batch size and x, y, (z) are spatial dimensions. In the batchgenerators data convention this corresponds to a 'seg' tensor without an explicit channel axis (single segmentation map per sample). The function reads image values and does not modify the input array in-place.
        classes (list): A list of class label values (in the same value space as elements of image) that defines the ordering of output channels. If classes is None (default), the function computes classes = np.unique(image) across the whole batch and uses that ordering. If classes contains labels not present in image, the corresponding output channel will be all zeros. classes must be provided as a list (or similar iterable) when a specific channel order is required for downstream tasks (for example, ensuring class channel ordering matches network output or loss expectations).
    
    Returns:
        numpy.ndarray: A new numpy array with dtype equal to image.dtype and shape [b, len(classes), x, y] for 2D or [b, len(classes), x, y, z] for 3D. The returned array is a binary one-hot encoding per sample and per class: for batch index b and class index i corresponding to class label c = classes[i], out[b, i, ...] == 1 where image[b, ...] == c and 0 elsewhere. The function returns a freshly allocated array and does not modify the input image.
    
    Behavior, defaults, and failure modes:
        - When classes is None the unique label values are determined by np.unique(image) computed over the entire batch; the ordering returned by np.unique determines output channel ordering.
        - The output dtype is set to image.dtype; the literal value 1 is written into the output array and will be cast to that dtype. For numerical stability and expected downstream behavior, it is typical to use integer or boolean-like dtypes for segmentation masks, but the function does not enforce a specific dtype.
        - The implementation iterates over the batch dimension and the provided classes (for b in range(batch) and for each class), so memory use is proportional to batch size and number of classes and CPU time increases with both. For very large batches or many classes this may be a performance bottleneck.
        - The function expects the input layout described above. If image has an unexpected shape (for example includes an explicit channel axis as in (b, c, x, y) or uses a different axis ordering), the output will be incorrect for the intended purpose; such mismatches may lead to logically wrong encodings though no explicit exception is raised by this code. If image does not have at least one batch dimension (i.e., image.shape[0] is missing), indexing will raise an IndexError.
        - If classes is provided but not a list-like iterable, behavior will depend on whether len(classes) and iteration work on the provided object; supplying a non-iterable will raise a TypeError.
    """
    from batchgenerators.augmentations.utils import convert_seg_image_to_one_hot_encoding_batched
    return convert_seg_image_to_one_hot_encoding_batched(image, classes)


################################################################################
# Source: batchgenerators.augmentations.utils.convert_seg_to_bounding_box_coordinates
# File: batchgenerators/augmentations/utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def batchgenerators_augmentations_utils_convert_seg_to_bounding_box_coordinates(
    data_dict: dict,
    dim: int,
    get_rois_from_seg_flag: bool = False,
    class_specific_seg_flag: bool = False
):
    """Convert per-pixel segmentation maps into bounding box annotations and per-lesion metadata used by detection/instance tasks (e.g., Mask R-CNN) in the batchgenerators augmentation pipeline.
    
    This function is used in the medical-image data-augmentation context (see README) to translate pixel-wise lesion annotations into a set of bounding boxes, per-lesion binary masks and class labels that are easier to consume by object-detection style networks or further augmentation steps. It inspects each sample's segmentation map, extracts connected lesion regions or label-encoded ROIs, computes axis-aligned bounding boxes with a one-voxel margin, and produces outputs placed back into the input data_dict so downstream transforms/augmenters can access them.
    
    Args:
        data_dict (dict): Input data dictionary as returned by the batch generator. Must contain a segmentation entry under the key 'seg' such that data_dict['seg'][b] yields a spatial label map for sample b. The repository README defines segmentation arrays typically as shape (b, c, x, y) for 2D or (b, c, x, y, z) for 3D; this function expects the per-sample slice data_dict['seg'][b] to be a single-channel spatial label map (2D: (x, y) or 3D: (x, y, z)). If your pipeline uses a channel dimension, provide a single-channel map (or the appropriate channel slice) in data_dict['seg'] before calling this function. Depending on get_rois_from_seg_flag (see below) data_dict is also expected to contain a 'class_target' entry: when get_rois_from_seg_flag is False, data_dict['class_target'][b] must be an iterable of per-label class targets (one element per integer ROI label present in data_dict['seg'][b]); when get_rois_from_seg_flag is True, data_dict['class_target'][b] is expected to be a single class label that applies to all connected components found in the binary segmentation.
        dim (int): Operating spatial dimensionality. Accepted values are 2 or 3. If dim == 2, bounding boxes are returned as (y1, x1, y2, x2). If dim == 3, bounding boxes are returned as (y1, x1, y2, x2, z1, z2). The function uses this to decide whether to compute and include the z-interval for each lesion.
        get_rois_from_seg_flag (bool): Default False. Controls how ROIs are interpreted from the provided segmentation:
            - False (default): Expect the segmentation to be a label map where each lesion/ROI has a unique integer label (1..n). The code will treat label value i as lesion index i and will read the corresponding per-lesion class target from data_dict['class_target'][b][i-1].
            - True: Expect the segmentation to be a binary foreground map where individual lesions are not uniquely labeled. In this case the function runs a connected-component labelling algorithm on data_dict['seg'][b] to split the foreground into individual ROIs on the fly; all resulting connected components are assigned the same class target derived from the (single) data_dict['class_target'][b] value of that sample. When this flag is True the original input 'class_target' entry for the dataset may be removed/overwritten by the function.
        class_specific_seg_flag (bool): Default False. Controls the form of the output segmentation stored back into data_dict['seg']:
            - True: Produce a class-specific label map where pixel values are remapped to class indices (background remains 0, lesion pixels are set to class_label where class_label equals the per-lesion class target + 1 as used internally by this function). This preserves multi-class information in the segmentation output.
            - False (default): Produce a binary foreground/background segmentation map where every non-background pixel is set to 1. Use this when downstream components expect only foreground masks.
    
    Behavior and side effects:
        - For every sample b in data_dict['seg'], the function inspects non-zero pixels. If no foreground pixels are present, it appends an empty bounding-box list for that sample, a single empty/zero roi_mask slice and a class label array containing [-1] for that sample to indicate "no lesion".
        - When lesions are found, the function either:
            - uses the integers in the provided label map (get_rois_from_seg_flag=False) to separate lesions, or
            - applies connected-component labelling (get_rois_from_seg_flag=True) to split a binary map into separate ROIs.
        - For each lesion/ROI the function computes the minimal axis-aligned bounding box by taking the min and max indices over the lesion voxel coordinates and then subtracting 1 from minima and adding 1 to maxima (i.e., an extra one-voxel margin). These coordinate values are not clipped inside the function; minima can therefore become negative if the lesion touches the image border after the subtraction step.
        - ROI class labels stored in the returned per-box labels are computed as data_dict['class_target'][b][rix] + 1 in the code. This means that the function shifts provided class targets by +1 so that 0 is reserved as the background class in outputs.
        - The function casts roi masks to numpy.uint8 before placing them into data_dict.
        - If get_rois_from_seg_flag is True the function removes the original input 'class_target' entry (data_dict.pop('class_target', None)) before constructing and writing the new per-box class list; in all cases the function overwrites/sets data_dict['class_target'] at the end with the per-sample list of per-box class labels generated during processing.
        - The function modifies data_dict in place by adding/replacing keys described in Returns below; it also returns the modified data_dict for convenience.
    
    Failure modes and input validation:
        - Missing keys: If data_dict lacks 'seg' the function will raise a KeyError. If get_rois_from_seg_flag is False and data_dict['class_target'] is missing or does not provide a per-label sequence matching the integer labels present in the segmentation, behavior is undefined and indexing errors or incorrect label assignments may occur.
        - Dimensionality mismatch: If the provided per-sample segmentation arrays do not have the expected number of spatial dimensions for the specified dim (2 or 3), numpy indexing/argwhere usage may raise errors.
        - Coordinate clipping: The function does not clamp bounding box coordinates to image extents. If lesions touch the image border the computed coordinates may be negative or exceed image size; callers should clamp coordinates if required by downstream consumers.
        - Type expectations: Outputs are numpy arrays (roi masks are cast to dtype uint8). The function does not convert other entries in data_dict apart from those listed below.
    
    Returns:
        dict: The same input data_dict object, modified in place and returned for convenience. New/updated keys added to data_dict are:
            - 'bb_target': numpy.array of length batch_size where each element is an array of bounding boxes for that sample. Per-sample shape is (n_boxes, 4) for dim==2 or (n_boxes, 6) for dim==3, with coordinate order (y1, x1, y2, x2[, z1, z2]). If no boxes are found for a sample an empty list/array is stored at that position.
            - 'roi_masks': numpy.array of per-sample lists/arrays of binary segmentation masks for each detected lesion. Each mask is a binary spatial array with dtype uint8. Per-sample structure is (n_boxes, y, x[, z]) or, for empty cases, a single zero-like slice as created in the function.
            - 'class_target': numpy.array of per-sample lists/arrays with class labels for each bounding box produced by this function. For samples without lesions this will be an array containing [-1]. Note that class labels produced by the function have been shifted by +1 compared to the input convention so that 0 denotes background in the returned outputs.
            - 'seg': the returned segmentation map out_seg which is either a class-specific remapped label map (if class_specific_seg_flag is True) or a binary foreground map (if class_specific_seg_flag is False). This replaces the original data_dict['seg'] entry.
        The function returns the modified data_dict so it can be threaded in pipelines, but the primary effect is the in-place augmentation of data_dict with the keys above.
    """
    from batchgenerators.augmentations.utils import convert_seg_to_bounding_box_coordinates
    return convert_seg_to_bounding_box_coordinates(
        data_dict,
        dim,
        get_rois_from_seg_flag,
        class_specific_seg_flag
    )


################################################################################
# Source: batchgenerators.augmentations.utils.elastic_deform_coordinates_2
# File: batchgenerators/augmentations/utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def batchgenerators_augmentations_utils_elastic_deform_coordinates_2(
    coordinates: numpy.ndarray,
    sigmas: list,
    magnitudes: list
):
    """Elastic deformation of a coordinate grid using smoothed random fields in the Fourier domain, intended for spatial augmentations in the batchgenerators pipeline (e.g., to simulate soft-tissue or other realistic deformations of image sampling grids). The function generates per-dimension random noise, applies a Fourier-domain Gaussian low-pass filter (via fourier_gaussian and FFT/IFFT), scales the resulting deformation fields by the provided magnitudes, normalizes them to the supplied magnitude range, and returns coordinates displaced by these deformation offsets. This is used downstream to map image sampling coordinates for elastic spatial augmentation in 2D/3D data augmentation workflows described in the README.
    
    Args:
        coordinates (numpy.ndarray): Input coordinate grid to be deformed. The first axis indexes coordinate dimensions (n_dim) and the remaining axes define the spatial grid shape; for example a 2D grid could have shape (2, x, y). Each entry is interpreted as the coordinate values along that dimension. The function computes offsets with the same spatial shape and returns coordinates + offsets. The array is treated as floating point during processing and the returned array will be floating point.
        sigmas (list): Controls the amount of Gaussian smoothing applied in the Fourier domain for each deformation component. Each element corresponds to the standard deviation (in frequency-space units used by fourier_gaussian) applied to the random field for one coordinate dimension. Per the implementation, if sigmas is not a tuple or list it will be broadcast into a list of length (len(coordinates) - 1); if a list is provided its length is expected to match the per-dimension requirements of the code (see Failure modes). Smoothing reduces high-frequency components of the random field and therefore controls the spatial scale of the deformation.
        magnitudes (list): Scaling factors for the final deformation amplitude for each coordinate dimension. Each element determines how strongly the normalized deformation field for that dimension is scaled before being added to coordinates. If magnitudes is not a tuple or list it will be broadcast into a list of length (len(coordinates) - 1) per the implementation. The code normalizes each sampled deformation field by its maximum absolute value and then scales it by the corresponding magnitude (numerical stability uses a +1e-8 term to avoid division by zero).
    
    Returns:
        numpy.ndarray: New coordinates of the same shape as the input coordinates array containing the original coordinates plus the computed elastic deformation offsets (offsets + coordinates). This array is a floating point ndarray representing the deformed sampling grid and can be used to resample images or segmentation maps. Note that the operation is stochastic (it uses numpy.random.random); to obtain reproducible results, the caller must control numpy's random seed before calling this function.
    
    Behavior, side effects, defaults, and failure modes:
        The function generates a random noise field per coordinate dimension using numpy.random.random, transforms it to the Fourier domain with np.fft.fftn, applies fourier_gaussian for low-pass filtering, and returns to the spatial domain with np.fft.ifftn(). The deformation for each dimension is normalized by its maximum absolute value and scaled by the corresponding magnitude; a small constant (1e-8) prevents division-by-zero if the maximum is zero. The implementation broadcasts non-list/tuple sigmas and magnitudes into lists of length len(coordinates) - 1; providing lists whose lengths do not match the expectations of the implementation may lead to IndexError or incorrect behavior. The function uses FFTs and intermediate arrays and can therefore consume significant memory for large spatial grids. The randomness is governed by numpy's global RNG; seed it externally for deterministic behavior. The function relies on the availability of fourier_gaussian in the execution environment.
    """
    from batchgenerators.augmentations.utils import elastic_deform_coordinates_2
    return elastic_deform_coordinates_2(coordinates, sigmas, magnitudes)


################################################################################
# Source: batchgenerators.augmentations.utils.get_organ_gradient_field
# File: batchgenerators/augmentations/utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def batchgenerators_augmentations_utils_get_organ_gradient_field(
    organ: numpy.ndarray,
    spacing_ratio: float = 0.10416666666666667,
    blur: int = 32
):
    """Calculates a 3D gradient vector field around a binary organ segmentation to support anatomy-informed data augmentation (used, for example, to simulate soft-tissue deformations in medical image augmentation pipelines).
    
    Args:
        organ (numpy.ndarray): Binary organ segmentation volume. This is the input mask from which the spatial gradient is computed. The array is converted to floating point internally (organ.astype(float)) so the original array is not modified. The function expects a 3D volume with three spatial axes; if a 2D array is provided, numpy.gradient will return only two gradient components and unpacking into three outputs will raise an error.
        spacing_ratio (float = 0.10416666666666667): Ratio of the axial spacing to the in-plane slice thickness (axial spacing / slice thickness). This scalar is required to correctly scale the gradient along the axial axis so that the resulting vector field reflects the physical anisotropy of voxel spacing in typical medical image volumes. The default equals 0.3125/3.0 (approximately 0.1041667) as used in the original implementation. Values <= 0 are invalid and will produce incorrect gradients or may cause the gaussian filter to behave unexpectedly.
        blur (int = 32): Kernel constant that controls the amount of Gaussian smoothing applied to the binary segmentation before computing gradients. Internally, gaussian_filter is called with sigma=(blur * spacing_ratio, blur, blur). Larger values increase the smoothing radius, producing a smoother, lower-magnitude gradient field concentrated over a wider boundary region; smaller values preserve sharper edges. Must be a positive integer; non-positive or non-integer values may lead to unexpected results.
    
    Returns:
        tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]: Three floating-point arrays (t, u, v) containing the components of the gradient field. Each returned array has the same shape as the input organ. t is the gradient along the axial axis (axis 0) and is scaled by spacing_ratio to account for axial/in-plane spacing differences; u and v are the gradients along the remaining two in-plane axes (axes 1 and 2). These components can be used as a vector field for anatomy-informed augmentations (e.g., to derive deformation directions or boundary-normal fields). No in-place modification of the input is performed. Potential failure modes include providing inputs with fewer than three spatial dimensions (causes unpacking error), arrays containing NaNs/Infs (which propagate through gaussian_filter and gradient), or invalid spacing_ratio / blur values as noted above.
    """
    from batchgenerators.augmentations.utils import get_organ_gradient_field
    return get_organ_gradient_field(organ, spacing_ratio, blur)


################################################################################
# Source: batchgenerators.augmentations.utils.mask_random_squares
# File: batchgenerators/augmentations/utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def batchgenerators_augmentations_utils_mask_random_squares(
    img: numpy.ndarray,
    square_size: int,
    n_squares: int,
    n_val: float,
    channel_wise_n_val: bool = False,
    square_pos: tuple = None
):
    """Masks a given number of squares in an image. This utility is part of the batchgenerators augmentation toolkit (used in medical image computing and general image augmentation pipelines) and is intended to simulate localized occlusions or missing regions by repeatedly applying square masks to an input numpy array. The function repeatedly delegates to mask_random_square to place and fill each square and returns the augmented image array for use in training/validation pipelines that expect batchgenerators-style image arrays.
    
    Args:
        img (numpy.ndarray): Image array to be augmented. In the batchgenerators context this is typically an array with a channel dimension (for example a single-sample image with shape (c, x, y) or a batch with shape (b, c, x, y) for 2D or (b, c, x, y, z) for 3D as described in the README). This argument is the data that will receive the square masks; the returned array contains the same data structure with n_squares masked regions applied. The function forwards this array to mask_random_square n_squares times, so callers should expect the returned array to reflect all applied masks. Depending on the implementation of mask_random_square, masking may modify the array in-place or return a new array; callers should use the returned value as the canonical augmented image.
        square_size (int): Size of each square mask expressed in image voxels/pixels along each spatial dimension. In practice this controls the spatial extent of simulated occlusions (for example, a square_size of 16 will produce 16x16 (2D) or 16x16x16 (3D) masked blocks). Must be an integer; providing a size larger than image spatial dimensions will generally cause mask_random_square to clip or raise an error depending on its implementation.
        n_squares (int): Number of squares to place in the image. The function applies mask_random_square exactly n_squares times in a loop, producing that many masked regions (positions may overlap). Typical use is to increase robustness of models by simulating multiple occlusions per sample. If n_squares is zero or negative, no masks will be applied; non-integer or invalid types will raise an error when used by mask_random_square.
        n_val (float): Fill value used when creating each square mask. This is interpreted as the numerical value written into the masked voxels/pixels (for example 0.0 to zero out regions or other floats to simulate noise/constant intensity). The exact handling of n_val in multi-channel data and how it is broadcast is delegated to mask_random_square; here it is passed through unchanged.
        channel_wise_n_val (bool): If False (default) the same n_val is used for all channels when filling masked squares. If True, instructs mask_random_square to perform channel-wise handling of n_val (for example sampling or applying different values per channel). This argument enables simulations of channel-specific occlusions or artifacts (useful for multi-modal medical images), but the precise channel-wise behavior (whether n_val is interpreted as a sequence or sampled per channel) is implemented in mask_random_square.
        square_pos (tuple): Optional fixed position for the square mask. If None (default) mask_random_square will choose a random location for each square. If provided, this tuple is forwarded to mask_random_square and typically specifies the spatial coordinates where the square should be placed (exact format and indexing convention follow mask_random_square's contract). Providing a position makes augmentation deterministic for that square; invalid positions (out of bounds, wrong dimensionality) will cause mask_random_square to raise an error.
    
    Returns:
        numpy.ndarray: The input image array with n_squares applied square masks. The returned array follows the same shape and dtype conventions as the input img; it is the augmented image that should be used downstream in the batchgenerators pipeline (e.g., fed into training or validation). Because the implementation delegates to mask_random_square, callers must treat the returned value as the authoritative result (the underlying array may or may not have been modified in-place).
    
    Raises:
        ValueError: If argument types or values are invalid (for example non-integer square_size or n_squares), or if provided square_pos is incompatible with the image spatial dimensions. Such errors may surface directly from mask_random_square; this function does not perform extensive validation beyond forwarding parameters and looping n_squares times.
        IndexError or other array-related errors: If square_size or square_pos cause attempted writes outside the array bounds, the underlying mask_random_square implementation may raise array indexing errors.
    """
    from batchgenerators.augmentations.utils import mask_random_squares
    return mask_random_squares(
        img,
        square_size,
        n_squares,
        n_val,
        channel_wise_n_val,
        square_pos
    )


################################################################################
# Source: batchgenerators.augmentations.utils.pad_nd_image
# File: batchgenerators/augmentations/utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def batchgenerators_augmentations_utils_pad_nd_image(
    image: numpy.ndarray,
    new_shape: list = None,
    mode: str = "constant",
    kwargs: dict = None,
    return_slicer: bool = False,
    shape_must_be_divisible_by: list = None
):
    """Pad an N-dimensional numpy image to at least the requested spatial shape, optionally returning a slicer to recover the original region. This function is used in the batchgenerators augmentation pipeline to ensure image arrays (for example the 'data' or 'seg' entries with shapes like (b, c, x, y) or (b, c, x, y, z) described in the README) meet minimum spatial size requirements and/or are adjusted to be divisible by network-friendly values before further processing (e.g., feeding into a CNN). Padding is computed symmetrically where possible and applied only to the last axes of the array if fewer target dimensions are provided than the image has.
    
    Args:
        image (numpy.ndarray): Input N-dimensional image array to be padded. In the batchgenerators context this is typically a numpy array holding a batch of images or segmentations (for example shape (b, c, x, y) for 2D or (b, c, x, y, z) for 3D). The function does not modify the input in-place; it returns a padded copy if padding is needed. All array axes are preserved; padding is only added (no cropping).
        new_shape (list or tuple or numpy.ndarray or None): Minimum desired shape for the last len(new_shape) axes of image. If provided, new_shape is interpreted as a per-axis minimum (new_min_shape) rather than an exact target: any axis where new_shape[i] is smaller than the existing image size will not be cropped and the original size for that axis is kept. new_shape may have fewer entries than image.ndim; in that case the last axes of image are the ones considered for padding. If new_shape is None, shape_must_be_divisible_by must be provided (see shape_must_be_divisible_by description). Common usage in batchgenerators: supply only spatial axes (e.g., (x, y) or (x, y, z)) while leaving batch and channel axes unchanged.
        mode (str): Passed directly to numpy.pad as the padding mode (see numpy.pad documentation). Typical value used in batchgenerators is "constant" to pad with constant values (e.g., zeros). Other numpy.pad modes (for example "reflect", "edge") are accepted and will affect how padding values are filled. Incorrect modes will raise the same errors numpy.pad raises.
        kwargs (dict or None): Additional keyword arguments forwarded to numpy.pad (for example for mode="constant" one may pass {'constant_values': 0}). If None (the default) the function uses {'constant_values': 0} so constant padding will fill with zeros by default. Invalid keys or values will result in numpy.pad raising an error.
        return_slicer (bool): If False (default) the function returns only the padded array. If True the function returns a tuple (padded_array, slicer) where slicer is a list of slice objects that can be applied to the padded array to extract the region corresponding to the original (pre-padding) image. The slicer length equals padded_array.ndim and for axes that were not considered for padding (leading axes when new_shape is shorter than image.ndim) the slicer covers the full axis. Use the returned slicer to crop back to the original content after processing the padded image (for example to discard padded borders after network prediction).
        shape_must_be_divisible_by (list or tuple or numpy.ndarray or int or None): Optional constraint used primarily for neural network prediction where spatial dimensions must be divisible by some number (for example downsampling factors of a U-Net). If provided and new_shape is given, the function will increase new_shape (never decrease below the original sizes) so that each considered axis is divisible by the corresponding entry in shape_must_be_divisible_by. If a single int is supplied, it will be broadcast to all considered axes. If new_shape is None, then shape_must_be_divisible_by must be a sequence (list/tuple/numpy.ndarray) and the function will use the current image spatial sizes as the base new_shape before enforcing divisibility. Implementation notes: the function converts inputs to numpy arrays internally as needed and applies a deterministic symmetric padding strategy (difference // 2 below, difference // 2 + difference % 2 above) so the padded image is as centered as possible. The code contains assertions that will fail if shape_must_be_divisible_by is None while new_shape is None, or if shape_must_be_divisible_by is provided as a sequence whose length does not match len(new_shape).
    
    Returns:
        numpy.ndarray: The padded image array when return_slicer is False and no error occurs. If no padding is required (requested minima already satisfied and divisibility constraints met) the original image object is returned unchanged.
        tuple (numpy.ndarray, list(slice)): If return_slicer is True, returns a tuple (padded_image, slicer). padded_image is the numpy.ndarray described above. slicer is a list of slice objects (one per axis of padded_image) that select the sub-array corresponding to the original image content (i.e., applying padded_image[tuple(slicer)] yields the original image region). This is useful in pipelines where you need to undo padding after processing (for example extracting the prediction region from a network output that operated on the padded array).
    
    Behavioral details, defaults, and failure modes:
        - The function never crops data: new_shape values smaller than the existing axis sizes are treated as minima and the original size for that axis is kept.
        - Padding is symmetric where possible: pad_below = floor((new - old)/2) and pad_above = ceil((new - old)/2). If the difference is odd, the extra voxel/element is put on the upper side (pad_above).
        - The function uses numpy.pad to perform padding; invalid mode or kwargs will raise numpy.pad errors.
        - If kwargs is None the default used is {'constant_values': 0} together with the provided mode (default "constant"), so constant-zero padding is the default behavior.
        - If new_shape is None you must provide shape_must_be_divisible_by as a sequence; otherwise the function asserts and raises AssertionError.
        - If shape_must_be_divisible_by is provided as a non-sequence scalar, it will be broadcast to all considered axes. If provided as a sequence its length must match len(new_shape) and a mismatch will cause an AssertionError.
        - If no padding is required (all per-axis differences are zero) the original array object is returned (no copy) and, when return_slicer is True, the slicer covers the full axes (i.e., it will reproduce the same array when applied).
        - This function is intended for use in data augmentation and preprocessing pipelines (see batchgenerators README), particularly to ensure consistency of spatial dimensions for neural network inputs and to accommodate augmentation operations that require minimum image sizes.
    """
    from batchgenerators.augmentations.utils import pad_nd_image
    return pad_nd_image(
        image,
        new_shape,
        mode,
        kwargs,
        return_slicer,
        shape_must_be_divisible_by
    )


################################################################################
# Source: batchgenerators.augmentations.utils.resize_multichannel_image
# File: batchgenerators/augmentations/utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def batchgenerators_augmentations_utils_resize_multichannel_image(
    multichannel_image: numpy.ndarray,
    new_shape: tuple,
    order: int = 3
):
    """Resize a multichannel image by resizing each channel independently and
    recombining the resized channels into a multichannel array. This function is
    used in the batchgenerators augmentation pipeline to resample per-sample
    channel data (for example modalities in a multi-modal medical image) when a
    new spatial resolution or shape is required. It preserves the input array's
    channel ordering and numeric dtype while performing interpolation on a
    floating-point representation.
    
    Args:
        multichannel_image (numpy.ndarray): Input image with channels as the first
            axis. Expected shape is (c, X, Y) for 2D data or (c, X, Y, Z) for 3D
            data, where c is the number of channels/modalities and X, Y, (Z) are
            spatial dimensions. This is the per-sample "data" representation used
            by batchgenerators (the README describes batch-level data as (b, c, ...),
            this function operates on the per-sample slice with shape beginning
            with c). Each channel will be resized independently and then stacked
            back into the same channel-first layout. The function does not modify
            the input array in place.
        new_shape (tuple): Target spatial shape for each channel given as a tuple
            of integers (X_new, Y_new) for 2D or (X_new, Y_new, Z_new) for 3D. The
            returned array will have shape (c, ) + new_shape. The length of this
            tuple must match the spatial dimensionality of multichannel_image
            (i.e., 2 entries for 2D input, 3 entries for 3D input). If this does
            not hold, the underlying resize implementation will raise an error.
        order (int): Interpolation order passed through to the underlying
            resize implementation (default: 3). In practice this controls the
            interpolation kernel (e.g., 0=nearest, 1=linear, 3=cubic). The value
            must be an integer supported by the resize routine; invalid values
            will cause that routine to raise an exception. The default (3) gives
            cubic interpolation commonly used for image resampling.
    
    Returns:
        numpy.ndarray: A new numpy array containing the resized multichannel image.
        The array has shape (c,) + new_shape and the same dtype as the input
        multichannel_image. Internally, each channel is converted to float for
        interpolation, resized using the module's resize function with clip=True
        and anti_aliasing=False, and then the stack is cast back to the original
        dtype. As a consequence, integer input types will be rounded/truncated
        according to numpy casting rules when converting back to the original dtype,
        which may cause loss of precision. No in-place modification is performed on
        the input array.
    
    Behavior, defaults, and failure modes:
        - Each channel is processed independently: useful for multi-modal medical
          images or multi-channel feature maps where channels represent distinct
          modalities and should not be mixed during interpolation.
        - The function uses clip=True and anti_aliasing=False when calling the
          underlying resize; this is the fixed behavior and cannot be changed via
          this function's parameters.
        - The return dtype matches the input dtype. Because interpolation is done
          on a floating-point copy, casting back to the original dtype can produce
          rounding or clipping for integer types.
        - The function expects the length of new_shape to match the spatial rank of
          multichannel_image (2 for 2D, 3 for 3D). Mismatched dimensionality,
          non-integer or non-positive entries in new_shape, or unsupported order
          values will cause the underlying resize implementation to raise an
          exception (e.g., ValueError or TypeError).
        - No GPU or device-specific behavior is assumed; the operation is purely
          CPU-based numpy operations and the module's resize function.
    """
    from batchgenerators.augmentations.utils import resize_multichannel_image
    return resize_multichannel_image(multichannel_image, new_shape, order)


################################################################################
# Source: batchgenerators.augmentations.utils.resize_segmentation
# File: batchgenerators/augmentations/utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def batchgenerators_augmentations_utils_resize_segmentation(
    segmentation: numpy.ndarray,
    new_shape: tuple,
    order: int = 3
):
    """Resizes a segmentation map for use in spatial augmentations (medical image augmentation pipeline).
    This function is intended for resizing discrete label maps (segmentation masks) in the batchgenerators
    augmentation pipeline used by the MIC@DKFZ codebase. To avoid interpolation artifacts that can occur when
    resizing integer label images with continuous interpolators (for example producing intermediate label values
    such as [0, 0, 2] -> [0, 1, 2]), this function either (a) applies nearest-neighbor resizing when order==0 or
    (b) converts the segmentation into a set of binary masks (one-hot / multihot per unique label), resizes each
    mask with the specified interpolation order, thresholds the resized masks at 0.5, and reconstructs a label map.
    The reconstructed map is returned with the same numpy.dtype as the input segmentation. This behavior is used
    in batchgenerators whenever the 'seg' entry of a data dictionary must undergo spatial transforms (see README).
    
    Args:
        segmentation (numpy.ndarray): Input segmentation map to be resized. This is a discrete label image (for
            example anatomical segmentation labels in medical imaging). The function does not modify the input array;
            it creates and returns a resized copy. The dtype of the returned array matches segmentation.dtype.
            The function requires that len(segmentation.shape) == len(new_shape) (same number of spatial dimensions);
            if this is not true an AssertionError is raised.
        new_shape (tuple): Target shape for the resized segmentation. The tuple must have the same length as
            segmentation.ndim. Values in new_shape are the sizes for each corresponding dimension of segmentation
            after resizing. No batch/channel semantics are imposed by this function; it operates on the entire array
            shape given.
        order (int): Interpolation order passed to skimage.transform.resize (same semantics as skimage). Default is 3.
            Behavior differs by value:
            - If order == 0: the function performs a direct resize of the whole segmentation (nearest-neighbor semantics),
              using skimage.transform.resize(..., order=0, mode="edge", clip=True, anti_aliasing=False) and casts
              the result back to the original dtype. This is a fast path suitable for nearest-neighbor-style resizing.
            - If order != 0: the function performs the one-hot / multihot strategy: it finds np.sort(pd.unique(segmentation.ravel()))
              to enumerate all unique labels, resizes the binary mask for each label with the given order (mode="edge",
              clip=True, anti_aliasing=False), thresholds each resized mask at 0.5 and assigns the corresponding label
              where the threshold is met. This avoids creating fractional label values from interpolation and is the
              recommended approach for higher-order interpolation orders.
    
    Returns:
        numpy.ndarray: The resized segmentation map as a numpy array with shape equal to new_shape and dtype equal
        to segmentation.dtype. For order==0 this is obtained by resizing and casting the whole array. For order!=0 this
        is reconstructed from resized binary masks for each unique label. No in-place modification of the input occurs.
    
    Notes and failure modes:
        - The function asserts that the number of dimensions in new_shape matches segmentation.ndim; mismatch raises
          AssertionError.
        - The threshold for reconstructed labels is 0.5. Very small regions or labels that produce maximum resized
          probabilities below 0.5 may be lost (i.e., a label may disappear after resizing). Users should be aware
          that the one-hot reconstruction can therefore change small connected components.
        - If two or more label masks overlap after thresholding, the label assigned last (based on sorted unique labels)
          will overwrite previous labels at those voxels. This is an implementation detail that can affect ties.
        - The function uses skimage.transform.resize with mode="edge", clip=True and anti_aliasing=False to avoid
          introducing constant border values (cval) and to match batchgenerators' treatment of segmentation resizing.
          This choice is deliberate (see README release notes) because constant border values are undesirable for
          segmentation labels.
        - Performance: for many distinct labels the one-hot strategy can be computationally and memory intensive
          because it resizes one binary mask per unique label. For large numbers of labels consider using order==0
          or other strategies appropriate for your use case.
        - The function relies on pandas' pd.unique and skimage.transform.resize APIs; ensure those dependencies are
          available in the environment.
    """
    from batchgenerators.augmentations.utils import resize_segmentation
    return resize_segmentation(segmentation, new_shape, order)


################################################################################
# Source: batchgenerators.augmentations.utils.uniform
# File: batchgenerators/augmentations/utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def batchgenerators_augmentations_utils_uniform(low: float, high: float, size: tuple = None):
    """Concise wrapper around numpy.random.uniform that ensures well-defined output when the lower and upper bounds are identical.
    
    Args:
        low (float): Lower bound of the interval from which to draw samples. In the batchgenerators augmentation context this represents the minimum value for a random augmentation parameter (for example the smallest possible rotation, scale, or intensity change). If low is equal to high this function treats the sampling as degenerate and returns that constant value instead of delegating to numpy.
        high (float): Upper bound of the interval from which to draw samples. In the batchgenerators augmentation context this represents the maximum value for a random augmentation parameter. When high == low the function returns the constant value low (or an array filled with low) rather than attempting to sample.
        size (tuple): Shape of the output array to produce when sampling. This matches the semantics of numpy.random.uniform's size argument. Default: None. In the augmentation pipeline, pass a size to obtain per-element independent random values (e.g., one parameter per image or per voxel); leave as None to obtain a scalar.
    
    Returns:
        float or numpy.ndarray: If low == high and size is None, returns the scalar float low. If low == high and size is not None, returns a numpy.ndarray of shape size filled with the constant value low. Otherwise returns the result of numpy.random.uniform(low, high, size): a scalar float when size is None or a numpy.ndarray of shape size when size is provided. Values sampled by numpy.random.uniform are drawn from the half-open interval [low, high) per numpy semantics.
    
    Behavior and side effects:
        This function has no side effects other than delegating to numpy.random.uniform when low != high. It is intended for use within the batchgenerators data augmentation pipeline to produce random augmentation parameters while gracefully handling degenerate ranges (low == high), which commonly occur when a user fixes an augmentation parameter to a single value. When degenerate, it avoids calling numpy.random.uniform (which would otherwise produce a deterministic but possibly unexpected result) and explicitly returns the constant value or an array filled with that constant.
    
    Failure modes and validation:
        The function does not perform explicit input validation beyond Python/numpy normal behavior. Non-numeric inputs for low, high, or non-tuple values for size will raise exceptions from numpy or Python (TypeError, ValueError) as appropriate. When low != high the behavior (including any range checks) is delegated to numpy.random.uniform.
    """
    from batchgenerators.augmentations.utils import uniform
    return uniform(low, high, size)


################################################################################
# Source: batchgenerators.dataloading.data_loader.default_collate
# File: batchgenerators/dataloading/data_loader.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def batchgenerators_dataloading_data_loader_default_collate(batch: list):
    """Default collate function used by batchgenerators to assemble a list of samples into a batched structure.
    This function is heavily inspired by torch.utils.data.default_collate and is intended for use inside the
    batchgenerators data loading / augmentation pipeline (for example in MultiThreadedAugmenter and custom
    DataLoaderBase implementations). It converts a list of per-sample objects (the argument `batch`) into a single
    batched object that can be passed to downstream augmentations or model training. In the context of batchgenerators
    this is typically used to form arrays with a leading batch dimension consistent with the README data convention
    (e.g., data/seg arrays with shape (b, c, x, y) for 2D or (b, c, x, y, z) for 3D).
    
    The function handles a small set of concrete element types (numpy arrays, Python numeric types, numpy numeric
    scalars, dict/OrderedDict, tuple/list and strings) and applies type-specific collating rules described below.
    Behavior is recursive for nested containers (dicts of tuples of arrays, etc.). The function enforces consistent
    structure across samples: for dict/OrderedDict inputs every sample must contain the same keys; for arrays the
    per-sample array shapes must be compatible for stacking along a new leading batch axis.
    
    Args:
        batch (list): A list of samples produced by a DataLoader or by user code. Each element of this list
            represents one sample and may be one of the concrete types handled by this function:
            numpy.ndarray, int / np.int64, float / np.float32, np.float64, dict or OrderedDict, tuple or list,
            or str. Typical practical usage in batchgenerators is that each sample is a dictionary with keys
            such as 'data' and optionally 'seg' where the per-sample 'data' array has shape (c, x, y) for 2D or
            (c, x, y, z) for 3D and collating produces an array with leading batch dimension (b, c, x, y(, z)).
            The function expects that all elements in the list share a compatible type and compatible shapes/keys
            where applicable; otherwise a numpy or Python exception will be raised.
    
    Returns:
        object: The collated batch. The exact returned type depends on the element type of the input list:
            - If the first element is a numpy.ndarray: returns a numpy.ndarray produced by np.vstack(batch).
              Practically, this stacks per-sample arrays along a new leading batch axis so the resulting shape
              corresponds to (b, ...) as used throughout batchgenerators. All arrays must be compatible for vstack.
            - If the first element is an integer (int or np.int64): returns a numpy.ndarray of dtype np.int32
              containing the input integers (np.array(batch).astype(np.int32)).
            - If the first element is a Python float or np.float32: returns a numpy.ndarray of dtype np.float32
              (np.array(batch).astype(np.float32)).
            - If the first element is np.float64: returns a numpy.ndarray of dtype np.float64
              (np.array(batch).astype(np.float64)).
            - If the first element is a dict or OrderedDict: returns a dict where each key maps to the collated
              result of the corresponding values across samples. This is computed recursively by calling
              default_collate([d[key] for d in batch]) for each key. All samples must contain the same keys.
            - If the first element is a tuple or list: returns a list whose i-th element is the result of
              collating the i-th elements of all samples. Implementation detail: the input list is transposed
              with zip(*batch) and each group is collated recursively.
            - If the first element is a str: returns the original input list of strings unchanged.
        The returned object is intended to be directly consumable by subsequent transforms and model inputs
        in the batchgenerators pipeline.
    
    Raises:
        TypeError: If the element type of batch[0] is not one of the supported types listed above. The function
            raises TypeError with the element type included in the error message.
        ValueError or numpy errors: If array shapes are incompatible for np.vstack or if dtype conversions fail,
            the underlying numpy operations will raise ValueError or other numpy exceptions. If dict inputs do not
            share identical keys across all samples a KeyError or related exception may be raised during the
            per-key collection step.
    
    Notes and side effects:
        - The function is pure with respect to Python state (it does not modify the input list or its elements);
          however numpy operations create and return new arrays.
        - For numeric scalar inputs the function normalizes dtypes to np.int32, np.float32, or np.float64 as shown,
          which is important for downstream consistency and for matching expectations in batchgenerators examples.
        - For dict/OrderedDict inputs the collate operation is recursive and will apply the same dtype/stacking logic
          to nested arrays or containers.
        - This implementation is intentionally narrow and mirrors the behavior and type-handling choices used in
          the original PyTorch default_collate: it does not attempt to handle arbitrary custom objects. If you
          need custom collating behavior, implement and use a custom collate function in your DataLoader pipeline.
    """
    from batchgenerators.dataloading.data_loader import default_collate
    return default_collate(batch)


################################################################################
# Source: batchgenerators.datasets.cifar.maybe_download_and_prepare_cifar
# File: batchgenerators/datasets/cifar.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def batchgenerators_datasets_cifar_maybe_download_and_prepare_cifar(
    target_dir: str,
    cifar: int = 10
):
    """maybe_download_and_prepare_cifar(target_dir: str, cifar: int = 10)
    Checks for a local CIFAR dataset in target_dir and downloads, extracts, consolidates, and saves it
    if the expected consolidated files are missing. This function is used by the CIFAR DataLoader in
    batchgenerators.datasets.cifar to ensure the CIFAR-10 or CIFAR-100 dataset is available in a simple,
    single-file format suitable for fast loading and downstream data augmentation pipelines described
    in the repository README (e.g., for training/testing image classification models or for examples
    that ship with batchgenerators).
    
    This function verifies the presence of two compressed NumPy files in target_dir: "cifar{cifar}_training_data.npz"
    and "cifar{cifar}_test_data.npz". If either is missing, it downloads the official CIFAR tarball from
    http://www.cs.toronto.edu/~kriz/cifar-{cifar}-python.tar.gz, extracts the original batch files,
    unpickles and reshapes the raw batch data into image arrays with shape (N, 3, 32, 32) and dtype
    numpy.uint8, consolidates the five training batches into a single training array, and writes two
    compressed .npz files containing data, labels and filenames. After successful creation of the .npz
    files the extracted folder and the downloaded tar.gz archive are removed to clean up disk space.
    The function prints a brief message ("downloading CIFAR{cifar}...") when it starts a download.
    
    Behavior, formats and practical significance:
    - Training data: the five data_batch_* files are read in order, each block is reshaped from the
      original flat representation to (samples_per_batch, 3, 32, 32) and cast to numpy.uint8, then
      vertically stacked to form a single array. For the standard CIFAR releases this yields 5 * 10000 =
      50000 training samples for cifar=10 (the function follows the original CIFAR python files layout).
      The consolidated training file saved is "cifar{cifar}_training_data.npz" and contains:
      - data: numpy.ndarray of dtype numpy.uint8 and shape (N_train, 3, 32, 32)
      - labels: numpy.ndarray of integers (converted from the original batch labels)
      - filenames: numpy.ndarray of strings (converted from the original batch filenames)
    - Test data: the single test_batch file is reshaped and saved to "cifar{cifar}_test_data.npz" and
      contains:
      - data: numpy.ndarray of dtype numpy.uint8 and shape (N_test, 3, 32, 32)
      - labels: a Python list of integers (created from the test batch labels in the original files)
      - filenames: a Python list of strings (from the original test batch)
    - File cleanup: after creating the two .npz files, the temporary extracted directory
      "cifar-{cifar}-batches-py" and the downloaded tarball "cifar-{cifar}-python.tar.gz" are removed
      using shutil.rmtree and os.remove respectively to free disk space.
    - Intended use: the produced .npz files are optimized for quick repeated loading by the CIFAR
      DataLoader in batchgenerators and for use in augmentation and training pipelines described in the
      repository README (e.g., examples and DataLoader implementations that expect the consolidated format).
    
    Args:
        target_dir (str): Path to an existing, writable directory where CIFAR will be downloaded,
            extracted and where the consolidated files will be written. The function expects this
            directory to exist and be writable by the running process; if it does not exist or is not
            writable, the function will raise an exception from the underlying I/O or download calls.
        cifar (int): Numeric CIFAR variant identifier (default 10). Typical values used with this
            function are 10 or 100 corresponding to CIFAR-10 and CIFAR-100 releases. The value is
            inserted into download URLs, extracted folder names and output filenames (e.g.
            "cifar10_training_data.npz" when cifar=10). The function does not perform additional
            validation beyond using this integer in file and URL paths.
    
    Returns:
        None: This function does not return a value. Its side effects are the creation of two compressed
        NumPy archives in target_dir named "cifar{cifar}_training_data.npz" and "cifar{cifar}_test_data.npz"
        (unless both already existed, in which case no download/extraction is performed), and the removal
        of intermediate extracted files and the downloaded tar.gz archive when new files are created.
    
    Failure modes and exceptions:
    - Network/download errors (e.g., URLError, HTTPError) can occur during urlretrieve and will propagate
      to the caller unless handled externally.
    - Corrupt or unexpected archive contents can raise tarfile.ReadError, pickle/unpickle exceptions or
      ValueError during reshaping/unpacking.
    - Filesystem errors such as PermissionError, OSError, or FileNotFoundError can occur when writing the
      tarball, extracting files, saving .npz archives, or removing temporary files.
    - If either of the target .npz files already exists, the function will skip downloading only when
      both files are present; if one is missing it will download and recreate both consolidated files,
      potentially overwriting existing ones.
    
    Notes:
    - The produced arrays and saved metadata follow the layout and conventions of the original CIFAR
      python distribution (3 color channels, 32x32 images, batch-based original packaging), and are
      intended to be consumed by the CIFAR DataLoader implementations included in batchgenerators for
      experiments and augmentations described in the project README.
    """
    from batchgenerators.datasets.cifar import maybe_download_and_prepare_cifar
    return maybe_download_and_prepare_cifar(target_dir, cifar)


################################################################################
# Source: batchgenerators.datasets.cifar.unpickle
# File: batchgenerators/datasets/cifar.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def batchgenerators_datasets_cifar_unpickle(file: str):
    """Load and return the Python object stored in a CIFAR-format pickled file.
    
    This function is a minimal utility used by batchgenerators.datasets.cifar to read the batch files provided by the CIFAR-10/CIFAR-100 dataset distribution (see http://www.cs.toronto.edu/~kriz/cifar.html). It opens the given filesystem path in binary mode and uses the Python pickle module to deserialize the stored object using encoding='bytes'. In the context of batchgenerators, the returned object is consumed by CIFAR dataset loaders to obtain image/label data for training, validation, and testing pipelines.
    
    Args:
        file (str): Filesystem path to the pickled file to load. This must be a string path pointing to a file on disk that was written with Python's pickle module (for example, the batch files distributed with CIFAR). The function opens this path in binary read mode ('rb') and attempts to unpickle its contents with encoding='bytes'. Passing a path to a non-pickle file, a missing path, or a path without read permission will raise the corresponding I/O or unpickling exceptions.
    
    Returns:
        dict: The Python object returned by pickle.load. For CIFAR batch files this is typically a dictionary (with byte-string keys because encoding='bytes') containing entries such as b'data', b'labels', and b'filenames' that hold the stored dataset arrays/lists. The exact structure depends on the file contents, but batchgenerators expects the CIFAR convention when using this helper.
    
    Behavior and side effects:
        The function reads from disk and deserializes data using pickle.load with encoding='bytes', causing dictionary keys (and any pickled string objects) to be bytes objects rather than str. It does not modify the input file. Unpickling arbitrary data is a security risk: do not call this function on untrusted files because arbitrary code execution can occur during unpickling.
    
    Failure modes:
        If the file path does not exist, a FileNotFoundError is raised. If the file cannot be opened due to permissions, a PermissionError may be raised. If the file is not a valid pickle or is corrupted/incompatible, pickle.UnpicklingError or other exceptions from the pickle module may be raised. Compatibility issues can also arise when unpickling data produced with different Python/pickle protocol versions; in such cases adjust the data source or serialization method instead of modifying this loader.
    """
    from batchgenerators.datasets.cifar import unpickle
    return unpickle(file)


################################################################################
# Source: batchgenerators.utilities.data_splitting.get_split_deterministic
# File: batchgenerators/utilities/data_splitting.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def batchgenerators_utilities_data_splitting_get_split_deterministic(
    all_keys: list,
    fold: int = 0,
    num_splits: int = 5,
    random_state: int = 12345
):
    """Deterministically split a list of patient identifiers (or numeric keys) into training and testing sets corresponding to a single fold of a K-fold split. This function is used in the batchgenerators data-loading/augmentation workflow to create reproducible train/test partitions for cross-validation experiments (for example, when evaluating augmentation strategies or model performance on medical imaging datasets where each element of all_keys corresponds to one patient or case).
    
    Args:
        all_keys (list): A list of patient identifiers or numbers to be split. Each element should be a comparable value (e.g., integers or strings) because the function first calls numpy.sort on list(all_keys) to obtain a deterministic ordering. all_keys is not modified in place; a sorted copy is used for splitting.
        fold (int): Index of the fold to return. Valid values are 0..num_splits-1. The function iterates over the KFold splits and returns the train/test keys for the iteration whose index equals fold. If fold is out of this range, no matching split will be selected and the function will raise an error at return time (see Failure modes).
        num_splits (int): Number of folds for K-fold splitting (n_splits passed to sklearn.model_selection.KFold). Typical use is cross-validation with values like 5 or 10. Must be an integer >= 2 and also <= number of samples after sorting; otherwise sklearn's KFold will raise a ValueError.
        random_state (int): Seed for the random number generator used by KFold when shuffle=True. Supplying the same random_state yields reproducible (deterministic) shuffled splits across runs, which is important for reproducible experiments in medical image augmentation and model evaluation. Default is 12345.
    
    Returns:
        tuple: A pair (train_keys, test_keys) where both elements are numpy.ndarray objects containing the keys selected for training and testing for the requested fold. The keys are taken from the sorted copy of all_keys and then indexed according to the KFold split for the specified fold. These arrays are suitable for slicing or indexing other data structures that map patient identifiers to image/label arrays.
    
    Behavior and side effects:
        The function creates a sorted copy of all_keys using numpy.sort(list(all_keys)), creates a sklearn.model_selection.KFold instance with shuffle=True and the provided random_state, and iterates through splits.split(all_keys_sorted) until the iteration index equals fold. It then selects and returns the corresponding train and test keys. The original all_keys object is not modified. No I/O is performed.
    
    Failure modes and notes:
        If num_splits < 2 or num_splits > number of elements in all_keys_sorted, sklearn.model_selection.KFold will raise a ValueError. If fold is not in the range [0, num_splits-1], the loop will not select any split and attempting to return train_keys and test_keys will raise an UnboundLocalError (local variable referenced before assignment). If elements of all_keys are not mutually comparable (for example, mixing incompatible types), numpy.sort may raise a TypeError. Ensure inputs conform to these constraints to obtain deterministic train/test splits for reproducible cross-validation.
    """
    from batchgenerators.utilities.data_splitting import get_split_deterministic
    return get_split_deterministic(all_keys, fold, num_splits, random_state)


################################################################################
# Source: batchgenerators.utilities.file_and_folder_operations.split_path
# File: batchgenerators/utilities/file_and_folder_operations.py
# Category: fix_docstring
# Reason: Schema parsing failed: Cannot generate JSON schema for split_path because the docstring has no description for the argument 'path'
################################################################################

def batchgenerators_utilities_file_and_folder_operations_split_path(path: str):
    """batchgenerators.utilities.file_and_folder_operations.split_path splits the given filesystem path string at every platform-specific path separator (os.sep) and returns the sequence of path components. This function is used in the batchgenerators codebase for file and folder operations where the caller needs all path segments (for example when parsing dataset directory hierarchies, constructing relative paths for data loaders, or comparing individual path components in augmentation pipelines). It differs from os.path.split in that it does not only split at the last separator but returns all components separated by os.sep.
    
    This function performs a simple, non-normalizing string split and does not consult the filesystem. It does not resolve symbolic links, collapse '.' or '..' segments, handle alternate separators (for example '/' vs '\\' on Windows) or perform OS-specific path normalization beyond using the current os.sep value. Because it calls the standard str.split method, certain corner cases produce empty-string components: a leading separator yields an initial empty string component, a trailing separator yields a final empty string component, and consecutive separators produce empty components between them. An empty input string yields a single-element list containing an empty string. There are no side effects (no file I/O, no modification of the filesystem) and the operation is pure and safe to call repeatedly.
    
    Args:
        path (str): The filesystem path to split into components. This must be a Python str (not a path-like object); the function splits the string at every occurrence of os.sep (the platform-specific separator such as '/' on POSIX or '\\' on Windows). In the context of batchgenerators, this parameter typically holds dataset or file paths used by data loaders and augmentation utilities. Passing a non-str value will result in an AttributeError when attempting to call split on the value.
    
    Returns:
        List[str]: A list of path components obtained by splitting the input string at each occurrence of os.sep. Each element is a substring of the original path between separators. Note the following practical behaviors: leading, trailing, or consecutive separators produce empty-string components in the returned list; the function does not normalize components (it preserves '.' and '..' as literal elements); and it only recognizes the single separator value given by os.sep, so mixed-separator paths are not normalized before splitting.
    """
    from batchgenerators.utilities.file_and_folder_operations import split_path
    return split_path(path)


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
