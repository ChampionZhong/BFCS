"""
Regenerated Google-style docstrings for module 'lungmask'.
README source: others/readme/lungmask/README.md
Generated at: 2025-12-02T00:20:10.305393Z

Total functions: 8
"""


import numpy

################################################################################
# Source: lungmask.utils.bbox_3D
# File: lungmask/utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def lungmask_utils_bbox_3D(labelmap: numpy.ndarray, margin: int = 2):
    """Compute bounding box of a 3D labelmap used in lungmask for cropping and postprocessing of segmentation volumes.
    
    This function inspects a multi-dimensional numpy labelmap (typically a 3D segmentation volume produced or consumed by the lungmask package) and computes per-axis minimum and maximum indices that enclose all non-zero labels. The bounding box is expanded by the integer margin on each axis and clipped to the labelmap extents. In the lungmask workflow this is commonly used to crop volumes to the lung region before further processing (for example to reduce computation during model inference, to crop input for visualization, or to restrict fusion operations between models such as LTRCLobes and R231). For numpy arrays following the package convention, the first axis is slices (z), the second is chest-to-back (y), and the third is right-to-left (x), so for a 3D labelmap the returned array corresponds to [zmin, zmax, ymin, ymax, xmin, xmax]. The returned upper bounds are exclusive (suitable for Python slicing).
    
    Args:
        labelmap (numpy.ndarray): Input labelmap. A numpy array containing integer or boolean labels where non-zero values indicate voxels of interest (e.g., lung labels produced by a segmentation model). The array may be 3D (typical) or n-D; the function computes bounds for each axis in order.
        margin (int): Margin to add to the bounding box on each axis. This integer is broadcast to all axes (the code creates a per-axis list [margin] * number_of_axes). Positive values expand the box, negative values will shrink it. Defaults to 2.
    
    Returns:
        numpy.ndarray: Bounding box as a 1D array of integer indices with length 2 * ndim. For a 3D input the format is [zmin, zmax, ymin, ymax, xmin, xmax]. Here bmin values are inclusive start indices (suitable for Python slice start), and bmax values are exclusive end indices (suitable for Python slice end). The dtype is integer and the values are clipped to the interval [0, size] per axis where size is labelmap.shape[axis].
    
    Behavior and side effects:
        This function is pure (no external side effects) and does not modify the input labelmap. It computes per-axis projections by testing any(labelmap) across the complementary axes and then finds the first and last True positions to derive bmin and bmax. The computed bmin is decremented by the per-axis margin and clipped to >= 0. The computed bmax is incremented by the per-axis margin + 1 and clipped to <= axis size so that the upper bound is exclusive for slicing.
    
    Failure modes and recommendations:
        If labelmap contains no non-zero elements (no voxels of interest), the internal indexing that extracts the first/last True locations will raise an IndexError (because there are no True positions). Callers should check numpy.any(labelmap) before calling this function or catch the exception to handle empty labelmaps. The function assumes labelmap is a numpy.ndarray; passing other array-like types may raise type or attribute errors.
    """
    from lungmask.utils import bbox_3D
    return bbox_3D(labelmap, margin)


################################################################################
# Source: lungmask.utils.crop_and_resize
# File: lungmask/utils.py
# Category: valid
################################################################################

def lungmask_utils_crop_and_resize(img: numpy.ndarray, width: int = 192, height: int = 192):
    """lungmask.utils.crop_and_resize crops a 2D CT slice to the detected body region and resizes the crop to the specified target size. This preprocessing step is used in the lungmask pipeline to focus downstream lung segmentation models (for example the U-net variants described in the README) on the relevant body area, reduce input dimensionality to the network input size, and normalize spatial extent across slices.
    
    Args:
        img (np.ndarray): 2D image to be cropped and resized. In the lungmask workflow this is typically a single CT slice represented as a NumPy array (intensity values, usually Hounsfield units for CT). The function expects a 2-dimensional array indexed as (rows, columns). If a 3D volume (slices, rows, columns) is available (see README numpy-array support), call this function per-slice because the implementation computes a 2D body mask and 2D bounding box.
        width (int): Target width (number of columns) to which the cropped image will be resized. Defaults to 192. Width must be a positive integer; non-positive values or non-integer types will raise an error via scipy.ndimage.zoom or cause undefined behavior.
        height (int): Target height (number of rows) to which the cropped image will be resized. Defaults to 192. Height must be a positive integer; non-positive values or non-integer types will raise an error via scipy.ndimage.zoom or cause undefined behavior.
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing two NumPy arrays:
            The first element is the resized image (np.ndarray) produced by cropping the input to the detected body bounding box and then resampling to exactly (height, width) using scipy.ndimage.zoom with linear interpolation (order=1). The resizing operation may smooth or slightly alter intensities due to interpolation; data type may be preserved or converted according to scipy.ndimage.zoom behavior.
            The second element is the cropping bounding box (np.ndarray) expressed in coordinates of the original input image before resizing. The bounding box follows skimage.measure.regionprops.bbox ordering for 2D images: (min_row, min_col, max_row, max_col), where min indices are inclusive and max indices follow Python slice semantics (exclusive). If no body region is detected by the internal simple_bodymask call, the function returns a bounding box equal to the full image extent (array-like equivalent to [0, 0, img.shape[0], img.shape[1]]).
    
    Behavior and side effects:
        The function computes a binary body mask using simple_bodymask(img) and determines a cropping bounding box from the first region returned by skimage.measure.regionprops applied to the labeled body mask. The image is then cropped with that box and resampled to the requested (height, width). The original input array is not modified in-place; a new resized array is returned.
        The implementation intentionally does not alter background intensity values outside the body (a commented-out line that would set background to -1024 is disabled because it produced artifacts in some narrow circular field-of-view cases). No files are read or written; the operation is purely in-memory.
    
    Failure modes and caveats:
        If img is not 2-dimensional, indexing and bbox computation will raise errors or produce incorrect results; apply this function only to single 2D slices. If img has zero length in either axis, or if width/height are invalid, scipy.ndimage.zoom may raise an exception. The bounding box is taken from the first region returned by regionprops on the labeled mask; in images with multiple disconnected body regions this may not be the largest region. Resizing uses linear interpolation (order=1), which can blur very small structures and slightly change intensity values relevant for quantitative tasks; consider this when preparing inputs for trained segmentation models.
    """
    from lungmask.utils import crop_and_resize
    return crop_and_resize(img, width, height)


################################################################################
# Source: lungmask.utils.keep_largest_connected_component
# File: lungmask/utils.py
# Category: valid
################################################################################

def lungmask_utils_keep_largest_connected_component(mask: numpy.ndarray):
    """lungmask.utils.keep_largest_connected_component: Return a binary mask that contains only the largest connected component from an input segmentation label map. This function is used in the lungmask pipeline to remove small disconnected islands (false positive fragments) that can appear in per-slice or per-volume lung segmentation outputs, keeping the primary contiguous lung region used for downstream processing (e.g., volume measurements, lobe assignment, or visualization).
    
    Args:
        mask (numpy.ndarray): Input label map produced by a segmentation model or preprocessing step. This numpy.ndarray is expected to contain integer labels or boolean values where non-zero (True) elements are considered foreground. The array may represent a single 2D slice or a 3D volume as used throughout the lungmask package. The function treats all non-zero values as foreground regardless of specific label values (for example, left/right lung labels 1/2 or lobe labels 1-5 are all considered foreground).
    
    Returns:
        numpy.ndarray: A binary numpy.ndarray (dtype bool) of the same shape as the input mask where True (or 1) indicates voxels/pixels belonging to the single largest connected component found in the input. All other locations are False. The returned array is a newly created array; the input mask is not modified in place.
    
    Detailed behavior, side effects, defaults, and failure modes:
        The implementation labels connected foreground regions using skimage.measure.label with its default connectivity and then computes region properties with skimage.measure.regionprops to determine each component's area. The component with the largest area (number of voxels/pixels) is kept; all other components are removed. If multiple components share the same maximum area, the function selects one of them according to numpy.argsort ordering (i.e., the last index in the sorted area array) — selection in the case of exact ties is therefore deterministic to numpy.argsort but not semantically defined as a preference for any particular spatial component. The function returns a boolean mask (mask == max_region) corresponding to the selected component.
    
        The function does not perform any intensity normalization, morphological operations, or label remapping beyond converting the chosen region to a binary mask. It has no side effects on external state or file I/O.
    
        If the input contains no foreground (all zeros or all False), skimage.measure.regionprops will yield no regions and the function will raise an IndexError when attempting to select the largest component. Callers should check for foreground presence (for example, using numpy.any(mask)) before invoking this function if empty inputs are possible in their workflow.
    """
    from lungmask.utils import keep_largest_connected_component
    return keep_largest_connected_component(mask)


################################################################################
# Source: lungmask.utils.load_input_image
# File: lungmask/utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def lungmask_utils_load_input_image(
    path: str,
    disable_tqdm: bool = False,
    read_metadata: bool = False
):
    """Loads an image from a filesystem path and returns it as a SimpleITK Image. If path points to a file, the function uses SimpleITK's ImageFileReader to load that file. If path points to a directory, the function searches for DICOM series in the directory (calls read_dicoms with original=False, primary=False) and selects the largest series by voxel count (product of each volume's GetSize()) when multiple series are present. This function is intended to produce a sitk.Image suitable as input to the lungmask inference pipeline (for example LMInferer.apply) and to downstream U-net models described in the package README.
    
    Args:
        path (str): File or folder path to be loaded. If path is a file, it will be read with SimpleITK ImageFileReader. If path is a folder, a DICOM series is expected and the function will search that folder for one or more DICOM volumes using the package's read_dicoms helper.
        disable_tqdm (bool): Disable tqdm progress bar. This flag is forwarded to read_dicoms when a folder is processed so that long DICOM loading operations do not display a progress bar when True. Defaults to False.
        read_metadata (bool): Read the metadata (including DICOM tags) from the input and store it in the returned sitk.Image. When loading a single file, the function iterates reader.GetMetaDataKeys() and copies each key/value into input_image.SetMetaData(key, value). When loading a DICOM folder, this flag is forwarded to read_dicoms so that metadata from the DICOM series is read and attached to the returned image. Defaults to False.
    
    Returns:
        sitk.Image: A SimpleITK Image containing the loaded volume. The returned image is ready to be used by the lungmask inference code (for example LMInferer.apply) and, if read_metadata was True, contains meta-data/DICOM tags accessible via the image's GetMetaData API.
    
    Behavior, side effects, defaults, and failure modes:
        - Logging: The function emits informational logs via logger.info when a file or folder is detected and a warning via logger.warning if multiple DICOM volumes are found and the largest is chosen.
        - Multiple DICOM series: If multiple DICOM series are present in the folder, the function selects the single volume with the largest number of voxels (computed as np.prod(v.GetSize())). This selection strategy is intended to pick the primary volume for segmentation but may not always match user intent if multiple relevant series exist.
        - No DICOMs found: If the path is a folder and read_dicoms returns zero volumes, the function calls sys.exit("No dicoms found!"), which terminates the Python process with that message. Callers that require non-terminating behavior should handle this externally (for example, verify inputs before calling).
        - I/O and decoding errors: If the specified file is not a readable image or SimpleITK fails to decode the file, SimpleITK will raise its usual exceptions propagated from reader.Execute(). The function does not catch these exceptions internally.
        - Types and expectations: The function expects path to be a filesystem path string. It returns a SimpleITK Image object (sitk.Image) and does not perform type conversion to numpy arrays; for numpy consumption, callers must convert the sitk.Image themselves. The function does not alter image spacing/origin/direction beyond what the underlying readers provide.
        - Performance: Loading large DICOM folders or volumes can be time consuming; disable_tqdm only controls the progress bar display and does not affect underlying I/O performance.
    """
    from lungmask.utils import load_input_image
    return load_input_image(path, disable_tqdm, read_metadata)


################################################################################
# Source: lungmask.utils.postprocessing
# File: lungmask/utils.py
# Category: valid
################################################################################

def lungmask_utils_postprocessing(
    label_image: numpy.ndarray,
    spare: list = [],
    disable_tqdm: bool = False,
    skip_below: int = 3
):
    """lungmask.utils.postprocessing
    Post-process a labeled lung segmentation volume by remapping small connected components to neighboring labels, keeping only the largest connected component per original label, and removing labels listed in a spare mapping. This function is used in the lungmask pipeline to clean and fuse outputs from different models (for example when fusing LTRCLobes and R231 results), to remove small false-positive regions, and to ensure coherent left/right or lobe labelings as described in the project README (two-label outputs: 1=Right lung, 2=Left lung; five-label lobe outputs: 1..5 correspond to specific lobes).
    
    Args:
        label_image (numpy.ndarray): Label image (integer valued) to be processed. This is the input segmentation volume produced by a lung segmentation model (e.g., U-net(R231) or U-net(LTRCLobes)). Each voxel value represents a label id (0 for background, positive integers for lung parts or lobes). The function expects non-negative integer labels; very large maximum label values increase memory for internal arrays since the algorithm allocates arrays sized by the maximum label. The input array itself is not modified in-place; a new array is produced and returned.
        spare (list): Labels that are treated as temporary/filler labels: components with these label ids will be remapped to neighboring non-spare labels during post-processing and will not appear in the final returned labeling. This is intended for use in label-fusion workflows (for example when a "filling" model supplies candidate regions that should be merged into existing lobes or lungs). Defaults to [] (note: the default is a mutable list object shared across calls; to avoid unexpected persistence between calls, pass an explicit list).
        disable_tqdm (bool): If True, progress display via tqdm is disabled. If False (default), tqdm will display a progress bar while iterating over connected components. This only affects user-visible progress output and does not change algorithmic behavior.
        skip_below (int): Threshold for very small connected components. Any connected component with area smaller than this value will not be considered for merging into neighbors and will be removed from the final labeling. This parameter is a runtime/performance optimization and defaults to 3. Components with area >= skip_below can be merged into the neighbor with which they share the largest border (unless that neighbor is also in spare).
    
    Returns:
        numpy.ndarray: Postprocessed volume with the same spatial shape as label_image. The returned array contains integer labels (the implementation converts outputs to dtype np.uint8) where:
            - small spurious components below skip_below have been removed,
            - for each original label only the largest connected component is preserved as the primary region and smaller components of that label are either merged into neighboring labels that share the largest boundary or removed,
            - any labels listed in spare have been mapped to neighbors and then suppressed (set to background) in the final result,
            - simple holes/voids are filled (per-slice area closing for single-slice inputs, volumetric fill for multi-slice inputs) to produce contiguous regions.
        The function may raise or behave unexpectedly if label_image contains negative values, non-integer types, or extremely large label ids (because internal arrays use uint8 for label remapping and allocate arrays based on the maximum label); in particular, label ids larger than 255 may be truncated when converted to the output dtype. The routine uses skimage.measure.regionprops, binary dilation, and area-based morphology; its runtime and memory usage scale with volume size and the maximum label value.
    """
    from lungmask.utils import postprocessing
    return postprocessing(label_image, spare, disable_tqdm, skip_below)


################################################################################
# Source: lungmask.utils.preprocess
# File: lungmask/utils.py
# Category: valid
################################################################################

def lungmask_utils_preprocess(img: numpy.ndarray, resolution: list = [192, 192]):
    """Preprocesses a 3D CT volume by intensity clipping, per-slice cropping to the body, and resizing each slice to a fixed in-plane resolution. This function is used in the lungmask pipeline to prepare numpy array volumes for U-net based lung segmentation models (for example U-net(R231) and LTRCLobes) by mapping raw Hounsfield Unit (HU) intensities into a stable range, removing image borders outside the body, and producing uniformly sized 2D slices that the models expect.
    
    Args:
        img (numpy.ndarray): Input CT volume to be preprocessed. Expected to be a 3-dimensional numpy array where the first axis indexes slices (axial planes), the second axis corresponds to chest-to-back, and the third axis corresponds to right-to-left (this ordering is the numpy array convention supported by lungmask as described in the README). Values are expected to be encoded in Hounsfield Units (HU). The function creates an internal copy of this array and does not modify the provided array in-place.
        resolution (list): Target in-plane size after preprocessing given as [width, height]. Defaults to [192, 192]. The first element is used as the width argument and the second as the height argument when calling the underlying crop_and_resize routine. The function expects a list-like object with exactly two integer-like elements; providing a differently shaped or non-indexable object will raise an indexing or type error.
    
    Behavior, defaults, and side effects:
        The function first makes a full numpy copy of img and then clips all intensity values to the HU range [-1024, 600]. Clipping to this range reduces outlier intensities and matches the intensity windowing used by the trained lung segmentation models. The function then iterates over slices along the first axis. For each slice it calls crop_and_resize(slice, width=resolution[0], height=resolution[1]) to (a) compute a crop bounding box that isolates the body region and (b) resize the cropped body region to the specified in-plane resolution. crop_and_resize is responsible for determining the exact cropping strategy and the bounding box format; this function preserves the order of slices so that the returned bounding boxes correspond elementwise to the returned slices. The default resolution [192, 192] matches the network input commonly used by the provided models; changing resolution changes the spatial size of each 2D slice fed to the model and may affect model performance and runtime. The function does not perform any disk I/O and returns its results in memory. If img is not a 3D array with slices along axis 0 (for example a 2D array or an array with a different axis ordering), the preprocessing may produce incorrect results or raise errors. crop_and_resize may itself raise errors for slices where a body region cannot be found or for invalid slice shapes; such exceptions propagate to the caller.
    
    Returns:
        Tuple[numpy.ndarray, numpy.ndarray]: A tuple containing two numpy.ndarray elements. The first element is the preprocessed image volume as a numpy.ndarray with shape (N, H, W), where N is the number of input slices (the length of the first axis of img), H is resolution[1] (height) and W is resolution[0] (width); pixel intensities in this array have been clipped to the interval [-1024, 600] and each slice was cropped and resized to the target in-plane resolution. The second element is a numpy.ndarray containing the cropping bounding box information for each slice in the same slice order; each element is the bounding box as returned by crop_and_resize for the corresponding slice. These bounding boxes can be used to map predictions or masks produced on the preprocessed slices back to the original slice coordinates.
    """
    from lungmask.utils import preprocess
    return preprocess(img, resolution)


################################################################################
# Source: lungmask.utils.reshape_mask
# File: lungmask/utils.py
# Category: valid
################################################################################

def lungmask_utils_reshape_mask(mask: numpy.ndarray, tbox: numpy.ndarray, origsize: tuple):
    """Reshapes and places a predicted 2D mask back into the original image coordinate space using a provided bounding box.
    This function is used in the lungmask segmentation pipeline to reverse a prior crop/resize operation: models often predict a mask on a cropped and rescaled field-of-view; reshape_mask rescales that predicted mask with nearest-neighbor interpolation and inserts it into a zero background of the original CT image size so that the segmentation aligns with the original image coordinates used by downstream processing or file output (e.g., SimpleITK images or ITK formats described in the README).
    
    Args:
        mask (numpy.ndarray): 2D mask array produced by the model or an intermediate step. This array is the mask to be resampled and is interpreted as covering the bounding box region given by tbox. The function uses nearest-neighbor interpolation (order=0) to preserve label integers during resampling. Practical role: this is typically a per-slice segmentation output (labels for lung, lobes, etc.) that must be mapped back to full-image coordinates for visualization, metric computation, or saving to the original image space.
        tbox (numpy.ndarray): 1D array specifying the bounding box in the original image coordinates that corresponds to the current mask. Expected format is [row_start, col_start, row_end, col_end] where row_end > row_start and col_end > col_start; values should be integers and refer to indices in the original image coordinate system. Practical role: tbox defines the target rectangle inside the full-size image where the resampled mask will be placed (the field of view the model processed).
        origsize (tuple): Tuple describing the original image size into which the resampled mask will be placed. For 2D masks this should be (rows, cols). In the lungmask pipeline this corresponds to the in-plane image dimensions of the CT slice or the 2D plane extracted from a numpy volume (note: the package documents a numpy volume axis convention when working with 3D arrays; this function operates on the corresponding 2D in-plane size). Practical role: origsize determines the shape of the returned array and the coordinate frame for the tbox insertion.
    
    Returns:
        numpy.ndarray: A new array of shape origsize containing the resampled mask placed at the location defined by tbox and zeros elsewhere. The function creates a zero background, rescales mask to match the tbox size using nearest-neighbor interpolation and assigns the resampled patch into res[tbox[0]:tbox[2], tbox[1]:tbox[3]]. Notes on behavior and failure modes: the returned array dtype follows NumPy assignment/upcasting rules (the function initializes a zero array and then assigns the resampled mask, which may change dtype); tbox values must define a valid rectangular region inside origsize and should be integer-valued; non-integer or out-of-bounds tbox entries, mismatched sizes between the rescaled mask and the tbox rectangle, or masks of unexpected dimensionality may raise errors or produce undefined placement/behavior. There are no in-place side effects on the input arrays; the function always allocates and returns a new array.
    """
    from lungmask.utils import reshape_mask
    return reshape_mask(mask, tbox, origsize)


################################################################################
# Source: lungmask.utils.simple_bodymask
# File: lungmask/utils.py
# Category: valid
################################################################################

def lungmask_utils_simple_bodymask(img: numpy.ndarray):
    """lungmask.utils.simple_bodymask computes a fast, heuristic binary body mask for a single CT slice by thresholding at -500 Hounsfield units (HU), performing morphological cleanup, keeping the largest connected component, and rescaling the result back to the input resolution. This function is used in the lungmask pipeline to isolate the patient body / chest region on a single 2D CT slice so subsequent lung segmentation steps can focus on the relevant image area and ignore background and small artifacts.
    
    Args:
        img (numpy.ndarray): 2D CT image representing a single axial slice encoded in Hounsfield units (HU). The array should be a two-dimensional NumPy array (height, width). The function assumes HU intensity values are present; if the input is not encoded in HU (e.g., arbitrary image intensities or externally scaled values), the hard-coded threshold at -500 HU will not give meaningful results.
    
    Returns:
        numpy.ndarray: Binary mask as a NumPy array with the same shape as the input image. The mask contains foreground body pixels (value 1 or True) and background pixels (value 0 or False). The returned mask is produced by the following deterministic steps implemented for practical use in the lung segmentation domain: the input slice is resized to 128x128 using nearest-neighbor interpolation (ndimage.zoom with order=0), thresholded at -500 HU to create an initial foreground, closed to remove small holes, hole-filled using a 3x3 structure element, eroded (2 iterations) to remove small objects and thin connections, labeled with connectivity=1, and reduced to the largest connected region (if any regions are found). The largest region is then dilated (2 iterations) and rescaled back to the original input shape using nearest-neighbor interpolation. If no connected regions are found after labeling, the function returns an all-zero mask of the input shape. The function does not modify the input array in-place.
    
    Behavioral notes and failure modes:
    - The threshold is fixed at -500 HU; this is a heuristic chosen to separate body tissue from air/background in typical chest CTs. It is not adaptive and may fail on images not encoded in HU or with extreme intensity shifts.
    - The function expects a single 2D slice. Passing higher-dimensional volumes (e.g., 3D stacks) is not supported and will produce unintended results because the function rescales based on img.shape and performs 2D morphological operations.
    - Rescaling uses nearest-neighbor interpolation (order=0) to preserve binary-like decisions during zoom operations; this can introduce aliasing when input resolution differs greatly from 128x128.
    - Morphological parameters are fixed: binary_closing, binary_fill_holes with a 3x3 structure, binary_erosion iterations=2, and binary_dilation iterations=2. These choices remove small objects and fill cavities but may also remove thin legitimate anatomy in severely cropped or low-quality slices.
    - The largest connected component selection assumes the body is the largest contiguous foreground region in the slice; in rare cases (e.g., highly cropped slices, large external artifacts) this assumption can select an incorrect region or yield an empty mask.
    - The function is lightweight and intended as a fast preprocessing heuristic within the automated lung segmentation workflow; it is not a replacement for more robust body segmentation methods when high accuracy is required.
    """
    from lungmask.utils import simple_bodymask
    return simple_bodymask(img)


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
