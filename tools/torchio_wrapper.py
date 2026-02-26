"""
Regenerated Google-style docstrings for module 'torchio'.
README source: others/readme/torchio/README.md
Generated at: 2025-12-02T00:15:30.620851Z

Total functions: 3
"""


import numpy

################################################################################
# Source: torchio.utils.get_batch_images_and_size
# File: torchio/utils.py
# Category: valid
################################################################################

def torchio_utils_get_batch_images_and_size(batch: dict):
    """Get names of image entries in a batch and the number of channels (size) per image.
    
    Args:
        batch (dict): Batch dictionary produced by a torchio SubjectsLoader when extracting
            data from a torchio SubjectsDataset. In the TorchIO data pipeline, each entry
            in this dictionary is expected to map a string image name (for example
            't1', 'flair', or 'label') to an image-like dictionary (the representation
            used internally by torchio.Image/SubjectsLoader). This function inspects the
            batch to locate those entries that are dictionaries containing the key
            constants.DATA. The value at constants.DATA is expected to be a sequence
            or tensor-like object supporting len() (in TorchIO this is typically a
            PyTorch tensor with channels-first shape, e.g. (C, W, H, D)). The function
            uses that len() to determine the per-image "size" (practically, the number
            of channels C). This parameter must be the raw batch mapping produced by
            the loader; passing other structures will likely result in no images found
            or a RuntimeError.
    
    Returns:
        tuple[list[str], int]: A tuple where the first element is a list of strings
        containing the keys (image names) from the input batch that appear to be
        TorchIO images (i.e., their value is a dict containing constants.DATA). The
        second element is an integer equal to len(value[constants.DATA]) for the last
        image-like dictionary found while iterating over batch.items(). In the
        TorchIO medical-imaging domain, this integer commonly corresponds to the
        number of channels per image (for example, 1 for single-contrast MRI, 3 for
        multi-channel inputs). Note that if multiple image entries have different
        lengths, the returned size will reflect the last matching entry encountered;
        callers that require consistency across images should verify that all
        returned names share the same size after calling this function.
    
    Raises:
        RuntimeError: If the batch does not contain any entries that are dictionaries
            with the constants.DATA key (i.e., it does not seem to contain any
            TorchIO Image representations). This typically indicates that the batch
            was not produced by a SubjectsLoader/SubjectsDataset pair, that the
            SubjectsLoader configuration filtered out images, or that the data
            structure has been altered; users should inspect the batch contents to
            diagnose the issue.
    """
    from torchio.utils import get_batch_images_and_size
    return get_batch_images_and_size(batch)


################################################################################
# Source: torchio.utils.get_subjects_from_batch
# File: torchio/utils.py
# Category: valid
################################################################################

def torchio_utils_get_subjects_from_batch(batch: dict):
    """torchio.utils.get_subjects_from_batch: Reconstruct a list of torchio.data.Subject instances from a collated minibatch dictionary produced by a SubjectsLoader.
    
    Args:
        batch (dict): Dictionary produced by a :class:`tio.SubjectsLoader` when collating samples from a :class:`torchio.SubjectsDataset`. In practice this dictionary contains entries for image fields and non-image fields. Image entries are themselves dictionaries containing per-batch tensors and metadata under keys used by the library (for example constants.DATA, constants.AFFINE, constants.PATH, constants.TYPE). Non-image entries are sequence-like values (e.g., lists or tensors) with one element per sample in the batch. This function expects the dictionary structure created by the library's collate function and uses get_batch_images_and_size(batch) to determine which keys correspond to image data and the batch size. Provide this exact collated output when calling the function; passing other dictionary shapes will likely raise errors.
    
    Returns:
        list: A Python list of torchio.data.Subject instances reconstructed from the input batch. For each index in the minibatch the function:
            - Builds a subject_dict by iterating over batch items. For keys identified as image names, it extracts the i-th element of the per-image tensors/metadata (data, affine, path, type), creates a torchio.data.ScalarImage or torchio.data.LabelMap instance (LabelMap is used when the stored type equals constants.LABEL, otherwise ScalarImage) with tensor=data, affine=affine, and filename=Path(path).name, and stores that image object under the image name in subject_dict. For non-image keys it takes the i-th value and stores it directly in subject_dict.
            - Constructs a torchio.data.Subject from subject_dict and appends it to the returned list.
            - If the batch contains a constants.HISTORY entry, it expects batch[constants.HISTORY] to be sequence-like with one list of applied transforms per sample; for each transform in that per-sample list the function calls transform.add_transform_to_subject_history(subject), which mutates the subject by recording applied transforms in its history.
    
    Behavior and side effects:
        - The function does not modify the input batch argument itself, but it creates new Subject and Image objects and may mutate those Subject instances when adding transform history.
        - The returned list preserves the minibatch ordering: the i-th element corresponds to the i-th sample in the collated batch.
        - The function relies on the presence and correct structure of image-related keys (e.g., constants.DATA, constants.AFFINE, constants.PATH, constants.TYPE) for image entries. If those keys are missing, have unexpected types, or the per-key sequences are shorter than the detected batch size, the function can raise KeyError, IndexError, or TypeError propagated from attempted indexing or attribute access.
        - There are no default behaviors for missing fields: the function assumes the collated batch follows the library's contract produced by SubjectsLoader/SubjectsDataset.
    
    Failure modes:
        - Passing a non-dict value for batch will typically raise a TypeError.
        - If the collated dictionary does not follow the expected structure (missing keys, incorrect nesting, or mismatched lengths), callers may see KeyError, IndexError, or TypeError.
        - If constants.TYPE values are not equal to the expected constants.LABEL string for label images, images will be created as ScalarImage objects; this follows the library behavior and is not coerced automatically.
    
    Practical significance:
        - In the TorchIO medical imaging pipeline this function is used to convert a minibatch produced for model training or evaluation back into a list of Subject objects suitable for writing to disk, inspection, visualization, or analysis of per-subject transform history. It is therefore an important utility when one needs to bridge batched tensors and the higher-level Subject abstraction used throughout the library.
    """
    from torchio.utils import get_subjects_from_batch
    return get_subjects_from_batch(batch)


################################################################################
# Source: torchio.visualization.get_num_bins
# File: torchio/visualization.py
# Category: valid
################################################################################

def torchio_visualization_get_num_bins(x: numpy.ndarray):
    """Get the optimal number of bins for a histogram using the Freedman–Diaconis rule.
    
    This function implements the Freedman–Diaconis heuristic to choose a bin width that balances histogram resolution and variability of the data. In the context of TorchIO's visualization tools for 3D medical images (for example, plotting intensity histograms of MRI scans to inspect intensity distributions and augmentation effects), this routine computes an integer number of bins intended to minimize the integral of the squared difference between the histogram (relative frequency density) and the underlying theoretical probability density (see the Freedman–Diaconis rule). Concretely, the function computes the 25th and 75th percentiles q25 and q75 of the input values (percentiles are computed over all elements of x, i.e., the input is effectively flattened), forms the interquartile range IQR = q75 - q25, computes the bin width as 2 * IQR * n**(-1/3) where n is the number of samples len(x), and returns round((x.max() - x.min()) / bin_width) as the number of bins.
    
    Args:
        x (numpy.ndarray): A NumPy array of numeric values (typically image intensity samples from TorchIO image objects) used to compute the histogram. The array is treated as a flat collection of values when computing percentiles and min/max. The array must be non-empty and should contain finite numeric values; NaNs or infinities in x will affect percentile and min/max computations and may lead to invalid results or exceptions.
    
    Returns:
        int: The computed number of histogram bins according to the Freedman–Diaconis rule. This integer is intended for use when plotting or analyzing intensity distributions in medical images; a larger value increases histogram resolution while a smaller value increases smoothing. The function has no side effects and is deterministic for a given input array. Failure modes: if x is empty, if the interquartile range (q75 - q25) is zero, or if invalid values (NaN, inf) are present, the intermediate bin_width computation may be zero or non-finite and the function may raise a runtime error (for example, division by zero) or return an unexpected value. The caller should validate x (non-empty, finite, and with non-zero IQR) when these conditions are a concern.
    """
    from torchio.visualization import get_num_bins
    return get_num_bins(x)


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
