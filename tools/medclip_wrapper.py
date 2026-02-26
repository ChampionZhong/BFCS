"""
Regenerated Google-style docstrings for module 'medclip'.
README source: others/readme/medclip/README.md
Generated at: 2025-12-02T00:16:52.532238Z

Total functions: 3
"""


import torch

################################################################################
# Source: medclip.prompts.generate_chexpert_class_prompts
# File: medclip/prompts.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def medclip_prompts_generate_chexpert_class_prompts(n: int = None):
    """Generate text prompts for each CheXpert classification task used by MedCLIP for prompt-based (zero-shot or few-shot) image classification.
    
    This function builds candidate natural-language prompts for each CheXpert class by taking the Cartesian product of three token groups (severity, subtype, location) defined in constants.CHEXPERT_CLASS_PROMPTS for each class key. The resulting prompt strings (one token from each group joined with spaces) are intended to be fed into MedCLIP text encoders, processed by utilities such as medclip.prompts.process_class_prompts, and used by PromptClassifier for ensemble or zero-shot CheXpert label prediction as shown in the project README.
    
    Args:
        n (int): Number of prompts to return per class. If n is None (the default), the function returns the full set of candidate prompts for each class. If n is an integer and 0 <= n < total_candidates, the function randomly samples n unique prompts from the full candidate pool for that class using random.sample. If n >= total_candidates the full candidate pool is returned. Note that n should be an integer or None; passing a non-integer may raise a TypeError on the comparison n < len(...) or a ValueError if random.sample is called with an invalid k (for example, negative n). For reproducible sampling, the caller must seed the global random module (random.seed) before calling this function.
    
    Returns:
        dict: A dictionary mapping each CheXpert class name (string) to a list of prompt strings. Each prompt string is constructed by concatenating one token from the class's severity group, one token from its subtype group, and one token from its location group (in that order) separated by single spaces. The keys of the returned dictionary are the same class names found in constants.CHEXPERT_CLASS_PROMPTS.
    
    Behavior, side effects, and failure modes:
        - The function iterates over constants.CHEXPERT_CLASS_PROMPTS.items() and expects each value to be a mapping containing exactly three iterable groups (accessed as list(v.keys()) then v[keys[0]], v[keys[1]], v[keys[2]]). If the mapping does not contain at least three groups, an IndexError or KeyError may be raised.
        - The function prints a status line for each class to standard output of the form "sample {num} num of prompts for {class} from total {total}", which can produce noisy output in batch runs.
        - Sampling uses the Python random module. To control determinism, callers must set random.seed externally; the function does not modify or return the random state.
        - If n is negative, random.sample will raise a ValueError. If n is not comparable to an integer, a TypeError may occur when comparing n with len(cls_prompts).
        - No external I/O or network access is performed; the function purely constructs and returns in-memory prompt lists.
    """
    from medclip.prompts import generate_chexpert_class_prompts
    return generate_chexpert_class_prompts(n)


################################################################################
# Source: medclip.vision_model.window_partition
# File: medclip/vision_model.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def medclip_vision_model_window_partition(x: torch.Tensor, window_size: int):
    """medclip.vision_model.window_partition partitions a 4-D image feature tensor into non-overlapping square windows. This operation is used in MedCLIP vision models (for example ViT or Swin-style components) to prepare local image patches for windowed self-attention and downstream contrastive learning between medical images and text (e.g., chest x-ray feature maps). The function reshapes and permutes the input so each output row is one window, which simplifies batched window-wise processing used in MedCLIP training and inference.
    
    Args:
        x (torch.Tensor): Input feature tensor with shape (B, H, W, C), where B is batch size, H and W are spatial height and width, and C is the number of channels. This tensor is the image-level feature map produced by a vision backbone in MedCLIP. The function expects a 4-D tensor; values, dtype, and device are preserved in the returned tensor. H and W must be integer multiples of window_size (see failure modes below).
        window_size (int): Size of the square window (number of pixels/patches per side). This is the same window granularity used by windowed attention components in MedCLIP vision models and must be a positive integer that divides both H and W without remainder.
    
    Returns:
        torch.Tensor: A contiguous tensor containing all non-overlapping windows with shape (num_windows * B, window_size, window_size, C), where num_windows = (H // window_size) * (W // window_size). Each row corresponds to one window extracted from the input batch in row-major spatial order. The returned tensor is suitable for batched window-wise attention or per-window processing in MedCLIP pipelines.
    
    Behavior, side effects, defaults, and failure modes:
        - The function does not modify the input tensor in-place; it returns a new tensor that is made contiguous before the final reshape.
        - Device and dtype of the input tensor are preserved in the output.
        - If x is not 4-D or does not have shape (B, H, W, C), the view/permute operations will raise a runtime error (e.g., RuntimeError from torch.view). The caller should ensure x has the expected dimensionality.
        - If H or W is not divisible by window_size, the integer division used in the reshape will be incorrect and torch.view will raise an error. The caller must ensure window_size divides both spatial dimensions exactly.
        - window_size must be a positive integer; passing non-positive values will result in undefined behavior or runtime errors during reshape.
        - No padding or cropping is performed by this function; any required padding to make H and W divisible by window_size must be applied prior to calling window_partition.
    """
    from medclip.vision_model import window_partition
    return window_partition(x, window_size)


################################################################################
# Source: medclip.vision_model.window_reverse
# File: medclip/vision_model.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def medclip_vision_model_window_reverse(
    windows: torch.Tensor,
    window_size: int,
    H: int,
    W: int
):
    """medclip.vision_model.window_reverse reconstructs a full image tensor from non-overlapping square windows. This function is used in MedCLIP vision models to reverse a prior window partitioning step (for example, the window partition used in window-based Vision Transformer or shifted-window attention implementations). In the MedCLIP pipeline this reconstruction is necessary to restore spatial layout of per-window image features so they can be processed as a single image tensor for downstream embedding, contrastive alignment with text, or prompt-based classification.
    
    Args:
        windows (torch.Tensor): Input tensor of windowed patches with shape (num_windows*B, window_size, window_size, C). Each entry is a window (patch) produced by a prior partitioning of B images into non-overlapping windows of spatial size window_size x window_size and channel dimension C. The function infers the batch size B from this first dimension using the relation B = int(windows.shape[0] / (H * W / window_size / window_size)).
        window_size (int): Size of each square window along both spatial axes. This is the same window_size that was used during the partitioning that produced windows. It determines how many spatial pixels from the original images are contained in each window.
        H (int): Original full image height (number of rows) before partitioning. H must be divisible by window_size so that an integer number of windows tile the height.
        W (int): Original full image width (number of columns) before partitioning. W must be divisible by window_size so that an integer number of windows tile the width.
    
    Behavior and practical details:
        The function infers the batch dimension B from windows.shape[0] and the provided H, W, and window_size using the formula B = int(windows.shape[0] / (H * W / window_size / window_size)). It then reshapes and permutes the tensor to reassemble the original spatial layout, returning a contiguous tensor of shape (B, H, W, C). The returned tensor preserves the dtype and device of the input windows tensor and is suitable for further processing by MedCLIP vision encoders or feature aggregation layers.
        This routine performs no in-place modification of the input; it constructs and returns a new tensor view (made contiguous) representing the reconstructed images.
    
    Failure modes and errors:
        The function requires H and W to be exact multiples of window_size. If H or W is not divisible by window_size, or if windows.shape is inconsistent with the provided H, W, and window_size (for example, an incorrect number of windows or mismatched channel count), the internal view/permute operations will fail and torch will raise a runtime error (for example, a RuntimeError indicating an incompatible shape for view). Callers should validate shapes before invoking this function to produce clear diagnostics in MedCLIP data preprocessing.
    
    Returns:
        torch.Tensor: Reconstructed image tensor of shape (B, H, W, C). B is inferred as described above. This tensor contains the same per-window data as the input windows but reorganized into full images matching the original spatial dimensions used in MedCLIP vision processing.
    """
    from medclip.vision_model import window_reverse
    return window_reverse(windows, window_size, H, W)


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
