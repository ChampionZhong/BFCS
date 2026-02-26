"""
Regenerated Google-style docstrings for module 'monai'.
README source: others/readme/monai/README.md
Generated at: 2025-12-02T01:51:35.644429Z

Total functions: 96
"""


import numpy
import torch

################################################################################
# Source: monai.apps.auto3dseg.utils.get_name_from_algo_id
# File: monai/apps/auto3dseg/utils.py
# Category: valid
################################################################################

def monai_apps_auto3dseg_utils_get_name_from_algo_id(id: str):
    """Extract the algorithm name from an algorithm identifier used by MONAI's auto3dseg utilities.
    
    This utility is used in MONAI (a PyTorch-based framework for medical imaging) to parse compact algorithm identifiers that follow a convention used in Auto3DSeg workflows, model bundles, and the MONAI Model Zoo. Given an identifier that encodes the algorithm name, the cross-validation fold, and other metadata (conventionally "name_fold_other"), this function returns the human-meaningful algorithm name portion for use in experiment naming, logging, grouping results by algorithm across folds, and other bookkeeping tasks in 3D medical image segmentation pipelines.
    
    Args:
        id (str): Algorithm identifier string that is expected to follow the convention "name_fold_other". The function splits this string on the first underscore character and returns the first segment as the algorithm name. The caller must pass a Python str; passing non-str types will raise a TypeError from the underlying string operation.
    
    Behavior and side effects:
        The function performs a simple, deterministic string operation equivalent to id.split("_")[0] and has no side effects (it does not modify input data or external state). If id contains no underscore characters, the entire id string is returned unchanged. If id is the empty string, an empty string is returned. If id begins with an underscore, the function will return an empty string as the first segment. The function does not validate that the returned name corresponds to a known algorithm; callers should perform additional validation if needed.
    
    Returns:
        str: The extracted algorithm name, defined as the substring of id before the first underscore. This value is intended for use as a human-readable algorithm identifier within MONAI Auto3DSeg workflows (e.g., for naming outputs, logs, or grouping per-algorithm results).
    """
    from monai.apps.auto3dseg.utils import get_name_from_algo_id
    return get_name_from_algo_id(id)


################################################################################
# Source: monai.apps.nnunet.nnunet_bundle.convert_monai_bundle_to_nnunet
# File: monai/apps/nnunet/nnunet_bundle.py
# Category: valid
################################################################################

def monai_apps_nnunet_nnunet_bundle_convert_monai_bundle_to_nnunet(
    nnunet_config: dict,
    bundle_root_folder: str,
    fold: int = 0
):
    """Convert a MONAI bundle to nnU-Net format for use in nnU-Net training and inference workflows in medical imaging. This function reads MONAI bundle checkpoint and metadata files from a bundle_root_folder, extracts and re-structures optimizer state and network weights, creates the expected nnU-Net folder layout under the nnUNet_results environment path, writes nnU-Net style checkpoint files (checkpoint_final.pth and checkpoint_best.pth), and copies required dataset and plan JSON files so the result can be consumed by nnU-Net training/evaluation code.
    
    Args:
        nnunet_config (dict): Configuration dictionary that tells how to map the MONAI bundle into nnU-Net project structure and naming. This function expects at minimum the key "dataset_name_or_id" whose value is a string giving the dataset name or ID used by nnU-Net. Optional keys are "nnunet_trainer" and "nnunet_plans" (both strings). If "nnunet_trainer" is absent, the function uses the default "nnUNetTrainer". If "nnunet_plans" is absent, the function uses the default "nnUNetPlans". The dataset_name_or_id is used (via maybe_convert_to_dataset_name) to compute the target nnU-Net model folder under the directory defined by the environment variable nnUNet_results; nnunet_trainer and nnunet_plans are used to compose the model-folder suffix "<trainer>__<plans>__3d_fullres". This parameter therefore controls target folder naming and where converted checkpoints and copied JSON metadata will be written for downstream nnU-Net training or inference.
        bundle_root_folder (str): Filesystem path to the MONAI bundle root folder (string). The function expects this folder to contain a models subfolder with the following structure and files: "nnunet_checkpoint.pth" (a base nnUNet checkpoint template), a fold-specific subfolder models/fold_{fold} containing epoch checkpoints named with the prefix "checkpoint_epoch=" and suffix ".pt" (e.g., checkpoint_epoch=123.pt) and best-checkpoint files named with the prefix "checkpoint_key_metric=" and suffix ".pt" (e.g., checkpoint_key_metric=0.832.pt), as well as models/dataset.json and models/plans.json. The function reads these files with torch.load (called with weights_only=True as in the source) and relies on the filename patterns to parse numeric epoch values and best metric identifiers. The path is also used as the source for copied JSON files when the target nnU-Net folder does not already contain them. This argument is required and must point to a MONAI bundle laid out as expected by MONAI bundle conventions used in the repository.
        fold (int): Fold index for cross-validation (integer, default 0). The fold value selects which fold-specific subfolder inside the MONAI bundle to use when locating the epoch and best-key-metric checkpoint files (the function uses models/fold_{fold}). The fold value is also used to create and write into the corresponding target subfolder named fold_{fold} under the constructed nnU-Net model folder. Use the same fold indexing convention that the MONAI bundle uses when it saved its checkpoints.
    
    Returns:
        None: This function does not return a value. Instead it produces side effects on the filesystem and may raise exceptions on errors. Side effects include creating the target nnU-Net model folder (under the path computed from the environment variable nnUNet_results and the dataset name/id, trainer and plans), creating the fold_{fold} subdirectory, writing two checkpoint files into that subdirectory named checkpoint_final.pth and checkpoint_best.pth (these are torch-saved dictionaries with keys "optimizer_state", "network_weights", "current_epoch", "logging", "_best_ema", and "grad_scaler_state" as constructed in the function), and copying dataset.json, plans.json, dataset_fingerprint.json, and nnunet_checkpoint.pth from the bundle or preprocessed locations into the target model folder when those files do not already exist.
    
    Behavior, side effects, defaults, and failure modes:
        - Environment dependencies: The function reads the environment variables nnUNet_results and nnUNet_preprocessed to determine target and preprocessed dataset paths. If these environment variables are not set or point to invalid locations, the function will raise a KeyError or may fail with filesystem errors. Ensure the environment variables are correctly configured in the execution environment before calling the function.
        - File I/O expectations: The function expects the MONAI bundle at bundle_root_folder to contain models/nnunet_checkpoint.pth (a base template), models/dataset.json, models/plans.json, and a folder models/fold_{fold} containing epoch checkpoints with filenames that start with "checkpoint_epoch=" and end with ".pt" and best-checkpoint files that start with "checkpoint_key_metric=" and end with ".pt". The function parses epoch numbers by stripping the known prefix "checkpoint_epoch=" and suffix ".pt" and converts the remaining substring to int; it parses best-key-metric identifiers by stripping "checkpoint_key_metric=" and ".pt" and sorting those identifiers lexicographically. If no epoch checkpoint files are found, the function will attempt to access epochs[-1] and raise an IndexError. If checkpoint files exist but do not follow the exact naming patterns, parsing will fail or produce incorrect results.
        - Checkpoint handling: The function loads a base nnunet_checkpoint dictionary from models/nnunet_checkpoint.pth, then replaces its "optimizer_state" with the optimizer state from the last epoch checkpoint in the MONAI bundle (the epoch determined by the latest numeric epoch extracted from the filenames). It copies network weights from the MONAI checkpoint into an ordered dictionary (odict) under the key "network_weights" for both the final and best checkpoints. For the "final" file, current_epoch is set to the final epoch number, logging is set using nnUNetLogger().get_checkpoint(), _best_ema is set to 0, and grad_scaler_state is set to None. For the "best" file, optimizer_state and network_weights come from the MONAI best checkpoint. The function saves the resulting nnunet_checkpoint dictionaries to checkpoint_final.pth and checkpoint_best.pth under the target fold folder using torch.save.
        - Metadata copying: If the target nnU-Net model folder does not already contain dataset.json, plans.json, dataset_fingerprint.json, or nnunet_checkpoint.pth, the function copies these files from the MONAI bundle models folder or from the nnUNet_preprocessed folder as appropriate. These copied JSON files are required by nnU-Net for dataset description, preprocessing plans, and dataset fingerprinting, and are copied only when missing to avoid overwriting existing nnU-Net model folders.
        - Logging: The function creates a nnUNetLogger instance and uses its get_checkpoint() output to populate the "logging" key in the saved final checkpoint. This integrates MONAI-converted checkpoints with nnU-Net logging expectations.
        - Error propagation: Typical failures include missing or malformed bundle files, inability to read or write files due to permissions, incorrect environment variable configuration, torch.load/save errors, and parsing errors if checkpoint filename patterns differ from the expected "checkpoint_epoch=" and "checkpoint_key_metric=" prefixes and ".pt" suffix. These errors are not swallowed; the function will raise the underlying exceptions to the caller.
        - Deterministic naming conventions: The function constructs the target nnU-Net model folder as <nnUNet_results>/<dataset_name>/<nnunet_trainer>__<nnunet_plans>__3d_fullres and creates a fold_{fold} subdirectory. This naming is important so that nnU-Net utilities can locate and use the converted checkpoints and metadata when training or running inference.
        - In-place modification: The function does not modify the original MONAI bundle files; it reads from the bundle_root_folder and writes new files into the nnU-Net results directory. However, it will copy files from the bundle into the nnU-Net model folder when those files are missing there.
        - Ordering and format: The function uses an ordered dictionary (odict) for network_weights to preserve parameter ordering expected by downstream nnU-Net components. It calls torch.save to produce PyTorch checkpoint files compatible with nnU-Net checkpoints as produced by the existing codebase.
    
    Practical significance in the MONAI/nnU-Net domain:
        - Use case: This function is intended for practitioners who have trained or exported a model using MONAI bundle conventions and want to evaluate, continue training, or deploy that model using nnU-Net tooling. It automates the conversion of MONAI-format checkpoints and metadata into the file and checkpoint structure required by nnU-Net, enabling interoperability between MONAI bundles and nnU-Net training/inference pipelines commonly used in medical image segmentation research and clinical-research workflows.
        - How to use: Ensure the MONAI bundle is complete and contains the expected models/* files, set the environment variables nnUNet_results and nnUNet_preprocessed to valid paths, provide an nnunet_config dict with dataset_name_or_id (and optionally nnunet_trainer and nnunet_plans), and call this function with the appropriate fold index. After successful execution, check the target nnU-Net folder for checkpoint_final.pth, checkpoint_best.pth, and required JSON metadata before using nnU-Net commands that expect that structure.
    """
    from monai.apps.nnunet.nnunet_bundle import convert_monai_bundle_to_nnunet
    return convert_monai_bundle_to_nnunet(nnunet_config, bundle_root_folder, fold)


################################################################################
# Source: monai.apps.nnunet.nnunet_bundle.convert_nnunet_to_monai_bundle
# File: monai/apps/nnunet/nnunet_bundle.py
# Category: valid
################################################################################

def monai_apps_nnunet_nnunet_bundle_convert_nnunet_to_monai_bundle(
    nnunet_config: dict,
    bundle_root_folder: str,
    fold: int = 0
):
    """Convert nnUNet model checkpoints and configuration to a MONAI bundle on disk for use
    in MONAI workflows and the MONAI Model Zoo. This function is used in the medical
    imaging domain to translate nnUNet v2 training outputs into the MONAI bundle layout
    (expected by MONAI examples and bundle-based inference/training workflows).
    
    Args:
        nnunet_config (dict): Configuration dictionary for nnUNet required by this converter.
            This dictionary must contain the key "dataset_name_or_id" (a dataset identifier or name
            used by nnUNet). It may optionally contain "nnunet_trainer", "nnunet_plans", and
            "nnunet_configuration" to override defaults. The function uses these values to
            locate the nnUNet model folder under the environment variable nnUNet_results.
            Specifically, after optional overrides the code uses:
            nnunet_trainer default "nnUNetTrainer", nnunet_plans default "nnUNetPlans",
            nnunet_configuration default "3d_fullres". The dataset identifier is passed through
            maybe_convert_to_dataset_name from nnunetv2.utilities.dataset_name_id_conversion
            to obtain the dataset_name used in the source results path. If "dataset_name_or_id"
            is missing, a KeyError will be raised by this function.
        bundle_root_folder (str): Root folder path where the resulting MONAI bundle files
            and folders will be created. The function writes into bundle_root_folder/models and
            into bundle_root_folder/models/fold_{fold}. Side effects include creating directories
            (mkdir with parents=True, exist_ok=True), writing PyTorch checkpoint files using torch.save,
            and conditionally copying plans.json and dataset.json from the located nnUNet results folder
            into bundle_root_folder/models if those files do not already exist. This parameter is
            interpreted as a file-system path string and must be writable by the process.
        fold (int): Fold number of the nnUNet model to convert (default is 0). The function will
            read checkpoint files from the nnUNet results folder under the subfolder "fold_{fold}"
            and will write model weights for this fold to bundle_root_folder/models/fold_{fold}/model.pt
            and best_model.pt. Provide the integer fold corresponding to the trained nnUNet fold you
            intend to convert.
    
    Returns:
        None: The function does not return a value. Its effect is purely side-effecting: it
        creates and writes files under bundle_root_folder, specifically:
        - Saves a minimal nnunet_checkpoint.pth at bundle_root_folder/models/ that contains
          inference_allowed_mirroring_axes, init_args, and trainer_name extracted from the nnUNet
          "checkpoint_final.pth".
        - Creates bundle_root_folder/models/fold_{fold}/ and saves model.pt containing
          {"network_weights": <weights from nnUNet checkpoint_final>} and best_model.pt containing
          {"network_weights": <weights from nnUNet checkpoint_best>}.
        - Copies plans.json and dataset.json from the discovered nnUNet model folder into
          bundle_root_folder/models if those files do not already exist there.
        Failure modes are raised as standard Python exceptions: accessing missing keys in nnunet_config
        raises KeyError; missing environment variable nnUNet_results raises KeyError; torch.load or
        torch.save will raise exceptions on missing/corrupted files or incompatible formats; shutil.copy
        and filesystem operations will raise OSError/PermissionError on I/O or permission failures. The
        caller should ensure nnUNet results are present under the path formed by os.environ["nnUNet_results"],
        the (possibly converted) dataset name, and the trainer__plans__configuration string described above,
        and that the process has appropriate filesystem permissions.
    """
    from monai.apps.nnunet.nnunet_bundle import convert_nnunet_to_monai_bundle
    return convert_nnunet_to_monai_bundle(nnunet_config, bundle_root_folder, fold)


################################################################################
# Source: monai.apps.nnunet.utils.analyze_data
# File: monai/apps/nnunet/utils.py
# Category: valid
################################################################################

def monai_apps_nnunet_utils_analyze_data(datalist_json: dict, data_dir: str):
    """Analyze training data metadata and example files to infer dataset properties used by MONAI nnU-Net workflows.
    
    This function inspects the provided MONAI-style datalist JSON and the raw data directory to determine two properties that are commonly required when configuring segmentation networks and training pipelines: the number of input channels (used to set the model input layer) and the number of foreground classes (used to set the network output channels and loss/metric configuration). It loads actual image and label files from disk using MONAI's LoadImage transform with the options image_only=True, ensure_channel_first=True, simple_keys=True to obtain tensor-like image objects with channels-first ordering when possible. It logs the inferred values via logger.info and returns them as Python ints. Typical usage in the MONAI/nnU-Net context is to call this before building a network or preparing training configuration so the architecture and label handling match the dataset.
    
    Args:
        datalist_json (dict): A parsed MONAI datalist JSON object following the common MONAI tutorials format (for example, produced by json.load on a .json datalist). The function expects this dict to contain a top-level "training" key whose value is an iterable (usually a list) of entries. Each training entry is expected to be a mapping that contains at least the keys "image" and "label" pointing to file paths relative to data_dir. This argument provides the dataset manifest that the function uses to locate example image files and all label files to compute statistics. If "training" is missing, empty, or entries lack the "image"/"label" keys, the function will raise a KeyError or IndexError when attempting to access those fields.
        data_dir (str): Path to the raw data directory (string). The function will join this directory with the relative paths found in datalist_json["training"][i]["image"] and ["label"] to locate files on disk. This argument should point to the root folder where the image and label files referenced by the datalist JSON are stored. If files are not found, MONAI's LoadImage will raise an I/O-related exception.
    
    Returns:
        tuple[int, int]: A 2-tuple of integers (num_input_channels, num_foreground_classes).
        num_input_channels: the inferred number of input channels for images in the dataset. Determined by loading the first training image entry and applying the rule used in the source code: if the loaded image tensor has 4 dimensions (img.dim() == 4) the first dimension size img.size()[0] is returned as the channel count; otherwise the function returns 1. In practical MONAI medical-imaging contexts, a 4-D tensor typically corresponds to a channels-first volumetric image (C, D, H, W), so this value is used to configure the model input channels.
        num_foreground_classes: the inferred number of foreground classes for the segmentation labels. Computed by iterating over all training label files listed in datalist_json["training"], loading each label image, computing seg.max() for each, converting to int, and taking the maximum across the dataset. This value therefore represents the largest integer label value present in the label images (commonly used when labels use 0 for background and 1..N for foreground classes) and is used to configure the number of output channels or classes for segmentation networks and loss functions.
    
    Behavior, side effects, defaults, and failure modes:
        - The function performs disk I/O: it loads the first training image and every training label file referenced by datalist_json using monai.transforms.LoadImage with image_only=True, ensure_channel_first=True, simple_keys=True. This can be time- and memory-consuming for large datasets.
        - The function logs the inferred values via logger.info for num_input_channels and num_foreground_classes.
        - If datalist_json["training"] is empty, accessing the first image will raise IndexError; if required keys ("training", "image", "label") are missing a KeyError will be raised. If referenced files are missing or unreadable, LoadImage will raise an I/O or decoding exception.
        - The num_foreground_classes value is computed as the maximum integer value found in label images; if labels are non-integer, non-contiguous, or encoded differently than 0..N this computed value may not equal the expected count of distinct foreground classes. The function does not verify label contiguity or compute the count of unique labels, it only returns the maximum label value observed.
        - The num_input_channels rule is conservative: when the loaded image does not have 4 dimensions the function assumes a single input channel (returns 1). This behavior matches the source code and is intended to handle typical 2D/3D channel-first image layouts in MONAI examples.
        - All returned values are plain Python ints suitable for use when configuring network input/output channel sizes in MONAI/nnU-Net training setups.
    """
    from monai.apps.nnunet.utils import analyze_data
    return analyze_data(datalist_json, data_dir)


################################################################################
# Source: monai.apps.nnunet.utils.create_new_data_copy
# File: monai/apps/nnunet/utils.py
# Category: valid
################################################################################

def monai_apps_nnunet_utils_create_new_data_copy(
    test_key: str,
    datalist_json: dict,
    data_dir: str,
    num_input_channels: int,
    output_datafolder: str
):
    """Create and organize a new copy of data to meet the directory layout and file naming expectations of nnU-Net V2 training and inference workflows. This function is used in MONAI-based preprocessing pipelines (for example, tutorials that provide a datalist .json) to convert an existing datalist_json and the raw image/label files in data_dir into a new dataset folder containing per-channel image NIfTI files, corresponding label NIfTI files, and an exported datalist.json that records the mapping to newly assigned case names.
    
    Args:
        test_key (str): key for test data in the data list .json. This exact string is used as the second top-level section in the input datalist_json that the function iterates over together with the fixed "training" section. Typical values are names used in MONAI tutorials (for example "testing" or "test"), and the function expects datalist_json[test_key] to be present.
        datalist_json (dict): original data list .json (required by most monai tutorials). This dictionary must contain at least a "training" key and the key specified by test_key, each mapping to a list. Each list element may be either a string (interpreted as an image path relative to data_dir) or a dict with an "image" key (path) and optionally a "label" key (path). The function reads these entries to locate and copy images and labels.
        data_dir (str): raw data directory. The function uses os.path.join(data_dir, <path-from-datalist>) to locate source image and label files referenced in datalist_json. Paths in datalist_json may be relative to data_dir or absolute; if files are missing or unreadable an exception from the underlying IO/LoadImage call will be raised.
        num_input_channels (int): number of input (image) channels. For each image loaded, the function expects at least this many channels along the first (channel) dimension after ensure_channel_first=True. It extracts channels 0..num_input_channels-1 and saves each as a separate NIfTI file using the original image affine. If num_input_channels exceeds the available channels in an image, an indexing error or related exception from the array operations will propagate.
        output_datafolder (str): output folder. The function writes files into subfolders under this path using the standard nnU-Net V2 layout names: "imagesTr" and "imagesTs" for image channels, and "labelsTr" and "labelsTs" for labels. It also repeatedly exports an updated "datalist.json" into output_datafolder that mirrors the "training" and test_key sections but adds a "new_name" field for each case. The required subdirectories must exist and be writable; the function does not create these directories itself in the source implementation and will fail if writes cannot be performed.
    
    Returns:
        None: This function does not return a Python value. Instead, it has observable side effects: it writes per-channel image NIfTI files named by a monotonic case identifier and a four-digit channel suffix (the suffix is generated by the code index = "_" + str(_l + 10000)[-4:], producing strings like "_0000", "_0001"), e.g., "case_0_0000.nii.gz", "case_0_0001.nii.gz", etc., into the appropriate imagesTr/imagesTs folder; it writes label NIfTI files named "case_<index>.nii.gz" into labelsTr/labelsTs when label paths are present in datalist_json; and it exports a reconstructed datalist.json into output_datafolder that contains the original entries augmented with "new_name" fields that map source items to their new case identifiers. The function preserves the original image affine when saving NIfTI files and casts label arrays to uint8 and squeezes singleton channel dimensions when appropriate.
    
    Behavioral notes, side effects, and failure modes: The function iterates over the "training" section and the section named by test_key from datalist_json. For each entry it assigns a new name using the pattern "case_<index>" where index is a running integer starting at 0. Images are loaded with monai.transforms.LoadImage(image_only=True, ensure_channel_first=True, simple_keys=True), converted to numpy, and channels are saved individually with nibabel.Nifti1Image using the original affine saved in the image metadata. Labels, when provided as dict entries with a "label" key, are loaded similarly, converted to numpy, cast to numpy.uint8, squeezed if shape indicates a singleton channel dimension (ndim == 4 and shape[0] == 1), and saved as a single NIfTI file. The datalist.json file is exported repeatedly during processing via ConfigParser.export_config_file to ensure resume-friendly state; the final exported file contains the training and test_key lists with added "new_name" entries. The function will propagate IO errors (missing files, permission errors), image loading or transform errors (mismatched channel counts), and KeyError if expected keys are absent from datalist_json. It does not perform validation beyond what the underlying loaders and numpy operations enforce, so callers must ensure datalist_json structure, data_dir contents, num_input_channels, and writable output subdirectories are correct for successful conversion.
    """
    from monai.apps.nnunet.utils import create_new_data_copy
    return create_new_data_copy(
        test_key,
        datalist_json,
        data_dir,
        num_input_channels,
        output_datafolder
    )


################################################################################
# Source: monai.apps.nnunet.utils.create_new_dataset_json
# File: monai/apps/nnunet/utils.py
# Category: valid
################################################################################

def monai_apps_nnunet_utils_create_new_dataset_json(
    modality: str,
    num_foreground_classes: int,
    num_input_channels: int,
    num_training_data: int,
    output_filepath: str
):
    """Create a new dataset.json file formatted for nnU-Net V2 from simple inputs.
    
    This utility constructs the minimal JSON structure that nnU-Net V2 expects for a dataset description and writes it to disk. It is intended for use in MONAI-based medical imaging workflows where researchers or engineers need to generate the required dataset metadata (channel names, label mapping, number of training cases, and file ending) programmatically so that nnU-Net V2 training/evaluation code can consume the dataset. The function maps input image modalities to channel indices, creates a label dictionary with a reserved background label (0) and sequential foreground class IDs (1..N), sets the number of training samples, and fixes the image file extension to ".nii.gz". The produced JSON is exported via ConfigParser.export_config_file(fmt="json", sort_keys=True, indent=4, ensure_ascii=False).
    
    Args:
        modality (str): Image modality or modalities describing each input channel. This may be a single string (e.g., "CT" or "MR") or an iterable/sequence of strings (e.g., ("CT", "T1", "T2")). Each element provides the human-readable name used to populate the "channel_names" mapping in the output JSON. The function uses ensure_tuple(modality) internally, so a single string will be treated as a one-element sequence. It is the caller's responsibility to ensure that the number of modality entries is at least num_input_channels; otherwise an indexing error will occur when assigning channel names.
        num_foreground_classes (int): Number of foreground semantic classes in the segmentation task (non-background classes). This integer controls how many entries are created under "labels" in the output JSON beyond the mandatory "background": 0 entry. Foreground classes are named "class1", "class2", ..., "class{N}" and assigned integer labels 1..N respectively. Must be a non-negative integer; if 0, only the "background": 0 mapping is produced.
        num_input_channels (int): Number of input image channels expected by the model (e.g., 1 for single-channel CT, 3 for multi-sequence MR inputs). This integer controls the number of entries created under "channel_names" in the output JSON, with keys "0", "1", ... up to num_input_channels-1 mapped to corresponding modality names. Must be a positive integer; providing a value larger than the number of modality entries will raise an indexing error during channel name assignment.
        num_training_data (int): Number of training cases available for this dataset. This integer is written to the "numTraining" field in the JSON and is used by downstream nnU-Net tooling to understand dataset size for planning folds and resource allocation. Must be a non-negative integer; the function does not validate that this number matches any external data directory contents.
        output_filepath (str): Filesystem path (including filename) where the generated dataset JSON will be written, e.g., "/path/to/raw_dataset/dataset.json". The function will call ConfigParser.export_config_file to write the JSON file with fmt="json", sort_keys=True, indent=4, and ensure_ascii=False. The caller must ensure the directory exists and is writable; typical failure modes include permission errors, invalid paths, or disk full conditions raised by the underlying file I/O or ConfigParser implementation.
    
    Returns:
        None: This function does not return a value. Its observable side effect is writing a JSON file at output_filepath containing the keys "channel_names", "labels", "numTraining", and "file_ending" (set to ".nii.gz"). On failure the function may raise exceptions from ensure_tuple, from indexing when modality entries do not cover num_input_channels, from invalid types (if non-integer counts are supplied), or from ConfigParser.export_config_file/file I/O (e.g., IOError, OSError) which the caller should catch if they need to handle write errors programmatically.
    """
    from monai.apps.nnunet.utils import create_new_dataset_json
    return create_new_dataset_json(
        modality,
        num_foreground_classes,
        num_input_channels,
        num_training_data,
        output_filepath
    )


################################################################################
# Source: monai.apps.pathology.utils.compute_isolated_tumor_cells
# File: monai/apps/pathology/utils.py
# Category: valid
################################################################################

def monai_apps_pathology_utils_compute_isolated_tumor_cells(
    tumor_mask: numpy.ndarray,
    threshold: float
):
    """Compute isolated tumor cell (ITC) labels from a labeled tumor segmentation mask.
    
    This function identifies Isolated Tumor Cells (ITCs) in a labeled tumor mask by measuring each labeled region's longest diameter (major axis length) using skimage.measure.regionprops and comparing it to a provided threshold. In pathology workflows (as used in MONAI's pathology utilities), ITCs are small tumor foci whose largest extent is below a configured threshold; this function returns the integer labels of regions considered ITCs so downstream code can count, remove, or further analyze these small regions in slide-level or patch-level pipelines.
    
    Args:
        tumor_mask (numpy.ndarray): A labeled tumor mask array where integer values encode region labels and 0 typically represents background. This function expects a NumPy ndarray containing labeled connected regions produced by a segmentation or labeling step. The labels returned correspond to the integer region identifiers in this mask (see Failure modes for expectations about label contiguity). The array is not modified by this function.
        threshold (float): The threshold used to classify a region as an ITC. A region whose major_axis_length (the longest diameter computed by skimage.measure.regionprops on the mask) is strictly less than this threshold is considered an isolated tumor cell (ITC). The threshold is expressed in the same units as regionprops.major_axis_length (mask pixel units if no spatial calibration is applied). This value must be a finite floating-point number; non-finite values may produce no matches or raise comparison-related errors.
    
    Returns:
        list[int]: A list of integer labels corresponding to regions in tumor_mask that are classified as isolated tumor cells (ITCs). The function constructs this list by iterating region indices from 1 to the maximum label value present in tumor_mask and including label i when that region's major_axis_length < threshold; therefore the returned integers match the label values used in tumor_mask. An empty list is returned when no regions meet the ITC criterion.
    
    Behavior, side effects, and failure modes:
        This function computes region properties via skimage.measure.regionprops and uses numpy.amax to obtain the maximum label in tumor_mask. It does not modify tumor_mask or any external state. It expects tumor_mask to be an integer-labeled mask suitable for regionprops. If tumor_mask contains non-contiguous labels (for example, labels skip values between 1 and the maximum label), the implementation indexes the regionprops list by position (properties[i]) while iterating up to the maximum label; when labels are missing this can lead to an IndexError because regionprops returns only existing regions. If tumor_mask has no foreground labels (maximum label is 0), the function returns an empty list. For very large masks or many labeled regions, computing regionprops may be computationally and memory intensive and could increase runtime significantly. Ensure threshold is finite and meaningful for the mask scale; otherwise the classification may be empty or incorrect.
    """
    from monai.apps.pathology.utils import compute_isolated_tumor_cells
    return compute_isolated_tumor_cells(tumor_mask, threshold)


################################################################################
# Source: monai.apps.pathology.utils.compute_multi_instance_mask
# File: monai/apps/pathology/utils.py
# Category: valid
################################################################################

def monai_apps_pathology_utils_compute_multi_instance_mask(
    mask: numpy.ndarray,
    threshold: float
):
    """Compute a multi-instance segmentation mask from a binary tumor mask for pathology images.
    
    Args:
        mask (numpy.ndarray): Binary tumor mask array representing foreground tumor pixels and background non-tumor pixels. In the pathology use case this function expects a 2D (whole-slide or patch) binary mask where tumor pixels are indicated (commonly as 1 or True) and background pixels are 0 or False. The mask is used as the input segmentation from which separate tumor instances (connected components) will be identified. The function does not modify the input array in place; it reads the array values to compute a distance transform, hole filling, and connected-component labeling.
        threshold (float): Distance threshold in pixel units used to produce an intermediate binary region from the distance transform. This value controls how interior regions are selected before hole filling: the function computes the Euclidean distance transform of the inverted binary mask, creates a binary map of locations where distance < threshold, fills holes in that binary map, and then labels connected components. In practical pathology workflows, adjust this threshold (>= 0) to control sensitivity to small cavities and to determine whether nearby structures merge into a single instance. Non-finite values (nan/inf) for threshold will raise an error from the distance transform or comparison operations.
    
    Returns:
        Any: A multi-instance mask object produced by skimage.measure.label applied to the hole-filled binary image. The returned object contains integer labels for each connected component (instance) so downstream instance-level analysis (for example per-instance measurements, cropping, or model training for instance-aware tasks in digital pathology) can identify and index individual tumor regions. Typical behavior: small threshold values may produce no labeled instances (all background), large threshold values may merge many regions into a single label, and invalid input (non-binary mask values, wrong dimensionality, missing scipy/skimage dependencies) will raise runtime errors from the underlying ndimage or measure functions.
    
    Behavior and side effects:
        The function performs these steps in order: (1) invert the binary mask via neg = 255 - mask * 255, (2) compute Euclidean distance transform on the inverted image using scipy.ndimage.distance_transform_edt, (3) create a binary map where distance < threshold, (4) fill holes in that binary map using scipy.ndimage.binary_fill_holes, and (5) label connected components with skimage.measure.label(connectivity=2). It relies on SciPy and scikit-image primitives; if those libraries are unavailable or if the input mask contains unexpected value ranges (not binary), the underlying calls will raise ImportError or value/shape-related errors. The function is intended for 2D pathology segmentation masks and uses connectivity=2 (8-connectivity) to define component adjacency in typical 2D use; using masks of different dimensionality may lead to unexpected connectivity behavior.
    """
    from monai.apps.pathology.utils import compute_multi_instance_mask
    return compute_multi_instance_mask(mask, threshold)


################################################################################
# Source: monai.apps.reconstruction.networks.nets.utils.floor_ceil
# File: monai/apps/reconstruction/networks/nets/utils.py
# Category: valid
################################################################################

def monai_apps_reconstruction_networks_nets_utils_floor_ceil(n: float):
    """monai.apps.reconstruction.networks.nets.utils.floor_ceil returns the mathematical floor and ceil of a floating-point input, intended as a small utility used by MONAI reconstruction network code to convert continuous quantities (for example, spatial coordinates, fractional pixel indices, or computed sizes) into discrete integer values required for image grid indices, array shapes, or padding calculations in medical imaging workflows.
    
    Args:
        n (float): A real-valued input number. In the MONAI reconstruction context this typically represents a continuous quantity such as a coordinate, dimension, or computed size that must be converted to integer indices or sizes for downstream array operations. The function treats this value according to Python's math.floor and math.ceil semantics: the return values are the largest integer less than or equal to n and the smallest integer greater than or equal to n, respectively.
    
    Returns:
        tuple[int, int]: A 2-tuple of integers (floor_n, ceil_n) where floor_n == math.floor(n) and ceil_n == math.ceil(n). Both integers are suitable for use as array indices, shape components, or other discrete parameters in reconstruction/network code within MONAI.
    
    Behavior and side effects:
        This function is pure and has no side effects; it performs only numeric conversion via the standard math module and returns its result. There are no defaults to configure.
    
    Failure modes:
        If n is not a real number, the underlying math.floor/math.ceil calls will raise a TypeError. If n is NaN, math.floor/math.ceil will raise a ValueError. If n is infinite, math.floor/math.ceil may raise an OverflowError when converting to an integer. Callers in MONAI reconstruction code should validate or sanitize continuous quantities (e.g., clamp values or check for finite numbers) before calling this utility when necessary.
    """
    from monai.apps.reconstruction.networks.nets.utils import floor_ceil
    return floor_ceil(n)


################################################################################
# Source: monai.apps.tcia.utils.match_tcia_ref_uid_in_study
# File: monai/apps/tcia/utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def monai_apps_tcia_utils_match_tcia_ref_uid_in_study(study_uid: str, ref_sop_uid: str):
    """Match the SeriesInstanceUID for a series in a TCIA study that contains the given SOPInstanceUID.
    
    This function is used in MONAI applications that integrate with The Cancer Imaging Archive (TCIA) to map a DICOM SOPInstanceUID (an individual image or instance) back to the SeriesInstanceUID (the DICOM identifier for the image series) within a given StudyInstanceUID. It performs TCIA metadata queries: first it retrieves all SeriesInstanceUID values for the study, then for each series it retrieves the SOPInstanceUID list and checks for a match. This is useful when one has a reference SOP instance and needs to know which series in the TCIA study contains that instance for downstream processing, loading, or indexing in MONAI workflows.
    
    Args:
        study_uid (str): StudyInstanceUID identifying the DICOM study in TCIA. This is the DICOM UID that scopes the search to a single study; it is passed to the TCIA metadata query "getSeries?StudyInstanceUID=<study_uid>" to enumerate SeriesInstanceUID values in that study.
        ref_sop_uid (str): SOPInstanceUID that identifies the reference DICOM instance (image) to locate. This UID is compared against the SOPInstanceUID lists returned for each series via the TCIA metadata query "getSOPInstanceUIDs?SeriesInstanceUID=<series_id>".
    
    Returns:
        str: The SeriesInstanceUID (a DICOM UID string) of the first series in the study that contains ref_sop_uid. If no series in the study contains the given SOPInstanceUID, returns an empty string ("") as a sentinel value.
    
    Behavior and side effects:
        - The function calls get_tcia_metadata twice in sequence: once to retrieve the list of SeriesInstanceUID values for the study, and then once per series to retrieve its SOPInstanceUID list. These calls perform network requests to the TCIA metadata API and are the primary side effects.
        - The function returns immediately when it finds the first matching series; it does not continue searching for additional matches. If multiple series contain the same SOPInstanceUID (unlikely in normal DICOM datasets), only the first match encountered is returned.
        - There is no caching of TCIA responses within this function; repeated calls will re-query TCIA and incur network latency proportional to the number of series in the study.
        - If study_uid or ref_sop_uid are empty or invalid UIDs, the TCIA API may return empty lists and the function will return an empty string. The function itself does not validate UID format beyond passing the strings to the metadata queries.
    
    Failure modes and exceptions:
        - Network errors, authentication failures, TCIA service errors, or any exceptions raised by get_tcia_metadata will propagate to the caller (the function does not catch exceptions). Callers should handle these exceptions according to their application's error-handling policy.
        - If the TCIA API responds with unexpected data shapes or types that violate get_tcia_metadata assumptions, the function may raise exceptions from the underlying metadata access code.
    
    Usage note:
        - Use this function when integrating MONAI pipelines with TCIA-hosted DICOM datasets to resolve which series contains a known SOP instance. Expect one initial metadata query plus up to N per-series metadata queries, where N is the number of series in the study.
    """
    from monai.apps.tcia.utils import match_tcia_ref_uid_in_study
    return match_tcia_ref_uid_in_study(study_uid, ref_sop_uid)


################################################################################
# Source: monai.apps.utils.get_filename_from_url
# File: monai/apps/utils.py
# Category: fix_docstring
# Reason: Schema parsing failed: Cannot generate JSON schema for get_filename_from_url because the docstring has no description for the argument 'data_url'
################################################################################

def monai_apps_utils_get_filename_from_url(data_url: str):
    """Get the filename from the URL link. This utility issues HTTP requests to determine a sensible file name for a remote resource referenced by data_url and is used in MONAI workflows that download assets (for example model weights, bundles, or example datasets in the MONAI Model Zoo and tutorials) so files can be saved with an appropriate, human-readable name.
    
    Args:
        data_url (str): The URL string pointing to a remote resource. This is the exact URL that will be probed to determine a filename. The function first issues an HTTP HEAD request (allowing redirects) to read response headers, then may issue an HTTP GET for specific hosts (Google Drive) to parse HTML. The value should be a valid URL formatted as a string; no other types are accepted.
    
    Returns:
        str: The filename extracted from data_url. Extraction behavior, in order of precedence, is:
            - If the HTTP HEAD response includes a Content-Disposition header that matches the regular expression 'filename="?([^";]+)"?', the first captured group is returned as the filename (this handles server-specified attachment filenames commonly used when downloading files).
            - If the URL contains "drive.google.com", the function issues an HTTP GET and, if the response Content-Type indicates HTML, parses the HTML with BeautifulSoup to find a <span class="uc-name-size"> element and returns the anchor text inside that element (this handles typical Google Drive file pages used in MONAI tutorials and bundles).
            - If neither header nor Google Drive HTML provides a name, the function falls back to returning _basename(data_url), i.e., the URL path's basename (a simple fallback suitable for saving files when the server does not specify a filename).
        The returned string is intended to be used directly as a local filename when saving downloaded resources.
    
    Behavior and side effects:
        The function performs network I/O: it uses requests.head(data_url, allow_redirects=True) and, for Google Drive links, requests.get(data_url). These calls are synchronous and blocking. The HEAD request follows redirects. The function inspects response headers and, if required, parses HTML using BeautifulSoup and a regular expression. No temporary files are created by this function; it only returns a filename string.
    
    Failure modes and exceptions:
        Any network error, invalid URL, missing dependencies (requests, BeautifulSoup), or unexpected parsing error will be caught and re-raised as a generic Exception with the prefix "Error processing URL: " followed by the original exception message. Callers in MONAI code should handle this Exception and may retry, provide a timeout wrapper, or fall back to user-supplied filenames. The function does not perform URL validation beyond what the requests library enforces and does not attempt to sanitize the returned filename for filesystem safety beyond returning the extracted string.
    """
    from monai.apps.utils import get_filename_from_url
    return get_filename_from_url(data_url)


################################################################################
# Source: monai.auto3dseg.utils.check_and_set_optional_args
# File: monai/auto3dseg/utils.py
# Category: fix_docstring
# Reason: Schema parsing failed: Cannot generate JSON schema for check_and_set_optional_args because the docstring has no description for the argument 'params'
################################################################################

def monai_auto3dseg_utils_check_and_set_optional_args(params: dict):
    """Convert a dictionary of optional parameters into a single command-line style string of the form " --key_1=value_1 --key_2=value_2 ...".
    
    This utility is used in MONAI's auto3dseg helper code to assemble optional arguments for command-line invocation patterns (for example, when constructing strings to pass into python-fire or bundle/Model Zoo launcher utilities). In the medical imaging Auto3DSeg workflow, it lets higher-level code represent optional CLI overrides as a Python dict and then convert them into the exact text fragment appended to a command that will modify runtime behavior (such as hyperparameters, input/output paths, or other optional flags).
    
    Args:
        params (dict): A mapping of option names to option values representing optional command-line arguments. Keys are used verbatim as the option names and are inserted into the output following two dashes (i.e., "--{key}="). Values may be non-dict scalar-like objects or lists. If a value is a list, it is converted by list_to_python_fire_arg_str to a string representation appropriate for python-fire style CLI arguments before being inserted. The function does not mutate the input dictionary. The iteration order follows Python's dict.items() order (insertion order for CPython 3.7+), so the produced option sequence preserves that order.
    
    Behavior and side effects:
        The function builds and returns a string that begins with a space and contains one " --key=value" fragment per item in params, concatenated in dict item order. Each value is converted to its string representation via Python formatting; list values are transformed by list_to_python_fire_arg_str. There are no external side effects such as IO or modification of params.
    
    Failure modes and errors:
        If any value in params is itself a dict, the function raises ValueError("Nested dict is not supported.") because nested dictionaries are not supported by the intended CLI encoding. Other object types are handled by their normal string conversion; if conversion produces characters that are not safe for your shell or CLI frontend, the caller is responsible for proper escaping before passing the value into this function.
    
    Returns:
        str: A single string containing the concatenated optional arguments in the form " --key_1=value_1 --key_2=value_2 ...". Note that the returned string begins with a leading space (because each option fragment is prefixed with " --"). This string is intended to be appended to an existing command line or passed to a CLI invocation helper within the MONAI Auto3DSeg tooling.
    """
    from monai.auto3dseg.utils import check_and_set_optional_args
    return check_and_set_optional_args(params)


################################################################################
# Source: monai.auto3dseg.utils.list_to_python_fire_arg_str
# File: monai/auto3dseg/utils.py
# Category: valid
################################################################################

def monai_auto3dseg_utils_list_to_python_fire_arg_str(args: list):
    """Convert a Python list into a single argument string formatted for use with python-fire.
    
    This utility is part of MONAI's auto3dseg utilities and is used in automated 3D segmentation workflows to serialize a list of values (for example, device indices, file paths, or other hyperparameter lists commonly used in medical imaging experiments) into one command-line argument that python-fire can receive. The function obtains each element's textual form via str(), joins those textual elements with commas, and wraps the whole result in single quotes so it can be passed as a single shell/CLI token to python-fire.
    
    Args:
        args (list): The list of values to convert into a python-fire argument string. Each element is converted by calling str(element). The input list is not modified by this function. Typical usage in the MONAI auto3dseg domain includes serializing lists of numeric IDs, file paths, or configuration tokens so they can be passed through a python-fire CLI; the function does not restrict the element types beyond requiring that they be stringable via str(). There are no default values. Side effects: none. The function does not escape commas or quotes inside individual element strings.
    
    Returns:
        str: A single-quoted string containing the comma-joined str() representations of the input list elements, intended to be passed as one argument to python-fire. For example, given args = ["a", "b", 3], the function returns "'a,b,3'". Failure modes: if an element's __str__ implementation raises an exception, that exception will propagate. Note that if element string representations contain commas or single quotes, the produced string may be ambiguous for python-fire parsing because the function does not perform additional escaping of element contents.
    """
    from monai.auto3dseg.utils import list_to_python_fire_arg_str
    return list_to_python_fire_arg_str(args)


################################################################################
# Source: monai.auto3dseg.utils.verify_report_format
# File: monai/auto3dseg/utils.py
# Category: valid
################################################################################

def monai_auto3dseg_utils_verify_report_format(report: dict, report_format: dict):
    """Verify that a runtime report dictionary follows a keys-only format specification used by MONAI Auto3DSeg.
    
    This function is used in the monai.auto3dseg utilities to validate that a report produced by an Auto3DSeg training/evaluation workflow (for example, aggregated metrics, configuration summaries, or bundle reports) contains the keys and nested list structure described by a keys-only format dictionary. It checks presence of keys and the expected nested-list shape described by report_format without inspecting or validating the concrete values or value types. The function performs a recursive comparison that treats a single-element list in report_format as the element-format for items of a corresponding list in report.
    
    Args:
        report (dict): A runtime report dictionary that contains concrete values and potentially nested structures produced by MONAI Auto3DSeg code paths (metrics, metadata, or other report entries). The function interprets this dict as the actual data to be validated: keys must exist at the same positions as in report_format and lists in this dict must have elements whose structure matches the single-element list specification in report_format.
        report_format (dict): A keys-only dictionary that defines the expected structure of report. Values in this dict are either non-list placeholders (only keys matter) or single-element lists that describe the expected structure of each element of a corresponding list in report. Practically, in MONAI Auto3DSeg this format is used to declare which keys and nested-list shapes a valid report must contain.
    
    Returns:
        bool: True if report contains every key from report_format and any nested lists in report match the single-element list shape described by report_format; False otherwise. More specifically, False is returned when a required key from report_format is missing in report, when a list in report is empty while the format expects an element-level specification, or when a nested verification fails. The function returns True even if report contains additional keys not present in report_format (extra keys are ignored).
    
    Behavior, side effects, and failure modes:
        The comparison is recursive: for each key in report_format the function checks existence in report. If the corresponding report_format value is a list and the report value is a list, the function expects report_format to be a single-element list describing the element format; it will then recursively verify the first element of report against that element-format. If report_format contains a list whose length is not exactly 1, the function raises a UserWarning with the message "list length in report_format is not 1" (this is a deliberate guard used in MONAI Auto3DSeg to enforce single-element list specifications). If either the format list or the report list is empty, the function returns False because there is no element to verify against the element-level specification. The function does not validate types, numerical ranges, tensor shapes, or device placement of values — it only checks keys and nested-list shape per the provided report_format. There are no other side effects beyond the possible UserWarning. Performance characteristics: the function performs a depth-first traversal of keys and nested single-element lists and returns early on the first mismatch.
    """
    from monai.auto3dseg.utils import verify_report_format
    return verify_report_format(report, report_format)


################################################################################
# Source: monai.data.image_writer.register_writer
# File: monai/data/image_writer.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def monai_data_image_writer_register_writer(ext_name: str, *im_writers):
    """Register ImageWriter classes for a filename extension so MONAI's image-writing
    mechanism can resolve a file extension to one or more appropriate ImageWriter
    implementations used in medical imaging workflows.
    
    This function is used in MONAI (a PyTorch-based framework for healthcare imaging)
    to associate a filename extension (for example "nii" for NIfTI files) with a
    tuple of ImageWriter classes that know how to serialize image data to that
    format. When client code calls the higher-level image writing utilities, the
    extension string is used as a key to look up the registered writers and decide
    which writer(s) to try. Registration is additive and order-sensitive: writers
    passed to this function are given higher priority and placed before any
    previously registered writers for the same extension.
    
    Args:
        ext_name (str): the filename extension of the image being registered, e.g.
            "nii" or ".nii.gz". As an indexing key it will be converted to a lower
            case string and a leading dot will be removed (the implementation uses
            f"{ext_name}".lower() and strips a leading "."), so case and an optional
            leading dot in ext_name do not change the registration key. In the MONAI
            domain this extension determines the serialization format used when
            saving medical images.
        im_writers (ImageWriter classes): one or multiple ImageWriter classes (one
            per positional argument) that implement MONAI's ImageWriter interface
            and can serialize image tensors / arrays to files of the given
            extension. The writers provided here are treated as high-priority and
            are prepended to any previously registered writers for the same
            extension; the order of arguments matters for selection priority.
    
    Returns:
        None: This function does not return a value. Its side effect is that it
        updates the module-level SUPPORTED_WRITERS registry (a mapping from the
        normalized extension string to a tuple of ImageWriter classes) so subsequent
        calls to MONAI image-writing utilities will consider the newly registered
        writers. If no writers are provided, the registry will include any existing
        writers unchanged. This function does not validate that the provided
        im_writers actually implement the ImageWriter interface; incorrect or
        non-callable objects may lead to runtime errors later when the writers are
        invoked.
    """
    from monai.data.image_writer import register_writer
    return register_writer(ext_name, *im_writers)


################################################################################
# Source: monai.data.image_writer.resolve_writer
# File: monai/data/image_writer.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def monai_data_image_writer_resolve_writer(ext_name: str, error_if_not_found: bool = True):
    """Resolve the available ImageWriter backends registered in MONAI for a given filename extension and return them as a sequence of writer callables.
    
    Args:
        ext_name (str): The filename extension used as the lookup key in the global writer registry SUPPORTED_WRITERS.
            This function converts the value to a lower-case string and strips a leading dot if present (for example,
            ".nii.gz" or "NII" will be normalized to "nii.gz" or "nii"). In the MONAI medical-imaging workflow this
            extension identifies the desired file format when saving images, so callers should pass the file extension
            associated with the intended image file type.
        error_if_not_found (bool): Whether to raise an error if no suitable image writer backend is available for the
            resolved extension. Defaults to True. If True and no available writer backends are found, this function raises
            monai.utils.module.OptionalImportError to indicate that required writer packages are missing or no backend is
            registered. If False and no backends are available, the function returns an empty tuple instead of raising.
    
    Returns:
        Sequence: A tuple (sequence) of available ImageWriter callables (classes or factory functions) that are usable in
        the current runtime environment for writing images with the resolved extension. Each element is a callable that
        can be instantiated to perform writing; attempting instantiation of a writer may trigger dependency checks
        (via monai.utils.module.require_pkg) and raise OptionalImportError if its optional dependencies are not installed.
        The returned sequence is cached into the global SUPPORTED_WRITERS mapping under the normalized extension key
        before being returned.
    
    Behavior, side effects, and failure modes:
        - If the global SUPPORTED_WRITERS registry is empty, this function calls init() to populate the registry before
          performing the lookup. This ensures writers contributed by MONAI or optional backends are discovered.
        - The function first normalizes ext_name to a lower-case string and removes a leading "." if present.
        - The registry lookup uses writers registered for the exact extension key and falls back to writers listed under the
          wildcard key EXT_WILDCARD if present (the code obtains default_writers = SUPPORTED_WRITERS.get(EXT_WILDCARD, ())).
        - For each candidate writer callable returned by look_up_option, the function attempts to instantiate it once:
          instantiation triggers any packaged dependency checks. If instantiation succeeds, the writer is considered
          available and included in the returned sequence. If instantiation raises OptionalImportError (missing optional
          dependency), that candidate is skipped. If instantiation raises any other exception, the writer is treated as
          present and included (the exception is not propagated by this function), because such exceptions indicate the
          writer exists but failed during initialization for reasons unrelated to missing optional packages.
        - The resolved tuple of writer callables is stored in SUPPORTED_WRITERS[fmt] (cache) as a side effect so future
          lookups for the same extension value return the cached sequence without repeating initialization logic.
        - If no writers are available after checking candidates and error_if_not_found is True, the function raises
          monai.utils.module.OptionalImportError with a message indicating no ImageWriter backend was found for the
          normalized extension. If error_if_not_found is False, the function returns an empty tuple instead of raising.
    """
    from monai.data.image_writer import resolve_writer
    return resolve_writer(ext_name, error_if_not_found)


################################################################################
# Source: monai.data.meta_obj.set_track_meta
# File: monai/data/meta_obj.py
# Category: fix_docstring
# Reason: Schema parsing failed: Cannot generate JSON schema for set_track_meta because the docstring has no description for the argument 'val'
################################################################################

def monai_data_meta_obj_set_track_meta(val: bool):
    """monai.data.meta_obj.set_track_meta sets whether MONAI associates metadata with data objects used throughout the MONAI medical imaging data pipeline.
    
    This function configures a module-level boolean flag that controls whether metadata (for example, spatial affine, voxel spacing, orientation, original filename, channel dimension, and other dataset-specific attributes commonly preserved in medical imaging workflows) is tracked and attached to data objects by using MONAI's MetaObj subclasses. When tracking is enabled, MONAI returns enhanced objects that carry metadata alongside the raw data; when tracking is disabled, MONAI returns standard data containers (for example, torch.Tensor and numpy.ndarray) with empty or no metadata. By default this flag is True; most users should leave metadata tracking enabled to preserve spatial and provenance information required for preprocessing, resampling, visualization, and evaluation in healthcare imaging workflows.
    
    Args:
        val (bool): Flag indicating whether to enable metadata tracking. If True, metadata will be associated with data objects via MetaObj subclasses and related MONAI APIs that construct enhanced objects. If False, MONAI will return standard data objects without metadata. This parameter directly sets the module-global _TRACK_META variable inside monai.data.meta_obj and thus affects subsequent MONAI data creation and transformation functions that consult that flag.
    
    Returns:
        None: This function does not return a value. Its effect is a side effect: it sets the module-level global flag _TRACK_META to the provided boolean value. After calling this function, subsequent MONAI operations that create or wrap data will observe the new setting and either attach metadata (when enabled) or return plain data objects with empty metadata (when disabled).
    
    Behavioral notes and failure modes:
    - Default and typical use: The default behavior in MONAI is to track metadata (val=True). Tracking metadata is important in medical imaging for maintaining spatial consistency and provenance across preprocessing, augmentation, and model I/O.
    - When disabling metadata (val=False): Useful for debugging or when metadata is not needed and you want plain arrays/tensors; however, disabling can cause downstream MONAI components, third-party code, or user code that expects metadata to behave incorrectly, lose spatial information, or raise errors.
    - Scope and side effects: The change is global to the monai.data.meta_obj module. It is not scoped to a dataset, transform, or thread; changing it at runtime will affect all subsequent MONAI data operations in the current Python process. Care should be taken when toggling this flag in multi-threaded or long-running processes to avoid inconsistent behavior.
    - Input expectations: The function signature expects a boolean value for val. Passing a non-boolean value is not documented in the source and therefore should be avoided; the annotation signals intended usage as a strict boolean toggle.
    """
    from monai.data.meta_obj import set_track_meta
    return set_track_meta(val)


################################################################################
# Source: monai.data.utils.collate_meta_tensor
# File: monai/data/utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def monai_data_utils_collate_meta_tensor(batch: list):
    """Collate a sequence of meta-tensor containers into a batched meta-tensor structure suitable for MONAI data pipelines.
    
    This function is used in MONAI (a PyTorch-based medical imaging framework) to assemble per-sample data items produced by a Dataset into a single batch that preserves both tensor data and associated metadata (for example spatial affine, origin, orientation stored in MONAI MetaTensor/MetaObj objects). It recursively inspects the first element of the provided sequence to determine how to collate: if elements are MONAI MetaObj instances, it delegates to collate_meta_tensor_fn to produce a single batched meta-tensor; if elements are mapping/dictionary-like it builds a dictionary whose values are the collated results for each key; if elements are tuples/lists it returns a list of collated results for each position; otherwise it falls back to torch.utils.data.dataloader.default_collate to produce a conventional tensor batch. This behavior enables DataLoader-style batching while preserving domain-specific metadata required for healthcare imaging workflows and downstream models.
    
    Args:
        batch (list): A list (sequence) of per-sample objects to collate into a batch. Each element in this list is expected to be one of:
            - A MetaObj instance (MONAI metadata wrapper): when the first element is a MetaObj, the function will call collate_meta_tensor_fn(batch) to combine all MetaObj items into a single batched meta-tensor that preserves per-sample metadata (affine, spacing, keys used in imaging pipelines).
            - A Mapping (e.g., dict): when the first element is mapping-like, the function assumes all elements share the same keys and returns a dict where each key maps to the result of recursively collating the list of values for that key. Note: if a later element is missing a key present in the first element, a KeyError will be raised when attempting to access d[k].
            - A tuple or list: when the first element is a tuple/list, the function returns a list whose i-th entry is the result of recursively collating the i-th entries from each element of batch. This preserves ordered container structures commonly used to store (image, label) pairs or multi-modal inputs.
            - Any other object type: if none of the above cases match (and no MetaObj is found during the recursive checks), the function falls back to torch.utils.data.dataloader.default_collate(batch) to produce a standard PyTorch-style batched tensor or nested structure.
            The function requires that the provided batch is a Sequence; if the argument is not a Sequence the function raises NotImplementedError. The function performs recursive inspection starting from the first element and recurses into mappings and sequences to locate MetaObj instances; if it reaches leaf objects without finding MetaObj it uses default_collate. No mutation of the input list is performed by this function, but the collated outputs may be new tensors/containers allocated in memory.
    
    Returns:
        Any: The collated batch. Possible return shapes/types (determined by runtime structure of batch and MONAI internals) include:
            - The output of collate_meta_tensor_fn(batch) when elements are MetaObj instances: a single batched MetaTensor-like object that combines data and metadata for use in MONAI training/evaluation.
            - A dict mapping keys to collated values when elements are mapping/dictionary-like: each value is the recursively collated batch for that key (useful for batching structured sample dictionaries containing image, label, and metadata).
            - A list of collated values when elements are tuple/list-like: preserves positional containers such as (image, label).
            - The result of torch.utils.data.dataloader.default_collate(batch) for leaf types where no MetaObj was detected: standard PyTorch batched tensors or nested lists/tuples as produced by default_collate.
        Errors from the delegated functions (collate_meta_tensor_fn or default_collate) propagate to the caller. Potential failure modes include NotImplementedError if batch is not a Sequence, KeyError when mapping keys are inconsistent across samples, and TypeError or other exceptions raised by default_collate for unsupported element types.
    """
    from monai.data.utils import collate_meta_tensor
    return collate_meta_tensor(batch)


################################################################################
# Source: monai.data.utils.collate_meta_tensor_fn
# File: monai/data/utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def monai_data_utils_collate_meta_tensor_fn(batch: list, collate_fn_map: dict = None):
    """Collate a sequence of MONAI MetaTensor-like objects into a single batched tensor that preserves per-sample metadata and records applied operations.
    
    This function is implemented for the MONAI medical-imaging data pipeline and is used to combine a list of per-sample tensors (for example, image volumes or labels) and their associated metadata into a single batched tensor for downstream model training or inference. It is called by collage_meta_tensor (internal MONAI utility) and therefore is not intended to be passed directly as a DataLoader collate_fn. The function delegates numerical/tensor stacking to torch.utils.data._utils.collate.collate_tensor_fn and then attaches three MONAI-specific attributes to the returned tensor: meta (a batched collection of per-sample metadata), applied_operations (a list of each sample's recorded transform history), and is_batch (a boolean marker set to True). This preserves important provenance information (e.g., spatial affine, original shape, spacing, keys) used in healthcare imaging workflows when composing batches for models.
    
    Args:
        batch (list): A sequence (Python list) of per-sample objects expected to have at least the attributes used by MONAI MetaTensor-like objects: .meta and .applied_operations. Each element is typically a MetaTensor or an object for which collate_tensor_fn can produce a batched tensor. The .meta attribute is expected to be a dict-like mapping of metadata keys to values for that sample (or a falsey value, in which case TraceKeys.NONE is used as a placeholder). The .applied_operations attribute is expected to be a list (or falsey) describing the transforms applied to that sample. This parameter is the primary input: collate_tensor_fn(batch) is called to produce the numeric batched tensor, and the metadata and applied operations are collected and attached to that result. Passing an empty list, items missing required attributes, or items whose .meta values are never dict-like can cause runtime errors described below.
        collate_fn_map (dict): Reserved mapping of custom collate functions keyed by data type or key name. Default is None. In the current implementation this parameter is accepted for API compatibility but is not consulted by collate_meta_tensor_fn; numerical collation is always performed by torch.utils.data._utils.collate.collate_tensor_fn and metadata collation uses MONAI's default_collate. Provide None to use the built-in behavior. The parameter exists to accommodate future extension points where different collate functions might be selected per-key or per-type.
    
    Returns:
        torch.Tensor-like: The object returned is the same tensor-like result produced by torch.utils.data._utils.collate.collate_tensor_fn(batch) (i.e., the batched numeric tensor), augmented with three additional attributes used by MONAI workflows: meta, applied_operations, and is_batch. meta is set to the result of default_collate applied to the per-sample meta dictionaries (or TraceKeys.NONE placeholders) and therefore contains batched metadata fields corresponding only to the keys common across samples. applied_operations is a Python list containing each sample's applied operations (or TraceKeys.NONE placeholders). is_batch is a boolean set to True to indicate the tensor represents a batch. These attributes are side effects attached to the returned object and are essential for tracking provenance and reverse-mapping predictions to original sample coordinates in medical imaging pipelines.
    
    Behavior, defaults, and failure modes:
        The function first calls torch.utils.data._utils.collate.collate_tensor_fn to produce a numeric batched tensor from batch. It then constructs meta_dicts by reading the .meta attribute from each element in batch and substituting TraceKeys.NONE where .meta is falsey. If at least one element yields a dict-like meta, the function computes the intersection of keys common to all dict-like meta entries and reduces each per-sample meta to only those common keys prior to batching; this ensures the batched meta contains only metadata fields that are present for every sample. The batched metadata is created by calling MONAI's default_collate on the list of per-sample meta dictionaries. The function collects applied_operations from each sample (using TraceKeys.NONE when falsey) and attaches this list to the returned tensor as applied_operations. The returned tensor is also annotated with is_batch = True.
        Default behavior assumes at least one sample in batch provides a dict-like .meta value. If batch is empty, or if none of the .meta entries are dict-like, the internal call to set.intersection with no arguments will raise a TypeError; therefore callers should ensure batch contains valid samples with at least one dict-like .meta. If any element in batch lacks the .meta or .applied_operations attribute, an AttributeError will be raised when the function attempts to access those attributes. Numerical collation failures (for example, incompatible tensor shapes that collate_tensor_fn cannot stack) will raise the same exceptions as torch's collate utilities. Because collate_fn_map is not used, passing a mapping has no effect in this implementation.
        This function mutates/augments only the returned tensor-like object by adding the meta, applied_operations, and is_batch attributes; it does not modify the input sample objects.
    
    Usage context and practical significance:
        In MONAI's medical imaging workflows (e.g., multi-dimensional CT/MRI preprocessing and training pipelines), MetaTensor objects carry both the numeric image data and metadata required to interpret spatial information and to undo transforms. collate_meta_tensor_fn ensures that when multiple samples are combined into a batch for model input, their metadata and transform provenance are batched and preserved in a consistent way. Downstream components (loss functions, post-processing, inverse transforms, metric calculations) rely on these attached attributes to map model outputs back to original image space and to interpret predictions in the clinical imaging context.
    """
    from monai.data.utils import collate_meta_tensor_fn
    return collate_meta_tensor_fn(batch, collate_fn_map)


################################################################################
# Source: monai.data.utils.dev_collate
# File: monai/data/utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def monai_data_utils_dev_collate(
    batch: list,
    level: int = 1,
    logger_name: str = "dev_collate"
):
    """monai.data.utils.dev_collate recursively collates a list-style batch while emitting detailed critical-level log messages to help debug PyTorch DataLoader collate problems in MONAI data pipelines for medical imaging. It is intended for interactive debugging of batching behavior (for example, when collating multi-dimensional NumPy arrays or PyTorch tensors representing medical images and related metadata) and reports the internal decisions and errors at the "critical" logging level so messages are easy to spot when handling exceptions.
    
    Args:
        batch (list): The input batch to collate. Expected to be a non-empty Python list where each element is one sample from a DataLoader. Elements may be torch.Tensor, numpy.ndarray / numpy.memmap, Python scalars (float, int, str, bytes), mappings (dict-like) or sequences (list/tuple) that represent nested sample fields (e.g., image arrays, labels, metadata). The function will index batch[0] to detect element types; passing an empty list will raise IndexError. The batch parameter corresponds to the "batch" argument a custom collate_fn receives in PyTorch DataLoader usage and is used in MONAI workflows for batching medical imaging data.
        level (int): Current recursion depth used purely for logging indentation and readability. The function constructs a prefix string of ">" repeated level times and includes that in each critical log message to indicate nested structure depth. Default is 1. This parameter controls only log formatting and is incremented by internal recursive calls; callers normally do not need to set it.
        logger_name (str): Name of the logger passed to logging.getLogger(...) used for all critical log outputs. Default is "dev_collate". Using a dedicated logger name allows integration with MONAI or user logging configurations so that critical diagnostics from dev_collate can be captured separately from other logs.
    
    Returns:
        When successful, the return value depends on the detected element type and the collate action:
        - torch.Tensor: If the batch elements are torch.Tensor, the function attempts to stack them along a new 0-th dimension and returns torch.stack(batch, 0). This is the standard behavior expected for batching image tensors in MONAI.
        - list: If elements are Python scalar types (float, int, str, bytes) or if the function is collating a sequence of scalars, it returns the original list (i.e., the batch) so downstream code receives a list of scalars for that field.
        - dict: If each element of the batch is a mapping (e.g., a dict of fields like {"image": tensor, "label": int, "meta": {...}} used in MONAI datasets), the function returns a dict where each key maps to the result of recursively collating the list of values for that key across the batch (useful for building batched dictionaries of image and metadata tensors).
        - nested list structure: If batch elements are sequences (lists/tuples), the function will check element lengths, log sizes, transpose the sequence-of-sequences via zip(*batch) and recursively collate each transposed group, returning a list whose entries are the recursively-collated results (this mirrors PyTorch's default collate behavior but with logging).
        - If the function converts numpy.ndarray or numpy.memmap elements, it first converts each to a torch tensor via torch.as_tensor(...) and then recurses; in that case the final return will follow the tensor-stacking behavior above.
        None: The function returns None implicitly when it encounters unsupported types, when it logs an error for inconsistent or unhandled types, or when a torch.stack operation raises a TypeError or RuntimeError (these exceptions are caught, logged at critical level, and the function returns None instead of propagating the exception). Callers should check for None to detect collate failures during debugging.
    
    Behavioral notes, side effects, and failure modes:
        - The function is diagnostic: it logs detailed critical-level messages describing each collate decision, encountered shapes, types, key-level recursion, and any errors. Logs include an indentation marker derived from level to show nesting depth.
        - For torch.Tensor inputs, TypeError and RuntimeError raised by torch.stack are caught; the function logs the exception message and relevant context (element types or shapes) and returns None instead of raising, making it suitable for use inside exception handlers or debugging sessions.
        - For numpy.ndarray or numpy.memmap arrays, elements are converted to torch tensors via torch.as_tensor and then processed recursively; no new data type conversions beyond those in the source code are introduced.
        - For sequence elements, the function attempts to compute len(...) on each element. If any element lacks __len__ and a TypeError is raised, the function logs the offending types and returns None.
        - If sequence element lengths differ across the batch, the function logs a critical message about inconsistent sizes but will still attempt to transpose and collate; callers should inspect logs to determine if the inconsistency is problematic.
        - The function assumes batch is non-empty and will raise IndexError if an empty list is supplied because it inspects batch[0] to determine element type.
        - The function does not raise new exception types beyond those already handled in the implementation; other unexpected exceptions may propagate to the caller.
        - Because logging is performed at the "critical" level, messages are intended to be prominent and may be captured by monitoring systems; use logger_name to route messages appropriately.
    
    Practical significance in MONAI:
        - This helper is used during development and debugging of MONAI data pipelines where batched inputs often include multi-dimensional medical image arrays (NumPy or PyTorch tensors), labels, and nested metadata. It helps diagnose mismatched shapes, inconsistent list lengths, unsupported types, and conversion issues when building batches for training or inference. See PyTorch's collate_fn documentation for the broader collate contract that dev_collate inspects and emulates: https://pytorch.org/docs/stable/data.html#working-with-collate-fn
    """
    from monai.data.utils import dev_collate
    return dev_collate(batch, level, logger_name)


################################################################################
# Source: monai.data.utils.remove_extra_metadata
# File: monai/data/utils.py
# Category: valid
################################################################################

def monai_data_utils_remove_extra_metadata(meta: dict):
    """Remove extra metadata keys from a MONAI metadata dictionary in-place.
    
    This function is part of the monai.data.utils utilities and is used in MONAI preprocessing and data handling workflows to remove keys that are considered "extra metadata" according to MONAI conventions. It determines which keys to remove by calling get_extra_metadata_keys(), then delegates the actual removal to remove_keys(data=meta, keys=keys). Typical uses include cleaning a sample's metadata before serialization, logging, or passing the metadata into training/evaluation components so that large, temporary, or implementation-specific entries do not propagate through the pipeline.
    
    Args:
        meta (dict): A dictionary containing metadata for a medical imaging sample or batch, following MONAI's metadata conventions. This mapping is modified in-place: any top-level keys in meta that match the set returned by get_extra_metadata_keys() will be removed from this same dict object. The function does not return a new dict. The values and nested structures of remaining keys are preserved. If meta is empty, the function performs no changes. The caller should provide a dict; passing an object that is not a dict may raise a TypeError or other exception from underlying operations.
    
    Returns:
        None: This function returns None and has the side effect of mutating the provided meta dict by removing the extra metadata keys. The mutation is immediate and visible to any other references to the same dict. To retain the original metadata, callers must pass a copy of meta (for example, meta.copy()) before calling this function. Calling this function multiple times is effectively idempotent for the same set of extra keys because keys already removed will not be present on subsequent calls. Failure modes include exceptions if meta is not a dict or if underlying helper functions raise errors; this function does not perform deep traversal of nested dicts beyond removing top-level keys that match get_extra_metadata_keys().
    """
    from monai.data.utils import remove_extra_metadata
    return remove_extra_metadata(meta)


################################################################################
# Source: monai.data.utils.remove_keys
# File: monai/data/utils.py
# Category: valid
################################################################################

def monai_data_utils_remove_keys(data: dict, keys: list[str]):
    """monai.data.utils.remove_keys removes one or more keys from a mapping in-place without returning a value.
    
    This utility function is intended for MONAI data-processing workflows (for example, preprocessing and transform pipelines in medical imaging) where sample dictionaries carry images, labels, and metadata and specific keys need to be discarded before further processing or saving. The function iterates over the provided list of keys and calls dict.pop(key, None) for each key so that missing keys are ignored silently. Because it modifies the input mapping directly, callers should not expect a new dictionary or any return value.
    
    Args:
        data (dict): The dictionary to be modified. In MONAI this is typically a sample or batch dictionary containing imaging tensors and associated metadata (e.g., image, label, meta_dict). The function operates on this object in-place; no copy is made. If an object that does not implement dict.pop is provided, a runtime exception (for example, AttributeError or TypeError) may be raised by the interpreter.
        keys (list[str]): An ordered collection of string keys to remove from data. For each key in this list, the function attempts to delete that key from data using data.pop(key, None). If a key is not present in data, it is ignored and no error is raised. The operation is performed in the order of this list; duplicate entries in keys will cause repeated pop attempts but have no additional effect after the first removal.
    
    Returns:
        None: This function has no return value. Its effect is the side effect of removing the specified keys from the provided data dictionary. After the call, data will have those keys removed (if they existed).
    """
    from monai.data.utils import remove_keys
    return remove_keys(data, keys)


################################################################################
# Source: monai.data.utils.worker_init_fn
# File: monai/data/utils.py
# Category: fix_docstring
# Reason: Schema parsing failed: Cannot generate JSON schema for worker_init_fn because the docstring has no description for the argument 'worker_id'
################################################################################

def monai_data_utils_worker_init_fn(worker_id: int):
    """monai.data.utils.worker_init_fn sets per-worker random seeds for MONAI dataset transforms when used as the PyTorch DataLoader worker initialization callback.
    
    This function is intended to be passed directly to PyTorch DataLoader via its worker_init_fn argument so that each worker process used for loading medical imaging data in MONAI-based training receives a distinct random seed. In MONAI workflows this avoids different workers producing identical random augmentations for the same samples, which preserves desired augmentation diversity and supports reproducible experiments across multiple workers. The function obtains PyTorch worker information via torch.utils.data.get_worker_info(), and then calls set_rnd(...) with the DataLoader worker's dataset and the worker-specific seed (worker_info.seed) so that the dataset-level random number generators used by MONAI transforms are initialized for that worker.
    
    Args:
        worker_id (int): The integer worker index supplied by PyTorch DataLoader when invoking a worker initialization callback. In the PyTorch DataLoader API this is typically 0-based and identifies the current worker process. In this implementation the parameter exists to match the DataLoader callback signature but is not referenced directly by the body of this function; the function instead queries torch.utils.data.get_worker_info() to obtain the actual worker-specific information and seed.
    
    Returns:
        None: This function does not return a value. Its primary effect is a side effect: it calls torch.utils.data.get_worker_info() to obtain a WorkerInfo object and then calls set_rnd(worker_info.dataset, seed=worker_info.seed) to set the random seed/state used by the dataset/transforms in that worker process. Successful execution ensures different workers use different RNG seeds for MONAI transforms. Possible failure modes include: torch.utils.data.get_worker_info() returning None when not invoked inside a DataLoader worker (which will cause an AttributeError when accessing dataset or seed), the dataset lacking the expected attributes or not being compatible with set_rnd (which may raise AttributeError or TypeError), or missing imports/definitions for torch or set_rnd in the runtime environment. The function has no default behavior beyond these side effects and must be used within the PyTorch DataLoader worker context to function as intended.
    """
    from monai.data.utils import worker_init_fn
    return worker_init_fn(worker_id)


################################################################################
# Source: monai.engines.utils.default_metric_cmp_fn
# File: monai/engines/utils.py
# Category: valid
################################################################################

def monai_engines_utils_default_metric_cmp_fn(current_metric: float, prev_best: float):
    """monai.engines.utils.default_metric_cmp_fn: Default comparator for scalar metrics used by MONAI training and evaluation engines. This function implements a strict "greater-than" comparison to determine whether the metric computed in the current evaluation round represents an improvement over the best metric observed in previous rounds. In MONAI workflows (see README), such a comparator is typically used by components that track the best model state, save checkpoints, or drive early stopping decisions based on evaluation metrics for medical imaging tasks (for example, validation Dice score or accuracy).
    
    Args:
        current_metric (float): Metric value of the current round computation. This is a scalar floating-point value produced by a metric calculation (for example, validation accuracy, Dice coefficient, or any other scalar metric used in MONAI evaluation). The value represents the performance of the model for the current epoch/iteration and is compared against prev_best to decide if an improvement occurred.
        prev_best (float): The best metric value observed in previous rounds to compare with. This is a scalar floating-point value representing the historically best recorded metric (for example, the highest validation Dice seen so far) used as the baseline for determining improvement.
    
    Returns:
        bool: True if current_metric is strictly greater than prev_best (current_metric > prev_best), indicating an improvement according to this maximization comparator; False otherwise. Equal values do not count as improvement. The returned boolean is intended to be consumed by MONAI engine utilities for actions such as saving a new best checkpoint or updating best-metric bookkeeping.
    
    Behavior, side effects, defaults, and failure modes:
        This function is a pure, side-effect-free comparator: it does not modify inputs or external state. There are no default values for parameters; both must be provided as floats. The comparison follows standard IEEE floating-point semantics: comparisons with NaN yield False (so a NaN current_metric will not be considered an improvement), and positive/negative infinity follow normal numeric ordering (e.g., float('inf') is greater than any finite prev_best). If the provided arguments are not comparable as floats (for example, types that do not support the ">" operator with float), Python will raise a TypeError or propagate the comparison behavior of the provided types; such cases are outside the intended usage and may lead to exceptions.
    """
    from monai.engines.utils import default_metric_cmp_fn
    return default_metric_cmp_fn(current_metric, prev_best)


################################################################################
# Source: monai.handlers.utils.stopping_fn_from_metric
# File: monai/handlers/utils.py
# Category: fix_docstring
# Reason: Schema parsing failed: ("Couldn't parse this type hint, likely due to a custom class or object: ", collections.abc.Callable[[<monai.utils.module.optional_import.<locals>._LazyRaise object at 0x7f69a0460280>], typing.Any])
################################################################################

def monai_handlers_utils_stopping_fn_from_metric(metric_name: str):
    """Returns a stopping function that reads a metric value by name from an Ignite Engine's state for use with ignite.handlers.EarlyStopping.
    
    This helper is used in MONAI training and evaluation workflows (medical imaging deep learning) to construct the score function required by ignite.handlers.EarlyStopping. MONAI engines (ignite.engine.Engine) commonly populate engine.state.metrics with validation or training metrics (for example "val_loss", "val_dice") after each epoch or iteration. The callable returned by this function simply accesses engine.state.metrics[metric_name] and returns that value, so the EarlyStopping handler can decide whether to stop based on that metric.
    
    Args:
        metric_name (str): The name of the metric key to retrieve from engine.state.metrics. This string should match a metric key that MONAI or user code has stored in the Ignite Engine's state.metrics mapping (for example "val_loss" or "val_mean_dice"). This parameter has no default and must be provided; it determines which metric the returned stopping function will read when called.
    
    Returns:
        Callable[[Engine], Any]: A function that accepts a single argument engine (an ignite.engine.Engine instance) and returns engine.state.metrics[metric_name]. The returned callable does not modify the engine or its state; it only reads the metrics mapping. Practical significance: pass this callable as the score_function argument to ignite.handlers.EarlyStopping so EarlyStopping can monitor the specified metric produced by MONAI training/evaluation loops.
    
    Failure modes and side effects:
        The returned function will raise a KeyError if engine.state.metrics does not contain metric_name. It may raise AttributeError if the provided object does not have state or metrics attributes, or if engine.state.metrics is not a mapping-like object. The function performs no in-place changes to the engine state. When using with EarlyStopping, ensure the metric values are comparable (e.g., numeric scalars) and set EarlyStopping's greater_is_better appropriately for the monitored metric.
    """
    from monai.handlers.utils import stopping_fn_from_metric
    return stopping_fn_from_metric(metric_name)


################################################################################
# Source: monai.inferers.merger.iterate_over_chunks
# File: monai/inferers/merger.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def monai_inferers_merger_iterate_over_chunks(
    chunks: tuple,
    cdata_shape: tuple,
    slice_tuple: tuple = ()
):
    """Iterate over regularly spaced chunks of a multi-dimensional array and yield index tuples that select each chunk.
    
    This function is a small utility used by MONAI's inferers.merger logic when reconstructing outputs from tiled or chunked inference over medical images. Given a chunk shape and the number of chunks in each dimension (cdata_shape), it yields, one at a time, tuples of Python slice objects that can be used to index into a NumPy or PyTorch array/tensor to extract the corresponding tile. The function is implemented recursively to support an arbitrary number of spatial dimensions and is memory-efficient because it yields slices lazily rather than allocating a full list of indices.
    
    Args:
        chunks (tuple): The size of each chunk along every dimension, provided as a tuple of integers. Each element chunks[d] is the length (number of elements) of a chunk along dimension d. In the MONAI inferers/merger context, these values represent the spatial tile size used during tiled inference or patch-wise processing of medical images.
        cdata_shape (tuple): The number of chunks along each dimension, provided as a tuple of integers with the same length as chunks. Each element cdata_shape[d] is the count of chunks along dimension d. In the merging workflow this typically equals the number of tiles the image was split into per dimension.
        slice_tuple (tuple): A prefix tuple of already-computed slice objects to be prepended to the yielded slice tuples. This parameter defaults to the empty tuple () and is used internally for the recursive construction of full multi-dimensional slice tuples; callers normally omit it. When provided, slice_tuple must itself be a tuple of slice objects corresponding to earlier (leading) dimensions.
    
    Raises:
        ValueError: If len(chunks) != len(cdata_shape). The function requires the chunk size specification and the chunk count specification to have identical dimensionality; otherwise the mapping from chunk indices to array slices is undefined.
    
    Returns:
        generator: A Python generator that yields tuples of slice objects. Each yielded value is a tuple whose length equals len(chunks) (and len(cdata_shape)); element d of the tuple is a slice(start, stop) selecting the d-th dimension interval for one chunk. The slice intervals follow Python slice semantics and are half-open [start, stop), where start = i * chunks[d] and stop = (i + 1) * chunks[d] for chunk index i along that dimension. The generator produces tuples in order where the first dimension (dimension 0) is iterated outermost and remaining dimensions are iterated recursively, making the output suitable for sequentially indexing and copying tiles when reconstructing large medical image volumes. No other side effects occur.
    """
    from monai.inferers.merger import iterate_over_chunks
    return iterate_over_chunks(chunks, cdata_shape, slice_tuple)


################################################################################
# Source: monai.losses.perceptual.medicalnet_intensity_normalisation
# File: monai/losses/perceptual.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def monai_losses_perceptual_medicalnet_intensity_normalisation(volume: numpy.ndarray):
    """monai.losses.perceptual.medicalnet_intensity_normalisation: Normalize a medical image volume to zero mean and unit variance following the MedicalNet preprocessing convention.
    
    This function implements the intensity normalization used in the MedicalNet project (see referenced source in original implementation). It computes the global arithmetic mean and standard deviation of all voxels in the provided n-dimensional medical image volume (for example, a 3D MRI or CT scan stored as a NumPy array) and returns a new array where each voxel intensity is shifted and scaled to have zero mean and unit variance. This normalization is commonly applied as a preprocessing step in deep learning workflows for medical imaging (as in MONAI) to stabilize training, make network weights more comparable across inputs, and reduce sensitivity to absolute intensity scales between studies or scanners.
    
    Args:
        volume (numpy.ndarray): An n-dimensional NumPy array containing the medical image intensities to normalize. The array represents voxel or pixel intensity values across spatial dimensions (and optionally channels). This parameter is used as the source data from which the mean and standard deviation are computed globally (over all elements of the array). The function does not modify this input in-place; it reads the values to compute statistics and produces a separate normalized array as output.
    
    Returns:
        numpy.ndarray: A NumPy array of the same shape as the input volume containing the normalized intensities computed as (volume - mean) / std, where mean and std are the scalar mean and standard deviation of all elements in the input. The returned array contains floating-point values (dtype determined by NumPy broadcasting rules) and preserves the spatial structure (shape) of the input.
    
    Behavior, side effects, and failure modes:
        - The mean and standard deviation are computed over all elements of the input array (global normalization), not per-slice or per-channel, consistent with the MedicalNet implementation referenced in the source.
        - The operation returns a new array and does not mutate the input volume.
        - If the input contains NaNs or infinities, the computed mean and standard deviation may be NaN or infinite, and the returned array will reflect those values (propagating NaNs/infs according to NumPy arithmetic).
        - If the standard deviation equals zero (all input values are identical), division by zero will occur and NumPy will produce inf or NaN values in the output; callers should check for zero variance and handle that case explicitly (for example, bypass normalization or add a small epsilon) if such behavior is undesirable.
        - If the input is an empty array, NumPy will emit a warning and produce NaNs for mean and std, and the returned array will contain NaNs.
        - No device or backend switching is performed; the function expects and returns a NumPy ndarray (CPU memory).
    """
    from monai.losses.perceptual import medicalnet_intensity_normalisation
    return medicalnet_intensity_normalisation(volume)


################################################################################
# Source: monai.metrics.confusion_matrix.check_confusion_matrix_metric_name
# File: monai/metrics/confusion_matrix.py
# Category: fix_docstring
# Reason: Schema parsing failed: Cannot generate JSON schema for check_confusion_matrix_metric_name because the docstring has no description for the argument 'metric_name'
################################################################################

def monai_metrics_confusion_matrix_check_confusion_matrix_metric_name(metric_name: str):
    """monai.metrics.confusion_matrix.check_confusion_matrix_metric_name: Check, normalize, and map a confusion-matrix metric name or alias to the canonical short name used by MONAI's confusion-matrix metric implementations.
    
    This function is used in MONAI's metrics and evaluation pipelines for healthcare imaging to allow flexible user-provided metric names (for example in configuration files, CLI arguments, or metric selection code) while ensuring the downstream confusion-matrix computation modules receive a single, canonical metric identifier. The routine performs two deterministic normalization steps on the input string: it replaces spaces with underscores and lowercases the result. It then maps common synonyms and longer descriptive names for confusion-matrix-derived metrics to their standardized short forms (for example, "sensitivity" -> "tpr", "precision" -> "ppv", "false_positive_rate" -> "fpr", "f1_score" -> "f1", etc.). The canonical names produced by this function correspond to the metric keys expected by MONAI's confusion-matrix metric implementations and therefore determine which formula is applied when computing the metric on medical image classification or segmentation results. This function has no side effects beyond returning the normalized name and does not modify the caller's objects.
    
    Args:
        metric_name (str): The user-supplied metric identifier or alias describing a confusion-matrix-derived metric. In the MONAI medical imaging context this string may be a human-readable name (e.g., "Sensitivity", "False Positive Rate", "F1 Score") or a short code. The function will normalize the string by replacing spaces with underscores and converting to lower case, then map known synonyms to a canonical short name used internally by MONAI (examples include "tpr", "tnr", "ppv", "npv", "fnr", "fpr", "fdr", "for", "pt", "ts", "acc", "ba", "f1", "mcc", "fm", "bm", "mk"). Providing a non-string value is not supported by the implementation and will raise a TypeError from the underlying string operations.
    
    Returns:
        str: The canonical, simplified metric name (one of the short identifiers used by MONAI confusion-matrix metrics) corresponding to the input alias. This returned value is intended to be passed directly to MONAI's metric computation routines to select the appropriate confusion-matrix-derived calculation. The function always returns a new string and does not mutate the input provided by the caller.
    
    Raises:
        NotImplementedError: If the normalized metric_name does not match any supported metric alias recognized by MONAI's confusion-matrix utilities, the function raises NotImplementedError with the message "the metric is not implemented." This indicates the caller supplied an unknown metric name and must either use a supported alias or extend the mapping in the MONAI metrics code.
    """
    from monai.metrics.confusion_matrix import check_confusion_matrix_metric_name
    return check_confusion_matrix_metric_name(metric_name)


################################################################################
# Source: monai.metrics.froc.compute_froc_score
# File: monai/metrics/froc.py
# Category: valid
################################################################################

def monai_metrics_froc_compute_froc_score(
    fps_per_image: numpy.ndarray,
    total_sensitivity: numpy.ndarray,
    eval_thresholds: tuple = (0.25, 0.5, 1, 2, 4, 8)
):
    """Compute the CAMELYON-style FROC score (average sensitivity at predefined false positive rates per image).
    
    This function is modified from the official CAMELYON16 challenge evaluation code and implements the challenge's second evaluation metric: the average sensitivity (true positive rate) evaluated at a set of predefined false positive rates per whole-slide image. It is intended for use in medical imaging detection pipelines (for example, lesion/metastasis detection on whole-slide histopathology images) where model outputs are aggregated at multiple detection thresholds to produce per-threshold average false positives per image and corresponding sensitivities. The function linearly interpolates the provided sensitivity curve at the requested false-positive-per-image thresholds and returns the arithmetic mean of those interpolated sensitivities. The implementation reverses the input arrays before interpolation to satisfy numpy.interp's requirement that the interpolation x-coordinates be in increasing order.
    
    Args:
        fps_per_image (numpy.ndarray): A one-dimensional numeric array containing the average number of false positives per image computed at a series of detection thresholds. Each element corresponds to a particular detection threshold and represents the expected false positives per whole-slide image for that threshold. This array must have the same length as total_sensitivity and represent the x-axis values for interpolation. In typical MONAI/CAMELYON workflows these values are produced by aggregating per-image false positive counts across a validation/test set at multiple detection thresholds.
        total_sensitivity (numpy.ndarray): A one-dimensional numeric array of the same length as fps_per_image containing the sensitivities (true positive rates) corresponding to each detection threshold. Each element is the fraction of true lesions correctly detected at the matching threshold. This array provides the y-axis values for interpolation. Both fps_per_image and total_sensitivity are reversed internally (via [::-1]) before interpolation so that numpy.interp receives increasing x-coordinates.
        eval_thresholds (tuple): A tuple of numeric false-positive-per-image target rates at which the function will evaluate (by linear interpolation) the sensitivity curve defined by fps_per_image and total_sensitivity. Defaults to (0.25, 0.5, 1, 2, 4, 8), which is the canonical set used in the CAMELYON16 challenge for reporting the averaged sensitivity metric. The tuple elements are treated as the x-coordinates at which to sample the sensitivity curve.
    
    Returns:
        Any: The function returns the arithmetic mean of the interpolated sensitivities evaluated at eval_thresholds. Concretely, the implementation calls numpy.interp(eval_thresholds, fps_per_image[::-1], total_sensitivity[::-1]) to obtain interpolated sensitivity values at each requested false-positive-per-image rate, then returns numpy.mean(...) of those interpolated values. There are no side effects; the function does not modify its inputs. Possible failure modes and usage notes: fps_per_image and total_sensitivity must be one-dimensional arrays of equal length; if they differ in length numpy.interp will raise an error. If fps_per_image contains NaN or non-numeric values, results will be undefined. If eval_thresholds contain values outside the range of fps_per_image, numpy.interp will return endpoint sensitivity values (no exception), which may bias the mean toward endpoint behavior; users should ensure eval_thresholds are chosen appropriately for their data. This function is primarily intended for evaluation in supervised medical image detection tasks following the CAMELYON-style FROC protocol.
    """
    from monai.metrics.froc import compute_froc_score
    return compute_froc_score(fps_per_image, total_sensitivity, eval_thresholds)


################################################################################
# Source: monai.metrics.utils.get_code_to_measure_table
# File: monai/metrics/utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def monai_metrics_utils_get_code_to_measure_table(spacing: tuple, device: str = None):
    """monai.metrics.utils.get_code_to_measure_table returns a lookup table that maps a neighbourhood code (an integer index that encodes a local voxel/pixel neighbourhood configuration) to the corresponding geometric measure used in MONAI metrics: contour length for 2D data or surface area for 3D data. This function is used in MONAI's medical-imaging metric computations to convert local neighbourhood encodings into physical boundary contributions (lengths or areas) that can be summed to compute segmentation contour lengths or surface areas, taking into account the physical spacing of the image voxels/pixels and the target device for computation.
    
    Args:
        spacing (sequence of 2 or 3 numbers): a sequence specifying the spacing of the spatial dimensions (for example voxel spacing in millimetres). The length of this sequence determines the spatial dimensionality: length 2 selects 2D behavior (contour length), length 3 selects 3D behavior (surface area). The values are used to scale the per-neighbourhood code contributions into physical units appropriate for the imaging domain.
        device (optional): device to put the table on. If provided, the returned table will be placed on this device; if None, the table will be created on the implementation's default device. The device argument allows the table to be created directly on a GPU or other target to avoid costly host-to-device transfers when used in downstream metric computations.
    
    Behavior and details:
        The function first infers the spatial dimensionality from len(spacing) and enforces that spacing represents either 2 or 3 spatial dimensions via internal helpers (ensure_tuple_rep and look_up_option). For 2D spacing the function calls create_table_neighbour_code_to_contour_length(spacing, device) to construct a lookup table of contour lengths for each neighbourhood code; for 3D spacing it calls create_table_neighbour_code_to_surface_area(spacing, device) to construct a lookup table of surface areas for each neighbourhood code. The returned table is organized so that the index equals the neighbourhood code and the stored value equals the geometric contribution (contour length or surface area) for that code, scaled by the provided spacing. This table is intended to be used by MONAI metric implementations that accumulate boundary contributions from local neighbourhood encodings to obtain global contour length or surface area measurements for segmentation outputs.
    
    Defaults and side effects:
        If device is None, the function will create the table on the implementation's default device (device selection is implementation-dependent). The function has no other side effects besides allocating and returning the lookup table on the chosen device.
    
    Failure modes:
        If spacing does not represent exactly 2 or 3 spatial dimensions (i.e., its length is not 2 or 3), the internal helpers will raise an error (for example ValueError) indicating invalid spatial dimensionality. If an invalid or unsupported device is provided, the table creation routines may raise device-placement related errors (for example TypeError or runtime errors from the underlying tensor library).
    
    Returns:
        table: a 1D mapping (array- or tensor-like) where each index is a neighbourhood code and each value is the corresponding contour length (for 2D spacing) or surface area (for 3D spacing), scaled according to spacing and placed on the requested device. The table is intended for use in MONAI's segmentation boundary metric calculations to convert neighbourhood codes into physical length/area contributions.
    """
    from monai.metrics.utils import get_code_to_measure_table
    return get_code_to_measure_table(spacing, device)


################################################################################
# Source: monai.networks.layers.convutils.polyval
# File: monai/networks/layers/convutils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def monai_networks_layers_convutils_polyval(coef: list, x: float):
    """monai.networks.layers.convutils.polyval evaluates a polynomial at a given variable using Horner's method; this utility is part of MONAI's convolution-related utilities (monai.networks.layers.convutils) and can be used wherever a polynomial defined by coefficient sequences must be computed in medical imaging pipelines, for example when mapping parameters or constructing analytic filter responses.
    
    Args:
        coef (sequence of float or torch.Tensor): Coefficients of the polynomial, provided as a 1D sequence (for example a Python list or a 1D torch.Tensor) where coef[0] is the coefficient of the highest-degree term and coef[-1] is the constant term. In the context of MONAI's convutils, these coefficients define the polynomial to be evaluated (e.g., polynomial kernels or parameter transforms). The function converts coef to a torch.Tensor with dtype torch.float and will use the device of x (if x is a torch.Tensor) when creating this tensor. If coef is zero-dimensional or has length < 1, the routine attempts to return a zero tensor matching x.shape (see Failure modes below).
        x (float or sequence of float or torch.Tensor): The variable(s) at which to evaluate the polynomial. This can be a Python scalar float, a sequence/array of floats, or a torch.Tensor. The function converts x to a torch.Tensor with dtype torch.float; if x is already a torch.Tensor, its device is used for the computation and for creating the coef tensor. The returned tensor will have a shape that corresponds to x after conversion (so supplying a tensor-shaped x produces an output of the same shape).
    
    Behavior and side effects:
        The polynomial is evaluated using Horner's method for numerical stability and efficiency: for coef of length n the result computed is coef[n-1] + x * (coef[n-2] + ... + x * (coef[1] + x * coef[0])).
        Both coef and x are converted to torch.Tensor with dtype torch.float. If x is a torch.Tensor, the conversion uses x.device so the result resides on the same device (CPU/GPU) as x; otherwise tensors are created on the default device.
        If coef is zero-dimensional or has length < 1, the implementation returns torch.zeros(x.shape). Because the conversion of x to a tensor happens after this check, passing a plain Python scalar float for x when coef is empty can raise an AttributeError (float has no attribute shape). To avoid this failure mode, provide a non-empty coef or pass x as a torch.Tensor or sequence with a shape attribute.
        No in-place modification of the inputs is performed; new tensors are allocated for the computation.
    
    Failure modes:
        - Supplying an empty coefficient sequence and a plain Python float for x can raise an AttributeError because the code attempts to access x.shape before converting x to a tensor.
        - Passing coefficient objects that cannot be converted to a float tensor will raise the corresponding torch.as_tensor conversion error.
        - Very large polynomial degrees or extreme values may lead to floating-point overflow or loss of precision; use appropriate scaling or higher-precision dtypes if necessary.
    
    Returns:
        torch.Tensor: a 1D torch tensor of dtype torch.float containing the evaluated polynomial values. The tensor is allocated on the device of x if x is a torch.Tensor; otherwise it is allocated on the default device. The returned tensor's shape corresponds to x after conversion to a torch.Tensor (for scalar x this will be a scalar tensor). The tensor values are the result of applying the polynomial (defined by coef) to each element of x using Horner's method.
    """
    from monai.networks.layers.convutils import polyval
    return polyval(coef, x)


################################################################################
# Source: monai.networks.layers.factories.adaptive_avgpooling_factory
# File: monai/networks/layers/factories.py
# Category: fix_docstring
# Reason: Schema parsing failed: ("Couldn't parse this type hint, likely due to a custom class or object: ", type[torch.nn.modules.pooling.AdaptiveAvgPool1d | torch.nn.modules.pooling.AdaptiveAvgPool2d | torch.nn.modules.pooling.AdaptiveAvgPool3d])
################################################################################

def monai_networks_layers_factories_adaptive_avgpooling_factory(dim: int):
    """monai.networks.layers.factories.adaptive_avgpooling_factory returns the PyTorch Adaptive Average Pooling class corresponding to a requested spatial dimensionality. This factory function is used within MONAI (a PyTorch-based medical imaging framework) to select the correct adaptive average pooling layer type when building neural network architectures that operate on 1D signals, 2D image slices, or 3D volumes (common modalities in healthcare imaging workflows).
    
    Args:
        dim (int): Desired spatial dimension of the adaptive average pooling layer. This integer selects which PyTorch class is returned: 1 selects nn.AdaptiveAvgPool1d for 1D signals, 2 selects nn.AdaptiveAvgPool2d for 2D images (e.g., slice-based processing), and 3 selects nn.AdaptiveAvgPool3d for 3D volumes (e.g., CT/MRI volumes). The parameter is required and has no default. The argument is used directly as an index (dim - 1) into the internal tuple of classes; therefore it must be an integer in the set {1, 2, 3} to succeed.
    
    Returns:
        type[nn.AdaptiveAvgPool1d | nn.AdaptiveAvgPool2d | nn.AdaptiveAvgPool3d]: The selected PyTorch AdaptiveAvgPool class corresponding to the requested dimensionality. The returned value is a class (not an instantiated nn.Module); callers should instantiate it (for example, with an output size argument) when constructing a network layer. This selection enables MONAI components and user code to programmatically obtain the correct pooling layer type for different medical imaging data dimensionalities.
    
    Behavior and side effects:
        The function has no side effects other than returning the class object. It is registered as a pool factory under the key "adaptiveavg" via the Pool.factory_function decorator in the source, allowing MONAI factory-based construction code to look up this class by name.
    
    Failure modes:
        If dim is not in {1, 2, 3}, an IndexError will be raised because the internal tuple lookup uses dim - 1. If a non-integer is passed, a TypeError may be raised during the tuple indexing operation. The function does not validate additional attributes (such as output sizes), which are the responsibility of the caller when instantiating the returned class.
    """
    from monai.networks.layers.factories import adaptive_avgpooling_factory
    return adaptive_avgpooling_factory(dim)


################################################################################
# Source: monai.networks.layers.factories.adaptive_maxpooling_factory
# File: monai/networks/layers/factories.py
# Category: fix_docstring
# Reason: Schema parsing failed: ("Couldn't parse this type hint, likely due to a custom class or object: ", type[torch.nn.modules.pooling.AdaptiveMaxPool1d | torch.nn.modules.pooling.AdaptiveMaxPool2d | torch.nn.modules.pooling.AdaptiveMaxPool3d])
################################################################################

def monai_networks_layers_factories_adaptive_maxpooling_factory(dim: int):
    """Factory that returns the PyTorch adaptive max pooling class corresponding to a specified spatial dimension (1, 2, or 3). This function is used by MONAI's pooling factory (registered as Pool.factory_function("adaptivemax")) to map a simple dimension identifier into the concrete PyTorch nn.AdaptiveMaxPoolXd class so network-building code in medical imaging workflows can instantiate the appropriate adaptive max-pooling module for 1D signals, 2D images (slices), or 3D volumes.
    
    This factory does not create a layer instance; it returns the class object for the appropriate AdaptiveMaxPool module. The returned class can be instantiated as a normal PyTorch module (for example, returned_type(output_size)) and then used in model definitions to perform spatial down-sampling via maximum pooling in MONAI models for healthcare imaging.
    
    Args:
        dim (int): desired dimension of the adaptive max pooling layer. This integer selects which PyTorch class is returned: 1 selects torch.nn.AdaptiveMaxPool1d for 1D signals, 2 selects torch.nn.AdaptiveMaxPool2d for 2D images (typical for slice-based medical imaging or 2D CNNs), and 3 selects torch.nn.AdaptiveMaxPool3d for 3D volumes (typical for volumetric medical imaging such as CT/MRI). The parameter is required and must be an integer; it is interpreted directly by indexing a tuple of the three classes in the order (1d, 2d, 3d).
    
    Returns:
        type[nn.AdaptiveMaxPool1d | nn.AdaptiveMaxPool2d | nn.AdaptiveMaxPool3d]: the PyTorch adaptive max pooling class corresponding to the requested spatial dimensionality. This is the class object (not an instantiated module); call it with the appropriate output_size argument to create an nn.Module instance that performs adaptive max pooling in the chosen dimensionality.
    
    Raises:
        IndexError: if dim is an integer greater than 3, the internal tuple lookup (types[dim - 1]) will raise IndexError. Users should pass 1, 2, or 3 to avoid this error.
        TypeError: if dim is not an integer type that supports subtraction and indexing (for example, a float or None), Python will raise a TypeError when attempting to use it as a list index (dim - 1). Note that dim values <= 0 will not raise here but will index the tuple using Python negative indexing semantics (for example, dim == 0 returns the 3D class); therefore only 1, 2, or 3 should be used for unambiguous behavior.
    
    Side effects:
        None. The function performs no I/O, does not modify global state, and only returns a class object. The practical effect in MONAI workflows is to provide a simple programmatic mapping from an integer dimensionality indicator to the concrete PyTorch adaptive max-pooling class used when constructing networks for medical imaging tasks.
    """
    from monai.networks.layers.factories import adaptive_maxpooling_factory
    return adaptive_maxpooling_factory(dim)


################################################################################
# Source: monai.networks.layers.factories.avgpooling_factory
# File: monai/networks/layers/factories.py
# Category: fix_docstring
# Reason: Schema parsing failed: ("Couldn't parse this type hint, likely due to a custom class or object: ", type[torch.nn.modules.pooling.AvgPool1d | torch.nn.modules.pooling.AvgPool2d | torch.nn.modules.pooling.AvgPool3d])
################################################################################

def monai_networks_layers_factories_avgpooling_factory(dim: int):
    """monai.networks.layers.factories.avgpooling_factory returns the PyTorch average pooling layer class corresponding to a requested spatial dimension. In the MONAI medical-imaging framework this factory is used to select the appropriate nn.AvgPoolNd class (1D, 2D, or 3D) for building network architectures that perform spatial downsampling or local averaging of feature maps (for example, when constructing encoder blocks or reducing resolution of volumetric medical images).
    
    Args:
        dim (int): Desired spatial dimension for the average pooling layer. This argument selects which PyTorch nn.AvgPoolNd class is returned: dim == 1 selects nn.AvgPool1d, dim == 2 selects nn.AvgPool2d, and dim == 3 selects nn.AvgPool3d. The value is interpreted as an integer index into the supported pooling dimensions and must be provided exactly as an integer.
    
    Returns:
        type[nn.AvgPool1d | nn.AvgPool2d | nn.AvgPool3d]: The PyTorch average pooling layer class object corresponding to the requested dimension. The mapping is: 1 -> nn.AvgPool1d, 2 -> nn.AvgPool2d, 3 -> nn.AvgPool3d. The returned object is a class (not an instantiated layer); callers should instantiate it with appropriate arguments such as kernel_size, stride, padding, etc., when constructing network modules in MONAI.
    
    Raises:
        IndexError: If dim is an integer outside the supported set {1, 2, 3}, the function attempts to index the internal tuple and an IndexError is raised. This indicates an unsupported pooling dimension was requested.
        TypeError: If dim is not an integer type that can be used as a tuple index (for example, a float or non-integer), a TypeError will be raised by the indexing operation.
    
    Behavior and side effects:
        This function performs no runtime side effects when called; it simply returns a reference to a PyTorch AvgPool class. At module import time, the decorator present on this function (Pool.factory_function("avg")) registers this factory under the key "avg" with MONAI's Pool factory mechanism, enabling lookup by pooling type name elsewhere in the MONAI codebase. The factory itself does not instantiate layers or modify global state beyond the decorator's registration at import.
    """
    from monai.networks.layers.factories import avgpooling_factory
    return avgpooling_factory(dim)


################################################################################
# Source: monai.networks.layers.factories.batch_factory
# File: monai/networks/layers/factories.py
# Category: fix_docstring
# Reason: Schema parsing failed: ("Couldn't parse this type hint, likely due to a custom class or object: ", type[torch.nn.modules.batchnorm.BatchNorm1d | torch.nn.modules.batchnorm.BatchNorm2d | torch.nn.modules.batchnorm.BatchNorm3d])
################################################################################

def monai_networks_layers_factories_batch_factory(dim: int):
    """monai.networks.layers.factories.batch_factory returns the PyTorch BatchNorm class corresponding to a requested spatial dimensionality (1D, 2D, or 3D). This factory function is used within MONAI's network-building utilities to select the appropriate torch.nn.BatchNorm layer class when constructing normalization layers for medical imaging models (for example, 2D for X-ray or slices and 3D for volumetric CT/MRI data). The function is decorated with Norm.factory_function("batch"), registering it under the "batch" normalization factory name used by MONAI.
    
    Args:
        dim (int): Desired spatial dimension of the batch normalization layer. Valid, intended values are 1, 2, or 3, which select torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, and torch.nn.BatchNorm3d respectively. This parameter is required and has no default. If a non-integer value is provided, a TypeError will be raised by the indexing operation; if an integer outside the intended set {1, 2, 3} is provided, behavior is undefined relative to the MONAI expectation: negative or zero integers may index the internal tuple via Python's negative indexing semantics (leading to an unexpected but valid BatchNorm class), while integers greater than 3 will raise an IndexError. Callers should therefore pass 1, 2, or 3 to obtain the intended class.
    
    Returns:
        type[nn.BatchNorm1d | nn.BatchNorm2d | nn.BatchNorm3d]: The torch.nn.BatchNorm class object corresponding to the requested dimension. This is a class (not an instantiated layer); to create a layer instance for use in a model, call the returned class with the required parameters (for example, returned_class(num_features)). There are no other side effects from this function itself beyond returning the class and its registration via the factory decorator.
    """
    from monai.networks.layers.factories import batch_factory
    return batch_factory(dim)


################################################################################
# Source: monai.networks.layers.factories.constant_pad_factory
# File: monai/networks/layers/factories.py
# Category: fix_docstring
# Reason: Schema parsing failed: ("Couldn't parse this type hint, likely due to a custom class or object: ", type[torch.nn.modules.padding.ConstantPad1d | torch.nn.modules.padding.ConstantPad2d | torch.nn.modules.padding.ConstantPad3d])
################################################################################

def monai_networks_layers_factories_constant_pad_factory(dim: int):
    """monai.networks.layers.factories.constant_pad_factory: Factory that returns the PyTorch nn.ConstantPad class for a specified spatial dimensionality (1, 2, or 3).
    
    This factory is used within MONAI preprocessing and network construction to select the appropriate constant-padding layer type for medical imaging data of different spatial dimensionalities. In the MONAI domain, 1D is typically used for sequential signals, 2D for slice-based images (e.g., X-rays or individual MRI/CT slices), and 3D for volumetric medical scans (e.g., full CT or MRI volumes). The function performs no tensor allocation or layer instantiation itself; it returns the class object (for example, nn.ConstantPad3d) so callers can instantiate a layer with specific padding and fill value (e.g., nn.ConstantPad3d(padding, value)) according to the PyTorch API.
    
    Args:
        dim (int): desired dimension of the constant padding layer. Must be 1, 2, or 3. The integer selects which PyTorch class is returned: 1 -> nn.ConstantPad1d, 2 -> nn.ConstantPad2d, 3 -> nn.ConstantPad3d. This parameter controls the spatial dimensionality of padding applied in downstream MONAI pipelines and network layers.
    
    Returns:
        type[nn.ConstantPad1d | nn.ConstantPad2d | nn.ConstantPad3d]: The selected PyTorch constant padding class corresponding to the requested dimensionality. This return value is a class (not an instantiated layer); to create a usable layer, instantiate it with the padding specification and constant value as defined by PyTorch (for example, returned_class(padding, value)).
    
    Raises:
        IndexError: if dim is not in the range 1..3, because the implementation indexes a fixed tuple of three classes.
        TypeError: if dim is not an integer type compatible with indexing (the function signature expects an int).
    """
    from monai.networks.layers.factories import constant_pad_factory
    return constant_pad_factory(dim)


################################################################################
# Source: monai.networks.layers.factories.conv_factory
# File: monai/networks/layers/factories.py
# Category: fix_docstring
# Reason: Schema parsing failed: ("Couldn't parse this type hint, likely due to a custom class or object: ", type[torch.nn.modules.conv.Conv1d | torch.nn.modules.conv.Conv2d | torch.nn.modules.conv.Conv3d])
################################################################################

def monai_networks_layers_factories_conv_factory(dim: int):
    """monai.networks.layers.factories.conv_factory: Return the PyTorch convolution module class appropriate for a specified spatial dimensionality used in MONAI medical imaging networks.
    
    This factory function maps an integer spatial dimensionality used in medical imaging deep learning workflows to the corresponding torch.nn convolution class. In MONAI, selecting the correct convolution dimensionality is essential when designing networks for different data modalities: 1 for temporal or 1D signal data, 2 for 2D image slices (e.g., individual radiology slices), and 3 for volumetric image data (e.g., CT or MRI volumes). The function is registered as the "conv" factory and is intended to be used when building or configuring network layers so that subsequent layer construction uses the appropriate nn.ConvXd class for the chosen dimension. The function performs no in-place side effects; it only returns a class object.
    
    Args:
        dim (int): Desired spatial dimensionality of the convolutional layer. Must be 1, 2, or 3 to select between torch.nn.Conv1d, torch.nn.Conv2d, and torch.nn.Conv3d respectively. This integer controls which convolution class is returned so callers can instantiate the returned class with the usual convolution constructor arguments (for example, channels and kernel size) appropriate to that dimensionality.
    
    Returns:
        type[nn.Conv1d | nn.Conv2d | nn.Conv3d]: The torch.nn convolution class corresponding to the requested dimensionality. Specifically, a call with dim == 1 returns nn.Conv1d, dim == 2 returns nn.Conv2d, and dim == 3 returns nn.Conv3d. The returned value is a class object (not an instance); callers must instantiate it to create a layer instance.
    
    Raises:
        IndexError: If dim is not in the set {1, 2, 3}, the internal lookup types[dim - 1] is out of range and an IndexError will be raised. This is the expected failure mode for unsupported dimensionalities.
        TypeError: If dim is not an integer type that can be used to index the internal tuple (for example, a float or None), a TypeError may be raised when used as an index.
    """
    from monai.networks.layers.factories import conv_factory
    return conv_factory(dim)


################################################################################
# Source: monai.networks.layers.factories.convtrans_factory
# File: monai/networks/layers/factories.py
# Category: fix_docstring
# Reason: Schema parsing failed: ("Couldn't parse this type hint, likely due to a custom class or object: ", type[torch.nn.modules.conv.ConvTranspose1d | torch.nn.modules.conv.ConvTranspose2d | torch.nn.modules.conv.ConvTranspose3d])
################################################################################

def monai_networks_layers_factories_convtrans_factory(dim: int):
    """Factory function in monai.networks.layers.factories that selects and returns the appropriate PyTorch transposed convolution class for a given spatial dimensionality. In the MONAI framework (a PyTorch-based library for deep learning in healthcare imaging), this factory is used when constructing network components that require learnable upsampling or decoder layers (for example, the decoder path of a segmentation model). The function returns the class (not an instance) corresponding to nn.ConvTranspose1d, nn.ConvTranspose2d, or nn.ConvTranspose3d so callers can instantiate the layer with desired channel counts, kernel sizes, strides, padding, and other convolutional parameters.
    
    Args:
        dim (int): Desired spatial dimensionality of the transposed convolutional layer. Valid, documented values are 1, 2, or 3, corresponding respectively to the PyTorch classes nn.ConvTranspose1d, nn.ConvTranspose2d, and nn.ConvTranspose3d. The integer expresses the number of spatial dimensions in the medical imaging data being processed (e.g., 2 for standard X/Y image slices, 3 for volumetric scans). This parameter is required and has no default. The function expects an integer; providing a value outside the set {1, 2, 3} will cause an indexing failure when selecting the class (see Failure modes). The function does not validate convolution hyperparameters; it only selects the appropriate class type.
    
    Returns:
        type[nn.ConvTranspose1d | nn.ConvTranspose2d | nn.ConvTranspose3d]: The PyTorch ConvTranspose class matching the requested dimensionality. The returned value is a class object (subclass of torch.nn.Module), not an instantiated layer. Callers should instantiate it with the usual nn.ConvTranspose arguments (for example, ConvClass(in_channels, out_channels, kernel_size, stride, padding, ...)) to obtain a runnable module to include in a MONAI network.
    
    Behavior, side effects, and failure modes:
        The function performs a pure selection operation with no side effects: it constructs a tuple of three PyTorch ConvTranspose classes and returns the element indexed by (dim - 1). Because of this direct indexing, passing dim values outside the valid range 1..3 will raise an IndexError. Passing a non-integer value that is not implicitly convertible to an integer may raise a TypeError or produce unexpected behavior; callers should pass an explicit int. This factory does not instantiate layers, allocate parameters, or move objects to devices; those actions occur when the caller instantiates the returned class and manages device/dtype. This function is intended to be used when programmatically assembling MONAI network architectures that must adapt to 1D, 2D, or 3D medical imaging data.
    """
    from monai.networks.layers.factories import convtrans_factory
    return convtrans_factory(dim)


################################################################################
# Source: monai.networks.layers.factories.dropout_factory
# File: monai/networks/layers/factories.py
# Category: fix_docstring
# Reason: Schema parsing failed: ("Couldn't parse this type hint, likely due to a custom class or object: ", type[torch.nn.modules.dropout.Dropout | torch.nn.modules.dropout.Dropout2d | torch.nn.modules.dropout.Dropout3d])
################################################################################

def monai_networks_layers_factories_dropout_factory(dim: int):
    """monai.networks.layers.factories.dropout_factory returns the PyTorch dropout layer class corresponding to a specified spatial dimensionality used in MONAI network construction for medical imaging models. In MONAI this factory is registered via the Dropout.factory_function decorator under the name "dropout" and is used to select the correct dropout class when building or configuring networks for 1D, 2D, or 3D imaging tasks (for example, time-series, slice-based, or volumetric medical image models) so that callers can instantiate the appropriate nn.Dropout, nn.Dropout2d, or nn.Dropout3d layer.
    
    Args:
        dim (int): Desired spatial dimension index for the dropout layer. This argument selects which PyTorch dropout class to return: 1 selects nn.Dropout (standard, element-wise dropout suitable for generic 1D features), 2 selects nn.Dropout2d (channel-wise/spatial dropout typically used for 2D image feature maps), and 3 selects nn.Dropout3d (channel-wise/spatial dropout for 3D volumetric feature maps). The parameter must be an integer; the function uses straightforward tuple indexing (types[dim - 1]) to perform the selection.
    
    Returns:
        type[nn.Dropout | nn.Dropout2d | nn.Dropout3d]: The PyTorch dropout class corresponding to the requested dimensionality. This return value is the class object (not an instantiated layer); callers should construct an instance by calling the returned class with appropriate arguments (for example, dropout probability). No in-place modification or side effects occur inside this factory itself beyond returning the class.
    
    Behavior and failure modes:
        The implementation maps dim == 1 to nn.Dropout, dim == 2 to nn.Dropout2d, and dim == 3 to nn.Dropout3d. The function relies on Python tuple indexing using dim - 1; therefore, passing integers outside the expected set {1, 2, 3} can produce unintended results due to Python's negative indexing semantics (for example, dim == 0 will select nn.Dropout3d) or raise an IndexError if the computed index is out of range for the internal tuple. Passing a non-integer type for dim will raise a TypeError during indexing. Callers should validate inputs and use dim values 1, 2, or 3 to get predictable, documented behavior suitable for MONAI network layer construction.
    """
    from monai.networks.layers.factories import dropout_factory
    return dropout_factory(dim)


################################################################################
# Source: monai.networks.layers.factories.instance_factory
# File: monai/networks/layers/factories.py
# Category: fix_docstring
# Reason: Schema parsing failed: ("Couldn't parse this type hint, likely due to a custom class or object: ", type[torch.nn.modules.instancenorm.InstanceNorm1d | torch.nn.modules.instancenorm.InstanceNorm2d | torch.nn.modules.instancenorm.InstanceNorm3d])
################################################################################

def monai_networks_layers_factories_instance_factory(dim: int):
    """monai.networks.layers.factories.instance_factory selects and returns the appropriate PyTorch Instance Normalization layer class for a specified spatial dimensionality used in MONAI medical-imaging networks.
    
    This factory function is registered via Norm.factory_function("instance") and is intended for use in MONAI model construction and layer factory patterns where the normalization layer class must be chosen based on the spatial dimension of the data (e.g., 1D biomedical signals, 2D image slices, or 3D volumetric scans). The function does not create an instance of the normalization layer; it returns the corresponding nn.InstanceNormXd class so the caller can instantiate it with the required arguments (for example, num_features, affine, track_running_stats) according to PyTorch's API. The mapping is: dim == 1 -> torch.nn.InstanceNorm1d, dim == 2 -> torch.nn.InstanceNorm2d, dim == 3 -> torch.nn.InstanceNorm3d.
    
    Args:
        dim (int): Desired spatial dimension for the instance normalization layer. This integer must be 1, 2, or 3, corresponding respectively to InstanceNorm1d, InstanceNorm2d, and InstanceNorm3d. The value determines which PyTorch nn.InstanceNormXd class is returned and therefore which normalization semantics are applied when processing 1D signals, 2D medical image slices, or 3D medical image volumes in MONAI pipelines.
    
    Returns:
        type[nn.InstanceNorm1d | nn.InstanceNorm2d | nn.InstanceNorm3d]: The PyTorch nn.InstanceNorm class corresponding to the requested spatial dimensionality. This return value is a class object (not an instantiated layer); callers must call it (for example, ReturnedClass(num_features, affine=True)) to create an nn.Module suitable for insertion into a MONAI network.
    
    Behavior and side effects:
        The function performs a simple lookup from an internal tuple of (nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d) using index dim - 1. There are no external side effects or global state changes beyond the factory registration performed by the decorator at definition time. The function is lightweight and intended to be used in model construction code where the correct normalization class must be selected programmatically.
    
    Failure modes:
        If dim is not one of the integers 1, 2, or 3, the index operation will raise an IndexError. If dim is not of integer type, Python will raise a TypeError when attempting to use it as a sequence index. Callers should validate or ensure dim is an int within {1, 2, 3} before calling this function to avoid these exceptions.
    """
    from monai.networks.layers.factories import instance_factory
    return instance_factory(dim)


################################################################################
# Source: monai.networks.layers.factories.instance_nvfuser_factory
# File: monai/networks/layers/factories.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def monai_networks_layers_factories_instance_nvfuser_factory(dim: int):
    """monai.networks.layers.factories.instance_nvfuser_factory returns a normalization layer class optimized for 3D instance normalization when available (apex.normalization.InstanceNorm3dNVFuser) and falls back to the appropriate torch.nn.InstanceNorm{1,2,3} classes for other dimensionalities or when the NVFuser implementation is not available. This factory is used in MONAI to select an efficient InstanceNorm layer for medical imaging deep learning pipelines, where 3D volumetric data (e.g., CT or MRI) often benefits from a specialized, CUDA-accelerated implementation.
    
    This function examines the requested spatial dimensionality and either:
    - returns the NVIDIA APEX NVFuser accelerated class apex.normalization.InstanceNorm3dNVFuser when dim == 3 and the NVFuser implementation is installed and importable; or
    - returns the corresponding torch.nn.InstanceNorm1d or torch.nn.InstanceNorm2d class when dim is 1 or 2; or
    - returns torch.nn.InstanceNorm3d when dim == 3 but the NVFuser implementation is not installed or not importable.
    
    Behavioral notes, practical significance, and side effects:
    This factory returns the layer class itself (not an instantiated layer). Callers must instantiate the returned class with appropriate constructor arguments (for example, affine and track_running_stats) consistent with torch.nn.InstanceNorm* semantics. The NVFuser implementation (apex.normalization.InstanceNorm3dNVFuser) is a customized autograd implementation provided by NVIDIA APEX that can be faster on CUDA for 3D volumes; it requires a CUDA-enabled environment and is not supported on Windows. Because the NVFuser variant uses custom autograd logic, it is currently not compatible with TorchScript; if TorchScript compatibility is required, use torch.nn.InstanceNorm3d instead.
    
    When the factory chooses a non-NVFuser fallback, it issues a Python warning via warnings.warn to inform the user about the fallback. If dim != 3 and dim is in {1, 2}, a warning indicates which torch.nn.InstanceNorm class will be used. If NVFuser is not installed or not importable, a warning indicates that torch.nn.InstanceNorm3d will be used instead. The function uses optional_import to import the NVFuser class; the returned value is the first element of optional_import(...)[0], i.e., the class object.
    
    Failure modes and limits:
    The function assumes dim is an integer representing spatial dimensionality. It is designed for dim values 1, 2, or 3. If dim is outside the range 1..3, the function will attempt to index an internal tuple and will raise an IndexError; callers should validate dim before calling if there is any chance it is outside this range. The NVFuser path requires that apex.normalization.InstanceNorm3dNVFuser be installed and importable; otherwise the function falls back to torch.nn.InstanceNorm3d. The NVFuser implementation is not TorchScript compatible and requires CUDA on a non-Windows OS; attempting to use it in a CPU-only environment, on Windows, or with TorchScript will fail or produce unsupported behavior.
    
    Installation reference:
    If you intend to use the NVFuser implementation, install NVIDIA APEX per its repository instructions: https://github.com/NVIDIA/apex#installation
    
    Args:
        dim (int): Spatial dimensionality requested for the InstanceNorm layer. This integer selects which normalization class to return: 1 returns torch.nn.InstanceNorm1d, 2 returns torch.nn.InstanceNorm2d, and 3 attempts to return the faster apex.normalization.InstanceNorm3dNVFuser class when available; if the NVFuser class is not available, torch.nn.InstanceNorm3d is returned. The value must be 1, 2, or 3; values outside this set will cause an IndexError due to internal indexing.
    
    Returns:
        type: A class object implementing instance normalization for the requested dimensionality. Possible returned classes are torch.nn.InstanceNorm1d, torch.nn.InstanceNorm2d, torch.nn.InstanceNorm3d, or apex.normalization.InstanceNorm3dNVFuser (the latter only when dim == 3 and the NVFuser implementation is installed and importable). The caller should instantiate the returned class to create a layer instance for use in MONAI model definitions.
    """
    from monai.networks.layers.factories import instance_nvfuser_factory
    return instance_nvfuser_factory(dim)


################################################################################
# Source: monai.networks.layers.factories.maxpooling_factory
# File: monai/networks/layers/factories.py
# Category: fix_docstring
# Reason: Schema parsing failed: ("Couldn't parse this type hint, likely due to a custom class or object: ", type[torch.nn.modules.pooling.MaxPool1d | torch.nn.modules.pooling.MaxPool2d | torch.nn.modules.pooling.MaxPool3d])
################################################################################

def monai_networks_layers_factories_maxpooling_factory(dim: int):
    """Max pooling layer class factory for 1D, 2D, or 3D spatial data used in MONAI network construction.
    
    Returns the PyTorch max pooling layer class corresponding to the requested spatial dimensionality so callers within MONAI (a PyTorch-based medical imaging deep learning framework) can instantiate pooling layers appropriate for their network architectures. In the medical imaging domain, max pooling layers reduce spatial resolution and help build hierarchical feature representations; this factory centralizes the selection of the correct torch.nn MaxPool class (imported as nn in this module) based on a simple integer dimension argument.
    
    Args:
        dim (int): Desired spatial dimension of the max pooling layer class. This argument selects which torch.nn MaxPool class is returned: 1 selects nn.MaxPool1d, 2 selects nn.MaxPool2d, and 3 selects nn.MaxPool3d. The value must be an integer corresponding to the supported pooling dimensionalities used in MONAI models for 1D, 2D, or 3D medical imaging data. Passing a value outside the range 1..3 will result in an IndexError at runtime; passing a non-integer value may raise a TypeError when used for indexing.
    
    Returns:
        type[nn.MaxPool1d | nn.MaxPool2d | nn.MaxPool3d]: The PyTorch MaxPool class for the requested dimensionality (not an instantiated layer). Specifically, this function returns the class object nn.MaxPool1d for dim == 1, nn.MaxPool2d for dim == 2, and nn.MaxPool3d for dim == 3. Callers should instantiate the returned class with appropriate arguments (for example, kernel_size, stride, padding) when constructing network layers. There are no side effects; the function does not modify global state.
    """
    from monai.networks.layers.factories import maxpooling_factory
    return maxpooling_factory(dim)


################################################################################
# Source: monai.networks.layers.factories.replication_pad_factory
# File: monai/networks/layers/factories.py
# Category: fix_docstring
# Reason: Schema parsing failed: ("Couldn't parse this type hint, likely due to a custom class or object: ", type[torch.nn.modules.padding.ReplicationPad1d | torch.nn.modules.padding.ReplicationPad2d | torch.nn.modules.padding.ReplicationPad3d])
################################################################################

def monai_networks_layers_factories_replication_pad_factory(dim: int):
    """Replication padding layer class selector for 1D, 2D, and 3D spatial tensors used by MONAI's layer factory.
    
    This function is used in MONAI (Medical Open Network for AI) to map a requested spatial dimensionality to the corresponding PyTorch replication padding layer class when the "replicationpad" pad type is requested from the Pad factory. In medical imaging workflows within MONAI, replication padding is commonly applied to multi-dimensional image tensors to extend boundaries by copying edge values; this helper returns the appropriate nn.ReplicationPad class so the caller or factory can instantiate a padding layer with concrete padding sizes for preprocessing or network layers.
    
    Args:
        dim (int): desired spatial dimensionality for the replication padding layer. Valid values are 1, 2, or 3 corresponding to nn.ReplicationPad1d, nn.ReplicationPad2d, and nn.ReplicationPad3d respectively. This parameter has no default and must be provided. If dim is outside the supported set (for example less than 1 or greater than 3), the function will raise an IndexError from the internal tuple lookup; if dim is not an integer, a TypeError or similar will occur when used as an index. The caller (or the MONAI Pad factory) is responsible for supplying a correct integer dimension as part of constructing padding layers for multi-dimensional medical image tensors.
    
    Returns:
        type[nn.ReplicationPad1d | nn.ReplicationPad2d | nn.ReplicationPad3d]: the PyTorch replication padding layer class corresponding to the requested spatial dimension. This function returns the class object (not an instantiated module); to create a usable padding layer, the caller should instantiate the returned class with the desired padding size(s), e.g., nn.ReplicationPad2d(padding). There are no other side effects beyond returning the class and registering this factory function via the Pad.factory_function decorator in the MONAI factory system.
    """
    from monai.networks.layers.factories import replication_pad_factory
    return replication_pad_factory(dim)


################################################################################
# Source: monai.networks.layers.factories.split_args
# File: monai/networks/layers/factories.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def monai_networks_layers_factories_split_args(args: tuple):
    """monai.networks.layers.factories.split_args normalizes an argument specification into a (type, kwargs) pair suitable for MONAI factory-style layer construction utilities.
    
    This function is part of MONAI's layer factory utilities used by network/layer factories to accept flexible type specifications from configuration or code. It accepts either a single string naming a layer/type (for example a key used with monai.networks.layers.Act) or an explicit pair consisting of a name or callable that identifies the object to instantiate and a dict of keyword arguments to pass to that object's constructor. The normalized output is intended for direct use with factory mappings that construct PyTorch/MONAI layer objects from a type-specifier and keyword arguments.
    
    Args:
        args (str or tuple): input arguments to be parsed. This must be either:
            - a string that names the desired type (for example "PRELU" as used by monai.networks.layers.Act), in which case the function returns that string and an empty dict of kwargs; or
            - a two-element tuple (name_obj, name_args) where name_obj is either a string naming the object type or a callable (for example a class or factory function) and name_args is a dict of keyword arguments to pass when constructing the object. The tuple form is used to provide explicit constructor parameters (for example ("PRELU", {"num_parameters": 1, "init": 0.25}) when instantiating an activation with specific settings).
    
    Returns:
        tuple: A pair (name_obj, name_args) where name_obj is the original string or callable identifying the object type and name_args is a dict of keyword arguments to pass to the object's constructor. If the input was a single string, name_args will be an empty dict. The returned pair is intended to be passed directly to factory lookup and construction code (for example monai.networks.layers.Act[name_obj](**name_args)).
    
    Raises:
        TypeError: If args is not a string and not a two-element tuple where the first element is either a string or a callable and the second element is a dict. The error message explains that layer specifiers must be single strings or pairs of the form (name/object-types, argument dict). This function performs no other side effects; it only validates and normalizes the input for downstream factory-based construction.
    """
    from monai.networks.layers.factories import split_args
    return split_args(args)


################################################################################
# Source: monai.networks.layers.weight_init.trunc_normal_
# File: monai/networks/layers/weight_init.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def monai_networks_layers_weight_init_trunc_normal_(
    tensor: torch.Tensor,
    mean: float = 0.0,
    std: float = 1.0,
    a: float = -2.0,
    b: float = 2.0
):
    """Initialize the given tensor in-place with samples from a truncated normal distribution.
    
    This function is used within MONAI for network weight initialization in medical imaging models (for example, initializing convolution and linear layer weights before training). It fills the provided n-dimensional torch.Tensor with random values drawn from a normal distribution with specified mean and standard deviation, but truncated to lie within the interval [a, b]. The implementation is based on the pytorch-image-models truncated normal routine (https://github.com/rwightman/pytorch-image-models) and performs the assignment under torch.no_grad to avoid tracking these operations in autograd. Typical usage is to call this once after creating parameter tensors to produce stable initial weights (defaults: mean=0.0, std=1.0, a=-2.0, b=2.0).
    
    Args:
        tensor (torch.Tensor): The n-dimensional tensor to be initialized. This tensor is modified in-place: its existing values are replaced by random samples from the truncated normal distribution. The function returns the same tensor object for convenience.
        mean (float): The mean (mu) of the underlying normal distribution from which samples are drawn before truncation. Default is 0.0. In the MONAI context, choosing an appropriate mean shifts the center of initialized weights for layers in imaging networks.
        std (float): The standard deviation (sigma) of the underlying normal distribution. Default is 1.0. Must be greater than zero; if std <= 0 a ValueError is raised because a non-positive standard deviation is invalid for a normal distribution.
        a (float): The minimum cutoff value for truncation. Default is -2.0. Values sampled from the normal distribution that fall below this cutoff are not retained; samples are effectively drawn so the final values lie within [a, b]. In practice, this prevents extreme outliers in initialized weights which can destabilize training.
        b (float): The maximum cutoff value for truncation. Default is 2.0. Must satisfy a < b; if a >= b a ValueError is raised. The interval [a, b] defines the inclusive bounds used for truncation.
    
    Returns:
        torch.Tensor: The same tensor object passed in via the tensor argument, after being filled in-place with values from the truncated normal distribution. Side effects: the input tensor is modified in-place, the operation is performed under torch.no_grad so it will not be part of autograd graph construction, and exceptions are raised for invalid std or invalid cutoff interval as described above.
    """
    from monai.networks.layers.weight_init import trunc_normal_
    return trunc_normal_(tensor, mean, std, a, b)


################################################################################
# Source: monai.networks.nets.efficientnet.get_efficientnet_image_size
# File: monai/networks/nets/efficientnet.py
# Category: valid
################################################################################

def monai_networks_nets_efficientnet_get_efficientnet_image_size(model_name: str):
    """monai.networks.nets.efficientnet.get_efficientnet_image_size returns the required input image spatial size (single dimension) for a specified EfficientNet model variant used in MONAI's imaging networks.
    
    Args:
        model_name (str): Name of the EfficientNet variant to query. This must match a key in the module-level efficientnet_params mapping used by the EfficientNet network constructors (for example "efficientnet-b0", ..., "efficientnet-b7"). In the MONAI medical-imaging context this string is used when selecting a pretrained or custom EfficientNet backbone for 2D image classification or encoding tasks so that input images can be resized or cropped to the correct resolution expected by the model.
    
    Returns:
        int: The image size for a single spatial dimension (height or width) required by the specified EfficientNet model. EfficientNet models in this implementation expect square inputs, so this single integer is the size for each spatial dimension. This is typically used to configure preprocessing transforms (resize, center crop) and the network input layer.
    
    Raises:
        ValueError: If model_name is not present in the module-level efficientnet_params mapping, a ValueError is raised. The error message lists the valid model names; callers should validate or catch this exception when model names may come from external configuration or user input.
    
    Notes:
        The function performs a dictionary lookup in efficientnet_params and extracts the resolution value from the parameter tuple (the third element in that tuple as used by the EfficientNet builder). There are no side effects. The returned type is exactly int as produced by the mapping; no conversion to other types is performed.
    """
    from monai.networks.nets.efficientnet import get_efficientnet_image_size
    return get_efficientnet_image_size(model_name)


################################################################################
# Source: monai.networks.nets.mednext.create_mednext
# File: monai/networks/nets/mednext.py
# Category: valid
################################################################################

def monai_networks_nets_mednext_create_mednext(
    variant: str,
    spatial_dims: int = 3,
    in_channels: int = 1,
    out_channels: int = 2,
    kernel_size: int = 3,
    deep_supervision: bool = False
):
    """Create a configured MedNeXt model instance for medical imaging tasks by selecting one of the predefined model-size variants. This factory constructs a MedNeXt encoder-decoder network (used in MONAI for deep learning on healthcare imaging data) with variant-specific expansion ratios and block counts optimized for different model capacities, and with shared common settings (residual connections enabled, group normalization, no global response normalization, and 32 initial filters). The created MedNeXt is intended for use in segmentation or classification pipelines that operate on multi-dimensional medical images (e.g., 2D or 3D volumes) and integrates with MONAI training and inference workflows.
    
    Args:
        variant (str): The MedNeXt variant to create. Must be one of 'S', 'B', 'M', or 'L' (case-insensitive). Each letter selects a predefined architecture size and complexity: 'S' (small) constructs a lightweight model with encoder_expansion_ratio=2, decoder_expansion_ratio=2, bottleneck_expansion_ratio=2, blocks_down=(2, 2, 2, 2), blocks_bottleneck=2, blocks_up=(2, 2, 2, 2); 'B' (base) constructs a medium model with encoder_expansion_ratio=(2, 3, 4, 4), decoder_expansion_ratio=(4, 4, 3, 2), bottleneck_expansion_ratio=4, blocks_down=(2, 2, 2, 2), blocks_bottleneck=2, blocks_up=(2, 2, 2, 2); 'M' (medium) increases depth and capacity with encoder_expansion_ratio=(2, 3, 4, 4), decoder_expansion_ratio=(4, 4, 3, 2), bottleneck_expansion_ratio=4, blocks_down=(3, 4, 4, 4), blocks_bottleneck=4, blocks_up=(4, 4, 4, 3); 'L' (large) constructs the highest-capacity model with encoder_expansion_ratio=(3, 4, 8, 8), decoder_expansion_ratio=(8, 8, 4, 3), bottleneck_expansion_ratio=8, blocks_down=(3, 4, 8, 8), blocks_bottleneck=8, blocks_up=(8, 8, 4, 3). Choosing a larger variant increases model parameters and representational capacity, which can improve accuracy on complex medical imaging tasks at the cost of greater memory and compute.
        spatial_dims (int): Number of spatial dimensions for the network convolutions and tensor operations. Defaults to 3. In practice, set this to 2 for 2D medical images (e.g., X-ray or slice-based tasks) or 3 for volumetric data (e.g., CT, MRI), and MONAI will configure convolutional and normalization layers accordingly.
        in_channels (int): Number of input channels in the input tensor. Defaults to 1. For example, set to 1 for single-channel grayscale volumes (typical in many medical imaging modalities) or to higher values when the input contains multi-channel data (e.g., multi-contrast MRI).
        out_channels (int): Number of output channels produced by the network's final layer. Defaults to 2. In segmentation workflows, out_channels commonly equals the number of segmentation labels or classes; in other tasks it represents the dimensionality of the model output expected by downstream loss/metric computations.
        kernel_size (int): Kernel size for convolutions used throughout the network. Defaults to 3. This integer determines the spatial support of convolutional filters and therefore influences the receptive field and local context captured by the model.
        deep_supervision (bool): Whether to enable deep supervision (intermediate auxiliary outputs during training). Defaults to False. When True, the model exposes intermediate outputs from decoder stages that can be used to compute auxiliary losses during training to improve gradient flow and training stability for deep architectures; when False, only the final output is produced.
    
    Returns:
        MedNeXt: A MedNeXt instance configured according to the requested variant and the provided arguments. The returned object is a PyTorch/ MONAI-compatible network configured for medical imaging tasks, with the following common settings applied by default: use_residual_connection=True, norm_type='group', global_resp_norm=False, init_filters=32. The constructed model can be used directly in MONAI training loops, evaluation, and exported/saved via standard PyTorch utilities.
    
    Raises:
        ValueError: If variant is not one of 'S', 'B', 'M', or 'L' (case-insensitive). The function validates the variant string and raises this error to prevent creating an undefined architecture.
    """
    from monai.networks.nets.mednext import create_mednext
    return create_mednext(
        variant,
        spatial_dims,
        in_channels,
        out_channels,
        kernel_size,
        deep_supervision
    )


################################################################################
# Source: monai.networks.nets.resnet.get_medicalnet_pretrained_resnet_args
# File: monai/networks/nets/resnet.py
# Category: fix_docstring
# Reason: Schema parsing failed: Cannot generate JSON schema for get_medicalnet_pretrained_resnet_args because the docstring has no description for the argument 'resnet_depth'
################################################################################

def monai_networks_nets_resnet_get_medicalnet_pretrained_resnet_args(resnet_depth: int):
    """get_medicalnet_pretrained_resnet_args: Return the configuration tuple (bias_downsample, shortcut_type) that matches the MedicalNet pretrained ResNet weight conventions for a given ResNet depth.
    
    This function is used in MONAI (a PyTorch-based medical imaging deep learning framework) when constructing ResNet variants that will load pretrained weights from MedicalNet/model zoo. It encodes the known conventions for whether the downsample convolution uses a bias term and which residual shortcut variant ("A" or "B") was used when the MedicalNet weights were produced. Using the values returned by this function ensures the model architecture matches the weight parameter shapes and naming expected by the pretrained checkpoint, avoiding mismatches when loading weights for medical imaging tasks (classification, segmentation, or feature extraction).
    
    Args:
        resnet_depth (int): The ResNet depth identifier (for example 18, 34, 50, 101, 152, 200) that specifies which ResNet variant is being constructed. This integer is used to select the MedicalNet weight convention: ResNet-18 and ResNet-34 use the alternative shortcut/type and bias convention encoded by this function, while other common depths use the default convention. The parameter is expected to be an integer corresponding to the network depth; passing a value of a different type may lead to unexpected membership behavior or TypeError if the object is unhashable.
    
    Returns:
        tuple: A 2-tuple matching the MedicalNet pretrained ResNet conventions.
            bias_downsample (bool): Indicates whether the downsample convolution layers in the ResNet residual blocks should include a bias term when building the model to match MedicalNet pretrained weights. For ResNet depths 18 and 34 this is True; for other tested depths (for example 10, 50, 101, 152, 200) this is False. This flag is used when instantiating downsample convolutions so that parameter shapes and counts align with the pretrained checkpoint.
            shortcut_type (str): A string value "A" or "B" naming the shortcut/projection variant used in the original MedicalNet-trained ResNet for the given depth. The value "A" is returned for depths 18 and 34 (alternate shortcut style), and "B" is returned for other depths. Matching this value ensures the residual connection implementation is compatible with the pretrained weights.
    
    Behavior and failure modes:
        This function performs a pure computation with no side effects. It determines values by checking whether resnet_depth is in the set {18, 34}. It does not raise exceptions for integer inputs; however, providing a non-integer or an unhashable object may produce unexpected results or a TypeError during the membership check. The mapping implemented here reflects conventions observed and tested for MedicalNet pretrained weights; if upstream MedicalNet weight conventions change for new ResNet depths, callers should verify and update usage accordingly.
    """
    from monai.networks.nets.resnet import get_medicalnet_pretrained_resnet_args
    return get_medicalnet_pretrained_resnet_args(resnet_depth)


################################################################################
# Source: monai.networks.nets.resnet.get_pretrained_resnet_medicalnet
# File: monai/networks/nets/resnet.py
# Category: valid
################################################################################

def monai_networks_nets_resnet_get_pretrained_resnet_medicalnet(
    resnet_depth: int,
    device: str = "cpu",
    datasets23: bool = True
):
    """monai.networks.nets.resnet.get_pretrained_resnet_medicalnet downloads and returns pretrained ResNet weights from the TencentMedicalNet repository on the Hugging Face Hub that are intended for medical imaging tasks and for use with MONAI ResNet model implementations.
    
    Args:
        resnet_depth (int): depth of the pretrained ResNet model to fetch. Supported integer values are 10, 18, 34, 50, 101, 152 and 200. This parameter selects which variant of ResNet was trained in the MedicalNet releases; the choice affects the architecture for which the returned state dictionary (model parameters) is compatible.
        device (str): device specifier used as the map_location argument for torch.load when loading the downloaded checkpoint into memory. Typical values are "cpu" or "cuda". This controls which device the tensors in the loaded checkpoint are moved to and is important when subsequently loading the returned state dictionary into a model on the same device.
        datasets23 (bool): if True, request weights that were trained on a larger collection of medical imaging datasets (23 datasets) when available. Not all resnet_depth values have a corresponding "_23dataset" file; if such a file is not available for the requested depth, the function will automatically try to download the standard (non-23dataset) weights for that depth. Defaults to True.
    
    Returns:
        dict: a pretrained state dictionary extracted from the downloaded checkpoint (checkpoint.get("state_dict")). This dictionary contains parameter name keys and tensor values suitable for loading into a compatible PyTorch/ MONAI ResNet model via model.load_state_dict(state_dict). The state dict is returned after torch.load(..., map_location=torch.device(device), weights_only=True) has loaded the checkpoint into memory.
    
    Raises:
        huggingface_hub.utils._errors.EntryNotFoundError: if the requested pretrained file is not found on the Hugging Face Hub for the constructed repository/filename pair. This occurs when neither the datasets23-specific filename nor the standard filename exists for the requested resnet_depth in the TencentMedicalNet repository on the hub.
        NotImplementedError: if resnet_depth is not one of the supported values [10, 18, 34, 50, 101, 152, 200]. The function validates this input and raises this error for unsupported depths.
    
    Behavior and side effects:
        The function constructs a repository id by concatenating the MedicalNet base name ("TencentMedicalNet/MedicalNet-Resnet") with the numeric resnet_depth (for example, "TencentMedicalNet/MedicalNet-Resnet34") and a filename based on the depth and the datasets23 flag (for example, "resnet_34_23dataset.pth" when datasets23 is True). It uses hf_hub_download to download the specified file into the local Hugging Face cache; this performs network I/O and may write to the local cache directory used by the Hugging Face Hub. If datasets23 is True but the corresponding file is not available, the function logs that the file was not available and retries with the standard filename (without the "_23dataset" suffix). After successful download, torch.load is called with map_location set to the provided device and weights_only=True to load only the model weights into memory; logger messages indicate download progress. The function then returns the "state_dict" entry from the loaded checkpoint. Consumers should ensure the returned state dict is compatible with their ResNet model implementation (matching layer names and architecture) before calling model.load_state_dict; mismatches when loading into a model may raise PyTorch errors. Network failures, permission issues, or problems reading the cached file may cause hf_hub_download or torch.load to raise additional exceptions that propagate to the caller.
    """
    from monai.networks.nets.resnet import get_pretrained_resnet_medicalnet
    return get_pretrained_resnet_medicalnet(resnet_depth, device, datasets23)


################################################################################
# Source: monai.networks.nets.swin_unetr.compute_mask
# File: monai/networks/nets/swin_unetr.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def monai_networks_nets_swin_unetr_compute_mask(
    dims: tuple,
    window_size: tuple,
    shift_size: tuple,
    device: str
):
    """Computes an attention region mask for shifted-window self-attention as used by Swin Transformer and the Swin-UNETR model in MONAI.
    
    This function builds a binary-region index map over a 2D or 3D input grid (height/width or depth/height/width) and converts that map into an attention mask that prevents cross-window attention after a cyclic shift. The implementation follows the approach in Liu et al., "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows" and the reference implementation in the Microsoft Swin-Transformer repository. In MONAI this mask is used by swin_unetr network components to ensure attention is computed only within the appropriate local windows for medical image segmentation tasks, and it is intended to be added to attention logits before softmax so masked positions receive a large negative bias (here -100.0) and contribute near-zero probability after softmax.
    
    Args:
        dims (tuple): Spatial dimension sizes of the input image/volume. For 3D volumes this must be a tuple of three integers (depth, height, width). For 2D images this must be a tuple of two integers (height, width). The function branches on len(dims) == 3 or len(dims) == 2; any other length is unsupported and will lead to an exception or undefined behavior.
        window_size (tuple): Local window size used to partition the input into non-overlapping windows before shifting. This must be a tuple of integers whose length matches len(dims). Each element corresponds to the window size along the respective spatial axis. Elements are used directly to create slicing ranges and must be valid integers for the given dims.
        shift_size (tuple): Shift size applied to create the shifted-window attention pattern. This must be a tuple of integers whose length matches len(dims) and corresponds elementwise to window_size. Values are used to compute the three-region slicing used to assign region indices. If shift_size and window_size lengths do not match dims length or contain invalid integers, the function will raise an error or behave incorrectly.
        device (str): Device specification passed to torch.zeros when allocating the intermediate img_mask tensor. This should be a device string accepted by PyTorch (for example 'cpu' or 'cuda') and determines where the returned attention mask tensor is allocated. The returned tensor will reside on the same device.
    
    Returns:
        torch.Tensor: An attention mask tensor suitable for adding to attention logits. The mask contains floating values of 0.0 for positions where attention is allowed (same local window after shifting) and -100.0 for positions that should be blocked. The mask is created on the provided device and is intended to be applied to attention score tensors so that masked positions contribute negligible weight after softmax. The exact shape depends on the window_partition output for the provided dims and window_size and matches the indexing used by the corresponding attention implementation in swin_unetr.
    
    Behavior, side effects, and failure modes:
        - The function allocates one or more intermediate tensors (img_mask and mask_windows) on the specified device; memory usage scales with the spatial dims and window_size and can be significant for large medical images/volumes.
        - The function supports only 2D (len(dims) == 2) and 3D (len(dims) == 3) spatial inputs, following the Swin Transformer pattern. Passing dims of any other length will lead to a NameError or other exception because the code does not handle those cases.
        - window_size and shift_size must be tuples of integers whose lengths equal len(dims). Mismatched lengths or non-integer entries will cause indexing errors or incorrect masks.
        - The mask values use -100.0 to strongly suppress attention after softmax; this value is chosen to be large negative but is hard-coded and not configurable in this function.
        - The function depends on the helper window_partition function; if that helper is unavailable or modified to return a different layout, the resulting attn_mask shape or semantics may change.
        - Intended use is to add this mask to attention logits before softmax in Swin-style shifted-window attention layers (e.g., in Swin-UNETR within MONAI) for medical imaging segmentation and related tasks.
    """
    from monai.networks.nets.swin_unetr import compute_mask
    return compute_mask(dims, window_size, shift_size, device)


################################################################################
# Source: monai.networks.nets.swin_unetr.filter_swinunetr
# File: monai/networks/nets/swin_unetr.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def monai_networks_nets_swin_unetr_filter_swinunetr(key: str, value: torch.Tensor):
    """filter_swinunetr
    Converts and filters parameter keys from a pretrained checkpoint (from the Disruptive Autoencoders / SSL weights referenced in the repository) so they can be loaded into the MONAI SwinUNETR model using utilities such as monai.networks.utils.copy_model_state. This function inspects the incoming state-dict key (string) and either returns a renamed key paired with the original tensor value or returns None to indicate the parameter should be skipped. It is intended to be used when adapting checkpoint key naming conventions (for example, checkpoints that use an "encoder." prefix) to MONAI's SwinUNETR naming convention (which expects a "swinViT." prefix and slightly different internal key layout). This function does not modify the tensor contents, move tensors across devices, or validate tensor shapes; those checks and any loading errors are handled by the caller (e.g., copy_model_state or torch.nn.Module.load_state_dict).
    
    Args:
        key (str): The key name from the source state dictionary. In practice this is a parameter name from a pretrained SSL checkpoint (for example keys that begin with "encoder." or the exact keys listed below). The function uses this string to decide whether to skip the parameter or to produce a renamed key compatible with MONAI SwinUNETR. Passing a non-string will raise a TypeError in typical usage because string operations (slicing and equality) are applied.
        value (torch.Tensor): The tensor value associated with the source state-dict key. This is the weight or bias tensor to be copied into the target model if the key is accepted. The function does not alter this tensor (it returns the same object when returning a (key, value) pair). The function does not check device, dtype, or shape; those compatibility checks are left to the loader that applies the returned mapping.
    
    Behavior and rules:
        - If key exactly matches any of the following strings, the function returns None to signal that those pretrained parameters should be skipped and not loaded into the MONAI SwinUNETR model:
            "encoder.mask_token"
            "encoder.norm.weight"
            "encoder.norm.bias"
            "out.conv.conv.weight"
            "out.conv.conv.bias"
          These entries are typically specific low-level or output parameters in the source checkpoint that do not map cleanly to the target model or should be initialized differently in MONAI workflows.
        - If key begins with the prefix "encoder.", the function returns a tuple (new_key, value) where new_key is constructed by prefixing "swinViT." and then concatenating slices of the original key to adapt the naming convention:
            - If the substring key[8:19] equals "patch_embed", new_key is "swinViT." + key[8:]. This preserves the remainder of the original key after the "encoder." prefix and is used to map patch embedding parameters.
            - Otherwise, new_key is "swinViT." + key[8:18] + key[20:]. This expression takes the characters of the original key from index 8 up to (but not including) 18, skips the characters at indices 18 and 19, and then appends the rest of the key from index 20 onward. This deterministic slicing is the exact transformation implemented to reconcile small naming-layout differences between the source checkpoint and MONAI SwinUNETR internal names.
        - For any key that does not match the above conditions (not one of the explicit skips and not starting with "encoder."), the function returns None to indicate the parameter should be ignored when copying state.
    
    Side effects:
        - No in-place modification of the provided value tensor is performed. The function returns either None or a tuple referencing the original tensor.
        - The caller is responsible for handling device placement, dtype conversion, and shape validation. If a returned (new_key, value) pair is applied to the target model and the tensor shape or dtype does not match the target parameter, the loader (e.g., copy_model_state) will surface an error or report the parameter as not loaded.
    
    Failure modes and notes:
        - Passing a non-string key will cause string operations to fail (TypeError). The function assumes key is a str as used by PyTorch state dictionaries.
        - The function does not guarantee semantic compatibility of parameters beyond key renaming. Even when a (new_key, value) pair is returned, loading may fail later due to mismatched tensor shapes, dtypes, or device placement.
        - The two-character skip performed for the non-patch_embed branch is an exact, index-based transformation kept from the original implementation; changing source checkpoint naming or unexpected key formats may require updating this function.
    
    Returns:
        tuple(str, torch.Tensor) or None: If the key is accepted and renamed for MONAI SwinUNETR, returns a tuple containing the new key name (str) and the original tensor value (torch.Tensor). If the key should be skipped (either because it is in the explicit exclusion list or does not start with "encoder."), returns None to indicate the parameter must not be copied into the target model.
    """
    from monai.networks.nets.swin_unetr import filter_swinunetr
    return filter_swinunetr(key, value)


################################################################################
# Source: monai.networks.nets.swin_unetr.get_window_size
# File: monai/networks/nets/swin_unetr.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def monai_networks_nets_swin_unetr_get_window_size(
    x_size: tuple,
    window_size: tuple,
    shift_size: tuple = None
):
    """Compute adjusted window size (and optional adjusted shift size) for Swin-style windowed attention given an input spatial size.
    
    This function implements the behavior used in the Swin Transformer family (see Liu et al., "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows") and in MONAI's Swin UNETR network: it ensures local window sizes do not exceed the corresponding input spatial dimensions and disables window shifting on any dimension where the window equals or exceeds the input size. The result is suitable for downstream use when partitioning image patches or feature maps into local windows for self-attention in medical imaging models.
    
    Args:
        x_size (tuple): The input spatial size along each dimension. In MONAI usage this is typically the shape of an image patch or intermediate feature map (per-dimension sizes). The function iterates over len(x_size) and uses each element to decide whether to clamp the corresponding window size; therefore x_size must be an indexable tuple with one entry per spatial dimension.
        window_size (tuple): The desired local window size along each dimension before adjustment. This tuple provides the initial per-dimension window sizes used by windowed attention. The function creates an internal copy and reduces any entry where the corresponding x_size dimension is smaller or equal, so window_size is not mutated in place.
        shift_size (tuple, optional): The window shifting size along each dimension, or None to indicate no shifting. If provided, the function creates an internal copy and sets the shift to 0 for any dimension where x_size[i] <= window_size[i] (because shifting is not meaningful when the window covers the whole dimension). If None (the default), no shift values are returned and the function returns only the adjusted window sizes.
    
    Behavior and side effects:
        The function converts the provided window_size (and shift_size, if given) to mutable internal lists, adjusts entries as described, and returns new tuples. It does not modify the input tuple objects passed in by the caller. When shift_size is None the function returns a single tuple; when shift_size is provided it returns a pair of tuples (adjusted_window_size, adjusted_shift_size).
        The returned tuples have one entry per dimension processed (the number of entries equals len(x_size)), and are intended to be used directly for constructing or configuring window partitioning and shifted-window attention in Swin-style models within MONAI.
    
    Failure modes and edge cases:
        If window_size or shift_size (when provided) are shorter than x_size, the function will raise an IndexError because it indexes those tuples up to len(x_size). If any argument is not an indexable tuple-like object, a TypeError or related exception may be raised. The function assumes the tuples represent per-dimension sizes and does not validate element types or ranges beyond performing comparisons and assignments.
    
    Returns:
        tuple: If shift_size is None, returns a single tuple containing the adjusted window sizes (one entry per dimension, matching len(x_size)). These values have been clamped so no window size exceeds the corresponding input dimension, which ensures valid window partitioning for Swin-style attention in MONAI models.
        tuple, tuple: If shift_size is provided, returns a pair (adjusted_window_size, adjusted_shift_size). The first element is the adjusted window sizes tuple as above. The second element is the adjusted shift sizes tuple where any dimension whose window was clamped to the input size has its shift set to 0, disabling shifting for that dimension.
    """
    from monai.networks.nets.swin_unetr import get_window_size
    return get_window_size(x_size, window_size, shift_size)


################################################################################
# Source: monai.networks.nets.swin_unetr.window_partition
# File: monai/networks/nets/swin_unetr.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def monai_networks_nets_swin_unetr_window_partition(x: torch.Tensor, window_size: tuple):
    """window partition operation used by Swin Transformer–based models (for example Swin UNETR in MONAI) to split an input feature map or volume into non-overlapping local windows. This function implements the partitioning described in "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows" (Liu et al.) and is intended for use in medical imaging deep-learning pipelines where local windowed self-attention is applied to 2D feature maps or 3D volumes.
    
    Args:
        x (torch.Tensor): Input tensor representing a batch of feature maps or volumes produced by a preceding network layer. For 3D volumes the expected shape is (B, D, H, W, C) where B is batch size, D is depth (slices), H is height, W is width and C is channel/features per voxel. For 2D feature maps the expected shape is (B, H, W, C). The function only supports inputs with rank 5 (3D volume) or rank 4 (2D map); other ranks will cause the internal reshape/permutation to fail (typically raising a runtime error). The input tensor is not modified in-place; the function returns a new tensor containing the partitioned windows (the implementation uses view/permute/contiguous operations and may return a new contiguous tensor sharing or copying memory as required by PyTorch).
        window_size (tuple): Local window size expressed as a tuple of integers matching the spatial dimensions of x. For a 3D input (B, D, H, W, C) provide (Wd, Wh, Ww) where each element divides the corresponding spatial dimension (D, H, W) exactly. For a 2D input (B, H, W, C) provide (Wh, Ww) where each element divides the corresponding spatial dimension (H, W) exactly. If the spatial dimensions are not divisible by the corresponding window sizes, the reshape/view operations will fail (raising a runtime error).
    
    Returns:
        torch.Tensor: A tensor containing the partitioned non-overlapping windows ready for window-wise attention or further per-window processing. For a 3D input with shape (B, D, H, W, C) and window_size (Wd, Wh, Ww), the returned tensor has shape (num_windows_total, Wd * Wh * Ww, C) where num_windows_total = B * (D // Wd) * (H // Wh) * (W // Ww). For a 2D input with shape (B, H, W, C) and window_size (Wh, Ww), the returned tensor has shape (num_windows_total, Wh * Ww, C) where num_windows_total = B * (H // Wh) * (W // Ww). This layout flattens each local window's spatial elements into the second dimension while preserving the feature/channel dimension C for attention or linear projections.
    """
    from monai.networks.nets.swin_unetr import window_partition
    return window_partition(x, window_size)


################################################################################
# Source: monai.networks.nets.swin_unetr.window_reverse
# File: monai/networks/nets/swin_unetr.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def monai_networks_nets_swin_unetr_window_reverse(
    windows: torch.Tensor,
    window_size: tuple,
    dims: tuple
):
    """monai.networks.nets.swin_unetr.window_reverse reconstructs a tensor of local windowed features back into the original spatial layout used by Swin Transformer style models (for example within the Swin-UNETR architecture used in medical imaging in MONAI). It reverses the window partitioning performed during windowed self-attention so that per-window feature vectors are rearranged into a contiguous volume or image that matches the original spatial dimensions.
    
    This function is used in the Swin-UNETR pipeline to reassemble processed local windows (the output of window-based attention or processing) into a full 2D image or 3D volume so downstream modules (decoders, up-samplers, loss computation) can operate on the restored spatial grid.
    
    Args:
        windows (torch.Tensor): A tensor containing features extracted or processed per local window. This tensor is expected to be the result of a corresponding window partition operation and to be laid out so that it can be reshaped into blocks for each batch and spatial grid of windows. The last dimension is treated as the feature/channel dimension and is preserved. The tensor's device and dtype are preserved by this operation. If the memory layout or total number of elements does not match the expected grouping implied by window_size and dims, a runtime error (view/reshape mismatch) will occur.
        window_size (tuple): A tuple of integers specifying the size of each local window along each spatial axis. For a 3D volume this is (ws_d, ws_h, ws_w); for a 2D image this is (ws_h, ws_w). The lengths of window_size and the spatial portion of dims must be consistent: when dims has length 4 (b, d, h, w) window_size must have three elements; when dims has length 3 (b, h, w) window_size must have two elements. Mismatched lengths or values that do not evenly divide the corresponding spatial dimensions in dims will produce a runtime error during reshape.
        dims (tuple): A tuple describing the target output spatial dimensions including the batch size. For 3D volumes provide (b, d, h, w); for 2D images provide (b, h, w). The function branches based on len(dims): when len(dims) == 4 it reconstructs a 3D volume of shape (b, d, h, w, C); when len(dims) == 3 it reconstructs a 2D image of shape (b, h, w, C). dims must exactly match the spatial dimensions from which windows were originally partitioned; otherwise the reshaping logic will fail and raise a runtime exception (an undefined local variable or view/reshape error).
    
    Behavior and side effects:
        This function performs tensor reordering using view, permute, contiguous, and view operations to map flattened windowed features back into the specified spatial grid. It does not perform in-place modification of user tensors; it returns a new tensor (or a view thereof) with the reconstructed spatial layout. It preserves the input tensor's dtype and device. The output places the feature/channel dimension last (i.e., channel-last ordering). The function only supports dims tuples of length 3 or 4; other lengths are not handled and will result in a runtime error because the internal variable for the output is not defined for those cases. The caller is responsible for ensuring that window_size elements evenly divide the corresponding spatial dimensions in dims and that windows was produced by a compatible partitioning routine.
    
    Returns:
        torch.Tensor: The reconstructed tensor with spatial dimensions restored. For a 3D-volume case (len(dims) == 4) the returned tensor has shape (b, d, h, w, C) where C is the inferred feature/channel size from the input windows. For a 2D-image case (len(dims) == 3) the returned tensor has shape (b, h, w, C). The channel/feature dimension is last; if a channel-first layout is required for subsequent PyTorch modules, the caller must permute dimensions after receiving this result.
    """
    from monai.networks.nets.swin_unetr import window_reverse
    return window_reverse(windows, window_size, dims)


################################################################################
# Source: monai.networks.nets.vista3d.vista3d132
# File: monai/networks/nets/vista3d.py
# Category: valid
################################################################################

def monai_networks_nets_vista3d_vista3d132(
    encoder_embed_dim: int = 48,
    in_channels: int = 1
):
    """monai.networks.nets.vista3d.vista3d132 returns a configured VISTA3D model instance implementing the exact network configuration used in the paper at https://arxiv.org/abs/2406.05285. This factory function builds a 3D image encoder (SegResNetDS2) and two task heads (PointMappingSAM and ClassMappingClassify), wires them into a VISTA3D model, and returns the assembled model for use in medical imaging workflows (for example, 3D segmentation, point-based mapping, and classification in healthcare imaging datasets). The implementation treats class indices larger than 132 as zero-shot (i.e., out-of-support classes are handled as unseen by the model).
    
    Args:
        encoder_embed_dim (int): Hidden embedding dimension used throughout the encoder and head feature interfaces. Practically, this value controls the number of output channels produced by the SegResNetDS2 encoder (passed as out_channels and init_filters) and the feature_size consumed by both PointMappingSAM and ClassMappingClassify. A larger encoder_embed_dim increases model capacity and memory usage; the default 48 is the value used in the referenced VISTA3D132 configuration. This value must be a positive integer; non-positive values or values that are incompatible with downstream deployment hardware may cause runtime errors (for example, allocation failures or shape mismatches when loading pretrained weights).
        in_channels (int): Number of input channels expected by the 3D image encoder (SegResNetDS2). In medical imaging, common values are 1 for single-channel modalities (e.g., CT or MRI intensity volumes) or 3 for multi-channel inputs. This argument is forwarded directly to SegResNetDS2(in_channels=...), so the tensor passed to the returned model during inference or training must have the same channel dimension. Mismatched channel counts will raise tensor-shape errors at runtime.
    
    Returns:
        VISTA3D: A VISTA3D model instance assembled as follows: the image encoder is a SegResNetDS2 with blocks_down=(1, 2, 2, 4, 4), norm="instance", out_channels and init_filters set to encoder_embed_dim, and dsdepth=1; the point_head is a PointMappingSAM constructed with feature_size=encoder_embed_dim, n_classes=512, last_supported=132 (this enforces the zero-shot behavior for class indices > 132); the class_head is a ClassMappingClassify with n_classes=512, feature_size=encoder_embed_dim, and use_mlp=True. The returned module contains trainable parameters and must be placed on the desired device (CPU/GPU) by caller code. No other side effects occur during construction beyond allocation of module objects and their parameter tensors.
    """
    from monai.networks.nets.vista3d import vista3d132
    return vista3d132(encoder_embed_dim, in_channels)


################################################################################
# Source: monai.networks.schedulers.rectified_flow.timestep_transform
# File: monai/networks/schedulers/rectified_flow.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def monai_networks_schedulers_rectified_flow_timestep_transform(
    t: torch.Tensor,
    input_img_size_numel: torch.Tensor,
    base_img_size_numel: int = 32768,
    scale: float = 1.0,
    num_train_timesteps: int = 1000,
    spatial_dim: int = 3
):
    """Applies a resolution-aware remapping to diffusion timesteps used by the rectified flow scheduler in MONAI. This function converts original timestep indices into adjusted timesteps that account for differences between the current input image voxel count and a reference (base) image voxel count. In the MONAI medical-imaging diffusion training workflow, this helps scale the effective progression through training timesteps when images have different spatial resolutions or sizes so that denoising or diffusion dynamics remain comparable across resolutions.
    
    Args:
        t (torch.Tensor): The original timestep(s) as a torch.Tensor of one or more scalar values representing discrete training timesteps or fractional timesteps. These are interpreted in the same units as num_train_timesteps (see below). The function first normalizes t by dividing by num_train_timesteps to obtain a fractional progress value in [0, 1] (for typical valid inputs).
        input_img_size_numel (torch.Tensor): The total number of spatial elements (H * W * D or other product of spatial axes) of the current input image as a torch.Tensor scalar. This represents the image resolution/size whose effect on diffusion dynamics we want to compensate for. It is compared to base_img_size_numel to compute a spatial scaling ratio.
        base_img_size_numel (int): Reference total number of spatial elements used during training or as a canonical size (default 32768, equal to 32 * 32 * 32). This integer is the denominator in the spatial ratio and should be the same reference used when designing or calibrating the scheduler. If base_img_size_numel is zero, a division-by-zero error will occur.
        scale (float): Additional multiplicative scaling factor applied to the spatial ratio before remapping timesteps (default 1.0). Use this to globally increase (>1.0) or decrease (<1.0) the influence of resolution differences on the timestep transform. A scale of 1.0 leaves the spatial ratio unmodified.
        num_train_timesteps (int): Total number of discrete training timesteps used by the diffusion/rectified flow process (default 1000). This value is used to normalize input t into a fractional progress value and to re-scale the transformed fractional progress back into the original timestep units. If set inconsistently with the training schedule, the mapping between fractional progress and timestep indices will be incorrect.
        spatial_dim (int): Number of spatial dimensions in the image (default 3). This integer is used as the root exponent (1.0 / spatial_dim) when converting voxel-count ratios into linear spatial scale ratios. spatial_dim must be non-zero; a zero or negative spatial_dim will produce a ZeroDivisionError or an invalid root, and non-positive inputs to the root may produce non-real results.
    
    Behavior and formula:
        1. Normalizes the input timestep(s): t_normalized = t / num_train_timesteps.
        2. Computes a spatial linear scale ratio from the voxel-count ratio: ratio_space = (input_img_size_numel / base_img_size_numel) ** (1.0 / spatial_dim).
        3. Applies the optional global scale: ratio = ratio_space * scale.
        4. Remaps the normalized timestep via a rational transform that compresses or expands progress depending on ratio: new_t_normalized = ratio * t_normalized / (1 + (ratio - 1) * t_normalized).
        5. Converts back to timestep units: new_t = new_t_normalized * num_train_timesteps.
    
    Practical significance:
        - When input images are larger (higher input_img_size_numel) than the base image, ratio_space > 1 and this function typically slows or stretches the effective progress through the diffusion timesteps so that denoising steps correspond to comparable spatial scales.
        - When input images are smaller, ratio_space < 1 and the mapping speeds up or compresses progress.
        - The rational mapping ensures the transformed fractional progress remains bounded and smoothly varies with t and ratio, avoiding simple linear rescaling that could push values outside valid ranges.
    
    Side effects and failure modes:
        - This function performs elementwise arithmetic on the provided torch.Tensor t and input_img_size_numel; it returns a new torch.Tensor and does not modify inputs in-place.
        - If base_img_size_numel is zero, a division-by-zero will occur.
        - If spatial_dim is zero, a ZeroDivisionError will occur when computing the root exponent.
        - If input_img_size_numel or base_img_size_numel are non-positive and spatial_dim produces a fractional root, the computation may yield NaNs or complex values; to produce meaningful real-valued ratios, use positive voxel counts and an appropriate integer spatial_dim.
        - The function assumes num_train_timesteps is positive; non-positive values will produce invalid normalization.
    
    Returns:
        torch.Tensor: The transformed timestep(s) as a torch.Tensor with the same broadcasting semantics as the arithmetic operations applied to the inputs. The returned tensor is in the same timestep units as the input t (i.e., remapped indices scaled by num_train_timesteps) and is intended to be used directly by the rectified flow scheduler or other diffusion-training code paths in MONAI to adjust timestep-dependent computations for differing image resolutions.
    """
    from monai.networks.schedulers.rectified_flow import timestep_transform
    return timestep_transform(
        t,
        input_img_size_numel,
        base_img_size_numel,
        scale,
        num_train_timesteps,
        spatial_dim
    )


################################################################################
# Source: monai.networks.trt_compiler.cuassert
# File: monai/networks/trt_compiler.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def monai_networks_trt_compiler_cuassert(cuda_ret: tuple):
    """monai.networks.trt_compiler.cuassert is an error-reporting helper used by MONAI's TensorRT compilation and CUDA-interfacing code to validate CUDA API call results and surface failures during medical imaging model compilation and inference workflows.
    
    Args:
        cuda_ret (tuple): A tuple representing the immediate return from a lower-level CUDA call used in the TensorRT compiler and CUDA-bound helpers. By convention the first element (index 0) is the CUDA return code (zero indicates success, non-zero indicates an error condition) and the optional second element (index 1) is the actual result value or handle produced by the CUDA call (for example a pointer, object, or status payload forwarded from the CUDA wrapper). This function expects a tuple with at least one element; passing an empty tuple or a non-tuple object may raise IndexError or TypeError respectively because the implementation indexes into the sequence.
    
    Behavior and side effects:
        The function inspects cuda_ret[0] to determine whether the CUDA call succeeded. If the return code is non-zero, cuassert raises a RuntimeError with a message prefixed by "CUDA ERROR:" and including the numeric return code, which is intended to halt the calling compilation/inference operation and propagate the error in MONAI's TensorRT/CUDA integration. If the return code is zero (success) and the input tuple contains a second element, that element is returned to the caller as the CUDA call result. If the return code is zero and no second element is present, the function returns None. Any additional elements beyond the second in the tuple are ignored by this function. The function performs no logging, state mutation, or CUDA resource management itself; its only side effect is raising RuntimeError on detection of a non-zero CUDA return code.
    
    Returns:
        object or None: If the CUDA return code (cuda_ret[0]) indicates success (zero) and a second tuple element exists, that second element is returned and represents the CUDA call's practical result (e.g., a handle or value to be used by subsequent MONAI/TensorRT code). If the return code indicates success but no second element is provided, returns None to indicate "no returned value" while still signaling success. If the return code indicates failure, the function does not return but instead raises RuntimeError; callers should handle this exception to propagate or recover from CUDA errors in MONAI's medical-imaging model compilation and inference pipelines.
    """
    from monai.networks.trt_compiler import cuassert
    return cuassert(cuda_ret)


################################################################################
# Source: monai.networks.trt_compiler.get_dynamic_axes
# File: monai/networks/trt_compiler.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def monai_networks_trt_compiler_get_dynamic_axes(profiles: list):
    """get_dynamic_axes(profiles) calculates a mapping of input/output tensor names to the axes that should be treated as dynamic when exporting a PyTorch model to ONNX, intended for use in MONAI workflows that convert models for TensorRT compilation. This function inspects a set of "profiles" describing minimum, optimal, and maximum sizes for each tensor dimension and returns the axis indices where the minimum and maximum differ. In the MONAI medical imaging domain, this allows onnx.export(..., dynamic_axes=...) to mark spatial or batch dimensions that can vary across inputs so downstream tools (ONNX runtime, TensorRT) can generate appropriate, optimized engines for variable-sized medical images.
    
    Args:
        profiles (list): A list of profile dimension specifications. Each element is expected to be a mapping (iterable of key/value pairs in the form used by calling code) where each key is a tensor name (string) and the corresponding value is a three-element sequence [min, opt, max] representing shapes or sizes for that tensor at the minimum, nominal (optional), and maximum profile points. The function iterates each profile and each key within a profile, compares the min (vals[0]) and max (vals[2]) entries elementwise, and records the index i of any dimension where vals[0][i] != vals[2][i] as a dynamic axis. An empty list or other falsy value for profiles causes the function to return an empty mapping immediately. The function does not validate element types beyond indexability; if profile entries do not follow the expected three-element, indexable structure, Python indexing or type errors may be raised.
    
    Returns:
        dict[str, list[int]]: A dictionary mapping tensor name strings to lists of integer axis indices that should be treated as dynamic in onnx.export. Each entry is added only if at least one axis differs between the profile's min and max shapes. If multiple profiles contain the same tensor key, later profiles will overwrite earlier entries for that key because the function assigns dynamic axes directly into the returned dictionary. If profiles is empty, an empty dictionary is returned. The function has no other side effects. Exceptions such as IndexError or TypeError may be raised if profile entries are malformed (for example, not having three indexable elements or non-indexable values).
    """
    from monai.networks.trt_compiler import get_dynamic_axes
    return get_dynamic_axes(profiles)


################################################################################
# Source: monai.transforms.lazy.utils.is_compatible_apply_kwargs
# File: monai/transforms/lazy/utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def monai_transforms_lazy_utils_is_compatible_apply_kwargs(kwargs_1: dict, kwargs_2: dict):
    """monai.transforms.lazy.utils.is_compatible_apply_kwargs checks whether two mappings of keyword arguments are compatible for combination by a lazy transform "apply" operation in MONAI's transform pipeline.
    
    This function is intended for use within MONAI's lazy transform system (used in medical imaging preprocessing and compositional transform APIs described in the project README) to decide if two per-call kwargs dictionaries can be merged or applied together when composing transforms. Each argument dictionary is expected to represent keyword arguments that would be passed to an individual transform's apply method during lazy execution (for example, per-item options carried through a composed sequence of transforms). The current implementation is a predicate function used by higher-level code that composes or merges kwargs before invoking apply.
    
    Args:
        kwargs_1 (dict): First mapping of keyword arguments intended for a transform's apply call. This parameter represents per-transform or per-item options produced earlier in a lazy pipeline. The function expects a dict as provided by calling code; no mutation is performed on this object by the function itself.
        kwargs_2 (dict): Second mapping of keyword arguments intended for a transform's apply call. This parameter represents additional per-transform or per-item options that might be merged with kwargs_1 when composing transforms. The function expects a dict as provided by calling code; no mutation is performed on this object by the function itself.
    
    Returns:
        bool: True if the two kwargs mappings are considered compatible for combination by an apply operation, False otherwise. In the context of MONAI's lazy transforms, a True return value means the calling code may safely merge or forward both dictionaries to the same apply invocation according to the pipeline's merging policy. A False return value indicates that the calling code should not merge them and should instead handle them separately (for example, by executing separate apply calls or resolving conflicts before merging). Note: the present implementation of this function unconditionally returns True, meaning it currently treats all dict inputs as compatible; callers relying on nuanced compatibility checks should either perform their own validation or update this function. There are no side effects from calling this function.
    """
    from monai.transforms.lazy.utils import is_compatible_apply_kwargs
    return is_compatible_apply_kwargs(kwargs_1, kwargs_2)


################################################################################
# Source: monai.transforms.lazy.utils.kwargs_from_pending
# File: monai/transforms/lazy/utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def monai_transforms_lazy_utils_kwargs_from_pending(pending_item: dict):
    """monai.transforms.lazy.utils.kwargs_from_pending extracts a dictionary of keyword arguments used to configure downstream lazy transforms from a "pending" transform item recorded in MONAI's lazy transform pipeline.
    
    This function is used in MONAI's lazy (deferred) transformation system for medical imaging preprocessing, where transform operations may collect metadata (such as interpolation mode, padding mode, target shape, and dtype) to be applied later when the actual tensor/image is materialized. The returned dictionary contains only the keys that should be forwarded as kwargs to the transform implementation (for example, resampling or resizing operations) so that composed, deferred transforms can be applied consistently across a preprocessing pipeline.
    
    Args:
        pending_item (dict): A mapping that represents a pending transform item captured by MONAI's lazy transform mechanism. The mapping may contain the constants LazyAttr.INTERP_MODE and LazyAttr.PADDING_MODE (these are always checked and will appear in the result, possibly with value None), and may optionally contain LazyAttr.SHAPE and LazyAttr.DTYPE. This function does not validate the semantic correctness of the values (for example, it does not enforce that a shape is an integer sequence or that dtype is a valid data type); it only extracts and returns the relevant keys/values for later use. If pending_item is not a dict (for example, None or other type), the function treats it as absent and returns an empty dict. This parameter is central to MONAI's compositional preprocessing design because it carries transformation configuration between stages without immediately applying the operation.
    
    Returns:
        dict: A dictionary suitable for passing as keyword arguments to downstream transform functions. The returned mapping will always include entries for LazyAttr.INTERP_MODE and LazyAttr.PADDING_MODE (their values will be taken from pending_item via pending_item.get(..., None), so they may be None if absent). Entries for LazyAttr.SHAPE and LazyAttr.DTYPE are included only if those keys are present in pending_item; they are not inserted with a default of None. If the input pending_item was not a dict, an empty dict is returned. The function has no side effects and does not modify the input pending_item; it merely constructs and returns a new dict. Failure modes: malformed or unexpected values for any returned key are not checked here and may cause errors later when the kwargs are applied to concrete transform implementations.
    """
    from monai.transforms.lazy.utils import kwargs_from_pending
    return kwargs_from_pending(pending_item)


################################################################################
# Source: monai.transforms.lazy.utils.requires_interp
# File: monai/transforms/lazy/utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def monai_transforms_lazy_utils_requires_interp(matrix: numpy.ndarray, atol: float = 0.001):
    """monai.transforms.lazy.utils.requires_interp checks whether a given affine transformation matrix can be implemented by simple axis operations (flip, permutation, pad/slice) or whether it requires voxel-wise interpolation during resampling.
    
    This function is used in MONAI preprocessing and lazy transform code to decide whether a spatial transform represented by an affine matrix can be realized by cheap array operations (no interpolation, e.g., memory-only permutation/flip) or must be performed with interpolation (resampling voxels), which is more computationally expensive and can change image intensities. The function inspects the translation column and the top-left submatrix of the affine matrix to determine if the transform is an integer-translation plus axis-permutation/flip (returns a mapping) or requires interpolation (returns None). Internally the input is converted to a NumPy array for numeric checks; the function does not modify the provided matrix.
    
    Args:
        matrix (numpy.ndarray): The affine matrix to check. This is the (N+1)x(N+1) homogeneous affine matrix typically used in spatial transforms for medical images in MONAI, where the top-left N x N submatrix encodes axis scaling/rotation/flip and the last column encodes translation. The function uses the matrix values to determine whether the transform corresponds exactly (within tolerance) to axis flips/permutations and integer translations. The matrix is converted to a NumPy array internally for numeric comparisons; the function does not mutate the original matrix argument.
        atol (float): Absolute tolerance used for numerical comparisons. This tolerance is applied when checking whether translation components are close to integers and whether submatrix entries are close to -1, 0, or 1. The default is AFFINE_TOL (0.001 in the signature), meaning values within this absolute difference are treated as exact integers or exact -1/0/1 in the decision logic.
    
    Returns:
        list[int] or None: If the affine matrix indicates that resampling can be achieved by simple axis operations (no voxel interpolation required), returns a Python list of integers that encodes the axis mapping including the channel axis. The returned list has length N+1 for an N-dimensional spatial matrix: the first element is 0 (the channel axis index), and each subsequent element gives the input axis index (0-based, counting channel as index 0) that should be mapped to the corresponding output spatial axis in order. For example, a 3D identity affine would return [0, 1, 2, 3]. If the matrix suggests interpolation is required (translation not integer within atol, submatrix contains non-zero entries other than ±1 or 0 within atol, or the mapping is not a one-to-one axis mapping), the function returns None. This None return indicates that a voxel-wise resampling operation (with interpolation) is necessary to apply the affine transform.
    """
    from monai.transforms.lazy.utils import requires_interp
    return requires_interp(matrix, atol)


################################################################################
# Source: monai.transforms.spatial.functional.convert_box_to_points
# File: monai/transforms/spatial/functional.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def monai_transforms_spatial_functional_convert_box_to_points(bbox: torch.Tensor, mode: int):
    """Convert an axis-aligned bounding box tensor to corner point coordinates usable by MONAI spatial transforms.
    
    This function is used in MONAI to convert per-box coordinate encodings (common in medical imaging tasks such as lesion/ROI annotation and localization) into explicit corner point coordinates that downstream components can sample, transform, or visualize. The function interprets each row of bbox according to a box mode resolved by get_boxmode(mode) and then uses the mode-specific boxes_to_corners routine to produce corner coordinates. For 2D boxes the output contains four 2D points per box; for 3D boxes the output contains eight 3D points per box. The function does not modify its inputs in-place and returns a new tensor.
    
    Args:
        bbox (torch.Tensor): Input bounding boxes with shape (N, C) where N is the number of boxes and C is the coordinate length per box. For 2D boxes C must be 4 with ordering [x1, y1, x2, y2]. For 3D boxes C must be 6 with ordering [x1, y1, z1, x2, y2, z2]. Each row represents one axis-aligned box in the image/volume coordinate space. The tensor's dtype and device are preserved in the output as returned by the underlying tensor operations. The function requires bbox to be a 2-dimensional tensor with at least one row; if bbox.ndim != 2, if C is not 4 or 6, or if bbox has zero rows, the routine will fail (for example, get_boxmode(mode) or tensor stacking will raise an error).
    
        mode (int): Integer code that specifies how to interpret the values in bbox. This integer is resolved by get_boxmode(mode) to a mode object that provides a boxes_to_corners method. The mode determines the coordinate ordering and any semantic interpretation required to convert the compact box representation into corner coordinates (for example, which indices correspond to minima/maxima along each axis). The caller must supply a mode value accepted by get_boxmode; an invalid mode will cause get_boxmode(mode) to raise an error.
    
    Returns:
        torch.Tensor: A tensor of corner point coordinates with shape (N, M, D) where N is the number of input boxes, D is the point dimensionality (2 for 2D, 3 for 3D), and M is the number of corner points per box (4 for 2D -> shape (N, 4, 2); 8 for 3D -> shape (N, 8, 3)). Each inner element is a coordinate vector [x, y] or [x, y, z] corresponding to a corner of the axis-aligned bounding box. The ordering of the returned corner points for each box follows the ordering produced by the resolved mode.boxes_to_corners implementation (for 2D the four corners form the rectangle edges in a consistent sequence; for 3D the eight corners are grouped by planes along the third axis). No in-place side effects occur on the input tensor.
    """
    from monai.transforms.spatial.functional import convert_box_to_points
    return convert_box_to_points(bbox, mode)


################################################################################
# Source: monai.transforms.spatial.functional.convert_points_to_box
# File: monai/transforms/spatial/functional.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def monai_transforms_spatial_functional_convert_points_to_box(points: numpy.ndarray):
    """Converts a set of corner points for rectangular (2D) or cuboid (3D) regions into axis-aligned bounding boxes.
    
    This function is intended for medical imaging workflows (see MONAI README) where annotations or detections are often provided as the corner points of a rectangle in 2D or a cuboid in 3D. Given a batch of such corner points, the function computes the axis-aligned bounding box that encloses each set of corners by taking the elementwise minimum and maximum along the corner dimension and concatenating them. The result is suitable for downstream preprocessing, cropping, augmentation, or evaluation steps that expect boxes in (min_coords, max_coords) form.
    
    Args:
        points (numpy.ndarray): Numeric array containing the corner coordinates for one or more boxes.
            For 3D cuboids, the expected shape is (N, 8, 3) corresponding to N boxes, each with 8 corner points
            and 3 coordinates (x, y, z) per corner. For 2D rectangles, the expected shape is (N, 4, 2)
            corresponding to N boxes, each with 4 corner points and 2 coordinates (x, y) per corner.
            The first dimension N represents the number of boxes in the batch. The function delegates the
            elementwise min/max reduction to MONAI's utils that unify NumPy/PyTorch operations, so the
            numeric semantics follow those underlying implementations.
            Practical significance: callers typically provide model predictions or annotated corners here;
            the function produces a compact axis-aligned box representation used widely in medical image
            preprocessing and metric calculations.
            Behavior and failure modes: if the input does not have one of the documented shapes or is not a
            numeric numpy.ndarray, the underlying reduction calls will raise an exception (for example,
            due to an invalid dimension index). The function does not validate coordinate ordering beyond
            taking per-coordinate minima and maxima; it assumes corners represent valid rectangle/cuboid corners.
    
    Returns:
        numpy.ndarray: Array of axis-aligned bounding boxes formed by concatenating the per-coordinate
        minima and maxima along the second axis. For 3D input (N, 8, 3) the output shape is (N, 6) and each
        row is ordered as (x_min, y_min, z_min, x_max, y_max, z_max). For 2D input (N, 4, 2) the output shape
        is (N, 4) and each row is ordered as (x_min, y_min, x_max, y_max). These boxes are returned as a
        numpy.ndarray with the same numeric semantics as the underlying min/max implementation. There are
        no side effects.
    """
    from monai.transforms.spatial.functional import convert_points_to_box
    return convert_points_to_box(points)


################################################################################
# Source: monai.transforms.spatial.functional.flip
# File: monai/transforms/spatial/functional.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def monai_transforms_spatial_functional_flip(
    img: torch.Tensor,
    sp_axes: tuple,
    lazy: bool,
    transform_info: dict
):
    """monai.transforms.spatial.functional.flip flips image data along specified spatial axes for use in MONAI preprocessing and data-augmentation pipelines for medical imaging. This function implements the flip eagerly (applies torch.flip to the tensor data) or lazily (registers a transform as metadata to be applied later in a TraceableTransform workflow) depending on the lazy flag. The function assumes channel-first tensors (channel dimension first) as used throughout MONAI transform utilities and constructs/updates an affine-like transform matrix describing the flip so downstream traceable metadata can track the spatial change.
    
    Args:
        img (torch.Tensor): Input image tensor to be flipped. In MONAI this is expected to be channel-first (shape convention C x D x H x W... for multi-dimensional medical images). The implementation also recognizes MetaTensor instances (a MONAI wrapper around torch.Tensor) and will read pending shape/rank and copy/produce metadata when MetaTensor is provided. The tensor contents are the image voxel/intensity values used for training, inference, or preprocessing.
        sp_axes (tuple): Spatial axes along which to flip. By convention this refers to spatial axes relative to the channel-first layout (i.e., axes corresponding to the dimensions after the channel dimension). If None, the function will flip over all spatial axes (behavior documented in the original implementation). Negative axis indices are permitted and count from the last spatial axis toward the first. A tuple of integers flips on each axis specified in the tuple. The provided axes are mapped internally to the tensor axes that include the channel dimension using MONAI's map_spatial_axes utility before the flip is performed.
        lazy (bool): When False (eager mode), the function performs the flip immediately by calling torch.flip and returns the flipped tensor (or a MetaTensor with updated metadata). When True (lazy mode), the function does not modify the image data; instead it records the flip as traceable metadata via TraceableTransform.track_transform_meta and returns that metadata or a MetaTensor carrying that metadata so the actual pixel/voxel-level flip can be applied later in a composed transform pipeline. Use lazy=True when building transform pipelines that should accumulate metadata for later application (e.g., for efficient I/O or delayed application), and lazy=False when you need the flipped image data immediately (e.g., for model input).
        transform_info (dict): Dictionary that accumulates transform-specific information across composed transforms in a pipeline. This dictionary is passed to TraceableTransform.track_transform_meta and is used to propagate, merge, or annotate metadata for downstream transforms, inverse operations, or logging. The function will update transform_info implicitly via the metadata object it creates; callers typically pass a mutable dict maintained by pipeline orchestration code.
    
    Returns:
        torch.Tensor or MetaTensor or dict: In eager mode (lazy=False) returns a new torch.Tensor containing the flipped image data; if the input was a MetaTensor the returned value will be a MetaTensor with metadata copied/updated to reflect the flip (affine-like transform and extra_info with the flipped axes). In lazy mode (lazy=True) returns either a MetaTensor with the new transform metadata attached (if the input/output type is MetaTensor) or the metadata dictionary produced by TraceableTransform.track_transform_meta (when a bare torch.Tensor is used). The returned metadata encodes the affine-like flip transform (a matrix with -1 scaling on flipped spatial axes and translation to preserve index coordinates) and an extra_info entry {"axes": sp_axes} for downstream tracking.
    
    Behavior, side effects, and failure modes:
        - Channel-first convention: The function treats the first tensor dimension as channel. Spatial size is derived from img.shape[1:] for a plain torch.Tensor or from img.peek_pending_shape() for a MetaTensor; incorrect input layout (not channel-first) will lead to incorrect axis mapping and results.
        - Axis mapping: The provided sp_axes are converted to tensor axes including the channel dimension via monai.transforms.utils.map_spatial_axes; invalid or out-of-range axis indices will propagate errors (for example, IndexError or value errors raised by the axis-mapping utility or torch.flip).
        - Metadata tracking: The function constructs an affine-like transform matrix and calls TraceableTransform.track_transform_meta to produce metadata. When MetaTensor input is used, metadata methods peek_pending_shape and peek_pending_rank are consulted; the function will copy metadata into the returned MetaTensor when appropriate.
        - Lazy vs eager: lazy=True avoids modifying pixel data and instead returns metadata to be applied later; callers must ensure later transforms or an executor apply the recorded transform if they need the pixel-level change. lazy=False performs the pixel flip immediately with torch.flip and returns the flipped tensor (metadata is still tracked and attached when MetaTensor is in use).
        - Copy semantics: torch.flip returns a new tensor; this function does not perform in-place mutation of the original tensor. When MetaTensor metadata is produced, copy_meta_from is used to propagate metadata to the returned MetaTensor.
        - Error conditions: Passing a non-tensor img, supplying sp_axes values that are not integers or contain invalid indices, or providing a transform_info that is not a dict may result in exceptions. The function relies on MONAI utilities (map_spatial_axes, TraceableTransform) and torch.flip; any errors from those calls can propagate to the caller.
        - Intended domain and use: This flip operation is intended for use in MONAI preprocessing and data-augmentation workflows for medical imaging (e.g., to augment training data or standardize orientations) and integrates with MONAI's traceable transform infrastructure for reproducible transforms and inverse operations.
    """
    from monai.transforms.spatial.functional import flip
    return flip(img, sp_axes, lazy, transform_info)


################################################################################
# Source: monai.transforms.spatial.functional.orientation
# File: monai/transforms/spatial/functional.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def monai_transforms_spatial_functional_orientation(
    img: torch.Tensor,
    original_affine: numpy.ndarray,
    spatial_ornt: numpy.ndarray,
    lazy: bool,
    transform_info: dict
):
    """monai.transforms.spatial.functional.orientation changes the orientation of a channel-first image tensor to the spatial orientation specified by spatial_ornt. This function is used in MONAI medical imaging pipelines to standardize or convert image voxel axis ordering and flips (for example, converting images to a canonical orientation across datasets). It supports eager execution (apply permutation and flips immediately) or lazy execution (record the transform metadata for deferred application), integrating with MONAI's MetaTensor and TraceableTransform metadata tracking.
    
    Args:
        img (torch.Tensor): Input image tensor to be reoriented. The function expects a channel-first layout (channels as the first dimension). In MONAI pipelines this is typically a torch.Tensor or a MONAI MetaTensor (a torch.Tensor subclass that carries metadata); when provided a MetaTensor, the function may peek pending shapes and will copy or update metadata via copy_meta_from. The practical role of img is the image data whose spatial axes (all dimensions after the channel dimension) will be permuted and/or flipped to match spatial_ornt.
        original_affine (numpy.ndarray): The original affine matrix associated with img that maps voxel coordinates to world (scanner) coordinates. This affine is recorded into the transform metadata (under extra_info) so that downstream consumers or provenance tracking know the image-to-world mapping before the orientation change.
        spatial_ornt (numpy.ndarray): A nibabel-style orientation array describing the target orientation of the spatial axes. This array follows nibabel.orientations conventions (see nibabel.orientations.*) and is used to compute the inverse orientation affine via nib.orientations.inv_ornt_aff and to determine axis permutation and axis flips (flip indicated by -1 in the second column). The argument defines how the input spatial axes should be reordered and flipped in the context of medical image orientation standardization.
        lazy (bool): If False, the function applies the orientation change immediately to the data: computes flips (torch.flip) and axis permutation (torch.permute) and returns the reoriented tensor with updated metadata. If True, the function does not perform the actual data manipulation; instead it constructs and returns metadata describing the transform (TraceableTransform.track_transform_meta) so the operation can be executed later (deferred execution). The default behavior in typical usage is eager (lazy=False) unless the pipeline explicitly defers transforms for performance or lazy-loading strategies.
        transform_info (dict): A dictionary containing provenance or contextual information for the applied transform(s). This function passes transform_info to TraceableTransform.track_transform_meta to compose a metadata record that includes the computed affine (inv_ornt_aff), resulting spatial size, original_affine (recorded under extra_info), and other bookkeeping such as orig_size and the incoming transform_info. In MONAI workflows, transform_info is used to maintain a chain of applied transforms for reproducibility and inverse operations.
    
    Returns:
        torch.Tensor: When lazy is False, returns the reoriented image tensor (a torch.Tensor). If the input was a MONAI MetaTensor, the returned tensor will be a MetaTensor with its metadata updated via copy_meta_from to include the tracked transform metadata (spatial size, affine, extra_info, orig_size, and transform_info). If lazy is True, the function does not necessarily return a data tensor: when a MetaTensor-compatible output is available, it returns that MetaTensor with metadata recording the pending orientation transform; when no MetaTensor is available (plain torch.Tensor input), the function returns the metadata object produced by TraceableTransform.track_transform_meta that describes the pending transform (so callers can store or apply it later). The returned tensor or metadata is intended for downstream MONAI components to either apply the orientation change immediately or to compose and execute deferred transforms.
    
    Behavior and side effects:
        The function computes the spatial transform using nibabel.orientations.inv_ornt_aff(spatial_ornt, spatial_shape) where spatial_shape is derived from the input image (peek_pending_shape for MetaTensor or img.shape[1:] for plain tensors). It constructs a full transpose (including the channel dimension) and identifies axes to flip based on spatial_ornt entries equal to -1. It records extra_info containing original_affine and then calls TraceableTransform.track_transform_meta to generate meta_info describing the operation. If lazy is False, the function will perform torch.flip on the identified axes and torch.permute according to the computed full_transpose; these operations mutate neither the input tensor in-place nor its metadata, but produce a new tensor (or MetaTensor) that copies metadata via copy_meta_from where applicable. If lazy is True, no data permute/flip is performed and only metadata is produced or attached for deferred application.
    
    Failure modes and constraints:
        The function assumes the input is channel-first and that spatial_ornt is compatible with the number of spatial axes of img (i.e., the number of rows in spatial_ornt should correspond to the number of spatial dimensions). Mismatched shapes between spatial_ornt and img spatial dimensions, or an original_affine with an incompatible shape, will lead to runtime errors (indexing, broadcasting, or function calls such as nib.orientations.inv_ornt_aff raising exceptions). The function relies on nibabel's orientation conventions; supplying a spatial_ornt that does not conform to nibabel.orientations expectations can produce incorrect transforms or errors. When using lazy=True, callers must handle the returned metadata object appropriately; attempting to treat that metadata as image data will be incorrect.
    
    Practical significance in MONAI:
        This function is a building block for preprocessing and augmentation pipelines in medical imaging tasks implemented with MONAI. It standardizes orientation across heterogeneous datasets, ensures consistent voxel-to-world relationships are tracked via metadata, and supports lazy/deferred execution to optimize I/O and memory usage in complex pipelines.
    """
    from monai.transforms.spatial.functional import orientation
    return orientation(img, original_affine, spatial_ornt, lazy, transform_info)


################################################################################
# Source: monai.transforms.spatial.functional.rotate90
# File: monai/transforms/spatial/functional.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def monai_transforms_spatial_functional_rotate90(
    img: torch.Tensor,
    axes: tuple,
    k: int,
    lazy: bool,
    transform_info: dict
):
    """Functional implementation of rotate90 used by MONAI for spatial rotations of channel-first medical image tensors. This function composes an affine rotation (90-degree increments) and optionally performs the numerical rotation on the tensor data or records the rotation lazily as transform metadata for later application. It is commonly used in preprocessing pipelines for healthcare imaging (2D slices or 3D volumes) where reproducible metadata tracking is required for downstream transforms, visualization, or inverse mapping.
    
    Args:
        img (torch.Tensor): Input image tensor to be rotated. The function assumes a channel-first layout (channels as dimension 0), so spatial dimensions are expected at dims 1..N. If img is a MONAI MetaTensor, the function will use MetaTensor APIs to read pending shape/rank and may return a MetaTensor with updated metadata; otherwise a plain torch.Tensor is used and metadata (when lazy) is returned as a separate object. Practical significance: callers should provide itk-like channel-first medical image tensors (e.g., CxHxW or CxDxHxW) so that spatial axes indexing and metadata bookkeeping align with MONAI conventions.
        axes (tuple): Two integers selecting the spatial plane to rotate. These indices refer to the dimensions of a channel-first tensor (for example, (1, 2) typically selects height and width for a 2D image). Negative values are allowed and count from the last axis toward the first axis. Internally the implementation also derives a zero-based spatial-axis pair (axes - 1) for shape bookkeeping; users must therefore supply axes consistent with channel-first indexing. Practical significance: use this to specify which two spatial axes (slice/row/column or depth/row/column) are rotated.
        k (int): Number of times to rotate by 90 degrees. Each increment represents an additional 90-degree rotation applied in the plane specified by axes. A value of 0 (or any integer congruent to 0 mod 4) results in an identity rotation, but metadata will still be updated/tracked. Practical significance: use small integer counts to perform common rotations (k=1 for 90°, k=2 for 180°, k=3 for 270°).
        lazy (bool): Flag indicating lazy (metadata-only) versus eager (data-modifying) operation. If False (the typical/eager mode), the function applies torch.rot90 to the tensor data and returns the rotated tensor (with metadata attached when a MetaTensor was given). If True (lazy mode, default behavior in higher-level pipelines), the function does not modify tensor data; instead it computes and returns the transformation metadata (or a MetaTensor whose metadata describes the pending rotation). Practical significance: lazy=True is used in pipelines that accumulate transforms for batched/incremental application or for later resolution on a different device. Note: the original implementation documents lazy default as False; higher-level transforms in MONAI may call this function with lazy=True when composing transformations.
        transform_info (dict): A dictionary containing the relevant information pertaining to an already-applied transform chain; this dictionary is passed to the internal TraceableTransform.track_transform_meta call to merge/augment transform history. The function returns or attaches updated transform metadata that includes the new rotation parameters under extra_info. Practical significance: this allows callers to maintain a chained record of spatial operations (important for inverse transforms, label mapping, or exporting transform provenance).
    
    Returns:
        torch.Tensor or MetaTensor or dict: If lazy is False, returns the tensor with the rotation applied: a torch.Tensor if the input was a plain tensor, or a MetaTensor (tensor with attached MONAI metadata) if the input was a MetaTensor; the returned MetaTensor contains updated metadata describing the rotation. If lazy is True, the function does not apply the numerical rotation; it returns a MetaTensor with updated metadata when the input is a MetaTensor, otherwise it returns the transform metadata object (as produced by TraceableTransform.track_transform_meta, typically a dictionary-like structure) instead of a rotated tensor. Practical significance: callers must handle both the rotated data and the metadata-only return forms depending on whether they requested eager or lazy behavior.
    
    Behavior, side effects, and failure modes:
        This function computes an affine that represents the requested sequence of 90-degree rotations and records shape changes for spatial dimensions; for k values of 1 or 3 the code swaps the two affected spatial dimensions in the recorded shape, matching the expected rotation of rectangular images. The function uses TraceableTransform.track_transform_meta to produce transform metadata (meta_info) that includes the affine, original and resulting spatial sizes, and an extra_info entry with axes (as zero-based spatial indices derived from the provided axes) and k. If lazy=True the data buffer is not transformed and meta_info is returned/attached; if lazy=False the function calls torch.rot90 to produce the rotated data and then attaches the meta_info to the output when possible.
        Potential errors arise when axes does not contain exactly two valid integer axis indices for the provided tensor shape (IndexError from tensor indexing or from torch.rot90), or when k is not an integer (TypeError). Passing axes values inconsistent with the channel-first assumption will rotate an unintended pair of dimensions. Large or negative k values are accepted as integers, but rotations are logically equivalent modulo 4 (k % 4). If the input is a MetaTensor, the function will use MetaTensor-specific APIs (peek_pending_shape/peek_pending_rank, copy_meta_from) to read or attach metadata; if those APIs are not present or the input does not conform to MONAI MetaTensor semantics, behavior will fall back to plain-tensor pathways and meta_info will be returned as a separate object.
        The function does not modify the supplied transform_info dictionary in place; transform_info is forwarded to the internal metadata tracker to produce new transform metadata describing this rotation.
    """
    from monai.transforms.spatial.functional import rotate90
    return rotate90(img, axes, k, lazy, transform_info)


################################################################################
# Source: monai.transforms.utils.check_boundaries
# File: monai/transforms/utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def monai_transforms_utils_check_boundaries(boundaries: list):
    """Check boundaries for Signal transforms.
    
    Validate that the provided boundaries argument is a list of exactly two float values used by MONAI Signal transforms. In the MONAI medical-imaging preprocessing domain, Signal transforms operate on 1D signals or time-series (for example, signals derived from imaging modalities or physiologic traces). This utility enforces that the transform receives a well-formed interval or pair of limits (lower and upper boundary) so downstream windowing, clipping, or scaling operations behave deterministically and consistently across a preprocessing pipeline.
    
    Args:
        boundaries (list): A list containing exactly two float values that represent the lower and upper boundary used by Signal transforms. The list must have length 2 and each element must be of Python type float. There is no default value; callers must pass this argument explicitly. This parameter's practical significance is to define the numeric interval applied by signal-level operations (e.g., clipping or windowing) in MONAI preprocessing workflows. If callers provide a value of a different type (for example, tuple, int-only values, nested lists, or strings), or a list with length not equal to 2, validation will fail.
    
    Returns:
        None: This function does not return a value on success; it performs validation only and has no side effects or state changes. On failure it raises a ValueError with the message "Incompatible values: boundaries needs to be a list of float." to indicate that the input is not a list of two floats. To fix the error, supply boundaries as a list of two float objects, e.g. [0.0, 1.0].
    """
    from monai.transforms.utils import check_boundaries
    return check_boundaries(boundaries)


################################################################################
# Source: monai.transforms.utils.img_bounds
# File: monai/transforms/utils.py
# Category: fix_docstring
# Reason: Schema parsing failed: Cannot generate JSON schema for img_bounds because the docstring has no description for the argument 'img'
################################################################################

def monai_transforms_utils_img_bounds(img: numpy.ndarray):
    """monai.transforms.utils.img_bounds computes the bounding indices of non-zero content along the first two axes of a NumPy array. It is intended for use in MONAI preprocessing pipelines (medical imaging) to find the first and last rows/columns (axis 0 and axis 1) that contain any foreground (non-zero) values so callers can derive a tight in-plane bounding box for cropping or centering operations.
    
    This function inspects axis 0 and axis 1 of the provided image array using numpy.any to detect non-zero elements. It returns a 1D numpy array of four integer indices: the minimum and maximum index where axis 0 contains any non-zero elements, followed by the minimum and maximum index where axis 1 contains any non-zero elements.
    
    Args:
        img (numpy.ndarray): Input image array used by MONAI preprocessing utilities. The array may be multi-dimensional; this function evaluates non-zero presence along the first two dimensions (axis 0 and axis 1). Elements are interpreted with NumPy truthiness (zero values are treated as background, non-zero values as foreground). This parameter is required and is not modified by the function.
    
    Returns:
        numpy.ndarray: A 1-D NumPy array with four integer indices in the order [min_idx_axis0, max_idx_axis0, min_idx_axis1, max_idx_axis1]. These indices identify the first and last positions along axis 0 and axis 1 that contain any non-zero elements and can be used to define an in-plane bounding box for cropping or region-of-interest extraction in medical image preprocessing.
    
    Raises:
        IndexError: If there are no non-zero elements along axis 0 or axis 1 (for example, the input is entirely zeros along one of these axes), numpy.where(...)[0] will be empty and indexing [[0, -1]] will raise an IndexError. Callers should validate that the image contains foreground content along both axes before relying on the returned indices.
    
    Notes:
        - The function has no side effects: the input array is not modified.
        - The indices refer to NumPy axis indexing (0-based) and correspond to the first two dimensions of the array; for 2D images this gives the expected row/column bounds, and for higher-dimensional arrays these bounds apply to the first two dimensions commonly used for in-plane operations in MONAI workflows.
    """
    from monai.transforms.utils import img_bounds
    return img_bounds(img)


################################################################################
# Source: monai.transforms.utils.in_bounds
# File: monai/transforms/utils.py
# Category: fix_docstring
# Reason: Schema parsing failed: Cannot generate JSON schema for in_bounds because the docstring has no description for the argument 'x'
################################################################################

def monai_transforms_utils_in_bounds(
    x: float,
    y: float,
    margin: float,
    maxx: float,
    maxy: float
):
    """monai.transforms.utils.in_bounds checks whether a 2D point (x, y) lies inside a rectangular valid region defined by a uniform border margin and the maximum coordinates. In the MONAI medical-imaging context this helper is typically used by spatial transforms and patch/ROI samplers to ensure candidate coordinates for cropping, patch extraction, or kernel placement lie within safe image boundaries.
    
    Args:
        x (float): The x coordinate of the point to test. In MONAI transforms this is the horizontal coordinate in the image coordinate system (for example, column index or continuous horizontal position). The value is compared as margin <= x < (maxx - margin) so the lower bound is inclusive and the upper bound is exclusive.
        y (float): The y coordinate of the point to test. In MONAI transforms this is the vertical coordinate in the image coordinate system (for example, row index or continuous vertical position). The value is compared as margin <= y < (maxy - margin) so the lower bound is inclusive and the upper bound is exclusive.
        margin (float): A non-negative border width subtracted from both image edges to define a safe inner rectangle. The rectangle tested is [margin, maxx - margin) on the x axis and [margin, maxy - margin) on the y axis. If margin == 0 the check reduces to membership in [0, maxx) × [0, maxy). If margin is negative the effective rectangle expands outside the nominal image bounds; the function does not clamp or validate margin beyond numeric comparisons.
        maxx (float): The maximum x coordinate (exclusive limit) of the image domain used to compute the inner rectangle. Practically this represents the image width extent in the same coordinate units as x and margin. The comparison uses (maxx - margin) as the exclusive upper bound.
        maxy (float): The maximum y coordinate (exclusive limit) of the image domain used to compute the inner rectangle. Practically this represents the image height extent in the same coordinate units as y and margin. The comparison uses (maxy - margin) as the exclusive upper bound.
    
    Returns:
        bool: True if and only if the point (x, y) is inside the rectangle defined by inclusive lower bounds at margin and exclusive upper bounds at (maxx - margin, maxy - margin); equivalently the function returns True when margin <= x < (maxx - margin) and margin <= y < (maxy - margin). In MONAI workflows a True result indicates the coordinate is safe for operations that require a full margin from image edges (for example, extracting a patch centered at the point). The function has no side effects.
    
    Notes on behavior and failure modes:
        - If maxx <= 2 * margin or maxy <= 2 * margin the computed exclusive upper bound is less than or equal to the lower bound and no finite (x, y) will satisfy the condition; the function will return False for all such points.
        - Inputs of NaN will cause the comparisons to evaluate to False and thus the function will return False for that point.
        - The function performs simple numeric comparisons only and does not perform type coercion, clamping, or device transfers; it is safe to call from transform logic but callers are responsible for providing coordinates and bounds in consistent units and ranges.
    """
    from monai.transforms.utils import in_bounds
    return in_bounds(x, y, margin, maxx, maxy)


################################################################################
# Source: monai.transforms.utils.is_positive
# File: monai/transforms/utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def monai_transforms_utils_is_positive(img: numpy.ndarray):
    """monai.transforms.utils.is_positive returns a boolean mask indicating which elements of a medical-image array are strictly greater than zero.
    
    Args:
        img (numpy.ndarray): Input numeric image array containing voxel or pixel intensities (for example, a 2D/3D medical image used in MONAI preprocessing pipelines). The array represents image data where positive values typically indicate foreground signal (e.g., soft-tissue intensity in MRI or positive Hounsfield units in CT) and non-positive values indicate background or absence of signal. The function performs an elementwise comparison against zero and does not modify this input array in place.
    
    Returns:
        numpy.ndarray: A boolean numpy.ndarray with the same shape as img and dtype numpy.bool_ where each element is True if the corresponding element in img is strictly greater than 0, and False otherwise. This returned mask is suitable for use as a binary segmentation/masking map in downstream MONAI transforms and workflows.
    
    Behavior, side effects, defaults, and failure modes:
        The operation is an elementwise comparison (img > 0) implemented via NumPy semantics and returns a new array; it has no side effects on the input. Any positive floating or integer value becomes True; zero, negative values, and values that do not satisfy the > 0 comparison become False. NaN values compare as False under NumPy's greater-than semantics, and positive infinity compares as True. The function expects a numpy.ndarray as declared; passing None or a non-numpy object may raise a TypeError or produce undefined behavior depending on NumPy's comparison rules. The function does not perform type coercion checks, clipping, or normalization of intensities—preprocessing (for example, scaling or conversion to numeric dtype) should be done before calling this utility if required by a specific MONAI pipeline.
    """
    from monai.transforms.utils import is_positive
    return is_positive(img)


################################################################################
# Source: monai.transforms.utils.paste
# File: monai/transforms/utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def monai_transforms_utils_paste(orig: numpy.ndarray, block: numpy.ndarray, loc: tuple):
    """monai.transforms.utils.paste pastes a smaller numpy.ndarray block into a larger numpy.ndarray origin array at a specified location, computing appropriate slice ranges and preserving the leading channel dimension commonly used in MONAI medical imaging data. This utility is used in MONAI preprocessing and augmentation pipelines to insert image patches or sub-volumes (block) into a reference image/volume (orig) at spatial coordinates (loc), for example when reconstructing a full volume from tiled patches or applying localized edits.
    
    Args:
        orig (numpy.ndarray): The destination array into which the block will be pasted. In MONAI workflows this typically represents a channel-first medical image or volume (channels on axis 0). The function modifies this array in-place by assigning values from block into computed spatial slices on orig. The return value may be a squeezed view of orig if the leading channel dimension is 1 (see Returns). orig must be indexable in the form used in the implementation (the code performs indexing like orig[:, orig_slices[0]]), so its shape and layout must be compatible with block and loc.
        block (numpy.ndarray): The source array to paste into orig. In MONAI usage this typically represents a patch or sub-volume extracted from an image or produced by a transform. The function computes slice objects for block (via the helper paste_slices) and assigns block[block_slices] into orig at the corresponding destination slices. The shape of block must be compatible with orig for the assignment to succeed.
        loc (tuple): The spatial starting coordinates (one per spatial axis) where the block is to be pasted into orig. loc is iterated together with block.shape and orig (via zip) and is used by the helper paste_slices to compute the actual slice ranges for both orig and block. loc values are treated as integer indices; providing coordinates outside the valid range relies on paste_slices to handle clamping or may result in IndexError or incorrect behavior.
    
    Returns:
        numpy.ndarray: The destination array after the block has been pasted. The returned object is the same underlying data as orig (orig is modified in-place by the assignment). If the leading dimension of orig equals 1, the function applies numpy.squeeze to remove that singleton channel dimension and returns the squeezed array; otherwise it returns orig unmodified except for the pasted region. Callers should be aware that a squeezed copy or view may be returned when orig.shape[0] == 1, while for other shapes the original array object is modified and returned.
    
    Notes on behavior and failure modes: This function delegates computation of precise slice objects to the helper paste_slices (called via map over zip(loc, block.shape, orig)), so boundary handling and clamping semantics depend on that helper. If loc, block.shape, and orig are incompatible (mismatched number of axes or shapes that prevent the assignment orig[:, orig_slices[0]] = block[block_slices[0]]), NumPy will raise errors such as IndexError or ValueError. The function assumes a channel-first layout when indexing the leading axis; callers must ensure their arrays follow the expected convention when using this utility in MONAI image-processing workflows.
    """
    from monai.transforms.utils import paste
    return paste(orig, block, loc)


################################################################################
# Source: monai.transforms.utils.paste_slices
# File: monai/transforms/utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def monai_transforms_utils_paste_slices(tup: tuple):
    """monai.transforms.utils.paste_slices computes two Python slice objects that specify how to paste a 1-D block (along the last axis) into a larger target volume. The function is intended for use in MONAI image-processing pipelines (PyTorch-based medical imaging workflows) when stitching or pasting extracted slices/blocks back into a full image volume: it returns the slice to select the destination region in the target and the corresponding slice to select the source region in the block so that overlapping regions are handled correctly.
    
    Args:
        tup (tuple): A 3-element tuple (pos, w, max_w) describing the paste operation. pos is the integer start index in the target where the block is intended to be placed; it may be negative to indicate the block starts before the target's 0 coordinate. w is the integer width (length) of the block along the last axis to paste. max_w is an object that exposes a shape attribute (for example, a NumPy array or a PyTorch tensor used in MONAI pipelines); its last dimension length (max_w.shape[-1]) defines the size of the target along the pasted axis and is used as the clipping bound. The function infers these roles from the tuple contents and computes clipped source and destination ranges so that out-of-bounds regions are handled without writing outside the target.
    
    Returns:
        tuple: A pair of Python slice objects (target_slice, block_slice). target_slice is slice(orig_min, orig_max) giving the indices along the target's last axis to be overwritten (orig_min is clamped to at least 0 and orig_max is clamped to at most the target size). block_slice is slice(block_min, block_max) giving the indices within the block to copy into the target; block_min is non-negative and block_max will be None when the block region extends to the end (so the slice should be interpreted to the block's end). These slices are computed so the copied region and destination region have equal lengths and represent only the overlapping portion. There are no side effects; the function is pure.
    
    Behavior notes and failure modes: The function expects tup to be a length-3 tuple-like object with numeric pos and w and a max_w that has a shape attribute. If max_w lacks a shape attribute, an AttributeError will be raised. If tup does not have three elements or elements are of incompatible types (for example non-integer pos/w), a TypeError or ValueError may be raised by the arithmetic operations. If w <= 0 or there is no overlap between the block and the target, the returned slices may represent an empty region (orig_min >= orig_max), which should be handled by the caller before attempting array assignment. The function only computes 1-D slices along the last axis; callers must integrate these with multi-dimensional indexing when pasting multi-axis blocks into a full image.
    """
    from monai.transforms.utils import paste_slices
    return paste_slices(tup)


################################################################################
# Source: monai.transforms.utils.rand_choice
# File: monai/transforms/utils.py
# Category: fix_docstring
# Reason: Schema parsing failed: Cannot generate JSON schema for rand_choice because the docstring has no description for the argument 'prob'
################################################################################

def monai_transforms_utils_rand_choice(prob: float = 0.5):
    """monai.transforms.utils.rand_choice returns a boolean that is True with probability prob and False otherwise. It is intended for use in MONAI transform pipelines and data-augmentation logic to make stochastic decisions (for example, whether to apply a particular spatial or intensity augmentation during preprocessing or training of medical imaging models).
    
    Args:
        prob (float): The probability of returning True. This parameter represents the desired probability that the function yields a True outcome; the default value 0.5 implements a 50/50 chance. Internally the function draws a single pseudorandom float from Python's standard library random.random() (which produces values in [0.0, 1.0)) and returns True when that draw is less than or equal to prob. Typical usage in MONAI is to pass a float in the closed interval [0.0, 1.0] where 0.0 means almost always False (True only if random.random() happens to be exactly 0.0, an extremely unlikely event) and 1.0 means always True. If prob > 1.0 the comparison will always be True; if prob < 0.0 the comparison will always be False. If prob is not a numeric type comparable to a float, the comparison may raise a TypeError.
    
    Returns:
        bool: A boolean indicating the random choice outcome. True signals that the probabilistic event occurred (for example, that a transform should be applied); False signals it did not. The function has the side effect of consuming one pseudorandom value from Python's global random number generator (random.random()). The result is non-deterministic unless the RNG is explicitly seeded via random.seed() or an equivalent mechanism prior to calling this function.
    """
    from monai.transforms.utils import rand_choice
    return rand_choice(prob)


################################################################################
# Source: monai.transforms.utils.scale_affine
# File: monai/transforms/utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def monai_transforms_utils_scale_affine(
    spatial_size: tuple,
    new_spatial_size: tuple,
    centered: bool = True
):
    """Compute a homogeneous affine scaling matrix that maps coordinates from an original spatial size to a new spatial size.
    
    This utility is used in MONAI preprocessing and spatial transforms for medical imaging workflows to build an (r+1) x (r+1) affine matrix that applies axis-wise scaling when an image or volume is resized. The function determines r as the maximum of the number of spatial dimensions in spatial_size and new_spatial_size, computes per-dimension scale factors as original_dim / max(new_dim, 1) to avoid division by zero, and constructs a square affine matrix suitable for homogeneous coordinate transformations used by downstream resampling and transform utilities. If centered is True (the default), the function adjusts the translation components so the scaling is performed about the image center; if False, scaling is performed about the origin/corner. The function does not modify input arguments and returns a new numeric matrix object.
    
    Args:
        spatial_size (tuple): The original spatial size of the image or volume. Each element corresponds to the size (number of voxels or pixels) along a spatial axis in the same order used throughout the MONAI spatial transform pipeline. This tuple is used as the numerator when computing per-axis scale factors (original_dim / max(new_dim, 1)). Supplying non-numeric entries will raise a TypeError or ValueError when floats are constructed from the values.
        new_spatial_size (tuple): The target spatial size to which the image or volume will be scaled. Each element corresponds to the target size along a spatial axis. For any target dimension value of 0, the function uses max(new_dim, 1) to avoid division by zero (i.e., a zero entry is treated as 1 for the scale computation). The number of elements may differ from spatial_size; r is computed as max(len(spatial_size), len(new_spatial_size)) and the returned affine matrix has size (r+1) x (r+1).
        centered (bool = True): Whether the scaling should be centered about the image center (True, default) or performed about the origin/corner (False). When True, the function modifies the translation components of the homogeneous affine matrix so that the geometric center of the image remains fixed after scaling. When False, the resulting affine applies pure scaling about the coordinate origin.
    
    Returns:
        numpy.ndarray: A square homogeneous affine scaling matrix of shape (r+1, r+1), where r = max(len(spatial_size), len(new_spatial_size)). The upper-left r x r block contains the diagonal scale factors for each spatial axis; the last column contains translation terms (zeros for corner-centered scaling or computed offsets for center-centered scaling), and the bottom row is [0, ..., 0, 1]. If spatial_size == new_spatial_size, the function returns the identity matrix of size (r+1) x (r+1). Failure modes include exceptions when inputs are not sequence-like or contain non-numeric values, or if an underlying helper (create_scale) raises an error for the provided arguments. The function has no side effects on its inputs.
    """
    from monai.transforms.utils import scale_affine
    return scale_affine(spatial_size, new_spatial_size, centered)


################################################################################
# Source: monai.transforms.utils.squarepulse
# File: monai/transforms/utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def monai_transforms_utils_squarepulse(sig: torch.Tensor, duty: float = 0.5):
    """monai.transforms.utils.squarepulse computes a periodic square-wave signal from an input time-like tensor using PyTorch operations. This implementation is intended for use in MONAI workflows (medical imaging preprocessing, synthetic signal generation, or augmentation pipelines) where a reproducible, PyTorch-native square pulse is required; it reproduces the behavior of scipy.signal.square by comparing the phase against a duty cycle fraction of a 2*pi period.
    
    Args:
        sig (torch.Tensor): Input tensor interpreted as a time variable (in radians). Each element of sig represents an instantaneous phase or time sample at which the square-wave value is evaluated. The function converts this input to a torch.Tensor (via convert_to_tensor) if it is not already one, and the output tensor y has exactly the same shape as sig so it can be used elementwise in downstream preprocessing or synthetic-data generation for medical imaging pipelines.
        duty (float): Fraction of the 2*pi period during which the output is high (value 1). duty is a scalar floating-point duty cycle with a default of 0.5 (50% duty cycle produces a symmetric square wave alternating between 1 and -1). Internally duty is converted to a tensor for broadcasting comparison; valid duty values are in the closed interval [0, 1] to produce the conventional square-wave values. If duty is outside [0, 1], the implementation treats those positions as invalid and leaves the corresponding output values at 0 (see Failure modes). The duty parameter controls the practical shape of the waveform used for synthetic temporal patterns or test signals in MONAI workflows.
    
    Behavior, side effects, defaults, and failure modes:
        - The function interprets the input sig values modulo a 2*pi period: it computes tmod = remainder(sig, 2*pi) so the waveform is periodic with period 2*pi in radians. This remainder is performed with torch.remainder, which yields non-negative remainders and therefore handles negative inputs by wrapping them into [0, 2*pi).
        - For elements where duty is within [0, 1], the output y at positions where tmod < duty * 2*pi is set to 1, and at other positions (within the valid duty range) set to -1. This reproduces the typical square-wave convention (high for the fraction duty of the period, low otherwise).
        - For elements where duty is outside the interval [0, 1], the implementation marks those positions as invalid and leaves the corresponding y values as 0. This design avoids raising an exception for out-of-range duty but signals invalid duty inputs by producing zeros in the output.
        - The default duty value is 0.5. When duty equals 0.0, the output will be -1 for all valid positions (no high interval); when duty equals 1.0, the output will be 1 for all valid positions (always high).
        - The function has no in-place side effects on the provided sig argument; it returns a new tensor y. Internally convert_to_tensor is used and may allocate a new tensor if conversion is required.
        - If sig cannot be converted to a torch.Tensor (for example, if convert_to_tensor is given an unsupported type), convert_to_tensor or subsequent torch operations may raise a TypeError or ValueError. The function itself does not perform explicit type validation beyond calling convert_to_tensor.
        - This function preserves the input shape and is intended to be used where elementwise, phase-dependent binary waveform values are needed in MONAI preprocessing, data augmentation, or synthetic signal tests.
    
    Returns:
        torch.Tensor: A tensor of the same shape as sig containing the square-wave values. For valid duty values in [0, 1], elements are either 1 (when the wrapped phase is within the duty fraction of the 2*pi period) or -1 (otherwise). For positions where duty is outside [0, 1], the corresponding elements are 0 to indicate an invalid duty input for that position. The returned tensor is suitable for direct use in PyTorch-based MONAI pipelines.
    """
    from monai.transforms.utils import squarepulse
    return squarepulse(sig, duty)


################################################################################
# Source: monai.transforms.utils.sync_meta_info
# File: monai/transforms/utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def monai_transforms_utils_sync_meta_info(key: str, data_dict: dict, t: bool = True):
    """monai.transforms.utils.sync_meta_info: Synchronize metadata and transform trace information between a MetaTensor stored under a key in a dictionary and the corresponding separate meta and transform entries used by MONAI transform pipelines.
    
    This function is used in MONAI (a PyTorch-based medical imaging deep learning framework) preprocessing and transform pipelines to keep an image/tensor's metadata and provenance of applied transforms consistent and discoverable. It ensures that the entry at data_dict[key] (expected to be or wrapped as a monai.data.MetaTensor) and the associated meta dictionary and transform-trace entries (derived via PostFix.meta(key) and TraceableTransform.trace_key(key)) reflect the same metadata and applied_operations. This is important for downstream operations that rely on accurate medical-imaging metadata (for example spatial metadata, affine, spacing) and for reproducible transform provenance in model training, validation, or inference workflows.
    
    Args:
        key (str): The dictionary key identifying the primary data item (for example an image tensor name) whose metadata and transform trace should be synchronized. This value is used as the base name to look up or create the companion entries: the meta dictionary at PostFix.meta(key) and the transform trace at TraceableTransform.trace_key(key). If this key is not present in data_dict, a KeyError will be raised when the function attempts to read data_dict[key].
        data_dict (dict): A mapping (standard Python dict expected) that contains at least the entry data_dict[key] (the tensor or object to be treated as a MetaTensor). The function will create or update companion entries inside a shallow copy of this mapping: an entry at PostFix.meta(key) holding metadata (obtained via monai.data.MetaTensor.get_default_meta() when missing) and an entry at TraceableTransform.trace_key(key) holding applied transform provenance (obtained via monai.data.MetaTensor.get_default_applied_operations() when missing). If the provided data_dict is not a mapping type compatible with the check used in the implementation, the original data_dict is returned unchanged (see Returns). Note that the function performs a shallow copy of the mapping before updates; the mapping object returned is a new dict, but the contained objects (such as the tensor stored under key) may be the same objects or wrapped into a MetaTensor and therefore may be shared or replaced.
        t (bool): Selection strategy flag for resolving conflicting applied transform information between the MetaTensor and the separate transform-trace entry. If True (default), the function selects the transform stack with greater length (more recorded applied operations) as the canonical applied_operations. If False, the function selects the transform stack with smaller length. The practical effect is that when multiple sources disagree about the sequence/provenance of transforms, t=True favors the more complete provenance, while t=False favors the smaller provenance record. If one source has no applied_operations (is empty or falsy) while the other has data, the non-empty source is used regardless of t.
    
    Returns:
        dict: A shallow copy of the original mapping with the following guarantees applied when the input mapping contains the required key:
        - data_dict[key] will be ensured to be a monai.data.MetaTensor (if it was not already, it will be wrapped into one using the original value).
        - A meta dictionary entry at PostFix.meta(key) will exist; if absent it will be created via monai.data.MetaTensor.get_default_meta(). The meta stored there will be updated with and prefer values from the MetaTensor.meta attribute.
        - A transform-trace entry at TraceableTransform.trace_key(key) will exist; if absent it will be created via monai.data.MetaTensor.get_default_applied_operations(). The applied_operations stored on both the MetaTensor and the transform-trace entry will be synchronized according to the selection strategy described by t.
        If the input data_dict is not a Mapping as checked by the implementation, the original data_dict object is returned unchanged.
    
    Side effects and failure modes:
        - The function makes a shallow copy of the provided mapping and returns that copy; it does not mutate the top-level mapping object passed in. However, it may wrap the value at data_dict[key] in a new monai.data.MetaTensor instance and assign shared nested objects, so in-place mutations of those nested objects can be visible outside the returned mapping.
        - If key is not present in the provided mapping, accessing data_dict[key] during the synchronization will raise a KeyError.
        - The function relies on MONAI internals: PostFix.meta, TraceableTransform.trace_key, monai.data.MetaTensor.get_default_meta, and monai.data.MetaTensor.get_default_applied_operations. If these symbols are unavailable or have changed, NameError or AttributeError may be raised.
        - The function assumes that the applied transform provenance can be compared by length (len); if applied_operations does not support len() the code may raise a TypeError.
        - The function does not introduce or accept new tensor/device/shape constraints beyond what monai.data.MetaTensor enforces; it is strictly a metadata and provenance synchronization utility within MONAI transform workflows.
    """
    from monai.transforms.utils import sync_meta_info
    return sync_meta_info(key, data_dict, t)


################################################################################
# Source: monai.transforms.utils.zero_margins
# File: monai/transforms/utils.py
# Category: fix_docstring
# Reason: Schema parsing failed: Cannot generate JSON schema for zero_margins because the docstring has no description for the argument 'img'
################################################################################

def monai_transforms_utils_zero_margins(img: numpy.ndarray, margin: int):
    """monai.transforms.utils.zero_margins returns True if all values within margin indices of the edges of img along dimensions 1 and 2 are zero, otherwise returns False. This utility is intended for MONAI preprocessing and transform checks in medical imaging workflows (for example verifying zero padding or cleared borders in channel-first image arrays used with PyTorch).
    
    Args:
        img (numpy.ndarray): Input array representing an image or batch of images. The function treats axis 1 and axis 2 as the spatial edge dimensions to inspect (i.e., the second and third dimensions of the array). Axis 0 is iterated over in aggregation (commonly channel or slice/depth in MONAI data layouts such as (C, H, W) or (C, D, H, W) where the check applies to the H and W axes). The array is not modified by this function; it is read-only for this check.
        margin (int): Non-negative integer number of indices measured from each edge along dimensions 1 and 2 to inspect for zeros. A margin of 0 results in an empty slice check and therefore returns True (no nonzero values found in zero-width margins). Negative margins are not the intended use; they will follow NumPy slicing semantics (which may produce unexpected results) and should be avoided. The function expects an int and will not validate non-integer types.
    
    Behavior and side effects:
        The function checks the values in the first margin indices from both the low and high edges of axis 2 (img[:, :, :margin] and img[:, :, -margin:]) and the first margin indices from both the low and high edges of axis 1 (img[:, :margin, :] and img[:, -margin:, :]). The checks are performed across all positions of axis 0. If any inspected element is nonzero, the function returns False; otherwise it returns True. There are no in-place modifications to img and no other side effects. For margin values greater than the length of an axis, NumPy slicing semantics return the full axis and the function will still operate without raising an IndexError. For margin == 0, NumPy produces empty slices and numpy.any on those slices yields False, so zero_margins returns True.
    
    Failure modes and errors:
        If img is not a NumPy ndarray or does not have at least three dimensions such that axes 1 and 2 exist, NumPy slicing or aggregation may raise an IndexError or TypeError; callers should ensure img is a numpy.ndarray with the expected dimensionality before calling this function. Passing non-integer types for margin may raise a TypeError from NumPy slicing or produce unexpected behavior. The function does not perform explicit type coercion or validation beyond relying on NumPy operations.
    
    Returns:
        bool: True if all inspected margin regions along dimensions 1 and 2 contain only zeros (indicating zero-valued borders), False if any inspected element is nonzero. This boolean result is useful in MONAI preprocessing pipelines to assert that padding or border-clearing operations produced zero margins prior to downstream transforms or model input.
    """
    from monai.transforms.utils import zero_margins
    return zero_margins(img, margin)


################################################################################
# Source: monai.transforms.utils_create_transform_ims.get_2d_slice
# File: monai/transforms/utils_create_transform_ims.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def monai_transforms_utils_create_transform_ims_get_2d_slice(
    image: numpy.ndarray,
    view: int,
    is_label: bool
):
    """monai.transforms.utils_create_transform_ims.get_2d_slice extracts a single 2D slice from a numpy image array. For a 2D input it returns the input unchanged; for an input with more than two dimensions it extracts the central slice along the specified axis (view) and squeezes that axis to produce a 2D array. When the image represents a label map (is_label=True), all voxel values equal to 0 are replaced with numpy.nan to mark background/no-label regions for downstream visualization or processing.
    
    This function is used in MONAI preprocessing/visualization utilities where 2D representations of 2D or 3D medical images are required (for example, rendering a central slice for inspection or debugging). It operates on numpy.ndarray objects and performs slicing and optional in-place modification of the returned array.
    
    Args:
        image (numpy.ndarray): Input image or label map stored as a numpy array. Expected to be either a 2D image (H x W) or a multi-dimensional image where one axis represents depth (for example a 3D volume D x H x W). The function uses image.ndim to decide behavior. If image is already 2D (image.ndim == 2), the exact same array object is returned (no copy is made).
        view (int): Axis index along which to take the central slice when image.ndim != 2. The function computes the center index as shape[view] // 2 and extracts that single-index slab, then squeezes the sliced axis to produce a 2D result. Must be a valid axis for image.shape; an invalid view (for example view >= image.ndim or view < -image.ndim) will raise an IndexError.
        is_label (bool): Flag indicating whether the input is a discrete label map. If True, after extracting the 2D slice the function replaces values equal to 0 with numpy.nan to indicate background/no-label. This replacement is done by assignment on the returned array and therefore may modify the original array if the returned object is a view of the input.
    
    Returns:
        numpy.ndarray: A 2D numpy array containing the extracted slice. If the input was 2D, this is the original array object; if the input had more than two dimensions, this is the central slice along the specified view axis with that axis removed (squeezed). If is_label is True, zeros in the returned array are replaced with numpy.nan.
    
    Behavior and side effects:
        - For 2D inputs the function returns the input array object directly; callers should be aware that any in-place modifications to the returned array will affect the original.
        - For ND inputs (ND != 2) the function extracts a single-index slice along axis view and squeezes that axis; the returned array may be either a view or a copy depending on the input array memory layout.
        - When is_label is True the code performs out[out == 0] = numpy.nan. Assigning numpy.nan into an integer-typed array will raise a ValueError because NaN cannot be represented in integer dtypes. To avoid this error, callers should convert the image to a floating dtype before calling (for example image = image.astype(float)).
        - An IndexError will be raised if view is not a valid axis for the input image. Other unexpected shapes (for example 1D arrays) are not the intended use case and may produce IndexError or unexpected results.
        - No device transfers (e.g., CPU<->GPU) are performed; this function operates only on numpy arrays in host memory.
    """
    from monai.transforms.utils_create_transform_ims import get_2d_slice
    return get_2d_slice(image, view, is_label)


################################################################################
# Source: monai.transforms.utils_create_transform_ims.get_data
# File: monai/transforms/utils_create_transform_ims.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def monai_transforms_utils_create_transform_ims_get_data(keys: list):
    """monai.transforms.utils_create_transform_ims.get_data: Download a small example medical imaging dataset (MarsAtlas), run a short MONAI dictionary-based transform pipeline on the image and label files, and return the resulting spatially padded dictionary of volumes ready for example/demonstration use.
    
    Args:
        keys (list): A list of dictionary keys used by the MONAI dictionary transforms in this function. In typical use this is [CommonKeys.IMAGE, CommonKeys.LABEL] (image key at index 0 and label key at index 1), and the order is significant: the first element (keys[0]) is treated as the image key whose transformed shape is used to compute padding. The function passes these keys exactly to the following MONAI transforms in sequence: LoadImaged(keys) to load NIfTI files into the input dictionary, EnsureChannelFirstd(keys) to guarantee a channel-first layout, ScaleIntensityd(CommonKeys.IMAGE) to scale intensities of the image key only, and Rotate90d(keys, spatial_axes=(0, 2)) to rotate both image and label. Provide keys that match the dictionary fields you expect in the returned dict; missing or misordered keys (for example if keys is empty or keys[0] is not the image key) will lead to runtime errors or incorrect padding.
    
    Returns:
        dict: A dictionary keyed by the same keys supplied in the keys argument. Each entry is the corresponding transformed volume after loading, channel-reordering, image-intensity scaling, rotation, and spatial padding. The function computes max_size = max(data[keys[0]].shape) from the first transformed volume (the image) and pads all specified keys to a cubic shape (max_size, max_size, max_size) via SpatialPadd(keys, (max_size, max_size, max_size)). The returned arrays are the array-like objects produced by MONAI dictionary transforms (the exact ndarray/tensor type is determined by the MONAI transforms and their configuration).
    
    Behavior, side effects, defaults, and failure modes:
        This function downloads and extracts a small example dataset named "MarsAtlas-MNI-Colin27.zip" from a fixed Dropbox URL into a cache directory determined by MONAIEnvVars.data_dir() if set, otherwise into a temporary directory created by tempfile.mkdtemp(). The function creates the cache directory if necessary and writes the zip file and extracted contents to disk; expect network usage and disk space consumption equal to the dataset size. The dataset used (MarsAtlas) is chosen because it contains one parcellated image for quick download and demonstration purposes; it is intended for examples and tutorials rather than production training. After download_and_extract, the function locates NIfTI files (*.nii) in the extracted folder and expects exactly the image and label files to be present and sortable; if the expected files are missing, not readable, or more files are present such that image/label selection fails, the function will raise an exception (propagating file I/O or glob-related errors). Network errors, permission errors, corrupted downloads, or failures inside MONAI transforms (for example incompatible file formats, unexpected array shapes, or an empty keys list) will also raise exceptions. The function applies a fixed rotation with spatial_axes=(0, 2) and scales intensity only for the image key; these behaviors are hard-coded and not configurable via parameters.
    """
    from monai.transforms.utils_create_transform_ims import get_data
    return get_data(keys)


################################################################################
# Source: monai.transforms.utils_create_transform_ims.get_stacked_2d_ims
# File: monai/transforms/utils_create_transform_ims.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def monai_transforms_utils_create_transform_ims_get_stacked_2d_ims(
    im: numpy.ndarray,
    is_label: bool
):
    """monai.transforms.utils_create_transform_ims.get_stacked_2d_ims: Extract the three orthogonal 2D views from a 3D medical image volume and return them as a stacked list of 2D images for downstream transforms, visualization, or model input preparation in MONAI pipelines.
    
    This function is used in MONAI pre-processing/transform utilities to produce the three orthogonal slices (one per spatial axis) from a volumetric numpy.ndarray. It calls get_2d_slice(im, i, is_label) for i in range(3) and returns the three resulting 2D images in axis order i = 0, 1, 2. The caller is expected to have ensured consistent image sizing (for example, by applying SpatialPadd earlier in a typical MONAI pipeline). The is_label flag is forwarded to get_2d_slice so that label-specific handling performed there (such as discrete processing or different interpolation rules) is preserved.
    
    Args:
        im (numpy.ndarray): The input medical image or label volume provided as a NumPy array. In MONAI workflows this is typically a 3D (or higher with channel dimension) image volume from which orthogonal 2D slices are extracted. The array must have sufficient spatial dimensions so that three orthogonal views are meaningful; sizes must be consistent across axes as required by downstream processing (SpatialPadd in the pipeline normally guarantees this).
        is_label (bool): A boolean flag indicating whether im represents a segmentation label map (True) or an intensity image (False). This flag is forwarded to get_2d_slice and controls label-specific handling performed by that function (for example, preserving discrete label values or using label-appropriate processing). It does not modify im in place; it only affects how each 2D slice is produced.
    
    Returns:
        list[numpy.ndarray]: A list of three 2D NumPy arrays corresponding to the orthogonal views extracted from im. The list length is exactly 3 and the elements are produced in index order i = 0, 1, 2 where each element is the 2D slice returned by get_2d_slice(im, i, is_label). These 2D arrays are suitable for use as input to 2D transforms, visualization utilities, or model components that expect orthogonal views from a volumetric medical image.
    
    Notes on behavior and failure modes:
        - The function performs no in-place mutation of im; it only reads im and returns new 2D arrays produced by get_2d_slice.
        - The function assumes the input volume has appropriate spatial dimensions; if im is not a NumPy array or does not provide at least the spatial axes expected by get_2d_slice, the underlying get_2d_slice call may raise an error.
        - Consistent spatial sizing across axes is required for correct orthogonal view extraction; in MONAI pipelines this is commonly enforced earlier by SpatialPadd or other padding/resampling transforms.
        - The exact per-slice processing (interpolation, value mapping, channel handling) is delegated to get_2d_slice and depends on how that function implements label vs. image behavior.
    """
    from monai.transforms.utils_create_transform_ims import get_stacked_2d_ims
    return get_stacked_2d_ims(im, is_label)


################################################################################
# Source: monai.transforms.utils_create_transform_ims.get_stacked_before_after
# File: monai/transforms/utils_create_transform_ims.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def monai_transforms_utils_create_transform_ims_get_stacked_before_after(
    before: numpy.ndarray,
    after: numpy.ndarray,
    is_label: bool = False
):
    """monai.transforms.utils_create_transform_ims.get_stacked_before_after returns the processed (stacked) representations of the provided "before" and "after" image arrays by delegating each array to get_stacked_2d_ims. This function is used in MONAI transform/visualization pipelines for medical imaging to produce 2D visual summaries of 2D or 3D image inputs so that "before" and "after" states of a transform can be compared side-by-side or inspected.
    
    Args:
        before (numpy.ndarray): The input image array representing the "before" state in a transform or preprocessing pipeline. In the MONAI medical-imaging context this is typically a 2D image or a 3D volume (for example, a single-channel CT/MR volume). When a 3D array is provided, it is expected that get_stacked_2d_ims will convert the volume into a single stacked 2D representation (for visualization). This function assumes that the spatial dimensions of this array match the corresponding dimensions of the `after` array; if they do not match, downstream processing in get_stacked_2d_ims may raise an exception.
        after (numpy.ndarray): The input image array representing the "after" state produced by a transform. This has the same intended format and role as `before`: typically a 2D image or 3D volume from medical-imaging workflows. When 3D, get_stacked_2d_ims will be applied to produce a stacked 2D visualization. The function requires that `after` and `before` have compatible spatial sizes; mismatched sizes are not handled here and will likely cause an error from the underlying stacking routine.
        is_label (bool): Flag indicating whether the provided arrays are label maps (segmentation masks) rather than continuous intensity images. Default is False. This flag is forwarded to get_stacked_2d_ims so that label-specific handling (for example, preserving discrete labels or using label-appropriate visualization rules) can be applied by that helper. The boolean controls processing semantics but no additional validation of label content is performed here.
    
    Returns:
        list: A two-element list [before_stacked, after_stacked] containing the outputs of get_stacked_2d_ims applied to `before` and `after`, respectively. Each element is the processed representation (typically a 2D image suitable for visualization) produced by get_stacked_2d_ims for the corresponding input. There are no in-place side effects on the input arrays in this function; errors or exceptions raised by get_stacked_2d_ims (for example, due to incompatible shapes, unsupported dtypes, or invalid content) are propagated to the caller.
    """
    from monai.transforms.utils_create_transform_ims import get_stacked_before_after
    return get_stacked_before_after(before, after, is_label)


################################################################################
# Source: monai.transforms.utils_create_transform_ims.pre_process_data
# File: monai/transforms/utils_create_transform_ims.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def monai_transforms_utils_create_transform_ims_pre_process_data(
    data: dict,
    ndim: int,
    is_map: bool,
    is_post: bool
):
    """Prepare a sample dictionary for a transform that requires 2D input by optionally extracting the central slice along the last dimension and returning either the full map or a single image/label array. This helper is used by monai.transforms.utils_create_transform_ims when constructing preprocessing transforms for medical imaging pipelines in MONAI; it converts multi-dimensional medical volumes (e.g., 3D stacks) to 2D by selecting the middle index of the last axis when ndim == 2 so downstream 2D-only transforms receive an appropriate 2D input.
    
    Args:
        data (dict): A mapping that represents a single sample used in MONAI transform pipelines. Keys are string identifiers (for example, CommonKeys.IMAGE or CommonKeys.LABEL) and values are numeric array/tensor objects that support attribute .shape and NumPy/PyTorch-style indexing (used here as v[..., v.shape[-1] // 2]). In the MONAI medical imaging domain, values typically represent multi-dimensional image volumes or label volumes where the last dimension is the slice/depth axis. This function reads values but does not mutate the original mapping object; it constructs a new mapping when extracting a 2D slice.
        ndim (int): The number of spatial dimensions required by the transform being prepared. If ndim == 2, the function converts multi-dimensional inputs into 2D by selecting the center index of the last axis for every entry in data. If ndim != 2, no slicing is performed and the original mapping is preserved for subsequent processing.
        is_map (bool): Flag indicating whether the caller expects the full mapping (a "map" transform) or a single array. If True, the function returns the (possibly sliced) mapping itself so map-style transforms can operate on all keys. If False, the function returns a single array selected by is_post (see below). Use True when building transforms that operate on entire sample dictionaries (e.g., map-style preprocessing).
        is_post (bool): When is_map is False, controls which single array is returned: if True, the function returns the label array located at the MONAI key CommonKeys.LABEL; if False, the function returns the image array located at CommonKeys.IMAGE. This flag is used by the transform-creation utilities to supply either input images or ground-truth labels to non-map (single-array) transforms during preprocessing or postprocessing.
    
    Returns:
        dict or array-like: If is_map is True, returns a mapping with the same keys as the input data; when ndim == 2 each mapping value has been reduced by selecting the central index of its last axis (v[..., v.shape[-1] // 2]), so the returned values are 2D slices appropriate for 2D transforms. If is_map is False, returns a single array-like object: data[CommonKeys.LABEL] when is_post is True, else data[CommonKeys.IMAGE]. The returned arrays are the result of indexing and may be views or new arrays depending on the underlying array/tensor implementation.
    
    Raises / failure modes:
        KeyError: If is_map is False and the expected key (CommonKeys.IMAGE or CommonKeys.LABEL) is not present in data, a KeyError will be raised when attempting to access data[...].
        AttributeError: If a value in data does not expose a .shape attribute, attempting to access v.shape[-1] will raise AttributeError.
        IndexError or ValueError: If the last dimension has size 0 or indexing with v[..., v.shape[-1] // 2] is invalid for a particular array/tensor shape, an IndexError or ValueError may be raised.
        Note: The function does not validate types beyond assuming mapping values support .shape and NumPy/PyTorch-style indexing; callers should ensure values are appropriate array/tensor objects used in MONAI workflows.
    """
    from monai.transforms.utils_create_transform_ims import pre_process_data
    return pre_process_data(data, ndim, is_map, is_post)


################################################################################
# Source: monai.transforms.utils_create_transform_ims.save_image
# File: monai/transforms/utils_create_transform_ims.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def monai_transforms_utils_create_transform_ims_save_image(
    images: list,
    labels: list,
    filename: str,
    transform_name: str,
    transform_args: dict,
    shapes: list,
    colorbar: bool = False
):
    """Save image to file, arranging "before" and "after" views in a compact grid and ensuring there is no whitespace around the edge. This utility is intended for MONAI transform visualization in medical imaging workflows: it displays grayscale image panels (one or more orthogonal views) arranged in rows (typically two rows: before and after a transform), optionally overlays segmentation/annotation label maps, composes a human-readable transform title from transform_name and transform_args, and writes the resulting figure to the filesystem using matplotlib. The function also configures matplotlib rendering (monospace font, dark background) to produce consistent figures suitable for inclusion in reports, debugging logs, or model documentation.
    
    Args:
        images (list): A 2D list-like structure of image arrays organized as rows and columns: outer list indexes rows (nrow = len(images); expected to be 2 for "Before" and "After" but any number is accepted), and each inner list contains the orthogonal view images for that row (ncol = len(images[0]); commonly 1 or 3). Each element is an array-like grayscale image object that supports .min(), .max(), and shape indexing (used to compute per-row vmin/vmax and height ratios). In MONAI usage, these are typically NumPy arrays or PyTorch tensors representing medical image slices or projections. The images determine the grid layout and intensity scaling per row.
        labels (list): A 2D list-like structure matching the layout of images (same outer/inner dimensions) containing label/annotation arrays to overlay on the corresponding image panels, or None to disable overlays. When provided, each label array is drawn with cmap="hsv", alpha=0.9, and interpolation="nearest" to visualize segmentation masks or region annotations used in medical imaging transform evaluation. If labels is not None but dimensions or indexing do not match images, the function may raise an IndexError or produce incorrect overlays.
        filename (str): Filesystem path (string) where the composed figure will be saved via matplotlib.figure.Figure.savefig. This must be a valid writable path; save failures (e.g., due to permission errors, invalid directory, or unsupported file extension) will raise the underlying matplotlib or OS exception (such as OSError).
        transform_name (str): The name of the transform being visualized (for example, "Flipd" or "Spacingd"). This is used as the base of the figure title, shown as transform_name followed by a parenthesized, comma-separated list of transform_args rendered according to rules below. In MONAI workflows, this helps document exactly which transform and parameters produced the "after" images.
        transform_args (dict): Dictionary of the transform's keyword arguments (parameter name -> value). This dictionary is converted into a human-readable title fragment: for each key/value pair the title contains key=value. Values of type str are quoted with single quotes, numpy.ndarray and torch.Tensor objects are represented as "[array]", callables are represented as "[callable]", and all other value types are converted with str(value). The resulting title is wrapped using textwrap.fill at 50 characters with a subsequent indent that aligns continuation lines under the opening parenthesis of the transform name. This behavior provides concise, informative titles for medical imaging experiment logs and figure captions.
        shapes (list): A list of strings (one per row) describing the image shapes or spatial metadata for each row (for example, "128x128x64" or other shape descriptors produced elsewhere in a MONAI pipeline). These strings are appended to the y-axis label for a row only when shapes[0] != shapes[1]; this helps highlight changes in image spatial dimensions produced by transforms (e.g., resampling, cropping).
        colorbar (bool = False): Whether to render a colorbar for each row's images. When True, a colorbar is added to the last column's axis of each row (if that column is the right-most subplot and colorbar is enabled). Default is False. Note that when colorbar is True, the code still suppresses some y-axis ticks for non-rightmost columns; when False, y-axis ticks are removed for all but the right-most column, and the right-most column's y-axis ticks are placed on the right and truncated to at most three visible labels.
    
    Behavior and side effects:
        This function configures matplotlib globally at the start by updating plt.rcParams to use a monospace font family and applying the "dark_background" style; these are global matplotlib settings and will affect subsequent plotting in the same Python process unless changed again. The function creates a matplotlib Figure with a GridSpec layout sized to the number of rows (nrow = len(images)) and columns (ncol = len(images[0])), and sets height_ratios proportional to the first column's pixel height for each row (hs = [float(r[0].shape[0]) for r in images]) to approximately match visual scale between rows.
        For each row, vmin and vmax are computed across all images in that row using i.min() and i.max() to produce a consistent intensity scaling per row. Each image is shown with ax.imshow(..., cmap="gray", vmin=vmin, vmax=vmax) and aspect set to "equal". If labels is not None, a corresponding label overlay is drawn on top of the image using hsv colormap, alpha=0.9, and nearest interpolation.
        The y-axis label for the first column is set to "Before" for row 0 and "After" for row 1 (or "After"/"Before" depending on row index), and if shapes differ between rows the corresponding shapes[row] string is appended on a new line to emphasize spatial changes. Axis frames are turned off, x-ticks are removed for all axes, and y-ticks are suppressed except for the right-most column (unless colorbar is True), where ticks are drawn on the right and only the first three tick labels are kept visible. The colorbar, if enabled, is attached to the last column's axis for each row.
        The title is constructed from transform_name and transform_args as described above, with trailing comma and space removed when transform_args is non-empty. The composed title is wrapped to 50 characters using textwrap.fill with subsequent indent aligned beneath the opening parenthesis and placed at the top-left of the figure (x=0.1, horizontalalignment="left").
        The function saves the figure to the provided filename via fig.savefig(filename) and then closes the figure via plt.close(fig) to free matplotlib resources.
    
    Failure modes and error conditions:
        The function does not perform exhaustive input validation. Common errors include IndexError or TypeError when images or labels do not have the expected nested list structure (outer list of rows, inner lists of same length), when inner elements do not support .min(), .max(), or shape indexing, or when shapes list length does not match the number of rows. File save errors (e.g., permission denied, invalid path) will propagate from matplotlib/OS. The function may raise exceptions if transform_args contains objects whose str() or callable checks raise errors; numpy.ndarray and torch.Tensor instances are handled specially but other array-like objects are converted via str(). Users should ensure passed arrays are numeric image arrays (e.g., NumPy arrays or PyTorch tensors) and that filename is a writable path.
    
    Returns:
        None: The function does not return a value. Its primary effect is the side effect of writing an image file to the provided filename and modifying global matplotlib rcParams/style. It also closes the created figure to release resources.
    """
    from monai.transforms.utils_create_transform_ims import save_image
    return save_image(
        images,
        labels,
        filename,
        transform_name,
        transform_args,
        shapes,
        colorbar
    )


################################################################################
# Source: monai.transforms.utils_create_transform_ims.update_docstring
# File: monai/transforms/utils_create_transform_ims.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def monai_transforms_utils_create_transform_ims_update_docstring(
    code_path: str,
    transform_name: str
):
    """monai.transforms.utils_create_transform_ims.update_docstring: Update a MONAI transform source or documentation file to include a pointer to the transform's example image hosted in the Project-MONAI DocImages repository.
    
    Args:
        code_path (str): Filesystem path to the file that contains the transform documentation to be updated. This should be a text file (typically a Python source or reStructuredText file) within the MONAI repository that includes the transform's documentation block. The function opens this path for reading and later for writing, and will overwrite the file in place if an insertion is performed.
        transform_name (str): The exact name of the transform whose documentation should be located and updated (for example, "RandRotate90"). The function searches the file for the literal occurrence of the backticked transform identifier, i.e., the sequence '`' + transform_name + '`', which is the convention in MONAI documentation to mark transform names.
    
    Behavior and purpose:
        This utility is used by MONAI maintainers and automation to ensure that each transform's documentation contains a reference to a representative example image illustrating that transform's effect. The function reads the file at code_path into memory as lines, locates the first line that contains the backticked transform_name, and treats that location as the start of the transform documentation block. It then inspects the line two lines after that start (doc_start + 2) to determine whether an image directive already exists. If an image directive starting with the reStructuredText image directive marker (".. image") is already present at that location, the function makes no changes and returns early.
    
        If the image is missing, the function inserts two lines immediately at that position: a reStructuredText image directive that points to the image hosted at "https://github.com/Project-MONAI/DocImages/raw/main/transforms/{transform_name}.png" and an indented ":alt:" line providing alternative text "example of {transform_name}". After insertion, the function asserts that exactly two lines were added to the file contents; if this invariant is violated, an AssertionError is raised. Finally, the function writes the modified line list back to code_path, overwriting the original file.
    
    Side effects:
        The function performs in-place modification of the file at code_path when an insertion is required. The write operation replaces the original file contents with the updated contents. The function constructs the remote image URL using the supplied transform_name and the fixed DocImages repository path, so the presence and correctness of that remote resource is assumed but not verified by this function.
    
    Failure modes and exceptions:
        A RuntimeError is raised if the function cannot find any line containing the backticked transform_name in the file; this indicates that the expected transform documentation block is not present in the target file. If the file is shorter than expected such that accessing the line at doc_start + 2 raises an IndexError, that exception will propagate (the function does not explicitly catch IndexError), indicating the file's layout does not match the expected structure. If the post-insertion length check fails (more or fewer than two lines added), an AssertionError is raised. Standard I/O exceptions (e.g., FileNotFoundError, PermissionError) may also be raised when opening or writing the file if the path is invalid or not writable.
    
    Returns:
        None: This function does not return a value. Its observable effect is the potential in-place modification of the file at code_path by inserting an image directive and alt text line linking to the example image for transform_name. If no modification is necessary, the function returns early without altering the file.
    """
    from monai.transforms.utils_create_transform_ims import update_docstring
    return update_docstring(code_path, transform_name)


################################################################################
# Source: monai.transforms.utils_pytorch_numpy_unification.unravel_indices
# File: monai/transforms/utils_pytorch_numpy_unification.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def monai_transforms_utils_pytorch_numpy_unification_unravel_indices(
    idx: list,
    shape: tuple
):
    """Compute unravelled coordinates for a sequence of flat indices and return them
    as a stacked NumPy array or PyTorch tensor. This utility is used in MONAI's
    pytorch/numpy unification helpers (monai.transforms.utils_pytorch_numpy_unification)
    to convert flattened indices (for example, indices produced by argmax over a
    flattened image or a flattened region of interest) into multi-dimensional
    coordinates that correspond to positions in medical imaging volumes or tensors
    used in MONAI transforms and post-processing.
    
    Args:
        idx (list): A list (sequence) of indices to unravel. Each element of this
            list should be an index or an array/tensor of indices compatible with
            the helper unravel_index implementation used by MONAI. Typical
            practical usage is a list of Python integer scalars, NumPy integer
            arrays, or torch.Tensor objects containing integer indices. The order
            of elements in this list is preserved in the output: the i-th element
            of the returned stacked result corresponds to the i-th element of
            this list. If the first element of idx is a torch.Tensor, the function
            uses torch.stack and returns a torch.Tensor; otherwise it uses
            numpy.stack and returns a numpy.ndarray. Passing an empty list will
            raise IndexError because the routine inspects idx[0] to determine the
            numeric library to use.
        shape (tuple): The shape of the target array or tensor for which indices
            are interpreted. This tuple enumerates the lengths of each axis in the
            same order as used in the array/tensor (the axis order in the returned
            coordinates matches the order of this tuple). In practical MONAI
            workflows, shape is typically the spatial shape of an image/tensor
            (for example the shape of a multi-dimensional medical image or model
            output). Values in idx must be valid flat indices for an array/tensor
            with this shape; otherwise the underlying unravel_index implementation
            will raise an error.
    
    Returns:
        NdarrayOrTensor: A stacked array or tensor of unravelled coordinates for
        the provided indices. The return value is either a numpy.ndarray (when
        idx[0] is not a torch.Tensor) or a torch.Tensor (when idx[0] is a
        torch.Tensor), preserving the numeric library used for the coordinate
        outputs. The first dimension of the returned object corresponds to the
        number of elements in idx (one entry per input index); each entry
        encodes the multi-dimensional coordinates produced by unravel_index for
        the given shape. No in-place modification of inputs occurs; the function
        produces a new stacked result.
    
    Raises/Failure modes:
        IndexError: If idx is empty (access to idx[0] fails).
        TypeError or ValueError: If elements of idx are heterogeneous in a way
            that prevents consistent stacking (for example mixing NumPy arrays and
            torch.Tensors), or if the individual coordinate outputs from unravel_index
            have incompatible shapes for stacking.
        Any exception raised by the underlying unravel_index implementation (for
            example when an index is out of bounds for the provided shape) may be
            propagated to the caller.
    
    Side effects:
        None. This function does not modify its inputs; it returns a new array or
        tensor. The returned container type (NumPy vs PyTorch) is chosen based on
        the type of the first element of idx to maintain consistency with MONAI's
        PyTorch/NumPy unification utilities.
    """
    from monai.transforms.utils_pytorch_numpy_unification import unravel_indices
    return unravel_indices(idx, shape)


################################################################################
# Source: monai.utils.component_store.is_variable
# File: monai/utils/component_store.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def monai_utils_component_store_is_variable(name: str):
    """Check whether a given string is a valid Python variable name and is not a reserved Python keyword.
    
    This utility is used in MONAI (a PyTorch-based framework for medical imaging deep learning) to validate candidate identifiers that may be used as variable names, attribute names, configuration keys, or registry/component names (for example when building or populating a component store of models, transforms, losses, or other reusable objects). Ensuring a name is both a valid identifier and not a keyword helps avoid syntax errors, prevents accidental shadowing of language keywords, and supports safe dynamic creation of attributes or generated code in MONAI workflows.
    
    Args:
        name (str): The string to test as a potential Python variable/identifier. This function evaluates the exact semantics of Python identifiers by calling str.isidentifier() and also checks against Python reserved words via keyword.iskeyword(). The parameter is expected to be a Python str representing a candidate name used in contexts such as registry keys, dynamic attribute names, or configuration fields within MONAI; callers should provide a str. Passing a non-str value may result in an AttributeError or TypeError because the underlying checks rely on string methods and the standard keyword module.
    
    Returns:
        bool: True if and only if the provided name is a valid Python identifier according to str.isidentifier() (allows letters, digits and underscores, not starting with a digit, and supports valid Unicode identifier characters) and the name is not a Python reserved keyword as determined by keyword.iskeyword(). Returns False if the name fails either of these checks. The function has no side effects and does not modify any global or external state.
    """
    from monai.utils.component_store import is_variable
    return is_variable(name)


################################################################################
# Source: monai.utils.dist.string_list_all_gather
# File: monai/utils/dist.py
# Category: valid
################################################################################

def monai_utils_dist_string_list_all_gather(strings: list[str], delimiter: str = "	"):
    """Utility to gather a list of Python strings from all processes in a distributed job and return the concatenated list of strings from every rank.
    
    This function is used in MONAI distributed training and utilities to share small pieces of textual information (for example file names, subject IDs, small metadata or logging messages) across multiple processes when running multi-GPU or multi-node workflows. It implements the pattern documented by PyTorch-Ignite's all_gather for strings: each rank joins its local list of strings into one long UTF-8 encoded byte sequence using a delimiter, uses a tensor-based all-gather (via ignite distributed APIs when ignite is available or native torch.distributed when initialized) and then decodes and splits the gathered byte sequences back into a flattened Python list ordered by rank (rank 0 results first, then rank 1, etc.). If no distributed backend is active (world size <= 1), the input list is returned unchanged.
    
    Args:
        strings (list[str]): a local list of UTF-8 text strings on the current process/rank to be gathered across all ranks. In MONAI workflows this typically contains small textual items such as image file names, case identifiers, or other per-process metadata that must be collected on every rank. The function does not modify this list in place; it only reads its contents. Each element will be converted to UTF-8 bytes internally before gathering.
        delimiter (str): a single string used to join the local list into one string before encoding for transport and to split the gathered strings back into elements after decoding. Default is "\t". The delimiter must be chosen so that it does not occur in the original string elements; if an element contains the delimiter, the split operation will produce extra elements and the reconstructed list will be incorrect. The delimiter is applied exactly as provided and is not validated or escaped by this function.
    
    Returns:
        list[str]: a flattened list of strings gathered from all ranks, ordered by rank (all strings from rank 0 first, then rank 1, etc.). The return contains the decoded UTF-8 strings reconstructed by splitting each gathered byte sequence on the provided delimiter. If the distributed world size is 1 or no supported distributed backend is active, this function returns the input strings list unchanged.
    
    Behavior and failure modes:
        - Backend selection: if the ignite distributed package is available, ignite APIs are used; otherwise, native torch.distributed APIs are used when torch.distributed.is_available() and torch.distributed.is_initialized() return True. If neither path is available or world size is 1, no inter-process communication is performed.
        - Encoding/decoding: strings are encoded to UTF-8 via bytearray(..., "utf-8") and transported as a torch.long tensor; decoding also uses UTF-8. All Python str values that are valid Unicode will encode/decode with UTF-8.
        - Delimiter collisions: if any input string contains the delimiter, the subsequent split will produce extra tokens and the reconstructed list will not match the original items. Choose a delimiter not present in the data or sanitize inputs beforehand.
        - Empty input: if the local strings list is empty, an empty string is used in the join step and the gathered result may include empty strings from other ranks; callers should handle empty-string items as needed.
        - No side effects on input: the function does not modify the provided strings list in place.
        - Intended use: designed for small textual payloads (identifiers, labels, short metadata). It is not intended for large binary blobs or very long text per element; for large data, prefer dedicated tensor- or file-based communication mechanisms.
    """
    from monai.utils.dist import string_list_all_gather
    return string_list_all_gather(strings, delimiter)


################################################################################
# Source: monai.utils.misc.list_to_dict
# File: monai/utils/misc.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def monai_utils_misc_list_to_dict(items: list):
    """monai.utils.misc.list_to_dict: Convert a list of "key=value" string items into a Python dictionary suitable for simple configuration parsing in MONAI workflows (for example, parsing command-line or bundle parameter overrides used in training/evaluation pipelines).
    
    This function accepts a sequence of text tokens where each token is expected to represent either a bare key ("a") or a key/value assignment ("key=value"). It is commonly used in MONAI to convert simple string-based parameter specifications into native Python types so they can be applied to configuration dictionaries for model training, inference, or pre-/post-processing pipelines.
    
    Behavior:
    - Each element in items is split at the first "=" into a key and a value. If an element contains no "=", the value for that key becomes None. This supports shorthand flags or presence indicators.
    - Surrounding whitespace and single-quote characters are removed from both keys and values by stripping the characters " \n\r\t'".
    - After splitting and stripping, the function attempts to convert the textual value to a native Python object using ast.literal_eval (so numeric literals, lists, dicts, tuples, and quoted strings will become their Python equivalents).
    - If ast.literal_eval raises a ValueError, the function then attempts to interpret the value as a boolean using distutils.util._strtobool; if that succeeds the boolean value is returned.
    - If both conversions fail, the original stripped string is used as the value.
    - Duplicate keys are considered an error: the function raises KeyError when the same key is encountered more than once.
    - If items is an empty list (no elements), an empty dictionary is returned.
    
    Limitations and failure modes:
    - The parameter type is list; passing a non-list is not documented and may lead to unexpected behavior. An empty list yields {}.
    - ast.literal_eval exceptions other than ValueError (for example, SyntaxError) are not explicitly caught by the implementation and will propagate to the caller.
    - The boolean conversion relies on distutils.util._strtobool semantics; values not recognized by that helper will not be converted to booleans and will fall back to the raw string.
    - Keys and values have only the characters " \n\r\t'" stripped; other surrounding characters are preserved.
    
    Args:
        items (list): A list of strings representing key or key=value entries. Each element should be a string such as "lr=0.001", "use_amp=True", or "tag". Keys are parsed into dictionary keys (strings). Values are parsed into Python objects when possible via ast.literal_eval, then boolean-converted via _strtobool if literal_eval fails, and otherwise left as the stripped string. If an element contains no "=", the corresponding dictionary value is None. Supplying an empty list returns an empty dict. This function is typically used in MONAI to parse simple configuration overrides for training, evaluation, or preprocessing pipelines.
    
    Returns:
        dict: A dictionary mapping parsed key strings to their corresponding values. Values will be one of:
            - a Python object produced by ast.literal_eval when the textual value represents a literal (numbers, lists, tuples, dicts, quoted strings),
            - a boolean produced by distutils.util._strtobool when literal_eval raised ValueError and the text matches known boolean forms,
            - the stripped string if neither conversion succeeded,
            - or None if the original item did not contain "=".
        Duplicate keys raise KeyError; other parsing exceptions (for example, SyntaxError from ast.literal_eval) may propagate to the caller.
    """
    from monai.utils.misc import list_to_dict
    return list_to_dict(items)


################################################################################
# Source: monai.utils.module.damerau_levenshtein_distance
# File: monai/utils/module.py
# Category: fix_docstring
# Reason: Schema parsing failed: Cannot generate JSON schema for damerau_levenshtein_distance because the docstring has no description for the argument 's1'
################################################################################

def monai_utils_module_damerau_levenshtein_distance(s1: str, s2: str):
    """Calculates the Damerau–Levenshtein distance between two strings, returning the minimum number of single-character edits (insertions, deletions, substitutions, or adjacent transpositions) required to transform s1 into s2. In the MONAI context this routine is used for spelling correction and string similarity tasks that arise in medical-imaging workflows (for example, normalizing metadata keys, matching label names, correcting typographical errors in annotations or clinical text before model training and analysis).
    
    This implementation follows the Damerau–Levenshtein definition (see https://en.wikipedia.org/wiki/Damerau–Levenshtein_distance) and is designed to be a small, dependency-free utility for preprocessing and validation steps in MONAI pipelines. It performs an exact computation using a dynamic programming table (implemented as a dict) and treats an adjacent transposition of two characters as a single edit operation.
    
    Args:
        s1 (str): The source string to be transformed. In MONAI usage this typically represents a label, metadata key, filename, or token extracted from clinical/annotation text whose spelling or format should be compared against a canonical or target string. The function uses Python string semantics (len, indexing); passing a non-str value will raise a TypeError when the function attempts string operations.
        s2 (str): The target string to compare against s1. In MONAI workflows this represents the canonical form, expected label, or corrected token to which s1 should be compared or aligned. Like s1, this must be a Python str; non-str inputs will cause exceptions.
    
    Returns:
        int: The Damerau–Levenshtein distance as a non-negative integer. This integer equals the minimal number of single-character edits (insertion, deletion, substitution, or adjacent transposition) required to convert s1 into s2. Practical behaviors and edge cases:
            - If s1 == s2 the function returns 0 immediately (no edits required).
            - If either s1 or s2 is the empty string (""), the distance equals the length of the other string (all insertions or deletions).
            - The algorithm has time complexity proportional to O(len(s1) * len(s2)) and uses a dynamic programming table whose size grows roughly with (len(s1)+2) * (len(s2)+2); very long strings may be slow or memory-intensive and could raise MemoryError in extreme cases.
            - There are no side effects: the function is pure and does not modify inputs or external state.
            - Invalid input types (non-str) will typically result in Python exceptions (e.g., TypeError) when string operations are attempted.
    """
    from monai.utils.module import damerau_levenshtein_distance
    return damerau_levenshtein_distance(s1, s2)


################################################################################
# Source: monai.utils.module.get_package_version
# File: monai/utils/module.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def monai_utils_module_get_package_version(
    dep_name: str,
    default: str = "NOT INSTALLED or UNKNOWN VERSION."
):
    """Get the installed version string for a given Python package name, or return a provided default message when the package is not available or does not expose a version.
    
    This function is used throughout MONAI (a PyTorch-based framework for medical imaging AI) to detect and report versions of optional or required third-party packages (for example, imaging libraries, dataset handlers, or utilities). It attempts to import the package by name and then read its __version__ attribute. The practical significance is to enable reproducible experiments, compatibility checks, diagnostic messages, and runtime guards within MONAI code paths that depend on specific dependency versions.
    
    Args:
        dep_name (str): The import name of the dependency to check (for example, "torch" or "nibabel"). This parameter identifies which installed Python package MONAI should attempt to load to obtain its version string. Providing the exact import name is required because the function calls an import helper to locate the package by this identifier.
        default (str): The string to return when the package cannot be loaded or when the loaded package does not have a __version__ attribute. The default value used in the implementation is "NOT INSTALLED or UNKNOWN VERSION." and is intended to make missing-version situations explicit in logs, configuration reports, or error messages produced by MONAI workflows.
    
    Returns:
        str: The package version string obtained from the package's __version__ attribute when the package import succeeds and the attribute exists. If the package cannot be imported or does not define __version__, the function returns the value provided in the default argument. Note: invoking this function may cause the target package to be imported into the current Python interpreter, which can execute module-level code and therefore has the side effect of loading that module into sys.modules. If the import attempt fails, the function does not raise an exception but returns the default value to allow calling code to handle missing dependencies gracefully.
    """
    from monai.utils.module import get_package_version
    return get_package_version(dep_name, default)


################################################################################
# Source: monai.utils.module.parse_version_strs
# File: monai/utils/module.py
# Category: fix_docstring
# Reason: Schema parsing failed: ("Couldn't parse this type hint, likely due to a custom class or object: ", collections.abc.Iterable[int | str])
################################################################################

def monai_utils_module_parse_version_strs(lhs: str, rhs: str):
    """monai.utils.module.parse_version_strs parses two version strings and returns two iterables of their parsed components (integers for numeric leading segments, otherwise strings). This helper is intended for basic version-component extraction used in MONAI workflows such as comparing package, dependency, or model-bundle version identifiers when deciding compatibility or branching behavior in medical imaging pipelines. It implements a minimal, packaging-independent strategy: remove any git-style build metadata after "+" and split on "."; each segment is stripped of surrounding whitespace and, if it begins with decimal digits, those leading digits are converted to an int while any trailing non-digit suffix in that segment is discarded.
    
    Args:
        lhs (str): The left-hand version string to parse. In MONAI contexts this is commonly the local or current version (for example, the installed MONAI package version or a model bundle version). The function will remove any "+" suffix and split the remainder on "." into segments; segments that begin with digits will be converted to int values representing their leading numeric component, and non-numeric segments will be returned as stripped strings. Supplying a non-str type will cause an AttributeError when the function attempts to call string methods.
        rhs (str): The right-hand version string to parse. In MONAI contexts this is commonly the reference or required version to compare against (for example, a required dependency version from metadata). The same parsing rules described for lhs apply: "+" suffix removal, "." splitting, whitespace stripping, and conversion of leading-digit segments to ints while preserving purely non-digit segments as strings.
    
    Returns:
        tuple[Iterable[int | str], Iterable[int | str]]: A pair of iterables (the implementation returns lazy map objects) corresponding to the parsed components of lhs and rhs, respectively. Each iterable yields elements that are either int (when a dot-separated segment begins with one or more ASCII digits; only the leading digit sequence is kept and converted to int) or str (the stripped original segment when no leading digits are present). For example, "1.2.3+gabcdef" becomes components 1, 2, 3; "1rc1.0" yields components 1, "rc1", 0 as implemented by this function's simplistic extraction rules. Note that the iterables are lazy; callers that need indexing, length, or repeated traversal should convert them to a concrete sequence (for example, list(iterable)). The function has no side effects (it does not modify inputs or external state).
    
    Behavioral notes and failure modes:
        - The function intentionally implements a lightweight parsing approach and is not a full Semantic Versioning (semver) or PEP 440 parser. It may produce results that differ from a strict semver/packaging.version parser for pre-release, post-release, or build metadata semantics.
        - Any "+" character and the trailing substring are removed from each input before parsing (this mirrors removing git/build metadata suffixes commonly appended to version strings).
        - For a dot-separated segment that contains digits followed by letters (for example "1rc1"), only the leading digit sequence is preserved and returned as an int (the trailing alphabetic suffix is dropped by design).
        - If an input is an empty string, the function will return an iterable yielding an empty-string segment (after stripping) following the same conversion logic.
        - The function expects string inputs (as declared in the signature); passing non-string values will raise standard Python errors when string operations are attempted.
        - Because the implementation uses a simple regular-expression-based extraction of leading digits, it may not handle locale-specific digit characters or unusual Unicode inputs as intended.
        - If the module-level regex match function used by this implementation is not available in the runtime environment, a NameError will be raised; under normal packaging within MONAI this function is available and no such error occurs.
    """
    from monai.utils.module import parse_version_strs
    return parse_version_strs(lhs, rhs)


################################################################################
# Source: monai.utils.module.version_geq
# File: monai/utils/module.py
# Category: valid
################################################################################

def monai_utils_module_version_geq(lhs: str, rhs: str):
    """Returns True if version `lhs` is later than or equal to version `rhs` according to a best-effort comparison of version strings used by MONAI to gate features and enforce dependency compatibility (for example, checking PyTorch or other package versions before enabling code paths). The function first normalizes inputs by casting them to Python strings, then attempts to use the packaging.version.Version class when available for a robust semantic comparison. If packaging.version is unavailable, it falls back to an element-wise comparison produced by parse_version_strs. This function is therefore useful within MONAI to programmatically decide whether the running environment satisfies a minimum or specific version requirement without raising on common invalid-version formats (it prefers a permissive result in that case).
    
    Args:
        lhs (str): Left-hand side version string to evaluate as the candidate or runtime version (e.g., the installed package version). The function casts this value to str at the start, so non-string inputs will be converted. In MONAI this is typically the version of a dependency or the library itself and is treated as the version that should be "later or equal" for the function to return True.
        rhs (str): Right-hand side version string representing the requirement or baseline version to compare against (e.g., a minimum required dependency version). This is also cast to str and represents the version that lhs is tested to be greater than or equal to.
    
    Returns:
        bool: True if lhs is considered later than or equal to rhs according to the comparison logic, False otherwise. When packaging.version is available, the function returns the result of pkging.Version(lhs) >= pkging.Version(rhs). If packaging.version raises pkging.InvalidVersion for either input, the function catches that specific exception and returns True (a permissive default used in MONAI to avoid failing version checks on nonstandard version strings). If packaging.version is not available, the function uses parse_version_strs(lhs, rhs) to obtain comparable components and compares them element-wise: if both components are integers they are compared numerically; otherwise components are compared lexicographically as strings. If all compared components are equal (including when zip stops at the shortest sequence), the function returns True. Side effects: none (pure function); defaults: no default parameters beyond casting inputs to str; failure modes: only pkging.InvalidVersion is handled specially (returns True); other exceptions raised by parse_version_strs or unexpected errors may propagate to the caller.
    """
    from monai.utils.module import version_geq
    return version_geq(lhs, rhs)


################################################################################
# Source: monai.utils.module.version_leq
# File: monai/utils/module.py
# Category: valid
################################################################################

def monai_utils_module_version_leq(lhs: str, rhs: str):
    """monai.utils.module.version_leq determines whether one version string (lhs) denotes an earlier or equal release than another version string (rhs).
    
    This function is used within MONAI for simple dependency and compatibility checks (for example, to determine whether a runtime or dependency version such as PyTorch satisfies a minimum or maximum required version). It accepts arbitrary inputs but first coerces them to Python str. It attempts to use the standard packaging.version.Version objects for robust semantic version comparison when the packaging library is available; if packaging.version is not available it falls back to a deterministic, segment-wise textual/numeric comparison implemented by parse_version_strs.
    
    Behavior details and practical significance:
    - The function returns True when lhs represents a version earlier than or equal to rhs, and False when lhs represents a later version than rhs. This boolean result is suitable for gating feature availability or enforcing dependency constraints in MONAI workflows.
    - Inputs are coerced to strings at the start (lhs and rhs are converted via str(lhs) and str(rhs)), so callers may pass objects that implement __str__.
    - When the optional packaging.version module is present (imported via optional_import), the function constructs packaging.version.Version(lhs) and packaging.version.Version(rhs) and returns the result of the <= operator on those Version objects. This provides standard semantic-version semantics when available and is the preferred comparison path.
    - If packaging.version.Version raises packaging.version.InvalidVersion for the provided strings, the implementation treats this as a conservative success and returns True. This behavior avoids blocking execution for unknown or nonstandard version formats when attempting to ensure compatibility in MONAI environments.
    - If packaging.version is not available (optional_import indicates absence), the function uses parse_version_strs(lhs, rhs) as a fallback. In the fallback:
        - The two version strings are parsed into corresponding sequences of segments (as returned by parse_version_strs).
        - The function iterates segment-wise over zipped segments from both sequences. For the first pair of unequal segments:
            - If both segments are integers, they are compared numerically (l < r).
            - Otherwise, the segments are compared lexicographically as strings (f"{l}" < f"{r}").
        - If all compared segments are equal for the length of the shorter parsed sequence, the function returns True. Practically, this means versions that are identical up to the shorter prefix are considered earlier-or-equal (for example, "1.2" is treated as <= "1.2.0" in the fallback logic).
    - The function has no side effects: it does not perform I/O, modify global state, or change its arguments. It is deterministic for the same inputs and available environment (presence or absence of packaging.version).
    
    Failure modes and error handling:
    - The function never intentionally raises exceptions for normal comparison of strings. If packaging.version is available but either lhs or rhs cannot be parsed into a valid packaging.version.Version, packaging.version.InvalidVersion is caught and the function returns True.
    - If parse_version_strs is unable to parse the provided strings and itself raises an exception, that exception will propagate; callers who expect to handle malformed custom version formats may need to validate inputs or handle exceptions from parse_version_strs.
    - The function does not validate that inputs conform to any specific versioning scheme beyond what packaging.version or parse_version_strs accept.
    
    Args:
        lhs (str): Left-hand side version string to compare. In MONAI this is typically the currently observed or candidate version (for example, the installed PyTorch or dependency version). The function will coerce lhs to str and interpret it as a version identifier; it returns True if lhs is earlier than or equal to rhs according to packaging.version semantics when available, otherwise by the fallback segment-wise comparison.
        rhs (str): Right-hand side version string to compare. In MONAI this is typically the target, minimum, or maximum version to compare against (for example, a required or tested dependency version). The function will coerce rhs to str and interpret it as a version identifier; it serves as the comparison target such that the function returns True when lhs is earlier-or-equal to this value.
    
    Returns:
        bool: True if lhs denotes a version that is earlier than or equal to rhs, False if lhs denotes a later version than rhs. When packaging.version is available, this is the result of packaging.version.Version(lhs) <= packaging.version.Version(rhs). If packaging.version.Version raises packaging.version.InvalidVersion during parsing, the function returns True. In the fallback path (packaging not available), the return value reflects a deterministic, segment-wise numeric-or-lexical comparison as described above.
    """
    from monai.utils.module import version_leq
    return version_leq(lhs, rhs)


################################################################################
# Source: monai.utils.type_conversion.dtype_numpy_to_torch
# File: monai/utils/type_conversion.py
# Category: fix_docstring
# Reason: Schema parsing failed: Cannot generate JSON schema for dtype_numpy_to_torch because the docstring has no description for the argument 'dtype'
################################################################################

def monai_utils_type_conversion_dtype_numpy_to_torch(dtype: numpy.dtype):
    """Convert a numpy dtype to the corresponding torch dtype used for PyTorch tensors in MONAI pipelines.
    
    This utility is used throughout MONAI to map numpy array data types (for example, from image pre-processing or data loading steps) to the torch.dtype that should be used when creating or interpreting PyTorch tensors for model input, loss computation, and other deep-learning operations in medical imaging workflows.
    
    Args:
        dtype (numpy.dtype): A numpy data type object (for example, numpy.float32, numpy.int16). This is the dtype of a numpy array or scalar that you want to convert into the equivalent PyTorch dtype. The function accepts numpy scalar dtypes (instances of numpy.dtype) and uses them to determine the corresponding torch.dtype. The value is used only to infer the torch dtype; no permanent data is copied or returned from the input.
    
    Returns:
        torch.dtype: The PyTorch dtype that corresponds to the provided numpy dtype (for example, torch.float32, torch.int16). The returned object represents the data type used by PyTorch tensors and is suitable for use when creating tensors or specifying tensor dtypes in MONAI model and transform code.
    
    Behavior and side effects:
        The function creates a temporary zero-dimensional numpy array with the provided dtype and calls torch.from_numpy on that array to obtain a torch tensor, then returns that tensor's dtype attribute. This transient creation is minimal in memory (a single scalar array and tensor) and does not persist references to the input data; the temporary objects are subject to normal Python garbage collection. The function does not perform any device placement or casting of existing arrays; it only maps dtype descriptors.
    
    Failure modes and notes:
        If the provided numpy dtype is not representable by PyTorch (for example, structured dtypes, object dtype, or certain uncommon numpy types), torch.from_numpy will raise a TypeError. Callers should ensure the dtype is a standard numeric or boolean numpy dtype supported by PyTorch before calling this function. The mapping is deterministic and follows PyTorch's numpy-to-torch interpretation used internally by torch.from_numpy.
    """
    from monai.utils.type_conversion import dtype_numpy_to_torch
    return dtype_numpy_to_torch(dtype)


################################################################################
# Source: monai.utils.type_conversion.get_numpy_dtype_from_string
# File: monai/utils/type_conversion.py
# Category: fix_docstring
# Reason: Schema parsing failed: Cannot generate JSON schema for get_numpy_dtype_from_string because the docstring has no description for the argument 'dtype'
################################################################################

def monai_utils_type_conversion_get_numpy_dtype_from_string(dtype: str):
    """monai.utils.type_conversion.get_numpy_dtype_from_string: Return a numpy.dtype object corresponding to a dtype name provided as a string. This utility is used in the MONAI medical-imaging deep learning codebase to convert textual dtype specifications found in configuration, metadata, or serialized bundle files into concrete numpy dtype objects (for example to allocate numpy arrays, cast data for preprocessing pipelines, or interoperate with PyTorch tensors).
    
    Args:
        dtype (str): A string naming the desired numpy dtype. The string should contain the dtype identifier recognized by NumPy, for example "float32". Qualified names that include module qualifiers separated by dots are accepted as strings because the implementation takes the final token after the last dot; for example "numpy.float32" or "torch.float32" (as strings) will be interpreted as "float32". The parameter type in the function signature is str and callers should pass a string. This function is intended to be used where dtype values are read or specified as text in MONAI workflows (for example, configuration files, bundle metadata, or logging), and the returned numpy.dtype will be used to control array allocation and casting semantics in preprocessing and model I/O.
    
    Returns:
        numpy.dtype: A numpy dtype object corresponding to the requested dtype name. The returned object is the dtype attribute of a zero-dimensional (scalar) numpy array created transiently using numpy.empty with the requested dtype name. Practically, this provides the canonical numpy dtype instance (for example numpy.float32) that downstream MONAI code can use to construct or cast arrays.
    
    Raises:
        TypeError: If the final token extracted from the input string is not recognized by NumPy as a valid dtype name, NumPy will raise a TypeError (or a similar exception) when attempting to create an empty array with that dtype. Callers should validate or catch exceptions when dtype strings may be malformed or originate from untrusted sources.
    
    Behavior and side effects:
        The function converts the input to a string form and takes the substring after the last dot character, then calls numpy.empty with shape [] and that substring as the dtype argument, and returns the dtype attribute of the created array. This allocates a temporary zero-dimensional numpy array; the allocation has negligible memory impact because the array is empty and is immediately discarded. There are no persistent side effects and the input string is not modified. There are no default dtype values; the function requires an explicit dtype string and will raise if that string does not correspond to a valid NumPy dtype.
    """
    from monai.utils.type_conversion import get_numpy_dtype_from_string
    return get_numpy_dtype_from_string(dtype)


################################################################################
# Source: monai.utils.type_conversion.get_torch_dtype_from_string
# File: monai/utils/type_conversion.py
# Category: fix_docstring
# Reason: Schema parsing failed: Cannot generate JSON schema for get_torch_dtype_from_string because the docstring has no description for the argument 'dtype'
################################################################################

def monai_utils_type_conversion_get_torch_dtype_from_string(dtype: str):
    """Get a torch.dtype corresponding to a textual dtype name.
    
    This function is used in MONAI to convert a textual dtype identifier commonly found in configuration files, model/bundle metadata, or user-supplied parameters into a PyTorch dtype object that can be applied when creating tensors, models, or performing dtype-sensitive operations in medical imaging deep learning workflows. The function implements this by delegating to two helpers: it first parses the input string into a numpy.dtype via get_numpy_dtype_from_string and then maps that numpy dtype to the equivalent torch.dtype via dtype_numpy_to_torch. Typical input strings are the Numpy-style dtype names such as "float32" or "int64"; for example, passing "float32" produces torch.float32. This function has no side effects beyond performing the conversion.
    
    Args:
        dtype (str): A textual dtype identifier. This string must represent a dtype name that get_numpy_dtype_from_string can parse (for example "float32", "int64"). In the MONAI domain, such strings commonly originate from user configuration, dataset metadata, or saved bundle settings and indicate the desired numeric precision for tensors and model parameters.
    
    Returns:
        torch.dtype: The PyTorch dtype object corresponding to the input string (for example, torch.float32 for "float32"). This value is intended for direct use when constructing torch.Tensor objects or setting module parameter dtypes in MONAI pipelines.
    
    Raises:
        Exception: If the input string cannot be parsed into a valid numpy.dtype or if no corresponding torch.dtype mapping exists, an exception raised by the underlying helper functions (for example a ValueError or TypeError) will be propagated. Passing a non-str value for dtype is not supported and will typically result in an error from the parsing helper. The exact exception type and message are determined by get_numpy_dtype_from_string and dtype_numpy_to_torch.
    """
    from monai.utils.type_conversion import get_torch_dtype_from_string
    return get_torch_dtype_from_string(dtype)


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
