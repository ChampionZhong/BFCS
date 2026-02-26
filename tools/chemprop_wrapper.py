"""
Regenerated Google-style docstrings for module 'chemprop'.
README source: others/readme/chemprop/README.md
Generated at: 2025-12-02T00:48:14.163427Z

Total functions: 4
"""


################################################################################
# Source: chemprop.utils.utils.make_mol
# File: chemprop/utils/utils.py
# Category: valid
################################################################################

def chemprop_utils_utils_make_mol(
    smi: str,
    keep_h: bool = False,
    add_h: bool = False,
    ignore_stereo: bool = False,
    reorder_atoms: bool = False
):
    """chemprop.utils.utils.make_mol builds an RDKit molecule (Chem.Mol) from a SMILES string for use in Chemprop's molecular featurization and model pipelines. This function parses the SMILES with configurable handling of explicit hydrogens, optional addition of hydrogens, optional removal of stereochemical information, and optional reordering of atoms by atom map numbers. It is used throughout Chemprop to convert string representations of molecules into RDKit Mol objects that downstream message-passing neural network code expects.
    
    Args:
        smi (str): A SMILES string representing the molecule to construct. This is the primary input used to create an RDKit molecule via Chem.MolFromSmiles. The SMILES should be a valid chemical string; if RDKit cannot parse it the function will raise a RuntimeError. Typical use is providing canonical or non-canonical SMILES from datasets that Chemprop trains or predicts on.
        keep_h (bool): Whether to preserve explicit hydrogens present in the input SMILES. When True, hydrogens that are explicitly written in the SMILES (e.g., [H]) are retained in the returned Chem.Mol. When False (default), the parser is instructed to remove explicit hydrogens during SMILES parsing (params.removeHs = True). Note: keep_h does not add hydrogens that are absent in the SMILES; it only preserves explicit hydrogens if they are present.
        add_h (bool): Whether to add implicit hydrogens to the constructed molecule after parsing. When True, Chem.AddHs is called on the parsed molecule to add explicit hydrogen atoms for all implicit hydrogens (default is False). This is useful when downstream featurization or models require explicit hydrogen atoms. If add_h is False, the returned molecule will remain without newly added hydrogens unless they were explicitly present in the SMILES and keep_h preserved them.
        ignore_stereo (bool): Whether to remove stereochemical information from the molecule after construction (default is False). When True, all atom chiral tags are set to CHI_UNSPECIFIED and all bond stereo values are set to STEREONONE using RDKit APIs, effectively ignoring R/S and cis/trans stereochemistry. Use this option when training or evaluating models that should not consider stereochemical differences or when input data contains inconsistent stereochemical annotations.
        reorder_atoms (bool): Whether to reorder the atoms in the returned molecule according to their atom map numbers (default is False). When True, the function collects atom map numbers via atom.GetAtomMapNum(), computes a new ordering using numpy.argsort on those map numbers, and applies Chem.rdmolops.RenumberAtoms to produce a molecule whose atom ordering follows ascending atom map numbers. This is useful when the SMILES atom order does not correspond to external atom mapping (e.g., "[F:2][Cl:1]") and a consistent atom index ordering is required for alignment with other data. NOTE: This option reorders atoms by their atom map numbers as implemented, but does not reorder the bond objects beyond what RenumberAtoms performs; users should verify resulting topology if they rely on specific bond object identities.
    
    Returns:
        Chem.Mol: An RDKit Chem.Mol object constructed from the input SMILES with the requested options applied. The returned molecule is suitable for Chemprop featurization and model input. If RDKit fails to parse the SMILES (Chem.MolFromSmiles returns None), the function raises a RuntimeError describing the invalid SMILES string instead of returning None.
    """
    from chemprop.utils.utils import make_mol
    return make_mol(smi, keep_h, add_h, ignore_stereo, reorder_atoms)


################################################################################
# Source: chemprop.utils.v1_to_v2.convert_hyper_parameters_v1_to_v2
# File: chemprop/utils/v1_to_v2.py
# Category: fix_docstring
# Reason: Schema parsing failed: Cannot generate JSON schema for convert_hyper_parameters_v1_to_v2 because the docstring has no description for the argument 'model_v1_dict'
################################################################################

def chemprop_utils_v1_to_v2_convert_hyper_parameters_v1_to_v2(model_v1_dict: dict):
    """chemprop.utils.v1_to_v2.convert_hyper_parameters_v1_to_v2 converts a saved Chemprop v1 model dictionary into a hyper_parameters dictionary formatted for Chemprop v2. This function is used during migration from Chemprop v1 to v2 (see the repository README and v1→v2 transition notes) to transform model hyperparameters, metric/loss identifiers, message-passing block dimensions, aggregation and predictor configuration into the v2 structure expected by the v2 training and inference codepaths.
    
    Args:
        model_v1_dict (dict): A dictionary produced by saving a Chemprop v1 model checkpoint. This dictionary must contain at minimum:
            - "args": an object (typically the argparse.Namespace or similar) with v1 CLI/training attributes referenced by this conversion: metric, warmup_epochs, init_lr, max_lr, final_lr, activation, bias, atom_messages, atom_descriptors_size, hidden_size, depth, dropout, undirected, aggregation, aggregation_norm (if aggregation == "norm"), features_generator, target_weights (optional), num_tasks, dataset_type, ffn_hidden_size, ffn_num_layers, and optionally loss_function.
            - "state_dict": a mapping of parameter names to tensors; the conversion reads shapes for "encoder.encoder.0.W_i.weight", "encoder.encoder.0.W_h.weight", and "encoder.encoder.0.W_o.weight" to infer message-passing dimensions d_h, d_v, and d_e.
            - "data_scaler" (required only if args.dataset_type == "regression"): a mapping containing "means" and "stds" used to build an UnscaleTransform for the predictor output transformation.
        The role of model_v1_dict is to provide the v1 model configuration and learned-parameter shapes so the converter can determine equivalent v2 hyperparameter values (for example, input/output dimensions, message-passing feature sizes, and which classes/registrations to reference). Practical significance: users running migration scripts should pass the exact saved v1 model dictionary produced by Chemprop v1; missing keys or unexpected types will cause the conversion to fail.
    
    Behavior and details:
        The function performs a pure transformation (it does not modify model_v1_dict) and returns a new dict hyper_parameters_v2 ready for use by v2 model factories/registries. Specific behaviors include:
        - Sets hyper_parameters_v2["batch_norm"] to False to reflect v1 defaults.
        - Maps the v1 metric string (args.metric) to a v2 registry key using a hardcoded renamed_metrics mapping (e.g., "auc" -> "roc", "prc-auc" -> "prc", "cross_entropy" -> "ce", "binary_cross_entropy" -> "bce", "mcc" -> "binary-mcc"). Some v1 metrics map to placeholder strings like "recall is not in v2"; such values are passed to MetricRegistry lookup and may be invalid in v2.
        - Builds hyper_parameters_v2["metrics"] as a single-element list by calling Factory.build with MetricRegistry[...], so the v2 MetricFactory/Registry machinery is used to instantiate the metric. This requires that the mapped metric key exist in the v2 MetricRegistry.
        - Copies learning-rate schedule parameters warmup_epochs, init_lr, max_lr, and final_lr from args_v1 into the top-level hyper_parameters_v2 keys to preserve optimizer/scheduler behavior across versions.
        - Infers message-passing dimensions from the saved parameter tensor shapes:
            d_h is taken from the first dimension of encoder.encoder.0.W_i.weight;
            d_v is derived from W_o.weight shape as W_o.shape[1] - d_h;
            d_e is computed differently depending on args_v1.atom_messages: if atom_messages is True, d_e = W_h.shape[1] - d_h; otherwise, d_e = W_i.shape[1] - d_v.
          These computed dimensions are set in hyper_parameters_v2["message_passing"], along with activation, bias, cls (BondMessagePassing or AtomMessagePassing chosen according to args_v1.atom_messages), d_vd (atom_descriptors_size), depth, dropout, and undirected. The practical significance is that the v2 message-passing block will have compatible internal sizes with the v1 trained weights, enabling loading or re-creation of equivalent architectures.
        - Converts aggregation settings into hyper_parameters_v2["agg"] with "dim" set to 0 (v1 always aggregates over atom features) and "cls" obtained from AggregationRegistry[args_v1.aggregation]; if aggregation == "norm", adds the "norm" parameter from args_v1.aggregation_norm.
        - Computes additional input feature dimensionality d_xd by summing contributions of any feature generators listed in args_v1.features_generator (200 if "rdkit" in generator name, 2048 if "morgan" in name). If features_generator is None, it is treated as an empty list.
        - Builds task_weights from args_v1.target_weights if provided (converting to a torch.tensor and unsqueezing to match expected shape); otherwise creates ones of length args_v1.num_tasks. These task_weights are used when constructing the predictor criterion (loss) via Factory.build.
        - Determines the loss function registry entry T_loss_fn by using getattr(args_v1, "loss_function", default) where default is selected from a hardcoded loss_fn_defaults mapping based on args_v1.dataset_type. This mirrors v1 behavior and ensures that regression/classification/multiclass/spectra dataset types map to sensible default losses if loss_function was not present in the saved v1 args.
        - Populates hyper_parameters_v2["predictor"] as an AttributeDict containing activation, cls set via PredictorRegistry[args_v1.dataset_type], criterion built via Factory.build(T_loss_fn, task_weights=task_weights), task_weights set to None (v2 stores the criterion with embedded weights), dropout, hidden_dim from args_v1.ffn_hidden_size, input_dim computed as hidden_size + atom_descriptors_size + d_xd, n_layers set to args_v1.ffn_num_layers - 1 (preserving v1 interpretation), and n_tasks from args_v1.num_tasks.
        - If args_v1.dataset_type == "regression", the function also adds hyper_parameters_v2["predictor"]["output_transform"] set to an UnscaleTransform built from model_v1_dict["data_scaler"]["means"] and ["stds"], preserving v1 target scaling behavior in v2 predictor output transformation.
    
    Defaults and side effects:
        - Defaults applied inside the function include treating a missing args_v1.features_generator as an empty list and using a default loss function based on dataset_type when args_v1.loss_function is absent (loss_fn_defaults).
        - The function has no side effects on the provided model_v1_dict; it returns a newly created dict mapping containing v2-style hyperparameter entries.
        - The function relies on external registries and factories (MetricRegistry, AggregationRegistry, PredictorRegistry, LossFunctionRegistry, Factory) and on classes/functions BondMessagePassing, AtomMessagePassing, AttributeDict, UnscaleTransform, and torch. These must be available in the runtime environment for the conversion to succeed.
    
    Failure modes and exceptions:
        - KeyError will be raised if expected top-level keys are missing from model_v1_dict (for example, "args", "state_dict", or, when required, "data_scaler"), or if required parameter names are not present in state_dict (encoder.encoder.0.W_i.weight, encoder.encoder.0.W_h.weight, encoder.encoder.0.W_o.weight).
        - AttributeError or TypeError may occur if args_v1 does not expose the required attributes or if those attributes have unexpected types (for example, non-iterable features_generator).
        - IndexError may occur when reading tensor shapes if the stored tensors do not have the expected number of dimensions.
        - Lookup errors (KeyError or custom registry errors) will occur if the mapped metric or registry keys (e.g., MetricRegistry[...] or PredictorRegistry[dataset_type] or LossFunctionRegistry[...]) are not present in the v2 registries. Notably, some v1 metrics are mapped to placeholder strings like "recall is not in v2"; attempting to build such metrics will result in registry lookup failures.
        - Any use of torch.tensor and Factory.build may raise their own runtime exceptions if inputs are invalid; these propagate to the caller.
    
    Returns:
        dict: A dictionary hyper_parameters_v2 containing the converted hyperparameters for Chemprop v2. The returned dictionary includes at least the following top-level keys with practical meanings mirroring v1 configuration:
            - "batch_norm": False (v1 default).
            - "metrics": a list with a built metric object produced by Factory.build(MetricRegistry[...]) corresponding to the v1 metric.
            - "warmup_epochs", "init_lr", "max_lr", "final_lr": scalar values copied from args_v1 to preserve scheduler settings.
            - "message_passing": an AttributeDict with keys activation, bias, cls (BondMessagePassing or AtomMessagePassing), d_e, d_h, d_v, d_vd, depth, dropout, undirected describing the v2 message-passing block dimensions and hyperparameters inferred from v1 weights and args.
            - "agg": a dict with "dim" and "cls" (and optionally "norm") describing aggregation behavior in v2 equivalent to v1 aggregation.
            - "predictor": an AttributeDict containing activation, cls (PredictorRegistry entry for the dataset type), criterion (loss built with task weights), task_weights (set to None in the returned structure because weights are embedded in criterion), dropout, hidden_dim, input_dim, n_layers, n_tasks, and, for regression, an "output_transform" UnscaleTransform built from v1 data_scaler means and stds.
        The returned dict is suitable for use by Chemprop v2 model construction and training codepaths; callers should validate registry keys and the presence of required state_dict keys before invoking this converter to avoid the failure modes described above.
    """
    from chemprop.utils.v1_to_v2 import convert_hyper_parameters_v1_to_v2
    return convert_hyper_parameters_v1_to_v2(model_v1_dict)


################################################################################
# Source: chemprop.utils.v1_to_v2.convert_model_dict_v1_to_v2
# File: chemprop/utils/v1_to_v2.py
# Category: fix_docstring
# Reason: Schema parsing failed: Cannot generate JSON schema for convert_model_dict_v1_to_v2 because the docstring has no description for the argument 'model_v1_dict'
################################################################################

def chemprop_utils_v1_to_v2_convert_model_dict_v1_to_v2(model_v1_dict: dict):
    """Converts a Chemprop v1 checkpoint dictionary (as loaded from a .pt file) into the v2 checkpoint dictionary format expected by Chemprop v2 and its PyTorch Lightning-based loading routines. This function is used during migration of trained molecular property prediction models from Chemprop v1 to Chemprop v2 so that v2 code can inspect and restore model parameters and hyperparameters produced by v1.
    
    Args:
        model_v1_dict (dict): A dictionary representing a saved Chemprop v1 model checkpoint, typically produced by torch.save(...) to a .pt file and then loaded with torch.load(...). In the Chemprop domain this dictionary should contain the serialized model parameters and v1 hyperparameter metadata that describe a message-passing neural network trained for molecular property prediction. The function expects v1-specific keys and formats handled by the helper converters; if the provided dict does not conform to the v1 checkpoint structure, helper converters may raise exceptions which propagate out of this function.
    
    Returns:
        dict: A new dictionary structured to match the v2 checkpoint conventions used by Chemprop v2 and PyTorch Lightning. The returned dictionary contains the following keys and meanings as set by this conversion routine:
            "epoch": None — placeholder for the training epoch in v2 checkpoints; set to None because v1 checkpoints converted here do not populate v2 epoch metadata.
            "global_step": None — placeholder for the global optimizer step in v2; set to None for converted v1 checkpoints.
            "pytorch-lightning_version": __version__ — the version identifier taken from the module namespace `__version__` and recorded to indicate the runtime/library version associated with the converted checkpoint.
            "state_dict": convert_state_dict_v1_to_v2(model_v1_dict) — a state dictionary of model tensors and parameter names converted from the v1 naming/format to the v2 naming/format; this is used by v2 to restore model weights for molecular property prediction networks.
            "loops": None — placeholder for PyTorch Lightning loop state in v2; left None for converted v1 checkpoints.
            "callbacks": None — placeholder for serialized callback states expected by v2; left None.
            "optimizer_states": None — placeholder for optimizer state dictionaries; left None because optimizer internals from v1 are not migrated here.
            "lr_schedulers": None — placeholder for learning-rate scheduler states; left None.
            "hparams_name": "kwargs" — a v2 metadata key set to the string "kwargs" to identify how hyperparameters are stored in the checkpoint.
            "hyper_parameters": convert_hyper_parameters_v1_to_v2(model_v1_dict) — a dictionary of hyperparameters converted from the v1 checkpoint format into the v2 expected format; these hyperparameters are used by v2 code to reconstruct model and training configuration.
    
    Behavior and side effects: This is a pure conversion function that constructs and returns a new dict; it does not write files or modify the input dict in place. It relies on the helper functions convert_state_dict_v1_to_v2 and convert_hyper_parameters_v1_to_v2 to translate v1-specific structures into v2 formats. Default placeholders (None) are intentionally used for v2 fields that do not have direct equivalents in v1; downstream v2 loading code should treat these as missing training-progress metadata. Failure modes: if model_v1_dict lacks expected v1 keys or contains unexpected formats, the helper converters may raise errors (for example, KeyError, TypeError, or ValueError); such exceptions are not caught here and will propagate to the caller.
    """
    from chemprop.utils.v1_to_v2 import convert_model_dict_v1_to_v2
    return convert_model_dict_v1_to_v2(model_v1_dict)


################################################################################
# Source: chemprop.utils.v1_to_v2.convert_state_dict_v1_to_v2
# File: chemprop/utils/v1_to_v2.py
# Category: fix_docstring
# Reason: Schema parsing failed: Cannot generate JSON schema for convert_state_dict_v1_to_v2 because the docstring has no description for the argument 'model_v1_dict'
################################################################################

def chemprop_utils_v1_to_v2_convert_state_dict_v1_to_v2(model_v1_dict: dict):
    """Converts a saved Chemprop v1 model dictionary (checkpoint) into a Chemprop v2 state dictionary suitable for loading into v2 model code. This utility is used during migration from Chemprop v1 to v2 (see README v1 -> v2 transition notes) for message passing neural networks applied to molecular property prediction; it remaps parameter key names and copies/reshapes certain metadata (for example, per-target output scaling for regression) so that v1 checkpoints can be consumed by v2 code.
    
    Args:
        model_v1_dict (dict): A v1 checkpoint dictionary as produced by torch.save/torch.load for Chemprop v1 models. This dictionary is expected to contain at least the keys "args" and "state_dict". "args" should be an object (typically argparse.Namespace) providing v1 hyperparameters accessed in the conversion (for example, ffn_num_layers, dataset_type, num_tasks, and optionally target_weights). "state_dict" should be a mapping from parameter name strings used in v1 (for example, "encoder.encoder.0.W_i.weight", "ffn.{i * 3 + 1}.weight" or "readout.{i * 3 + 1}.weight") to their parameter tensors. For regression models, model_v1_dict should also include "data_scaler" with "means" and "stds" arrays. This argument represents the entire stored model checkpoint from Chemprop v1 and is required for reconstructing a v2-compatible state dictionary for continued training or inference in v2.
    
    Returns:
        dict: A v2-style state dictionary mapping Chemprop v2 parameter name strings to tensors and other parameter-like objects that can be passed to a v2 model's load_state_dict. Key behaviors and contents:
          - Message passing weights are remapped from v1 keys:
              "encoder.encoder.0.W_i.weight" -> "message_passing.W_i.weight"
              "encoder.encoder.0.W_h.weight" -> "message_passing.W_h.weight"
              "encoder.encoder.0.W_o.weight" -> "message_passing.W_o.weight"
              "encoder.encoder.0.W_o.bias"   -> "message_passing.W_o.bias"
            These parameters are the learned linear transforms used in the v1 message passing encoder and in v2 are expected under the "message_passing" namespace for the graph neural network used in molecular property prediction.
          - Predictor feed-forward network (FFN/readout) parameters are remapped to "predictor.ffn.{layer}.{submodule}.(weight|bias)". The conversion handles the v1.6 rename from "ffn" to "readout": if a key like "readout.1.weight" exists in the v1 state_dict, parameters are read from the "readout" namespace; otherwise they are read from the "ffn" namespace. The code iterates over args.ffn_num_layers from the v1 args and uses the same layer indexing logic as v1 (suffix = 0 for the first layer, 2 for subsequent layers) to place weights/biases in the v2 predictor FFN layout. This preserves the predictor architecture mapping between v1 and v2.
          - For regression models (when args.dataset_type == "regression"), the v1 per-target normalization statistics are copied from model_v1_dict["data_scaler"]["means"] and ["stds"] into v2 keys:
              "predictor.output_transform.mean" and "predictor.output_transform.scale"
            These are converted to torch.float32 tensors and unsqueezed along dimension 0 to match the v2 module's expected shape. This preserves the practical significance of output scaling used by v1 for continuous property prediction.
          - Task weights for the predictor criterion are set under "predictor.criterion.task_weights". If args.target_weights exists in the v1 args (added in v1 change #183), those values are converted into a tensor and used; otherwise a tensor of ones with length args.num_tasks is created. The tensor is unsqueezed along dimension 0 to match v2 expectations.
          - The returned dictionary contains only the remapped keys described above (and their tensor values), not the full v1 checkpoint object. It is intended to be passed to a Chemprop v2 model's load_state_dict to restore learned parameters and relevant metadata for prediction.
    
    Behavior, defaults, and failure modes:
        - The function does not modify the input model_v1_dict; it constructs and returns a new dictionary (no in-place side effects).
        - The function relies on the presence and correct structure of model_v1_dict. If expected keys are missing (for example, "args", "state_dict", or "data_scaler" for regression) a KeyError will be raised by the function when attempting to read them. If the "args" object does not expose required attributes (ffn_num_layers, dataset_type, num_tasks) an AttributeError will be raised. If target_weights is present but not convertible to a tensor, a TypeError may be raised by torch.tensor.
        - The conversion explicitly handles both pre-v1.6 ("ffn") and v1.6+ ("readout") naming conventions for the predictor layers. For models that used a custom or nonstandard v1 checkpoint layout, the mapping may not be correct and manual inspection or a custom migration may be required.
        - The function uses torch.tensor to construct new tensors for output transform statistics and task weights; therefore the runtime environment must have PyTorch available. The dtype for copied means and stds is set to torch.float32 and the tensors are unsqueezed along dimension 0 to match v2 expectations.
        - Because the function recreates only the parameters and a few predictor-related metadata tensors, any other v1-only runtime metadata (custom training state, optimizer state, or deprecated keys) will not be present in the returned v2 state dict and must be handled separately if needed.
    
    Practical significance:
        - Use this function when migrating saved Chemprop v1 model checkpoints to Chemprop v2, for example to continue training, perform inference with v2 code, or ensemble models across versions. It automates the common key renames and metadata reshaping required by the v2 model implementation for molecular property prediction.
    """
    from chemprop.utils.v1_to_v2 import convert_state_dict_v1_to_v2
    return convert_state_dict_v1_to_v2(model_v1_dict)


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
