"""
Regenerated Google-style docstrings for module 'bioemu'.
README source: others/readme/bioemu/README.md
Generated at: 2025-12-02T00:16:13.444270Z

Total functions: 4
"""


################################################################################
# Source: bioemu.get_embeds.shahexencode
# File: bioemu/get_embeds.py
# Category: fix_docstring
# Reason: Schema parsing failed: Cannot generate JSON schema for shahexencode because the docstring has no description for the argument 's'
################################################################################

def bioemu_get_embeds_shahexencode(s: str):
    """bioemu.get_embeds.shahexencode computes and returns the lowercase hexadecimal SHA-256 digest of a given input string. In the BioEmu codebase this helper is intended for creating deterministic, filesystem-safe identifiers and cache keys used during embedding and MSA generation (for example, names derived from protein sequences, single-sequence FASTA content, MSA query URLs, or local paths used by the Colabfold setup described in the README).
    
    Args:
        s (str): The input text to encode. In the BioEmu domain this is typically a protein amino-acid sequence, a single-sequence FASTA record, an MSA query string, model or file path, or any other textual key used for embedding/metadata caching. The function obtains bytes by calling s.encode() (the source uses the default Python encoding call, which uses UTF-8 for str objects) and then computes the SHA-256 digest of those bytes.
    
    Returns:
        str: A lowercase hexadecimal string representing the SHA-256 digest of the UTF-8 encoding of s. The returned string is 64 hex characters long and is suitable for use as a deterministic identifier or cache key. Behavior and failure modes: the function is pure (no side effects) and deterministic (same s always yields the same output). If a non-str object is passed (contrary to the annotated signature), calling s.encode() may raise an AttributeError or TypeError; empty strings are valid and will produce the well-defined SHA-256 digest of the empty byte sequence. The SHA-256 digest provides a very low probability of accidental collisions, making it appropriate for cache/identifier purposes in BioEmu, but it is not presented here as a security mechanism. The implementation uses hashlib.sha256 internally as in the source code.
    """
    from bioemu.get_embeds import shahexencode
    return shahexencode(s)


################################################################################
# Source: bioemu.hpacker_setup.setup_hpacker.ensure_hpacker_install
# File: bioemu/hpacker_setup/setup_hpacker.py
# Category: fix_docstring
# Reason: Schema parsing failed: Cannot generate JSON schema for ensure_hpacker_install because the docstring has no description for the argument 'envname'
################################################################################

def bioemu_hpacker_setup_setup_hpacker_ensure_hpacker_install(
    envname: str = "hpacker",
    repo_dir: str = "/mnt/petrelfs/zhongzhanping/.hpacker"
):
    """Ensures that the HPacker tool and its runtime dependencies are installed inside a conda environment named by envname. This function is used by BioEmu's side-chain reconstruction and MD-relaxation pipeline to prepare an isolated conda environment containing hpacker and its dependencies (the hpacker conda environment is required before running bioemu.sidechain_relax). The function locates the conda installation, checks whether the target environment already exists, and if not invokes the hpacker installation script to create and populate the environment.
    
    Args:
        envname (str): Name of the conda environment to check or create. This is the environment under which hpacker and its dependencies will be installed; the default value in the function signature is 'hpacker'. In the BioEmu workflow this environment name is used to isolate hpacker and avoid contaminating the user's base environment. If this environment already exists under the detected conda prefix, the function performs no installation actions.
        repo_dir (str): Filesystem path passed to the hpacker install script where hpacker sources, caches, or related repository files should be placed; the default value in the function signature is '/mnt/petrelfs/zhongzhanping/.hpacker'. This path is forwarded to the external installation script (HPACKER_INSTALL_SCRIPT) and may be created or written to during installation.
    
    Returns:
        None: This function does not return a value. Its primary effects are side effects: locating the conda prefix via get_conda_prefix(), inspecting the conda "envs" directory for envname, and—if envname is absent—invoking the external HPACKER_INSTALL_SCRIPT via a subprocess call with arguments [\"bash\", HPACKER_INSTALL_SCRIPT, envname, repo_dir]. On successful installation the function exits normally. On failure the subprocess output is captured and an AssertionError is raised with the captured stdout/stderr included in the error message, so callers will see the install script log. Typical failure modes include missing or misconfigured conda (get_conda_prefix() may return an unexpected path or the envs directory may be absent), the install script not being present or not executable, network errors during package download, insufficient filesystem permissions or disk space to create the environment or write to repo_dir, or package installation/build failures reported by the install script. Note that, per the BioEmu README guidance, conda must be on PATH for the installation to succeed and installing GPU-accelerated dependencies may also require appropriate CUDA drivers (e.g., CUDA12) on the host. If you prefer a different environment name or install location before the first invocation of the side-chain pipeline, set the HPACKER_ENV_NAME environment variable or provide alternative arguments to this function as appropriate.
    """
    from bioemu.hpacker_setup.setup_hpacker import ensure_hpacker_install
    return ensure_hpacker_install(envname, repo_dir)


################################################################################
# Source: bioemu.seq_io.check_protein_valid
# File: bioemu/seq_io.py
# Category: fix_docstring
# Reason: Schema parsing failed: Cannot generate JSON schema for check_protein_valid because the docstring has no description for the argument 'seq'
################################################################################

def bioemu_seq_io_check_protein_valid(seq: str):
    """Checks that an input protein sequence string consists only of the canonical 20 IUPAC single-letter amino acid codes used by BioEmu.
    
    This function is a lightweight validator used throughout the BioEmu codebase (for example, prior to sampling structure ensembles and before embedding or MSA processing) to ensure that downstream components receive canonical protein sequences. It iterates over each character in the provided sequence string and verifies membership in the module-level IUPACPROTEIN set (the standard 20 amino acid single-letter codes). The function does not modify its input, does not access files, and is intended only for validating sequence contents; it does not parse FASTA files or accept sequence metadata.
    
    Args:
        seq (str): Protein sequence to validate. This must be a Python string where each character is the single-letter IUPAC code for one of the 20 standard amino acids. In the BioEmu sampling and embedding pipeline, this represents the primary-sequence input used to condition the generative model; providing non-canonical or ambiguous characters (for example characters outside the 20 IUPAC codes) will cause validation to fail and prevent downstream sampling/embedding steps.
    
    Returns:
        None: This function has no return value. Its practical effect is to raise an exception on invalid input to halt processing early and provide a clear error message; on success it simply returns to the caller allowing subsequent BioEmu processing to proceed.
    
    Raises:
        AssertionError: If any character in seq is not a member of the standard IUPAC 20 amino acid single-letter codes, an AssertionError is raised. The AssertionError message produced by the current implementation includes the offending character, for example "Sequence conteins non-valid protein character: X", where X is the first invalid character encountered. This early failure mode prevents malformed sequences from reaching sampling, embedding, or side-chain reconstruction routines that assume canonical residues.
    """
    from bioemu.seq_io import check_protein_valid
    return check_protein_valid(seq)


################################################################################
# Source: bioemu.utils.format_npz_samples_filename
# File: bioemu/utils.py
# Category: fix_docstring
# Reason: Schema parsing failed: Cannot generate JSON schema for format_npz_samples_filename because the docstring has no description for the argument 'start_id'
################################################################################

def bioemu_utils_format_npz_samples_filename(start_id: int, num_samples: int):
    """Format a canonical filename for a batch of sample records saved as a NumPy .npz file.
    
    This function is used by the BioEmu sampling pipeline to produce deterministic, parseable filenames for saved batches of generated protein-structure samples. The produced filename encodes the zero-padded start index of the first sample in the batch and the computed one-past-last (exclusive upper bound) index for the batch, allowing downstream tools and users to infer the global sample index range contained in the file.
    
    Args:
        start_id (int): The integer global index of the first sample in the batch. In the BioEmu sampling context, this is the starting position in the global sequence of samples produced for a given run (for example, when saving batches of generated protein backbone frames). This value is formatted with zero padding to seven digits. Passing a non-integer (e.g., a string or float) will raise a TypeError from the integer format specifier; negative integers are accepted by the formatter but will include a minus sign and may be semantically invalid for sample indices.
        num_samples (int): The integer number of samples contained in the batch. This is used to compute the exclusive upper bound of the batch index range as start_id + num_samples, which is the second index encoded in the filename. This value must be an integer; non-integer values will raise a TypeError when formatting. A value of zero is permitted and will produce a filename whose start and end indices are equal (an empty batch).
    
    Returns:
        str: A filename string following the pattern "batch_{start:07d}_{end:07d}.npz", where {start:07d} is start_id zero-padded to seven digits and {end:07d} is start_id + num_samples zero-padded to seven digits. Example: start_id=0 and num_samples=10 -> "batch_0000000_0000010.npz". Note that the implementation uses a fixed width of seven digits (zero-padded) and the code comment assumes this is sufficient for typical BioEmu runs (i.e., on the order of up to ~1,000,000 samples). Extremely large integers will still be converted to decimal strings without truncation but will exceed the intended zero-padded width. There are no side effects; the function only returns the formatted filename.
    """
    from bioemu.utils import format_npz_samples_filename
    return format_npz_samples_filename(start_id, num_samples)


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
