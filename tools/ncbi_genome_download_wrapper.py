"""
Regenerated Google-style docstrings for module 'ncbi_genome_download'.
README source: others/readme/ncbi_genome_download/README-CN.md
Generated at: 2025-12-02T00:36:36.100613Z

Total functions: 21
"""


################################################################################
# Source: ncbi_genome_download.core.argument_parser
# File: ncbi_genome_download/core.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def ncbi_genome_download_core_argument_parser(version: str = None):
    """ncbi_genome_download.core.argument_parser creates and returns a fully configured argparse.ArgumentParser for the ncbi-genome-download command-line interface. This parser defines the CLI arguments and help text used by the tool described in the README for downloading genome files from NCBI (for example: selecting taxonomic groups such as "bacteria" or "viral", choosing file formats like "fasta", filtering by taxids/genera/assembly levels, and controlling download behavior such as parallelism, caching, and progress display). The parser wiring uses NgdConfig.get_default(...) to populate default values and NgdConfig.get_choices(...) to populate choice lists for options where appropriate, and it declares deprecated aliases using DeprecatedAction. The returned parser is intended to be used by higher-level functions (for example, ncbi_genome_download.download()) to parse command-line arguments and drive the download workflow described in the README.
    
    Args:
        version (str): Optional string to use as the version text printed by the parser's -V/--version action. When provided, this value is passed directly to argparse as the version shown when the user invokes the version flag; when None, no custom version string is supplied (callers should pass a string when a meaningful version display is required). This parameter does not alter any other parser behavior.
    
    Detailed behavior and practical significance:
        The function constructs an argparse.ArgumentParser and registers a comprehensive set of options used by the ncbi-genome-download utility. Important behaviors configured by the parser include:
        - The positional argument groups selects one or more NCBI taxonomic groups to download; its default and allowed choices are provided by NgdConfig.get_default('groups') and NgdConfig.get_choices('groups') respectively. This directly controls which major organism categories (for example, bacteria, viral) the downstream download logic will operate on.
        - Multiple filtering options (section, file_formats, assembly_levels, genera, strains, species_taxids, taxids, assembly_accessions, refseq_categories, type_materials) are registered. Each of these uses NgdConfig.get_default(...) for a default value and, where appropriate, NgdConfig.get_choices(...) for valid choices. These flags are the primary mechanism for users to limit which assemblies and files are selected for download according to the README examples (for example, selecting only "complete" assembly_level or providing a comma-separated list of genera).
        - Boolean flags and behaviors such as --fuzzy-genus, --fuzzy-accessions, --flat-output, --human-readable, --progress-bar, --dry-run and --no-cache are provided to control matching behavior, output layout (human-readable hierarchy or flat dump), progress display, whether a dry-run should be performed (no genome files downloaded), and whether assembly summary caching is used. The --no-cache option's help references a module-level CACHE_DIR constant to explain where cached summaries would otherwise be stored.
        - Parallelism and retry behavior are exposed via --parallel and --retries; these options are typed (int) so the downstream download engine can use them to run multiple simultaneous downloads and to retry on transient NCBI connection failures.
        - A metadata output option (--metadata-table) allows writing a tab-delimited table with genome metadata as a side-effect instead of or in addition to writing raw genome files.
        - The parser includes deprecated aliases via DeprecatedAction (for example --genus -> --genera and --refseq-category -> --refseq-categories). If DeprecatedAction is unavailable or misconfigured, constructing the parser may raise an error at runtime.
        - Help texts for many options explicitly state default values (populated by NgdConfig) and acceptable comma-separated list formats, matching examples shown in the README (for example, "fasta,assembly-report" for formats or "complete,chromosome" for assembly levels). These help strings are used by users to understand how to invoke the tool to reproduce common workflows described in the README.
    
    Side effects and environment dependencies:
        - The function reads configuration defaults and choice lists from NgdConfig via NgdConfig.get_default(...) and NgdConfig.get_choices(...). If NgdConfig raises exceptions when queried (for example, due to missing configuration data), this function will propagate those exceptions.
        - The function references module-level names such as DeprecatedAction and CACHE_DIR; those symbols must exist in the module environment when this function is called or a NameError will be raised.
        - The function only constructs and returns the parser; it does not perform any file I/O, network access, or parsing of command-line arguments. No downloads or state changes occur as a result of calling this function alone.
    
    Failure modes:
        - If NgdConfig or other referenced symbols (DeprecatedAction, CACHE_DIR) are not defined or raise errors, argument_parser will raise the underlying exception (NameError, AttributeError, or the original NgdConfig error).
        - If callers pass a non-string value for version, argparse may raise a TypeError or produce unexpected output when the version flag is used; therefore pass a str or None as per the signature.
        - Invalid or inconsistent configuration returned by NgdConfig (for example, get_choices returning values inconsistent with downstream expectations) will not be validated here beyond argparse's built-in choice enforcement; such issues may cause runtime filtering to select no assemblies or to behave unexpectedly during actual downloads.
    
    Returns:
        argparse.ArgumentParser: A configured ArgumentParser instance that exposes the complete set of command-line options used by the ncbi-genome-download tool. The caller should invoke parser.parse_args(...) (or equivalent) to obtain parsed argument values and then pass those values into the download logic described in the README.
    """
    from ncbi_genome_download.core import argument_parser
    return argument_parser(version)


################################################################################
# Source: ncbi_genome_download.core.convert_ftp_url
# File: ncbi_genome_download/core.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def ncbi_genome_download_core_convert_ftp_url(url: str):
    """ncbi_genome_download.core.convert_ftp_url converts an FTP URL string into an HTTPS URL string by replacing the initial "ftp://" scheme with "https://".
    
    Args:
        url (str): The input URL string representing a resource location commonly produced or encountered when mirroring or referencing NCBI genome FTP directories (for example an assembly or sequence file location). In the ncbi-genome-download workflow this function is used to normalize URLs so consumers that prefer or require HTTPS mirrors can request the same resource over HTTPS. The function expects a Python str containing the full URL; it performs a literal, case-sensitive substitution of the first occurrence of the substring 'ftp://' with 'https://'. If the input already begins with 'https://' or with a different scheme (for example 'http://', 'sftp://', or 'ftps://'), the string will be left unchanged except where the exact lowercase 'ftp://' appears as the first occurrence. The function does not validate the rest of the URL, resolve hostnames, or perform network operations.
    
    Returns:
        str: A new URL string produced by replacing the first exact, case-sensitive occurrence of the prefix 'ftp://' with 'https://'. This return value is intended for use by HTTP/HTTPS download code in the ncbi-genome-download project so that downloads can be attempted over HTTPS mirrors instead of FTP. No side effects occur; the original input string is not modified. If a non-string object is passed for url, the call will fail at runtime (for example, by raising an AttributeError when attempting to call replace on a non-str), since only str is supported.
    """
    from ncbi_genome_download.core import convert_ftp_url
    return convert_ftp_url(url)


################################################################################
# Source: ncbi_genome_download.core.create_dir
# File: ncbi_genome_download/core.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def ncbi_genome_download_core_create_dir(
    entry: dict,
    section: str,
    domain: str,
    output: str,
    flat_output: bool
):
    """ncbi_genome_download.core.create_dir creates (if necessary) and returns the filesystem directory used to store files for a single genome assembly entry when downloading NCBI genomes with ncbi-genome-download. In the ncbi-genome-download workflow this function determines the output location for a download based on the requested section (for example 'refseq' vs 'genbank' indicated via the --section option in the tool), the domain or dataset grouping used by the tool (for example groupings like 'bacteria' or 'viral' shown in the README usage), and the assembly accession identifier found in the provided entry. The directory returned is intended to hold all files for that single assembly and mirrors the logical layout used by the downloader unless a flat output layout is requested.
    
    Args:
        entry (dict): A metadata dictionary describing a single assembly entry from the NCBI assembly summary. This function expects entry to contain the key 'assembly_accession' whose string value is used as the final path component when flat_output is False. The assembly_accession is the unique assembly identifier (for example "GCF_000005845.2") and is critical for creating per-assembly subdirectories so downloaded files for different assemblies do not collide.
        section (str): The dataset section name used by ncbi-genome-download (for example 'refseq' or 'genbank' as set with the --section option). This value is incorporated into the output path to separate files for different NCBI dataset sections and to reproduce a readable directory hierarchy for downstream users.
        domain (str): The biological domain or group name that categorizes the genome set being downloaded (for example group names such as 'bacteria', 'viral', or other groupings described in the README). This value is used as an intermediate path component so files are organized by domain/group.
        output (str): The base output directory path supplied by the caller where downloaded files and per-assembly directories should be created. This string is joined with section, domain, and assembly_accession (unless flat_output is True) to form the final directory path. The path may be relative or absolute as provided; no normalization beyond os.path.join is performed.
        flat_output (bool): When False (the default layout used by the downloader), the function creates a nested directory structure os.path.join(output, section, domain, entry['assembly_accession']) so each assembly has its own folder. When True, the function does not create an assembly-specific subdirectory and uses the provided output path directly, placing all files in a single flat directory. Use flat_output to request a non-hierarchical layout when desired.
    
    Returns:
        str: The filesystem path (as a string) to the directory that was created or that already existed and will be used to store files for the given assembly entry. This is the exact path passed to os.makedirs (either output or os.path.join(output, section, domain, assembly_accession)) and should be used by callers as the target directory for moving or writing downloaded files.
    
    Behavior, side effects, and failure modes:
        The function has the side effect of creating directories on the filesystem using os.makedirs, creating any missing intermediate directories in the path. If the target directory already exists and is a directory, the function is a no-op for creation and returns the existing directory path. If the path exists but is not a directory (for example a file with the same name), or if the filesystem raises an error other than EEXIST for an existing directory, the underlying OSError is re-raised to the caller. If entry does not contain the 'assembly_accession' key when flat_output is False, a KeyError will be raised before any directory creation. Other possible failure modes include permission errors, insufficient disk space, or other filesystem-level OSError conditions which will propagate to the caller. The caller should handle these exceptions appropriately in the context of parallel downloads or retries.
    """
    from ncbi_genome_download.core import create_dir
    return create_dir(entry, section, domain, output, flat_output)


################################################################################
# Source: ncbi_genome_download.core.create_readable_dir
# File: ncbi_genome_download/core.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def ncbi_genome_download_core_create_readable_dir(
    entry: dict,
    section: str,
    domain: str,
    output: str
):
    """Create the a human-readable directory for a single assembly entry and return its full path.
    
    This function is used by the ncbi-genome-download tool when the user requests a human-readable layout (for example via the --human-readable option described in the README). It constructs a directory path that parallels a readable, taxonomically organized view of NCBI assemblies and ensures that the directory exists on disk. The created directory is intended to be the location where link files (or other human-oriented files) for the given assembly entry can be placed; creating the directory itself does not create links. For non-viral domains the path is built from genus, species and strain labels derived by calling get_genus_label(entry), get_species_label(entry) and get_strain_label(entry). For the viral domain the path uses the entry['organism_name'] (with spaces replaced by underscores) and get_strain_label(entry, viral=True) instead.
    
    Args:
        entry (dict): A dictionary representing a single assembly/entry from an NCBI assembly summary. This dict must contain the fields expected by the label helper functions used here (for example 'organism_name' is required when domain == 'viral'). In practical use within ncbi-genome-download, entry is an assembly summary line parsed into a dict and provides the taxonomic and naming information used to form genus, species and strain labels that are meaningful to users browsing downloaded genomes.
        section (str): The NCBI section name (for example 'refseq' or 'genbank') used as part of the human-readable directory hierarchy. This string appears verbatim as a directory component under the top-level human_readable directory so that the resulting path reflects the source section of the assembly.
        domain (str): The high-level domain/group name (for example 'bacteria', 'fungi', 'viral') used as part of the directory hierarchy. When domain == 'viral' a different branch is taken: the organism name from entry['organism_name'] (spaces replaced with underscores) is used instead of genus/species labels, and get_strain_label is called with viral=True. The domain value is used to group assemblies by biological domain in the human-readable layout.
        output (str): The base output directory (filesystem path) under which the human_readable directory tree will be created. The final directory is constructed by joining output, 'human_readable', section, domain and the taxonomic labels described above. This must be a writable path on the host filesystem for directory creation to succeed.
    
    Returns:
        str: The absolute (or relative, depending on the provided output) filesystem path to the created human-readable directory for this entry (the variable full_output_dir in the implementation). This path is returned after ensuring the directory exists.
    
    Raises:
        OSError: If os.makedirs fails for reasons other than the directory already existing (for example permission denied, invalid path, or other filesystem errors), the original OSError is propagated. If the directory already exists and is a directory, the function silently succeeds. Note that this function only creates the directory structure — callers should handle creation of any links or files placed within the returned directory. Also be aware of cross-platform considerations mentioned in the project README: subsequent creation of symbolic links into this directory may not be supported on some Windows filesystems or older Windows versions.
    """
    from ncbi_genome_download.core import create_readable_dir
    return create_readable_dir(entry, section, domain, output)


################################################################################
# Source: ncbi_genome_download.core.create_symlink
# File: ncbi_genome_download/core.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def ncbi_genome_download_core_create_symlink(local_file: str, symlink_path: str):
    """Create a relative symbolic link for a downloaded genome file when a symlink path is provided.
    
    This function is used by the ncbi-genome-download package to create human-readable directory layouts that mirror the NCBI FTP/assembly structure (the --human-readable option). Instead of copying genome files into those human-readable locations, this function creates a relative symbolic link that points from the requested symlink location back to the actual saved file in the output folder, saving disk space and enabling a directory structure that is easy for users to inspect. The function normalizes the provided paths, removes any existing file or link at the target symlink path, computes a relative path from the symlink location up to the output folder, and creates a symlink pointing to the normalized local file. If no symlink path is given, the function does nothing and returns success. Note that some platforms and filesystems (for example, some Windows configurations) may not support symbolic links or may require elevated privileges; ncbi-genome-download's README documents this limitation.
    
    Args:
        local_file (str): Relative path to the actual file saved in the download output folder. This value is expected to be a filesystem path string (for example beginning with './' as used by the calling code) that points to the downloaded genome file within the output directory structure. The function will normalize this path (os.path.normpath) before constructing the relative link target.
        symlink_path (str): Relative path where the symbolic link should be created within the human-readable directory layout. This is a filesystem path string (for example beginning with './' as used by the calling code) describing the location of the desired symbolic link. If symlink_path is None, the function will skip symlink creation and return success; if a file or link already exists at symlink_path, it will be unlinked (removed) before the new symlink is created.
    
    Returns:
        bool: success code. Returns True after successfully performing normalization, removal of any existing target at symlink_path, and creation of the relative symbolic link (or after doing nothing when symlink_path is None). The function performs filesystem side effects: it may remove an existing file or link at symlink_path and will create a new symlink pointing to the computed relative path to local_file. On failure (for example permission errors, unsupported filesystem for symlinks, or other OS-level errors), the underlying os.unlink or os.symlink calls will raise an exception (e.g., OSError, PermissionError), which propagates to the caller.
    """
    from ncbi_genome_download.core import create_symlink
    return create_symlink(local_file, symlink_path)


################################################################################
# Source: ncbi_genome_download.core.create_symlink_job
# File: ncbi_genome_download/core.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def ncbi_genome_download_core_create_symlink_job(
    directory: str,
    checksums: dict,
    filetype: str,
    symlink_path: str
):
    """Create a DownloadJob configured to create a filesystem symlink for an already-downloaded genome file.
    
    This function is used by the ncbi-genome-download core to build a job that, when executed by the download/job runner, will create a symbolic link in a human-readable directory layout that points to an already-downloaded local file. It is intended for use with the "--human-readable" behavior described in the README where directory trees mirror NCBI layout and links are used to avoid duplicating downloaded files.
    
    Args:
        directory (str): The local directory path where the downloaded file currently resides. This is joined with the resolved filename to form the absolute path to the existing file (local_file). In the download pipeline this typically corresponds to the on-disk folder that already contains the downloaded genome artifact.
        checksums (dict): A mapping used to locate the downloaded filename and its checksum. The function calls get_name_and_checksum(checksums, pattern) to find the filename that matches the file-ending pattern for the requested filetype. Practically, this dict should be the checksums/assembly-report-derived mapping of filenames to checksum values produced during the download/scan phase; if no entry matching the pattern exists, the underlying get_name_and_checksum call will raise an exception which is propagated to the caller.
        filetype (str): A short identifier for the requested file format (for example the formats handled by ncbi-genome-download such as "fasta" or "genbank"). The function uses NgdConfig.get_fileending(filetype) to compute the filename pattern/ending to look up in checksums so the correct file variant is chosen for symlinking.
        symlink_path (str): The target directory path where the created symbolic link should live (the human-readable layout). The returned DownloadJob will be configured to create a symlink at os.path.join(symlink_path, filename) that points to the local file at os.path.join(directory, filename). This function does not validate that symlink_path exists; such validation or directory creation, if required, must be handled by the caller or by the job runner.
    
    Returns:
        DownloadJob: A DownloadJob instance configured to create a symlink rather than download content. The returned job is constructed as DownloadJob(None, local_file, None, full_symlink) where the job's source URL and checksum fields are set to None to indicate no remote fetch is required; local_file is the existing file to link to and full_symlink is the path to the symlink to be created. The function itself has no filesystem side effects (it does not create the symlink); executing the returned DownloadJob is what performs the symlink creation. Exceptions from NgdConfig.get_fileending, get_name_and_checksum, or the DownloadJob constructor are propagated to the caller.
    """
    from ncbi_genome_download.core import create_symlink_job
    return create_symlink_job(directory, checksums, filetype, symlink_path)


################################################################################
# Source: ncbi_genome_download.core.download_file_job
# File: ncbi_genome_download/core.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def ncbi_genome_download_core_download_file_job(
    entry: dict,
    directory: str,
    checksums: dict,
    filetype: str = "genbank",
    symlink_path: str = None
):
    """Generate a DownloadJob configured to download a single file for an NCBI assembly entry.
    
    This function is used by the ncbi-genome-download pipeline to turn a parsed assembly metadata entry (from an NCBI assembly summary) and a pre-parsed checksum mapping into a DownloadJob object that worker code can execute to actually fetch a genome-related file (for example a GenBank, FASTA, or assembly-report file) from the NCBI FTP site. It does not perform any network I/O itself; it only constructs the download URL, determines the local target path, looks up the expected checksum for integrity verification, and optionally computes a human-readable symlink target path that mirrors NCBI's directory layout.
    
    Args:
        entry (dict): A dictionary representing one assembly metadata row as used by the ncbi-genome-download pipeline. This dict MUST contain the key 'ftp_path' whose value is the original NCBI FTP directory URL for the assembly (for example "ftp://ftp.ncbi.nlm.nih.gov/.../GCF_..."). The function uses entry['ftp_path'] to derive the base HTTP/FTP URL from which the file will be fetched. Missing or malformed 'ftp_path' will raise a KeyError or cause URL conversion functions to fail.
        directory (str): Local filesystem directory path where the downloaded file should be saved. The function will join this directory with the chosen filename to form the local target path passed into the returned DownloadJob. This function does not create directories or verify writability; invalid or non-writable directories will cause downstream download operations to fail when the DownloadJob is executed.
        checksums (dict): A mapping of filenames or checksum-record keys to checksum values parsed from the assembly's checksum file. The function consults this mapping (via get_name_and_checksum with a pattern derived from filetype) to select the exact filename to download and the expected checksum string used for integrity verification. If no matching filename is found for the requested filetype pattern, get_name_and_checksum will raise an exception (e.g., ValueError), which propagates from this function.
        filetype (str): The requested file format/role to download, expressed as a canonical name used by NgdConfig.get_fileending (default: "genbank"). Typical values used by the ncbi-genome-download tool include "genbank", "fasta", "assembly-report", etc. This determines the pattern/extension used to locate the correct filename in checksums and to build the remote filename. The default "genbank" means the function will look for the GenBank-format assembly file entry in the checksums mapping.
        symlink_path (str): Optional local filesystem directory path where a human-readable symlink should be created to mirror the NCBI directory layout. If provided (not None), the doc job will be created with a symlink target (full_symlink) equal to os.path.join(symlink_path, filename). The function does not create the symlink itself; it only records the intended symlink path inside the returned DownloadJob. If None (the default), no symlink target is set on the DownloadJob.
    
    Returns:
        DownloadJob: An instance of DownloadJob initialized with four pieces of information: the full remote URL to download (constructed from entry['ftp_path'] and the chosen filename), the local file path (os.path.join(directory, filename)), the expected checksum string for integrity verification (from checksums), and an optional full_symlink path (os.path.join(symlink_path, filename) or None). The DownloadJob is a declarative object used by the rest of the ncbi-genome-download code to perform and validate the actual file transfer; this function does not perform the transfer itself.
    
    Behavior, side effects, defaults, and failure modes:
        This function performs only deterministic, side-effect-free string and path computations and calls helper utilities: NgdConfig.get_fileending(filetype) to determine the file-ending/pattern for the requested filetype; get_name_and_checksum(checksums, pattern) to select the exact filename and expected checksum; and convert_ftp_url(entry['ftp_path']) to convert the NCBI FTP directory URL into a usable base URL. Side effects: none (it does not create files, directories, or network connections). Default behavior: if filetype is omitted, 'genbank' is used. Failure modes: KeyError if entry lacks 'ftp_path'; exceptions from NgdConfig.get_fileending, get_name_and_checksum, or convert_ftp_url if inputs are invalid; downstream failures when the returned DownloadJob is executed if the local directory is non-existent or not writable, or if network access to the constructed URL fails; checksum mismatches will be detected later when the DownloadJob is executed.
    """
    from ncbi_genome_download.core import download_file_job
    return download_file_job(entry, directory, checksums, filetype, symlink_path)


################################################################################
# Source: ncbi_genome_download.core.downloadjob_creator_caller
# File: ncbi_genome_download/core.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def ncbi_genome_download_core_downloadjob_creator_caller(args: tuple):
    """ncbi_genome_download.core.downloadjob_creator_caller: Thin wrapper used to invoke the internal create_downloadjob function by unpacking a tuple of positional arguments. This function exists to provide a single-argument callable suitable for use as the target function in multiprocessing worker pools (for example, multiprocessing.Pool.map or Pool.imap) when building parallel download jobs for NCBI genome assemblies. In the ncbi-genome-download toolchain this wrapper is used to convert an iterable of argument tuples into calls that construct download job descriptions for assemblies (the objects/records that represent work items used to download genome files such as FASTA or GenBank from NCBI FTP locations).
    
    Args:
        args (tuple): A tuple of positional arguments to be unpacked and passed directly to ncbi_genome_download.core.create_downloadjob via create_downloadjob(*args). The exact contents and order of this tuple must match the signature expected by create_downloadjob; in typical usage these positional values represent the metadata and parameters required to create a download job for a genome assembly (for example: an assembly summary row or identifiers, the FTP path to the assembly files, local output directory, requested formats/filters, and any other parameters used by create_downloadjob). This wrapper does not validate or mutate the tuple; it only unpacks it. Because this function is intended for use with multiprocessing, the tuple and its contents must be picklable if passed between processes. If args is not an iterable suitable for unpacking or does not match the expected arity for create_downloadjob, a TypeError (or the error raised by create_downloadjob) will propagate.
    
    Returns:
        object: The exact return value produced by ncbi_genome_download.core.create_downloadjob(*args). In the ncbi-genome-download domain this return value is the created download-job descriptor or result produced by create_downloadjob (the item that the worker pool produces for subsequent processing or execution of the actual download). No additional transformation is applied by this wrapper; any side effects, logging, or exceptions are those of create_downloadjob. If create_downloadjob raises an exception (for example due to invalid arguments, I/O errors when inspecting remote FTP paths, or other validation failures), that exception propagates to the caller of downloadjob_creator_caller.
    """
    from ncbi_genome_download.core import downloadjob_creator_caller
    return downloadjob_creator_caller(args)


################################################################################
# Source: ncbi_genome_download.core.get_genus_label
# File: ncbi_genome_download/core.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def ncbi_genome_download_core_get_genus_label(entry: dict):
    """ncbi_genome_download.core.get_genus_label returns the genus (first token of the organism name) for an NCBI assembly summary entry. This function is used by the ncbi-genome-download tool to extract the genus string from an assembly entry's organism_name field so that entries can be grouped, filtered, or matched against user-specified genera (for example the --genera option described in the README).
    
    Args:
        entry (dict): A single assembly summary entry as provided by the NCBI assembly summary files or by upstream parsing code in ncbi_genome_download. The entry is expected to be a mapping that contains the key 'organism_name' whose value is the taxonomic organism name string (for example "Escherichia coli"). The practical role of this parameter is to supply the full organism name from which the genus is extracted; the function does not validate or enrich taxonomy data beyond simple string parsing.
    
    Returns:
        str: The genus name extracted from entry['organism_name'] by splitting the string on the ASCII space character (' ') and returning the first token. For a typical well-formed organism_name like "Escherichia coli" this returns "Escherichia". This return value is used within the download tool to match and filter assemblies by genus and to build human-readable directory labels.
    
    Behavior, side effects, defaults, and failure modes:
        This function performs no I/O and has no side effects; it is a pure string-extraction helper. It uses the literal split(' ') operation on the organism_name value, which means that leading spaces or multiple consecutive spaces can produce empty string tokens (for example a leading space yields an empty first token). If the organism_name contains only a single word, that word is returned as the genus. The function does not perform any taxonomic validation, normalization, or lookup against NCBI taxonomic databases — it only returns the textual first token. If the entry does not contain the 'organism_name' key a KeyError will be raised. If entry is not a mapping type or if entry['organism_name'] is not a string, a TypeError or AttributeError may occur. Callers that require robust behavior should ensure the entry includes a cleaned, non-empty organism_name (for example by trimming whitespace) before calling this function.
    """
    from ncbi_genome_download.core import get_genus_label
    return get_genus_label(entry)


################################################################################
# Source: ncbi_genome_download.core.get_name_and_checksum
# File: ncbi_genome_download/core.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def ncbi_genome_download_core_get_name_and_checksum(checksums: list, end: str):
    """Extract a full filename and its checksum from a checksums list for a file whose name ends with the given suffix.
    
    This function is used by the ncbi-genome-download tool to identify the exact archive or sequence file name and the associated checksum entry from a parsed checksums listing (the checksums metadata that accompanies NCBI assembly or FTP directory listings). The returned filename and checksum are intended for use when verifying a downloaded file against the authoritative checksum provided by NCBI. The function handles typical NCBI file-naming edge cases such as CDS and RNA FASTA files that themselves include the substring used by plain genomic FASTA names, avoiding false positive matches.
    
    Args:
        checksums (list): A list of checksum entries as produced from parsing an NCBI checksums/MD5 file or equivalent metadata. Each entry is expected to be a mapping-like object (dictionary) with at least the keys 'file' and 'checksum', where 'file' is the filename string found on the NCBI FTP/assembly directory and 'checksum' is the corresponding checksum string. The function scans this list in order and returns the first entry whose 'file' value matches the suffix constraint described by end.
        end (str): The filename suffix to match (for example a file ending such as '.fna.gz' or a format-specific ending returned by NgdConfig.get_fileending()). This is matched using str.endswith, so it must be the literal trailing substring expected at the end of the target filename. The value is typically one of the file-ending identifiers used by ncbi_genome_download (for example the values returned from NgdConfig.get_fileending('cds-fasta') or NgdConfig.get_fileending('rna-fasta')).
    
    Behavior and side effects:
        The function iterates over the provided checksums list and selects the first entry whose 'file' value ends with the provided end string. To avoid incorrect matches for filenames like '..cds_from_genomic.fna.gz' or '..rna_from_genomic.fna.gz' that also end in a more general genomic suffix, the function explicitly excludes entries that end with the CDS or RNA fasta endings when the requested end is a different (more general) suffix. This logic relies on NgdConfig.get_fileending('cds-fasta') and NgdConfig.get_fileending('rna-fasta') to obtain canonical CDS/RNA endings. There are no external side effects: the function does not modify the input list or global state and does not perform I/O.
    
    Defaults:
        No default values are used; both arguments are required.
    
    Failure modes:
        If no entry in checksums has a 'file' value that satisfies the matching rules, the function raises ValueError('No entry for file ending in {!r}'.format(end)). If entries in checksums do not have the expected 'file' or 'checksum' keys, a KeyError may be raised during access. The caller should ensure the checksums list is derived from a properly parsed NCBI checksum/MD5 listing and that end is the exact suffix to match.
    
    Returns:
        tuple: A 2-tuple (filename, expected_checksum). filename (str) is the full filename string from the matching checksums entry as found on the NCBI FTP/assembly directory. expected_checksum (str) is the corresponding checksum string from that entry. These values are intended for downstream verification of a downloaded file against the authoritative checksum.
    """
    from ncbi_genome_download.core import get_name_and_checksum
    return get_name_and_checksum(checksums, end)


################################################################################
# Source: ncbi_genome_download.core.get_species_label
# File: ncbi_genome_download/core.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def ncbi_genome_download_core_get_species_label(entry: dict):
    """ncbi_genome_download.core.get_species_label: Extract the species label (the species epithet) from an NCBI assembly summary entry.
    
    This function is used in the ncbi-genome-download tool to derive a simple species identifier from the assembly summary "organism_name" field provided by NCBI. The typical "organism_name" value is a scientific name string such as "Escherichia coli str. K-12" or "Streptomyces coelicolor A3(2)". The function performs a simple whitespace split of that string and returns the second token (index 1), which represents the species epithet in common two-word scientific names. This lightweight parsing is useful for creating human-readable directory labels, grouping downloaded assemblies by species, or generating short species tags when building NCBI mirror-like directory structures as described in the project README.
    
    Args:
        entry (dict): A single assembly summary entry represented as a dictionary (as produced by parsing NCBI assembly_summary files in ncbi-genome-download). This dictionary MUST contain the key 'organism_name' whose value is expected to be the organism scientific name as a string. The function splits entry['organism_name'] on spaces and returns the second whitespace-separated token as the species label. If the provided entry is missing 'organism_name' or that value is not a string, attempting to call this function will raise the underlying Python exception (for example, KeyError if the key is absent or AttributeError/TypeError if the value does not support split). If 'organism_name' is present but contains fewer than two whitespace-separated tokens (for example an empty string or a single-word name), the function returns the placeholder string 'sp.' to indicate that no distinct species epithet could be extracted.
    
    Returns:
        str: The species label extracted from entry['organism_name'] (the second token after splitting on spaces). If the organism_name contains fewer than two tokens, returns the placeholder 'sp.'. There are no other side effects; the function does not modify the input dictionary. Note that this is a simple string-based extraction and does not perform taxonomic validation against NCBI taxonomic identifiers.
    """
    from ncbi_genome_download.core import get_species_label
    return get_species_label(entry)


################################################################################
# Source: ncbi_genome_download.core.get_strain
# File: ncbi_genome_download/core.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def ncbi_genome_download_core_get_strain(entry: dict, viral: bool = False):
    """ncbi_genome_download.core.get_strain: Extract a strain identifier string from a single NCBI assembly summary entry.
    
    This function is used by the ncbi-genome-download tool to derive a human-meaningful strain label from an assembly summary entry returned by NCBI. The derived strain label is typically used for naming output files or organizing downloaded genomes by strain. The function examines the entry in a prioritized order to find the most specific available strain information: it first inspects the 'infraspecific_name' field (handling values of the form "key=value" by taking the value after the final '='), then the 'isolate' field, and finally, when the entry's 'organism_name' consists of more than two whitespace-separated words and the viral flag is False, it treats all words from the third onward as the strain (joining them with single spaces). If none of these produce a non-empty string, the function falls back to returning the assembly accession from the 'assembly_accession' field. Empty strings in the checked fields are treated as absent. The viral parameter disables the organism_name-based extraction because viral organism names should not be parsed in this way for strain derivation. The function performs no I/O and has no side effects; it operates purely on the provided dictionary and returns a string.
    
    Args:
        entry (dict): Assembly summary entry as a Python dictionary corresponding to one line/record from an NCBI assembly summary file. This dictionary is expected to contain at least the keys 'infraspecific_name', 'isolate', 'organism_name', and 'assembly_accession' (access via entry['key'] is performed). The practical significance is that these keys are the standard columns from NCBI assembly metadata used to determine strain-level labels for genome downloads; if any of these keys are missing, a KeyError will be raised by the function.
        viral (bool): Flag indicating whether the entry represents a viral genome. When False (default), the function will attempt to extract a strain from 'organism_name' when that name contains more than two words by joining words from the third onward. When True, the function will not use 'organism_name' for strain extraction, which is important in the viral domain where organism_name tokenization does not reliably represent strain. This parameter has no side effects beyond controlling the extraction logic.
    
    Returns:
        str: A string identifying the strain to be used for naming or grouping the assembly. This will be, in priority order: the value after the last '=' in entry['infraspecific_name'] if non-empty; entry['isolate'] if non-empty; the concatenation of words from the third onward of entry['organism_name'] (joined by single spaces) if that name has more than two words and viral is False; otherwise entry['assembly_accession']. The function returns the assembly accession as a last-resort identifier to ensure a non-empty, unique label. Possible failure modes: KeyError if required keys are missing from entry; the function does not validate or normalize characters beyond the described splitting and joining.
    """
    from ncbi_genome_download.core import get_strain
    return get_strain(entry, viral)


################################################################################
# Source: ncbi_genome_download.core.get_strain_label
# File: ncbi_genome_download/core.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def ncbi_genome_download_core_get_strain_label(entry: dict, viral: bool = False):
    """Clean up and normalize a strain name extracted from an NCBI assembly metadata entry so it can be safely used in file names and directory names created by ncbi-genome-download.
    
    Args:
        entry (dict): A dictionary representing a single assembly/record metadata entry as produced or consumed by ncbi-genome-download internal code paths (for example, parsed rows from an NCBI assembly summary). This function calls the package-internal get_strain(entry, viral) to extract the raw strain name from this metadata dictionary. The practical role of entry is to provide the original strain information as supplied by NCBI so that the returned label can be used when saving genome files or building human-readable directory structures described in the README.
        viral (bool): When True, instructs the internal get_strain call to use viral-specific extraction logic (if any) for entries from viral groups; when False (the default) standard non-viral strain extraction is used. This flag mirrors the group-specific behavior used elsewhere in ncbi-genome-download when selecting and organizing viral versus non-viral genomes.
    
    Returns:
        str: A sanitized strain label derived from the strain value returned by get_strain(entry, viral). The returned string has leading and trailing whitespace removed and the following characters replaced with underscores: space (' '), semicolon (';'), forward slash ('/'), and backslash ('\\'). The returned label is intended for practical use as a filesystem- and filename-friendly identifier when storing downloaded genomes (for example, as part of filenames or directory names created by ncbi-genome-download).
    
    Behavior and failure modes:
        This function performs no I/O and has no side effects on the entry dictionary; it only calls get_strain(entry, viral) and applies in-memory text normalization. If get_strain returns a non-string value (for example, None) or entry lacks the expected fields, cleanup will attempt to call string methods and may raise exceptions such as AttributeError or TypeError, or get_strain may raise KeyError or other errors originating from metadata extraction. The function only replaces the specific characters listed above; other potentially problematic characters are left unchanged by this function and should be handled elsewhere if further sanitization is required. Default behavior corresponds to viral=False.
    """
    from ncbi_genome_download.core import get_strain_label
    return get_strain_label(entry, viral)


################################################################################
# Source: ncbi_genome_download.core.get_summary
# File: ncbi_genome_download/core.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def ncbi_genome_download_core_get_summary(
    section: str,
    domain: str,
    uri: str,
    use_cache: bool
):
    """Get the NCBI assembly_summary.txt for a given section and domain and return it as an in-memory text stream.
    
    This function is used by the ncbi-genome-download tool (see README) to obtain the assembly summary file that lists available assemblies for a given group (for example "bacteria", "viral") and domain (for example "refseq", "genbank"). The returned StringIO contains the full contents of the NCBI assembly_summary.txt file (tab-delimited metadata used by ncbi-genome-download to filter assemblies by taxid, assembly level, refseq category, etc.). The function implements a simple caching mechanism to avoid repeated downloads within a 1-day window and logs its actions to the "ncbi-genome-download" logger.
    
    Args:
        section (str): The top-level NCBI section name used in the URL path and cache filename (for example "bacteria" or "viral"). This selects which organism group’s assembly_summary.txt to fetch and is used to form the cache filename "{section}_{domain}_assembly_summary.txt" stored under the module CACHE_DIR.
        domain (str): The NCBI domain segment used in the URL path and cache filename (for example "refseq" or "genbank"). Combined with section to determine both the remote path "{uri}/{section}/{domain}/assembly_summary.txt" and the local cache filename.
        uri (str): The base URI of the NCBI FTP/HTTP host to query (for example "ftp://ftp.ncbi.nlm.nih.gov/genomes" or an HTTP mirror). This value is concatenated with section and domain to form the full URL that is requested via requests.get.
        use_cache (bool): Whether to use and maintain a local cache copy of the assembly summary. If True, the function will: check for an existing cache file in CACHE_DIR named "{section}_{domain}_assembly_summary.txt" and use it if it exists and is newer than 1 day; if no valid cache exists it will download the remote file and write it to that cache file (creating CACHE_DIR if necessary). If False, the function will always fetch from the remote URL and will not read or write the cache.
    
    Behavior, side effects, defaults, and failure modes:
        - Cache naming and location: the cache filename is "{section}_{domain}_assembly_summary.txt" and is placed in the module-level CACHE_DIR. The cache freshness threshold is one day (timedelta(days=1)); files older than this are treated as stale.
        - When use_cache is True and a fresh cache exists, the function reads the cache using codecs.open with encoding="utf-8" and returns a StringIO constructed from the cached Unicode text. When use_cache is True and no fresh cache exists, the function issues an HTTP(S)/FTP GET using requests.get to the URL "{uri}/{section}/{domain}/assembly_summary.txt", writes the response text to the cache using UTF-8 encoding, and returns a StringIO of the response text.
        - When use_cache is False, the function requests the URL and returns a StringIO of the response text and does not read or write the cache.
        - Directory creation: if use_cache is True and the cache directory does not exist, the function attempts os.makedirs(CACHE_DIR). If os.makedirs raises OSError with errno equal to the platform-specific "file exists" error (errno 17 on POSIX), that error is ignored; other OSError values are re-raised to the caller.
        - Network and response handling: the function calls requests.get and uses req.text as the content. It does not validate HTTP status codes or the structure of the returned text; callers should validate that the returned assembly_summary.txt content is correct for their needs. Network errors or other exceptions raised by requests.get (for example subclasses of requests.exceptions.RequestException) will propagate to the caller.
        - Logging: the function logs debug/info messages to the "ncbi-genome-download" logger indicating cache checks, cache use, and download attempts.
        - Encoding: cache files are read and written using UTF-8. The in-memory StringIO contains Unicode text as produced by requests.text and the UTF-8-decoded cache file.
        - The function relies on module-level CACHE_DIR being defined; if CACHE_DIR is not writable or causes unexpected filesystem errors, file operations may raise OSError.
    
    Returns:
        StringIO: An in-memory text file-like object (io.StringIO) containing the complete contents of the requested assembly_summary.txt. The caller can iterate over lines or read() this object to obtain the tab-delimited assembly metadata used elsewhere in ncbi-genome-download to select and filter genome assemblies. Side effects may include creating CACHE_DIR and writing a cache file when use_cache is True and a fresh cache is not present.
    """
    from ncbi_genome_download.core import get_summary
    return get_summary(section, domain, uri, use_cache)


################################################################################
# Source: ncbi_genome_download.core.grab_checksums_file
# File: ncbi_genome_download/core.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def ncbi_genome_download_core_grab_checksums_file(entry: dict):
    """ncbi_genome_download.core.grab_checksums_file — Retrieve the MD5 checksums file for a single NCBI assembly entry.
    
    Retrieves the remote "md5checksums.txt" file for a single assembly entry dictionary (as used throughout ncbi-genome-download). This function is intended to be used by the ncbi-genome-download download/process pipeline (see README) to obtain per-assembly MD5 checksum listings published by NCBI (RefSeq/GenBank) so callers can verify integrity of downloaded genome files.
    
    Args:
        entry (dict): A mapping representing a single assembly record produced by the ncbi-genome-download code paths (for example, one line parsed from an NCBI assembly summary). This dict must contain the key 'ftp_path' whose value is the FTP URL string pointing to the assembly directory on NCBI's FTP server (e.g. "ftp://ftp.ncbi.nlm.nih.gov/..."). The function calls convert_ftp_url(entry['ftp_path']) to obtain an HTTP(S) URL to that directory and then requests the "md5checksums.txt" file from that location. Providing an entry without 'ftp_path' will raise a KeyError; providing a non-string or malformed ftp_path may lead to unexpected behavior from convert_ftp_url or the HTTP request.
    
    Returns:
        str: The raw text content of the retrieved "md5checksums.txt" file as returned by requests.get(...). In the typical NCBI workflow this is a plain-text list of filenames and their MD5 checksums used to validate downloaded genome/assembly files. Note that this function does not parse the checksum file or perform any verification itself; it only returns the HTTP response body.
    
    Behavior, side effects, and failure modes:
        - Network I/O: This function performs a blocking HTTP GET request to the URL constructed from the entry's ftp_path. It will generate network traffic and may be subject to network latency, timeouts, or remote server rate limits.
        - Exceptions: requests.get(...) may raise requests.exceptions.RequestException (or subclasses) on connectivity problems; callers should catch these if they need to handle transient network errors. A missing 'ftp_path' key raises KeyError. convert_ftp_url may also raise errors for malformed FTP URLs.
        - HTTP status handling: The implementation returns response.text unconditionally; it does not call raise_for_status(). If the HTTP response is an error page (non-200), the returned text may be empty or contain an HTML error message rather than a valid checksum list. Callers that require robust validation should check the response status code (by performing their own request or modifying the function) or validate the returned text format before using it to verify files.
        - No local side effects: This function does not write files, modify global state, or cache results; it only returns the response text. If persistent storage or caching is desired, the caller must implement it.
        - Usage context: In the ncbi-genome-download project this function is used to fetch MD5 checksum lists so the download pipeline or downstream scripts can compare checksums and ensure downloaded genome files match the checksums provided by NCBI.
    """
    from ncbi_genome_download.core import grab_checksums_file
    return grab_checksums_file(entry)


################################################################################
# Source: ncbi_genome_download.core.has_file_changed
# File: ncbi_genome_download/core.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def ncbi_genome_download_core_has_file_changed(
    directory: str,
    checksums: dict,
    filetype: str = "genbank"
):
    """ncbi_genome_download.core.has_file_changed determines whether a genome-related file in a local download directory is missing or has an MD5 checksum that differs from the expected checksum, indicating the file has changed and may need re-downloading or verification.
    
    This function is used by the ncbi-genome-download pipeline (see README) to validate downloaded genome files (by default GenBank files) against expected checksums derived from assembly summary metadata. It resolves the file extension for the requested filetype via NgdConfig.get_fileending(filetype), obtains the target filename and the expected checksum using get_name_and_checksum(checksums, pattern), then computes the actual MD5 checksum of the file on disk with md5sum and compares the two.
    
    Args:
        directory (str): Path to the local directory where the downloaded file is expected to reside. In the ncbi-genome-download workflow this is typically the per-organism or per-assembly output directory where downloaded GenBank/FASTA/other files are stored. The function constructs the full path by joining this directory with the filename returned by get_name_and_checksum.
        checksums (dict): A mapping containing checksum metadata for assemblies or files. This dict must be in the format expected by get_name_and_checksum(checksums, pattern); the function will call that helper to extract a filename (relative to directory) and the expected checksum string. In practice this dict is produced from parsed assembly summaries or cached checksum records used by the downloader.
        filetype (str): Type/format identifier for the file whose checksum is to be verified (default: "genbank"). This value is passed to NgdConfig.get_fileending(filetype) to determine the file extension or naming pattern (for example genbank, fasta, assembly-report), so choosing the correct filetype ensures the function checks the right file produced by ncbi-genome-download.
    
    Returns:
        bool: True if the file is considered changed, False if the existing file matches the expected checksum. A return value of True means either the file does not exist in the specified directory (interpreted as changed/missing) or the MD5 checksum computed from the on-disk file differs from the expected checksum extracted from the checksums argument. A return value of False indicates the on-disk file's checksum matches the expected checksum and no re-download is required.
    
    Behavior and side effects:
        This function performs no file modifications; it only reads filesystem metadata and file contents to compute an MD5 digest. If the target file is missing, the function returns True without attempting to create any files. The function relies on NgdConfig.get_fileending, get_name_and_checksum, and md5sum to determine filename and compute checksums; any exceptions raised by those helpers (for example KeyError, ValueError, I/O errors when reading files, or custom errors from NgdConfig) will propagate to the caller. Use callers in the ncbi-genome-download pipeline to catch and handle such exceptions as needed.
    """
    from ncbi_genome_download.core import has_file_changed
    return has_file_changed(directory, checksums, filetype)


################################################################################
# Source: ncbi_genome_download.core.md5sum
# File: ncbi_genome_download/core.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def ncbi_genome_download_core_md5sum(filename: str):
    """ncbi_genome_download.core.md5sum calculates the MD5 checksum of a local file and returns its hexadecimal digest string. This function is used by the ncbi-genome-download toolset to verify file integrity for downloaded genome and assembly files (for example, to compare against NCBI-provided md5sums or to detect corrupted transfers), and is implemented to be memory-efficient and deterministic for reproducible integrity checks.
    
    Args:
        filename (str): Path to the file on disk to be hashed. In the ncbi-genome-download domain this is typically a downloaded genome FASTA, GenBank/RefSeq assembly file, or related metadata file. The function opens the file in binary mode and reads it in fixed-size chunks (4096 bytes) to avoid loading the entire file into memory; therefore it works for very large genome files. The provided filename must be a valid file path accessible to the running process; passing a directory, a non-existent path, or a path without read permission will raise the underlying I/O exception (FileNotFoundError, PermissionError, or OSError) originating from Python's open() or file read operations.
    
    Returns:
        str: Lowercase hexadecimal MD5 digest of the file contents (the same format used by common md5sum utilities and by NCBI md5 manifest files). The return value is deterministic for a given file content and can be compared directly against NCBI-provided checksums to confirm successful and uncorrupted downloads. No modifications are made to the input file; the function only reads the file and closes it before returning.
    """
    from ncbi_genome_download.core import md5sum
    return md5sum(filename)


################################################################################
# Source: ncbi_genome_download.core.need_to_create_symlink
# File: ncbi_genome_download/core.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def ncbi_genome_download_core_need_to_create_symlink(
    directory: str,
    checksums: dict,
    filetype: str,
    symlink_path: str
):
    """Check whether a human-readable symlink should be created for an already-downloaded NCBI genome file. This function is used by the ncbi-genome-download pipeline when the user requests a human-readable directory layout (for example via the --human-readable option described in the README). It determines, without modifying the filesystem, whether a symlink in the requested symlink_path directory should be created that points to the real file in directory. The decision is based on the file name derived from checksums for the requested filetype and on the current state of an existing symlink (if any).
    
    Args:
        directory (str): Path to the directory that contains the actual downloaded file. In the ncbi-genome-download domain this is typically the per-assembly download directory where the file (e.g., a FASTA or GenBank file) resides; the function constructs the absolute path to the target file by joining this directory with a filename selected from checksums. This argument must be the same directory used when the file was downloaded so the resulting symlink target will point to the correct file.
        checksums (dict): A mapping of candidate filenames to checksum values as produced by the tool's checksum-parsing routines (for example parsed from an MD5/sha checksum file or assembly metadata). The function calls get_name_and_checksum(checksums, pattern) with a pattern derived from filetype to select the single filename that should be checked/linked. If the expected filename is not present in this mapping, the called helper may raise an error (see failure modes).
        filetype (str): Logical file type identifier (for example the same format names used by the CLI such as 'fasta', 'genbank', 'assembly-report', etc.). This value is passed to NgdConfig.get_fileending(filetype) to produce a filename pattern (file ending) used to select the appropriate filename from checksums. The chosen filename determines which file in directory would be the symlink target.
        symlink_path (str): Destination directory in which the symlink should be created to provide a human-readable layout (typically a parallel directory tree mirroring NCBI structure). If the caller passes None for this parameter, the function will treat symlink creation as not requested and immediately return False. When a string path is provided, the function inspects the filesystem to decide if a symlink with the chosen filename exists and already points exactly to the target file.
    
    Behavior, side effects, defaults, and failure modes:
        This function performs read-only checks on the provided inputs and on the filesystem: it calls NgdConfig.get_fileending(filetype) to derive a filename pattern, calls get_name_and_checksum(checksums, pattern) to obtain the target filename, and inspects the candidate symlink path using os.path.islink and os.readlink. It does not create or modify any files or symlinks itself; it only returns a boolean decision.
        If symlink_path is None the function returns False immediately (no symlink creation is needed because the caller did not request a human-readable layout).
        If a symlink already exists at the expected symlink location and os.readlink(symlink) returns a path that is exactly equal to the constructed full file path (os.path.join(directory, filename)), the function returns False because no creation is necessary.
        Otherwise the function returns True indicating that the caller should create or recreate the symlink.
        The function depends on exact string equality between the existing symlink target and the constructed full filename; an existing symlink that points to an equivalent but differently expressed path (for example a relative path, a different absolute canonicalization, or a symlink created before directory layout changes) will be treated as different and the function will indicate a symlink should be created (True).
        Possible exceptions propagated to the caller include those raised by NgdConfig.get_fileending, by get_name_and_checksum (for example if the requested pattern is not found in checksums), and by os.readlink/os.path operations (for example OSError for unreadable or broken symlinks). Race conditions are possible if the filesystem is modified concurrently (files or symlinks added/removed between this check and any subsequent creation attempt); callers should handle such races when creating the symlink.
    
    Returns:
        bool: True if a symlink should be created or recreated in symlink_path that points to the file determined from checksums and filetype; False if no symlink creation is necessary (either because symlink_path is None, or because an existing symlink already points exactly to the expected file).
    """
    from ncbi_genome_download.core import need_to_create_symlink
    return need_to_create_symlink(directory, checksums, filetype, symlink_path)


################################################################################
# Source: ncbi_genome_download.core.parse_checksums
# File: ncbi_genome_download/core.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def ncbi_genome_download_core_parse_checksums(checksums_string: str):
    """ncbi_genome_download.core.parse_checksums parses the textual contents of a checksum file (as produced in NCBI FTP assembly directories) and returns a structured list of checksum-to-filename mappings suitable for use by the ncbi-genome-download verification and bookkeeping logic.
    
    Args:
        checksums_string (str): The complete contents of a checksum file as a single string (for example, the result of reading an MD5/SHA checksum file downloaded from an NCBI FTP directory: open(path).read()). Each non-empty line is expected to contain two whitespace-separated fields: a checksum value and a filename. The function treats the first whitespace-separated token as the checksum and the second as the filename and will strip a leading "./" from filenames to normalize paths to the relative form used by ncbi-genome-download. Passing a non-string value will cause an attribute error when split() is called; callers should provide a string.
    
    Returns:
        list: A list of dictionaries, each dictionary representing one parsed entry from the checksum file. Each dictionary has two keys: 'checksum' (str) containing the checksum token extracted from the line, and 'file' (str) containing the normalized filename (with any leading "./" removed). If no valid lines are found, an empty list is returned.
    
    Behavior and side effects:
        The function splits the input string on newline characters and iterates over lines. Empty lines are skipped. For each non-empty line, line.split() is used to obtain whitespace-separated tokens; the implementation expects exactly two tokens (checksum and filename). If a line does not split into the expected two tokens (for example, if a filename contains internal whitespace or the line is malformed), a ValueError is caught, the line is skipped, and a debug-level message is emitted to the "ncbi-genome-download" logger indicating the skipped line. The function performs no I/O, does not modify files on disk, and has no other side effects beyond logging. It preserves the exact checksum string as read (no validation of checksum format, length, or algorithm is performed by this function). Memory usage is proportional to the size of checksums_string and the number of parsed entries.
    """
    from ncbi_genome_download.core import parse_checksums
    return parse_checksums(checksums_string)


################################################################################
# Source: ncbi_genome_download.core.parse_summary
# File: ncbi_genome_download/core.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def ncbi_genome_download_core_parse_summary(summary_file: str):
    """Parse the NCBI assembly summary TSV file and return a reader that yields each summary row as a dictionary; this is used by ncbi-genome-download to inspect available assemblies and their metadata (for example ftp paths, taxonomy IDs, RefSeq/GenBank categories, assembly levels, and type-material relationships) when deciding which genome files to download.
    
    Args:
        summary_file (str): Path to a local NCBI assembly summary file in TSV format (for example an "assembly_summary.txt" file downloaded or cached by ncbi-genome-download). The function expects a filesystem path string pointing to a plain-text TSV file produced by NCBI that contains one header row followed by tab-separated assembly metadata rows. Providing any other form (for example an already-open file handle) is not supported by the function signature.
    
    Returns:
        SummaryReader: An object constructed by SummaryReader(summary_file) that behaves like a csv.DictReader: it is an iterator that yields one dict per assembly row where keys are the column names from the TSV header and values are the corresponding string fields from each row. Practically, callers (such as the main download logic) iterate this reader to examine metadata (taxids, assembly_accession, ftp_path, refseq_category, assembly_level, relation to type material, etc.) and apply filters to select which genome files to fetch.
    
    Behavior and side effects:
        This function does not perform network activity or download genome data; it only delegates to SummaryReader(summary_file) to open and parse the given local TSV file. The only side effect is opening and reading the provided file via the underlying SummaryReader. The function returns immediately with the reader object; actual parsing/iteration of rows typically happens when the caller iterates the returned reader.
    
    Failure modes and errors:
        If the provided path does not point to an existing readable file, the underlying file-open operation will raise an I/O-related exception (for example FileNotFoundError or an OSError). If the file contents are not a well-formed NCBI assembly summary TSV (for example missing header or unexpected encoding), iteration over the returned reader or its construction may raise parsing or decoding errors propagated from the underlying CSV/reader implementation. The function itself performs no additional validation of column semantics; callers should validate required columns (such as "ftp_path" or "taxid") before using the row dictionaries for downstream selection or downloading.
    """
    from ncbi_genome_download.core import parse_summary
    return parse_summary(summary_file)


################################################################################
# Source: ncbi_genome_download.metadata.get
# File: ncbi_genome_download/metadata.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def ncbi_genome_download_metadata_get(columns: list = None):
    """Get or create the ncbi_genome_download.metadata.MetaData singleton used by ncbi-genome-download to represent and manage genome assembly metadata.
    
    This function centralizes access to a single MetaData instance for the process. In the ncbi-genome-download domain this MetaData instance encapsulates which metadata columns are tracked for NCBI genome assembly records (for example, the columns parsed from assembly summary files used when selecting and downloading genomes). Calling get ensures that the same MetaData object is returned for all callers in the same process, avoiding repeated re-parsing or re-initialization of metadata state.
    
    Args:
        columns (list): An explicit list of metadata column names to use when creating the MetaData object. If None, the function uses the module-level _DEFAULT_COLUMNS value. The provided list controls which assembly-report/summary fields the MetaData instance will track and expose to the rest of the ncbi-genome-download code that filters and selects assemblies for download. This parameter is considered only when the MetaData singleton is created for the first time; on subsequent calls the existing singleton is returned and this argument is ignored.
    
    Returns:
        MetaData: The singleton MetaData instance used by ncbi-genome-download. The returned object is either an existing global _METADATA instance or a newly created MetaData(columns) assigned to the global _METADATA. This object is used throughout the package to inspect, cache, and filter assembly metadata when building download lists and directory structures for NCBI genomes.
    
    Behavior and side effects:
        - If columns is None, the function substitutes the module-level _DEFAULT_COLUMNS before creating the MetaData instance.
        - On first call, this function instantiates MetaData(columns) and assigns it to the module-global _METADATA; this may trigger whatever initialization MetaData implements (for example, loading cached assembly-summary data, performing I/O, or parsing files).
        - On subsequent calls, the pre-existing singleton _METADATA is returned unchanged; any columns argument provided on later calls is ignored and does not alter the already-created MetaData.
        - The function affects global state by setting the module-level _METADATA when creating the singleton.
    
    Failure modes and errors:
        - If MetaData(columns) raises an exception during construction (for instance, due to invalid column values, I/O failures, or other initialization errors), that exception will propagate to the caller.
        - The columns argument is expected to be a list or None. Passing other types may cause MetaData to raise a TypeError or other validation error depending on MetaData's implementation.
        - This function does not attempt to re-create or replace an existing singleton; to change metadata columns after creation, the process must replace or reset the global _METADATA by other means (not provided by this function).
    """
    from ncbi_genome_download.metadata import get
    return get(columns)


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
