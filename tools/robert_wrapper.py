"""
Regenerated Google-style docstrings for module 'robert'.
README source: others/readme/robert/README.md
Generated at: 2025-12-02T01:29:39.175618Z

Total functions: 38
"""


import numpy

################################################################################
# Source: robert.aqme.filter_aqme_args
# File: robert/aqme.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def robert_aqme_filter_aqme_args(aqme_db: str):
    """robert.aqme.filter_aqme_args removes AQME-specific argument columns from a CSV file used in the AQME preprocessing step of the ROBERT package.
    
    This function is used in the AQME-related preprocessing pipeline of ROBERT (Refiner and Optimizer of a Bunch of Existing Regression Tools) to sanitize CSV inputs by removing columns that represent AQME arguments. Removing these columns prevents AQME-specific parameters from being treated as feature columns in downstream regression, machine-learning, or reporting steps handled by ROBERT.
    
    Args:
        aqme_db (str): Path to the CSV file to be filtered. This is the filesystem path (string) pointing to the CSV that contains potential AQME argument columns. The function opens this file with pandas.read_csv using UTF-8 encoding, so the path should reference a readable text CSV file encoded in UTF-8.
    
    Returns:
        None: This function does not return a Python object. Instead, it performs in-place modification of the file at the path given by aqme_db as a side effect: it reads the CSV into a pandas.DataFrame, drops any columns whose lowercased column name appears in the module-level collection aqme_args (a collection of AQME argument names used to identify AQME-specific columns), removes the original file from disk with os.remove(aqme_db), and writes the filtered DataFrame back to the same path using DataFrame.to_csv with header=True and index=None. The practical significance is that the original CSV is replaced by a cleaned version that no longer contains AQME argument columns, making it ready for ROBERT's downstream processing.
    
    Behavior, defaults, and failure modes:
        - The function reads the CSV with pandas.read_csv(..., encoding='utf-8'). If the file is not UTF-8 encoded, pandas may raise a UnicodeDecodeError.
        - Column filtering logic compares column.lower() against aqme_args, so aqme_args is expected to contain lowercase names of AQME-specific columns; mismatch in casing between aqme_args entries and CSV headers may prevent intended columns from being removed.
        - The function deletes the original file path via os.remove(aqme_db) before writing the sanitized version. This is destructive: if an error occurs after removal and before the new CSV is written, the original data may be lost. There is no atomic replace nor backup created by this function.
        - Possible exceptions include FileNotFoundError if aqme_db does not exist, pandas.errors.EmptyDataError or other pandas I/O exceptions if the CSV is invalid or unreadable, PermissionError if the process lacks permission to remove or write the file, and NameError if the module-level variable aqme_args is not defined. Users should ensure appropriate backups and permissions before invoking this function.
        - The function drops columns using pandas.DataFrame.drop(column, axis=1) and writes the resulting DataFrame back with to_csv; column order and non-AQME data are preserved except for the removed columns.
    
    Practical notes for users:
        - Use this function when you need to prepare CSV inputs coming from AQME or other pre-processing tools so that ROBERT's feature-engineering and model-fitting routines do not misinterpret AQME argument columns as model features.
        - Verify that aqme_args (the set/list of AQME argument names) is correctly populated and uses the expected lowercase format to ensure correct filtering.
        - Because the operation overwrites the original file, consider creating a file copy or version control checkpoint before calling this function when working with irreplaceable data.
    """
    from robert.aqme import filter_aqme_args
    return filter_aqme_args(aqme_db)


################################################################################
# Source: robert.generate_utils.detect_best
# File: robert/generate_utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def robert_generate_utils_detect_best(folder: str):
    """robert.generate_utils.detect_best checks a folder of CSV result files produced by ROBERT regression experiments and selects the file combination that yielded the best training result, then copies that CSV and its associated "_db.csv" file into a "Best_model" location by replacing the path segment "Raw_data" with "Best_model".
    
    This function is used in the ROBERT pipeline to automatically detect which combination of model/features/parameters produced the most favorable training metric (as reported in each CSV) and to preserve both the selected result CSV and its paired database CSV for later inspection or reporting. It expects CSV files written by earlier ROBERT steps where each CSV contains at least an "error_type" column and a "combined_{error_type}" column (for example "combined_mae" when error_type is "mae"). The function reads files with UTF-8 encoding, ignores files that include "_db" in their filename when computing the scoring metric, and preserves index alignment by inserting NaN for those entries.
    
    Args:
        folder (str): Path to the directory that contains the CSV result files produced by ROBERT (for example a directory named "Raw_data" with files like "experiment1.csv" and "experiment1_db.csv"). The function will glob for "{folder}/*.csv", open each non-"*_db.csv" file with pandas.read_csv(encoding='utf-8'), and expect that the first row provides 'error_type' and a corresponding combined_{error_type} column. The provided folder should be reachable by the running process and contain the expected CSV files; if it does not, file system errors (e.g., FileNotFoundError) may be raised.
    
    Returns:
        None: The function does not return a value. As a side effect, it copies the detected best result CSV and its matching "_db.csv" file into paths derived by replacing the substring "Raw_data" with "Best_model" in their original full file paths (using shutil.copyfile). If the detected best file's path does not contain the substring "Raw_data", the replacement still occurs and may produce a target path that does not exist, which will then raise a file system error. The function also may raise exceptions in these situations: when no CSV files are found in folder (ValueError or IndexError when accessing results), when required columns ('error_type' or the matching combined_{error_type}) are missing in a CSV (KeyError), when the first row is empty or malformed (IndexError), or when copy operations fail because the source or destination paths are invalid (FileNotFoundError, PermissionError, or shutil-related errors).
    
    Behavior and failure modes:
        - Detection: The function builds a list of CSV files in folder via glob. For files whose filename contains "_db" the function appends numpy.nan to an internal error list to keep indices aligned with file_list. For other files it reads the CSV (pandas.read_csv with UTF-8) and extracts the value at column name "combined_{error_type}" using the 'error_type' value found in the first row (results_model['error_type'][0]); that value is appended to the error list.
        - Selection rule: If the first CSV inspected reports an error_type whose lowercase form is 'mae' or 'rmse' the function selects the file with minimal numeric error using numpy.nanmin(errors) to ignore NaN placeholders. For any other error_type it selects the file with maximal numeric error using numpy.nanmax(errors). The use of the first inspected CSV's error_type determines whether smaller or larger is considered better for all files; inconsistent error_type values across CSVs may lead to incorrect selection or runtime errors.
        - Copying: After selecting the best file by index, the function constructs the path for the paired "_db.csv" by taking the chosen CSV filename, removing the ".csv" suffix and appending "_db.csv". It then copies both files to new paths obtained by replacing "Raw_data" with "Best_model" in their original paths. Existing destination files will be overwritten by shutil.copyfile.
        - Assumptions: The CSV format and column naming conventions (presence of 'error_type' and combined_{error_type}) are required. The function assumes that the first row of each non-"*_db" CSV contains canonical metadata used for selection. It assumes the folder argument points to the directory containing the CSVs output by ROBERT. If these assumptions are violated, the function will raise standard Python exceptions described above.
    
    Practical significance in the ROBERT domain:
        - In ROBERT's regression workflows, multiple experimental combinations generate per-combination CSV reports. This function automates identification of the best-performing combination according to the training metric recorded in those CSVs and copies both the result CSV and its associated database CSV into a canonical "Best_model" location for downstream use (report generation, reproducibility checks, or manual review).
    """
    from robert.generate_utils import detect_best
    return detect_best(folder)


################################################################################
# Source: robert.report_utils.calc_penalty_r2
# File: robert/report_utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def robert_report_utils_calc_penalty_r2(r2_val: float):
    """robert.report_utils.calc_penalty_r2 computes a small integer penalty for a model's coefficient of determination (R²) using predetermined thresholds. This penalty is intended to be used by ROBERT's report and scoring utilities to reduce the overall quality score of regression models with low predictive performance (for example, when assembling the ROBERT report or combining R² with other metrics such as MCC).
    
    This function implements a deterministic, pure scoring rule: it inspects the numeric R² value provided and returns a negative integer penalty according to fixed thresholds. The thresholds reflect conventions used within the ROBERT reporting pipeline: R² < 0.5 is considered poor and receives a larger penalty; 0.5 ≤ R² < 0.7 is considered marginal and receives a smaller penalty; R² ≥ 0.7 receives no penalty. There are no side effects, no internal state changes, and no defaults beyond the numerical thresholds coded in the function.
    
    Behavior notes:
    - The function performs simple numeric comparisons; it does not validate that r2_val lies within [0, 1]. Inputs outside the typical R² range (e.g., negative values or values > 1) will be compared against the thresholds and penalized accordingly.
    - If r2_val is float('nan'), comparisons with NaN evaluate to False in Python and the function will return 0 (no penalty).
    - If r2_val is not a numeric type that supports comparison with floats, a TypeError (or comparable exception) will be raised by the comparison operations.
    
    Args:
        r2_val (float): The coefficient of determination (R²) produced by a regression model. In the ROBERT context this is the model-level R² used to assess predictive performance in cheminformatics or related regression tasks. The function uses this numeric value to decide a penalty contribution for report scoring: lower R² values indicate worse fit and produce larger negative penalties.
    
    Returns:
        int: An integer penalty to be applied to an aggregate score. The function returns -2 when r2_val < 0.5 (poor performance), -1 when 0.5 <= r2_val < 0.7 (marginal performance), and 0 when r2_val >= 0.7 (no penalty). The returned penalty is intended to be combined with other metric penalties (for example, an MCC penalty computed elsewhere) when generating ROBERT evaluation reports.
    """
    from robert.report_utils import calc_penalty_r2
    return calc_penalty_r2(r2_val)


################################################################################
# Source: robert.report_utils.calc_score
# File: robert/report_utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def robert_report_utils_calc_score(
    dat_files: dict,
    suffix: str,
    pred_type: str,
    data_score: dict
):
    """Calculates the ROBERT score for a given dataset/model split and updates the provided score dictionary used by ROBERT report generation.
    
    This function is part of the ROBERT package (Refiner and Optimizer of a Bunch of Existing Regression Tools) and combines scores produced by the prediction and verification steps into a single composite "robert_score" used in reports and downstream decision-making in cheminformatics machine-learning workflows. It first augments the provided data_score dictionary by calling get_predict_scores on the file referenced by dat_files['PREDICT'] and get_verify_scores on dat_files['VERIFY']. Then, depending on whether the prediction task is regression ('reg') or classification ('clas'), it computes component scores (reading existing keys from data_score with safe defaults) and assigns a final robert_score_{suffix} entry in data_score. The function clamps negative final scores to zero to avoid producing negative ROBERT scores.
    
    Args:
        dat_files (dict): Mapping of file roles to file paths or file-like identifiers used by the prediction and verification steps. In practice this dict must contain the keys 'PREDICT' and 'VERIFY' (the function accesses dat_files['PREDICT'] and dat_files['VERIFY']) so that get_predict_scores and get_verify_scores can read the corresponding outputs. This parameter directs which files are used to compute or augment prediction/verification scores that are then combined into the ROBERT score.
        suffix (str): A string suffix that is appended into score keys inside data_score (for example, 'cv', 'test', or any other identifier used by the caller). The function constructs keys such as f'robert_score_{suffix}', f'cv_score_combined_{suffix}', f'r2_cv_{suffix}', etc., using this suffix to locate or store the relevant components for the specific split or configuration.
        pred_type (str): Prediction task type indicator that controls which aggregation logic is used. Accepted/expected values (as used in the ROBERT codebase) are 'reg' for regression tasks and 'clas' for classification tasks. If 'reg', the function aggregates a set of regression-specific component scores. If 'clas', it computes a classification-specific gap score based on the absolute difference between r2_cv_{suffix} and r2_test_{suffix} and then aggregates classification components. If pred_type is neither 'reg' nor 'clas', the function will still call get_predict_scores and get_verify_scores but will not add a robert_score_{suffix} entry.
        data_score (dict): Mutable mapping of score names to numeric values that already contains some evaluation metrics (for example values created earlier by prediction/verification steps). The function updates this dict in place by calling get_predict_scores(dat_files['PREDICT'], suffix, pred_type, data_score) and get_verify_scores(dat_files['VERIFY'], suffix, pred_type, data_score) and then by adding or updating keys such as f'robert_score_{suffix}', f'diff_mcc_score_{suffix}' (classification only), and other combined-score keys. Missing component keys are treated as zero via data_score.get(..., 0) when computing aggregates.
    
    Behavior and side effects:
        The function mutates and returns the same data_score dict passed in. For regression (pred_type == 'reg') it computes robert_score_{suffix} by summing the following component keys (each looked up with a default of 0 if absent): cv_score_combined_{suffix}, test_score_combined_{suffix}, cv_sd_score_{suffix}, diff_scaled_rmse_score_{suffix}, flawed_mod_score_{suffix}, sorted_cv_score_{suffix}. For classification (pred_type == 'clas') it first computes the absolute rounded difference diff_mcc = round(abs(mcc_test - mcc_cv), 2) where mcc_cv is read from r2_cv_{suffix} and mcc_test from r2_test_{suffix}, then sets diff_mcc_score_{suffix} to 2 if diff_mcc < 0.15, to 1 if 0.15 <= diff_mcc <= 0.30, and to 0 otherwise. The final classification robert_score_{suffix} is the sum of cv_score_combined_{suffix}, test_score_combined_{suffix}, flawed_mod_score_{suffix}, sorted_cv_score_{suffix}, diff_mcc_score_{suffix}, and descp_score_{suffix} (each defaulting to 0 if absent). In both branches any negative aggregated robert_score is clipped to 0 before being written to data_score.
    
    Failure modes and important notes:
        If dat_files does not contain the keys 'PREDICT' or 'VERIFY', the initial calls dat_files['PREDICT'] or dat_files['VERIFY'] will raise a KeyError. The function relies on the helper functions get_predict_scores and get_verify_scores to augment data_score; those helper functions may raise their own exceptions if the referenced files are missing, malformed, or if underlying libraries fail. When component keys are absent in data_score, they are treated as zero via dict.get defaulting, preventing KeyError for missing score components. pred_type values other than 'reg' or 'clas' will result in get_predict_scores/get_verify_scores being called but no robert_score_{suffix} being assigned by this function.
    
    Returns:
        dict: The same data_score mapping passed in, updated in place with augmented prediction/verification entries from get_predict_scores and get_verify_scores and with a computed robert_score_{suffix} entry for 'reg' or 'clas' tasks (and a diff_mcc_score_{suffix} entry for classification). The returned dict contains the final composite score values used by ROBERT report generation and downstream analysis.
    """
    from robert.report_utils import calc_score
    return calc_score(dat_files, suffix, pred_type, data_score)


################################################################################
# Source: robert.report_utils.combine_cols
# File: robert/report_utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def robert_report_utils_combine_cols(columns: list):
    """robert.report_utils.combine_cols constructs an HTML fragment that arranges a sequence of column contents into a single horizontal (multi-column) line using inline CSS flex layout. In the ROBERT project this helper is used when assembling pieces of the HTML-based report (for example the ROBERT_report.pdf styling pipeline) so that multiple data elements can be displayed side-by-side as equally weighted columns in report sections.
    
    Args:
        columns (list): A list (sequence) of objects whose textual representation will be placed into separate column containers. Each item in this list is inserted into a child <div> element via Python string formatting (f'{column}'), so the object's string form (str(column) or its __format__ result) will appear verbatim inside the generated HTML. The caller is responsible for providing the columns in the desired order; no additional escaping or sanitization is performed by this function.
    
    Returns:
        str: A single string containing an HTML fragment. The fragment wraps each provided column value inside a child <div style="flex: 1;">...</div> and encloses all children within a parent <div style="display: flex;">...</div>. This returned HTML is intended for inclusion in report HTML templates or further processing into PDFs. There are no side effects (the function does not modify input objects or external state).
    
    Behavior, defaults, and failure modes:
        - The function always returns a string even when the input list is empty; for an empty list it returns a parent flex <div> with no children.
        - Elements of columns are not validated or escaped: if an element contains HTML markup, that markup will be embedded verbatim in the output (possible XSS or rendering issues if untrusted content is used). Callers should perform escaping or sanitization when necessary.
        - If the provided columns argument is not iterable in a manner compatible with "for column in columns" (for example, if columns is None or an integer), Python will raise a TypeError when iterating. The signature documents columns as a list, so callers should pass a list-like sequence.
        - Non-string objects in the list are converted to their string representation using Python's formatting mechanism; no explicit type conversion is enforced by this function.
        - The produced HTML uses minimal inline CSS suited for simple report layouts; consumers that require different styling should post-process the returned string or supply already-styled HTML fragments as column elements.
    """
    from robert.report_utils import combine_cols
    return combine_cols(columns)


################################################################################
# Source: robert.report_utils.css_content
# File: robert/report_utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def robert_report_utils_css_content(csv_name: str, robert_version: str):
    """robert.report_utils.css_content returns a complete CSS stylesheet string tailored for ROBERT PDF reports, with the provided CSV filename and ROBERT version interpolated into header/footer content.
    
    This function is used by the ROBERT report generation pipeline to produce the CSS that an HTML-to-PDF renderer will apply when creating the ROBERT PDF report (see project README instructions for installing system libraries required by the PDF report workflow). The returned stylesheet encodes page geometry (A4, 2cm margin), page headers/footers (including page numbering, a bottom-left ROBERT version label, and a top-right CSV filename), global typography (Helvetica with fallbacks), logo/image selectors, and several layout utility classes (.dat-content, .img-PREDICT, hr rules). The csv_name and robert_version parameters are interpolated verbatim into CSS content rules so they appear in the generated PDF (csv_name appears at @top-right; robert_version appears at @bottom-left).
    
    Args:
        csv_name (str): CSV filename or short identifier to display in the report top-right header. In the ROBERT domain this is typically the input data filename used to produce the report (for example "dataset.csv"). The value is inserted verbatim into a CSS content rule (content: "{csv_name}"); therefore it should be plain text without unescaped double-quote characters or other characters that would break CSS string syntax. The function does not perform sanitization or escaping, so providing strings containing quotes or CSS-special characters may produce invalid CSS and cause PDF rendering failures.
        robert_version (str): ROBERT version string to display in the report bottom-left footer (for example "1.2.0"). This value is interpolated verbatim into a CSS content rule (content: "ROBERT v {robert_version}") so it becomes part of the generated PDF footer. As with csv_name, the function does not escape or validate this value; supply a plain text version identifier to avoid CSS syntax errors.
    
    Returns:
        str: A single CSS stylesheet as a plain Python string. The stylesheet is ready to be passed to a CSS-aware HTML-to-PDF rendering step used by ROBERT. The returned CSS includes:
            - @page rules setting size: A4 and margin: 2cm.
            - Footer and header regions with fixed positioning: page counter ("Page X of Y") at bottom-right, ROBERT version at bottom-left, an empty centered rule used as a top/bottom separator (border-top: 3px solid black), "ROBERT Report" at top-left, and the provided csv_name at top-right.
            - Global font-family set to "Helvetica", Arial, sans-serif and font-size/line-height defaults.
            - Selectors for image sources expected by ROBERT reports (e.g., img[src="Robert_logo.jpg"], img[src*="Pearson"], img[src*="PFI"]) and layout classes such as .dat-content and .img-PREDICT.
            - Horizontal rule styles (hr.black and hr).
        Side effects: None (the function is pure and deterministic). It only returns the CSS string; it does not perform I/O, write files, or validate availability of referenced images. Failure modes: supplying non-string types or strings containing double quotes or other characters that break CSS string syntax can produce invalid CSS and cause the downstream HTML-to-PDF renderer to fail. Ensure that any images referenced by the stylesheet (e.g., "Robert_logo.jpg") are accessible to the renderer at PDF generation time.
    """
    from robert.report_utils import css_content
    return css_content(csv_name, robert_version)


################################################################################
# Source: robert.report_utils.detect_predictions
# File: robert/report_utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def robert_report_utils_detect_predictions(module_file: str):
    """Detect whether a module file documents predictions coming from an external CSV test set and extract the associated metadata used by ROBERT report generation.
    
    Args:
        module_file (str): Path to a text module file produced or consumed by ROBERT (for example, a report-like .py/.txt file or a metadata file used when generating the ROBERT_report.pdf). The function opens this file for reading with UTF-8 encoding and scans its lines for well-known markers that ROBERT uses to record external test-set information: a line containing "- Target value:" to indicate the name of the predicted property, a line containing "- Names:" to indicate the column header used for sample identifiers, and a line containing "External set with predicted results:" to indicate the path to a CSV file with predictions. This parameter must be a file system path string; the function reads the entire file into memory and performs simple whitespace/token parsing as described below.
    
    Returns:
        tuple: A 4-tuple with the detection flag and extracted metadata, in the exact order returned by the function:
            csv_test_exists (bool): True if a line containing the literal substring "External set with predicted results:" was found; False otherwise. In the context of ROBERT, True indicates that an external CSV test set with model predictions is available and should be considered when assembling reports or computing summary statistics.
            y_value (str): The extracted target value name from the last encountered line containing the literal substring "- Target value:". Extraction logic: the function splits the line at ":" and joins all parts after the first ":", then strips leading and trailing whitespace. If no "- Target value:" line is found, this is the empty string ''. This string is used by ROBERT to label the predicted quantity (for example, a chemical property or experimental target) in plots, tables, and report sections.
            names (str): The extracted column name for sample identifiers from the last encountered line containing the literal substring "- Names:". Extraction logic: the function splits the line on whitespace and uses the last token (line.split()[-1]). If no "- Names:" line is found, this is the empty string ''. In ROBERT workflows, this value indicates which CSV column holds sample names/IDs when reading the external predictions file.
            path_csv_test (str): The extracted path to the external CSV file from the last encountered line containing the literal substring "External set with predicted results:". Extraction logic: the function splits the matching line on whitespace and returns the last token (line.split()[-1]). If that marker is not present, this is the empty string ''. When non-empty and csv_test_exists is True, this path is intended to point to a CSV file whose rows contain sample identifiers and predicted values that ROBERT can include in its final PDF report or downstream analysis.
    
    Behavior, side effects, defaults, and failure modes:
        The function performs a read-only scan of the specified file and has no other side effects. It reads the entire file into memory (calls readlines()), so extremely large files may increase memory usage. If multiple matching marker lines appear in the file, the function keeps the values from the last matching line encountered for each marker (later occurrences overwrite earlier ones). If a marker is not found, its corresponding return value is set to the default: False for csv_test_exists and '' (empty string) for each of the three metadata strings. The extraction uses simple string operations: "- Target value:" uses colon-based splitting and a final .strip(), while "- Names:" and "External set with predicted results:" use whitespace splitting and take the final token; therefore, unexpected spacing or additional tokens may affect the extracted results. The function will propagate I/O and decoding errors from Python's open/read operations (for example, FileNotFoundError if module_file does not exist, or UnicodeDecodeError if the file is not valid UTF-8).
    """
    from robert.report_utils import detect_predictions
    return detect_predictions(module_file)


################################################################################
# Source: robert.report_utils.format_lines
# File: robert/report_utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def robert_report_utils_format_lines(
    module_data: str,
    max_width: int = 122,
    cmd_line: bool = False,
    one_column: bool = False,
    spacing: str = ""
):
    """Format text block lines from a module string into HTML <pre> elements suitable for ROBERT report output, applying simple HTML escaping for the sequence "R2" (to render as R<sup>2</sup>), wrapping lines to a specified width, and optionally preparing a single-column variant with configurable indentation spacing.
    
    This function is used by the ROBERT reporting utilities (see README) to convert raw module or documentation text into justified HTML fragments that can be embedded into the project report (for example, ROBERT_report.pdf or HTML report sections). It accepts the raw text content as a single string, wraps each original line using Python's textwrap.fill, replaces occurrences of the literal substring "R2" with the HTML-safe superscript form "R<sup>2</sup>", and returns a single string containing one or more <pre style="text-align: justify;">...</pre> blocks. The output is intended for downstream inclusion in report-generation pipelines that expect HTML fragments.
    
    Args:
        module_data (str): Raw input text to format. This should be the full text block (for example, contents read from a module docstring or a file) with lines separated by newline characters. The function calls module_data.split('\n'), so passing a non-str will raise an AttributeError.
        max_width (int): Maximum column width used for text wrapping (passed to textwrap.fill as width). Defaults to 122. When cmd_line is True, the function uses max_width - 5 for wrapping to allow for a small margin appropriate for command-line style displays. If an invalid width (e.g., negative) is provided, textwrap.fill may raise a ValueError.
        cmd_line (bool): If True, format each line using a slightly smaller wrap width (max_width - 5) to reflect narrower command-line style output. If False (default), use max_width directly. This flag affects only the width passed to textwrap.fill and does not change other HTML structure.
        one_column (bool): If False (default), return a two-column-compatible concatenation of formatted <pre> blocks (in practice, simply the joined formatted blocks). If True, post-process the joined HTML blocks into a one-column variant: prepend spacing*3 at the start of non-tag lines and insert spacing*3 immediately after opening <pre ...> tags that contain content. Use this when inserting the text into a single-column region of the ROBERT report.
        spacing (str): String used to build indentation when one_column is True. The function multiplies this string by 3 (spacing*3) when inserting indentation. If spacing is the default empty string, no extra indentation is added. This parameter has no effect when one_column is False.
    
    Returns:
        str: A single string containing the formatted HTML fragments. For the default two-column mode (one_column=False) this is the concatenation of per-line <pre style="text-align: justify;">...</pre> blocks where each original line has been wrapped to the requested width and "R2" replaced with "R<sup>2</sup>". In one-column mode (one_column=True) the returned string is a post-processed variant with spacing*3 inserted as indentation at the start of content lines and immediately after opening <pre> tags, with newline handling preserved so the string is ready for insertion into a single-column region of a report.
    
    Behavior, defaults, and failure modes:
        - The function does not perform general HTML escaping beyond replacing the literal substring "R2" with "R<sup>2</sup>" to render R-squared in reports; other HTML-sensitive characters are not modified.
        - Each input line is wrapped independently using textwrap.fill with subsequent_indent set to an empty string.
        - The first formatted block is treated specially to avoid a leading newline at the very start of the returned string; subsequent blocks are prefixed/handled so that blocks appear separated when concatenated.
        - If module_data is not a str, an AttributeError occurs when calling split; callers should ensure they pass a str.
        - If max_width is set to an invalid value (e.g., non-integer or negative), textwrap.fill may raise TypeError or ValueError.
        - The function has no external side effects (does not read or write files); it only returns a formatted string for downstream use in ROBERT report generation.
    """
    from robert.report_utils import format_lines
    return format_lines(module_data, max_width, cmd_line, one_column, spacing)


################################################################################
# Source: robert.report_utils.get_col_score
# File: robert/report_utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def robert_report_utils_get_col_score(
    score_info: str,
    data_score: dict,
    suffix: str,
    spacing: str,
    eval_only: bool
):
    """get_col_score
    Gather and format the HTML column that summarizes a model score for inclusion in the ROBERT report (PDF/HTML). This function is used by the robert.report_utils module to assemble a small HTML fragment that displays the model title, model type, partition/proportion label, descriptor point counts, and an HTML-formatted block of evaluation text. In the ROBERT workflow (see README), the output is intended to be embedded into the final report (the ROBERT_report.pdf/HTML) to present the score and brief metadata for either a "No PFI" or "PFI" model.
    
    Args:
        score_info (str): An HTML-formatted string containing the detailed evaluation text or metric block to be inserted in the column. This value is placed near the bottom of the returned HTML fragment and is expected to already contain any necessary HTML tags (for example line breaks, paragraphs, or inline formatting) because the function does not sanitize or reformat its content. Supplying non-HTML plain text is allowed but may produce suboptimal rendering in the report.
        data_score (dict): A dictionary with model-specific metadata and precomputed strings required to populate the column. The function reads the following keys from this dict: f'robert_score_{suffix}' (a printable score string inserted into the title), 'proportion_ratio_print' (a string containing a substring "-  Proportion " that is split to obtain the partitions ratio), 'ML_model' (the model name placed in the fragment), and f'points_descp_ratio_{suffix}' (the descriptor points counts for the chosen suffix). Missing keys will raise a KeyError. The dictionary is supplied by the reporting pipeline in ROBERT and typically originates from model-evaluation routines.
        suffix (str): A string indicating which score column to build; the code recognizes and handles exactly two domain-specific values: 'No PFI' or 'PFI'. When suffix == 'No PFI' the function uses the global title_no_pfi template to build the caption; when suffix == 'PFI' it uses the global title_pfi template. If suffix has any other value, caption will not be defined and the function will raise an error (UnboundLocalError). The suffix is also used to select the per-suffix keys inside data_score described above.
        spacing (str): A short HTML snippet or spacing string that is interpolated into inline paragraph templates used for consistent visual spacing in the generated HTML fragment. The caller (report generation code) supplies spacing to control indentation or inline spacing in the final report. The function does not validate this string beyond inserting it into the HTML templates ML_line_format and part_line_format.
        eval_only (bool): When False (the default reporting mode), the function builds a title_line using the caption derived from the appropriate global title template and the robert_score_{suffix} value; when True the function overrides the title and returns a concise fixed title_line 'Summary and score of your model (No PFI)'. In practical use, eval_only True is intended to produce a compact summary suitable for evaluation-only displays. This flag only affects the title text; all other fields are still read from data_score.
    
    Behavior, side effects, defaults, and failure modes:
        The function constructs three internal HTML templates (ML_line_format, part_line_format, and column) and fills them with values drawn from spacing and data_score. It relies on the presence of two global string templates, title_no_pfi and title_pfi, which must be defined in the module namespace; if they are missing a NameError will be raised. The function expects data_score['proportion_ratio_print'] to contain the substring '-  Proportion ' and splits on that substring to extract the partitions_ratio; if the substring is absent, an IndexError will occur. If any of the required data_score keys are missing a KeyError will occur. The function performs no I/O, no HTML sanitization, and no type coercion beyond Python's f-string conversion; callers should ensure the input types match the signature. There are no side effects such as writing files or modifying global state (other than reading global title templates).
    
    Returns:
        str: A single string containing an HTML fragment. The fragment contains a bolded title line (either derived from title_no_pfi/title_pfi and the robert_score_{suffix} value, or the fixed summary when eval_only is True), a line with "Model = <ML_model> · <partitions_ratio>", a line with "Points(train+validation):descriptors = <points_descp_ratio_{suffix}>", the provided score_info HTML block, and a trailing paragraph spacer. The returned HTML is intended to be embedded directly into the ROBERT report generation pipeline.
    """
    from robert.report_utils import get_col_score
    return get_col_score(score_info, data_score, suffix, spacing, eval_only)


################################################################################
# Source: robert.report_utils.get_col_text
# File: robert/report_utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def robert_report_utils_get_col_text(type_thres: str):
    """robert.report_utils.get_col_text returns an HTML-formatted column string containing fixed abbreviation mappings used in the report "score" and "abbreviation" sections of the ROBERT PDF/HTML report. This helper assembles a sequence of hard-coded abbreviation descriptions (for example, "ACC: accuracy", "RF: random forest", "SHAP: Shapley additive explanations") into paragraph (<p>) elements with specific inline styles so the resulting string can be embedded directly into the report layout produced by ROBERT (a library for bridging machine learning and chemistry workflows described in the project README).
    
    The function is used during report generation to present concise explanatory text for abbreviations and threshold labels that appear in model evaluation and method-description sections of ROBERT-generated reports. It returns a single str containing multiple HTML paragraph elements; the first paragraph uses a larger negative top margin to reduce spacing relative to section headers and subsequent paragraphs use a smaller negative top margin to compact the column.
    
    Args:
        type_thres (str): A case-sensitive selector that chooses which predefined abbreviation group to format into an HTML column. Accepted literal values (taken from the source code) are 'abbrev_1', 'abbrev_2', and 'abbrev_3', each corresponding to a distinct hard-coded list of abbreviation HTML fragments used by the report. 'abbrev_1' includes items such as '<strong>ACC:</strong> accuracy' and '<strong>GP:</strong> gaussian process'. 'abbrev_2' includes items such as '<strong>KN:</strong> k-nearest neighbors' and '<strong>R2:</strong> coefficient of determination'. 'abbrev_3' includes items such as '<strong>RF:</strong> random forest' and '<strong>SHAP:</strong> Shapley additive explanations'. The caller must supply one of these exact strings; the function does not accept other values, does not perform normalization (for example, lowercasing or stripping), and is case-sensitive.
    
    Returns:
        str: An HTML string representing a vertical column of abbreviation lines ready for inclusion in a ROBERT report. Each list element is wrapped in a <p> element: the very first element is wrapped with the inline style '<p style="text-align: justify; margin-top: -25px;">' and all subsequent elements use '<p style="text-align: justify; margin-top: -8px;">'. Each paragraph is closed with '</p>' and a trailing newline character. The returned string has no further escaping or sanitization applied; it is intended to be embedded into report HTML or converted to PDF by the surrounding report generation code.
    
    Behavior, side effects, defaults, and failure modes:
        The function has no external side effects: it does not read or write files, modify global state, or perform I/O. Behavior is deterministic and entirely driven by the input selector and the hard-coded lists in the source. If type_thres matches one of the documented literals, the corresponding non-empty list of abbreviation fragments is iterated in order and concatenated into the output string; the order of items is preserved and is significant for the visual layout of the report. If type_thres does not match 'abbrev_1', 'abbrev_2', or 'abbrev_3', the local variable abbrev_list is never defined and the function will raise an UnboundLocalError when it attempts to iterate over abbrev_list. The function does not validate or escape HTML content in the hard-coded fragments; callers should not provide external or untrusted HTML via type_thres (type_thres is a selector only), and the function is not designed to accept user-provided abbreviation lists.
    """
    from robert.report_utils import get_col_text
    return get_col_text(type_thres)


################################################################################
# Source: robert.report_utils.get_col_transpa
# File: robert/report_utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def robert_report_utils_get_col_transpa(
    params_dict: dict,
    suffix: str,
    section: str,
    spacing: str
):
    """get_col_transpa generates an HTML snippet (column) that summarizes model-related parameters for the Reproducibility section of a ROBERT report.
    
    Args:
        params_dict (dict): Dictionary of model/run parameters produced or consumed by ROBERT report utilities. This dict is expected to contain keys such as 'type' (model family type, e.g., 'reg' or 'clas'), 'error_type' (used to exclude a combined error entry), 'model' (short code for the chosen estimator, e.g., 'RF', 'GB'), and optionally 'params' (a string representation of a Python dict containing estimator hyperparameters). The function reads these keys to decide which parameters to include in the returned HTML and to map short model codes to human-readable sklearn-like names. Pass the exact params_dict used by other ROBERT report code; missing required keys (for example 'type' or 'error_type') will raise a KeyError in this function.
        suffix (str): Caption selector for the column. The function recognizes the literal values 'No PFI' and 'PFI' and will select global caption variables title_no_pfi and title_pfi respectively to render the bold caption line in the HTML. If a different string is passed, the caption variable will not be set and a NameError or UnboundLocalError may occur; callers should pass exactly 'No PFI' or 'PFI' as used by the ROBERT reporting pipeline.
        section (str): Logical section selector that controls which keys from params_dict are rendered into the column. The function checks for values such as 'model_section' and 'misc_section'. When section == 'model_section', the function will (1) translate a 'model' code from params_dict into a human-readable sklearn model name according to an internal mapping (models_dict), and (2) if a 'params' key is present it will parse it via ast.literal_eval to extract and render individual hyperparameter lines. When section == 'misc_section', only keys listed in the function's misc_params list (['type','error_type','kfold','repeat_kfolds','seed']) are rendered. Use the same section strings used by ROBERT report generation code to ensure consistent output.
        spacing (str): A string used for spacing/indentation inserted into rendered HTML lines (for example a number of spaces). spacing is concatenated into inline HTML text to visually indent caption and parameter lines in the generated column. Provide the spacing string used elsewhere in the report generator to preserve consistent layout.
    
    Behavior, side effects, defaults, and failure modes:
        The function constructs three HTML paragraph templates with inline CSS to reduce vertical spacing between lines and uses spacing to indent text. It excludes a set of keys built from params_dict['error_type'] (an excluded combined error key) and a fixed list of keys defined in excluded_params (including 'train', 'X_descriptors', 'y', 'error_train', 'cv_error', 'names'). It uses params_dict['type'] to determine whether the model family is a Regressor (when value == 'reg') or a Classifier (when value == 'clas') and maps short model codes in params_dict['model'] (e.g., 'RF', 'GB', 'NN') to human-readable sklearn-like names via an internal models_dict. If the 'params' key is present and section == 'model_section', the function expects params_dict['params'] to be a string that represents a Python dict (for example "{'n_estimators': 100}") and parses it with ast.literal_eval; if the string is not a valid Python literal this will raise a ValueError or SyntaxError. The function reads global caption variables title_no_pfi and title_pfi; if these globals are not defined in the module namespace a NameError will be raised. The function iterates over params_dict keys in insertion order (Python dict ordering) and appends matching entries as HTML lines; keys in excluded_params are omitted. There are no external side effects such as file I/O or network access; the only observable effect is the returned HTML string. Errors you may encounter include KeyError for missing expected keys in params_dict, ValueError/SyntaxError from ast.literal_eval on malformed 'params', and NameError/UnboundLocalError when suffix is not one of the two expected literals or when caption globals are undefined.
    
    Returns:
        str: A single HTML-formatted string containing a small "column" suitable for inclusion in the ROBERT Reproducibility section. The string includes a bold caption line determined by suffix, an optional sklearn model name line derived from params_dict['model'] when section == 'model_section', and one or more parameter lines rendered from params_dict (either parsed hyperparameters from 'params' or miscellaneous run parameters). This HTML is intended to be directly embedded into the report generation pipeline (for example concatenated into larger HTML used to produce the report PDF).
    """
    from robert.report_utils import get_col_transpa
    return get_col_transpa(params_dict, suffix, section, spacing)


################################################################################
# Source: robert.report_utils.get_csv_metrics
# File: robert/report_utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def robert_report_utils_get_csv_metrics(file: str, suffix: str, spacing: str):
    """robert.report_utils.get_csv_metrics retrieves the "External test" metrics line from a ROBERT PREDICT dat file and returns a small HTML fragment that can be embedded in the ROBERT report. This function is used by the ROBERT report generation pipeline to extract and present external test performance information (the "csv_test" results recorded by the PREDICT module) as an underlined heading plus an optional metrics paragraph formatted with inline CSS.
    
    Args:
        file (str): Path to the PREDICT dat file to read. The function opens this file with encoding='utf-8' and reads all lines into memory. The file is expected to contain sections whose headings include the literal string 'o  Summary of results' and lines of the form '-  External test : <metrics>'. If the file is missing or unreadable, the underlying open/read will raise the usual I/O exceptions (for example FileNotFoundError, PermissionError, or UnicodeDecodeError). If the file ends before the function's 15-line local search window, an IndexError may be raised by the parser.
        suffix (str): A case-sensitive selector that controls which "Summary of results" block is scanned. The function recognizes exactly two meaningful values used by the ROBERT reports: 'No PFI' and 'PFI'. When suffix == 'No PFI', the code looks for a "Summary of results" line that also contains the literal 'No_PFI:'; when suffix == 'PFI', it looks for a "Summary of results" line that does not contain 'No_PFI:'. If suffix is any other string, the function will not match the expected blocks and will therefore not extract metrics (resulting in an empty string return). This parameter is important in the ROBERT domain because it selects whether metrics come from the PFI (permutation feature importance) variant or the No_PFI variant of the model summary.
        spacing (str): A string used as indentation inside the returned HTML fragment. The function inserts spacing twice (spacing*2) before the underlined "External test metrics" heading and before the metrics text if present. Typical values in ROBERT report templates are combinations of non‑breaking spaces or other whitespace strings used to align HTML content; the function does not validate this string and will use it verbatim.
    
    Behavior and side effects:
        The function reads the entire file into memory and iterates line by line. For each line that contains 'o  Summary of results' it applies the suffix selection logic described above. Once a matching summary header is found, it inspects up to the next 15 lines (a local window of lines[i:i+15]) searching for the first occurrence of '-  External test : '. If it encounters the literal 'o  SHAP' within that window before finding an External test line, it stops searching within that window. When an External test line is found, the function extracts the substring starting at character index 25 of that line (lines[j][25:]) and treats that substring as the metrics text. The function then constructs an HTML fragment that begins with a paragraph containing an underlined "External test metrics" heading and, if metrics were extracted, a second paragraph with the extracted metrics and specific inline CSS (text-align: justify; margin-bottom: 35px). No files are written; the only side effect is reading from disk. The function performs exact string comparisons and slicing as implemented; it does not perform additional validation or normalization of the extracted metrics text.
    
    Failure modes and limitations:
        The function assumes the PREDICT dat file follows ROBERT's expected formatting. If the '-  External test : ' line is missing, or the summary blocks do not match the suffix selection, the function returns an empty string. The extraction uses a fixed slice lines[j][25:], so if the External test line is shorter than 25 characters the extracted result may be an empty string or a truncated value. If the file is truncated near a matched "Summary of results" header such that the code attempts to access lines beyond the end of the list, an IndexError may be raised. The suffix matching is exact and case-sensitive; providing different capitalization or variants will prevent extraction.
    
    Returns:
        str: An HTML fragment (str) suitable for embedding in the ROBERT report. If a matching External test line is found, the returned string contains an underlined "External test metrics" heading inside a <p> tag with inline CSS and a second <p> tag containing the extracted metrics text prefixed by spacing*2. Example structure when metrics are found: '<p style="...">{spacing*2}<u>External test metrics</u></p><p style="...">{spacing*2}{metrics_text}</p>'. If no matching metrics are found the function returns an empty string ('').
    """
    from robert.report_utils import get_csv_metrics
    return get_csv_metrics(file, suffix, spacing)


################################################################################
# Source: robert.report_utils.get_csv_pred
# File: robert/report_utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def robert_report_utils_get_csv_pred(
    suffix: str,
    path_csv_test: str,
    y_value: str,
    names: str,
    spacing: str
):
    """Get the HTML snippet that summarizes external test predictions from a PREDICT csv_test file.
    
    Args:
        suffix (str): Selector for which csv_test variant to read. In the ROBERT workflow this must be either 'No PFI' or 'PFI' (as used by the PREDICT step and report conventions). When 'No PFI' the function selects a CSV file whose filename contains '_No_PFI.csv'. When 'PFI' the function selects a CSV file that contains '_PFI.csv' but does not contain '_No_PFI.csv'. This value controls which on-disk csv_test file is used to build the predictions table and also affects table left margin in the returned HTML.
        path_csv_test (str): File-system path (string) that points to the csv_test file location used by PREDICT. The function derives the folder to search by combining the current working directory (os.getcwd()) with os.path.dirname(path_csv_test) and then globbing for '*.csv' in that folder. This must be a path consistent with the PREDICT/csv_test layout in ROBERT projects; if the derived folder has no matching files or no file matching the suffix rules, the function will fail when attempting to read the CSV file.
        y_value (str): Name of the target column for the property predicted by the model (e.g., an experimental property name used by ROBERT). The CSV is expected to contain predicted values in a column named '{y_value}_pred' and optionally prediction standard deviations in '{y_value}_pred_sd'. If the CSV lacks a column exactly equal to y_value, the function will create y_value from '{y_value}_pred' (this supports cases where only predictions exist, such as some external test outputs). The function uses y_value to determine which columns to read, how to sort, and how to label the table columns in the generated HTML.
        names (str): Name of the column in the CSV that holds the sample identifiers (molecule names, entry IDs, or similar). This column is displayed as the first column in the HTML table. Long names (>12 characters) are truncated to the first 9 characters plus '...' for display purposes in both the header and individual rows.
        spacing (str): A string used for visual indentation in the leading HTML paragraphs (for example one or more non-breaking spaces or plain spaces). The function repeats this string to control the indentation of the paragraph lines that precede the HTML table in the returned snippet.
    
    Returns:
        str: An HTML fragment (string) containing a short explanatory paragraph and a styled HTML table that lists external test predictions sorted by predicted value in descending order. Practical details encoded in the returned string:
            - The fragment begins with a justified paragraph indicating "External test predictions (sorted, max. 20 shown)" and notes which csv_test variant was used (No_PFI or _PFI.csv). The spacing parameter is inserted into those paragraphs.
            - Table CSS is embedded inline to set borders, padding, and justification.
            - The table width is set to 91% and its left margin is 0 when suffix == 'No PFI' or 27 pixels otherwise.
            - Column headers show the names column, the true y_value column (if present), and either '{y_value}_pred ± sd' when '{y_value}_pred_sd' is present in the CSV or '{y_value}_pred' otherwise. Header strings longer than 12 characters are truncated to 9 characters plus '...'.
            - Rows are sorted by '{y_value}_pred' in descending order and formatted so numeric values are rounded to two decimal places. If '{y_value}_pred_sd' is absent (e.g., classification outputs), a list of zeros is used internally for sd and only the '{y_value}_pred' value is displayed.
            - If there are more than 20 entries, the function shows the top 10, then an ellipsis row, then the bottom 10 (i.e., a maximum of 20 rows displayed with an ellipsis in the middle).
            - Individual name strings longer than 12 characters are truncated to 9 characters plus '...' in table rows.
            - The function reads the CSV using pandas.read_csv(..., encoding='utf-8'), so pandas must be available and the CSV must be readable with UTF-8 encoding.
        Side effects and failure modes:
            - The function performs file-system operations: it uses os.getcwd(), os.path.dirname(path_csv_test), and glob.glob to find CSV files in the derived folder, then calls pandas.read_csv to load the selected file. These operations can raise exceptions such as FileNotFoundError, pandas parsing errors, or an UnboundLocalError if no csv file matching the suffix selection logic is found (csv_test_file will be undefined). Such exceptions are not caught inside the function.
            - The function assumes the CSV contains a column named '{y_value}_pred' (predictions). If that column is missing, KeyError will occur when attempting to access it; if the explicit true y_value column is missing, the function substitutes '{y_value}_pred' into y_value for display as described above.
            - The function relies on exact filename conventions ('_No_PFI.csv' and '_PFI.csv') and exact column-name conventions ('{y_value}_pred' and optional '{y_value}_pred_sd' and the names column). Deviations from these conventions will cause the selection, formatting, or reading logic to fail or produce incomplete output.
            - The returned HTML is intended for inclusion in ROBERT reports (for example embedded in the PDF/HTML report generation pipeline) and not for use as raw machine-readable data.
    """
    from robert.report_utils import get_csv_pred
    return get_csv_pred(suffix, path_csv_test, y_value, names, spacing)


################################################################################
# Source: robert.report_utils.get_metrics
# File: robert/report_utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def robert_report_utils_get_metrics(file: str, suffix: str, spacing: str):
    """Get the formatted summary of regression metrics from a PREDICT ".dat" file for inclusion in ROBERT reports.
    
    This function opens a PREDICT dat file, locates the block that begins with the line containing the literal text "o  Summary of results" and either the presence or absence of the substring "No_PFI:" depending on the suffix argument, extracts the nearby lines that contain the numeric summary reported by PREDICT, applies light post-processing (remove the first 8 characters of each metric line, replace the literal token "R2" with the HTML-safe "R<sup>2</sup>", and optionally prefix lines with the provided spacing), then wraps the concatenated lines inside an HTML <pre> element styled with "text-align: justify; margin-top: 10px;". The returned HTML fragment is intended to be embedded in ROBERT-generated reports (HTML or HTML-to-PDF pipelines) so the raw PREDICT summary appears as a preformatted, readable block in the final report.
    
    Args:
        file (str): Path to the PREDICT ".dat" file to read. This file is opened for reading with UTF-8 encoding and read entirely into memory. The function has no internal error handling for file access or decoding errors, so callers should expect FileNotFoundError if the path does not exist and UnicodeDecodeError if the file cannot be decoded as UTF-8.
        suffix (str): Selector string that controls which "Summary of results" block is extracted. This function only recognizes the exact values 'No PFI' and 'PFI' as used by the PREDICT output conventions in this codebase. If suffix == 'No PFI', the function looks for a "Summary of results" line that also contains "No_PFI:". If suffix == 'PFI', the function looks for a "Summary of results" line that does not contain "No_PFI:". If the requested pattern is not found in the file, the function will produce an empty summary block rather than raising an error.
        spacing (str): A string used as indentation when formatting lines for the 'PFI' suffix. When suffix == 'PFI', spacing is concatenated twice at the start of each extracted metric line (i.e., the function inserts spacing + spacing before the trimmed metric text). For suffix == 'No PFI' the spacing argument is ignored. spacing must be provided as a string; the function does not validate its contents.
    
    Returns:
        str: An HTML fragment (type str) containing a single <pre> element with inline style 'text-align: justify; margin-top: 10px;' whose inner text is the concatenated, post-processed metric lines extracted from the PREDICT dat file. The returned string is ready to be embedded into ROBERT report templates. If the function fails to find a matching "Summary of results" block, the returned HTML fragment will contain an empty preformatted block. No file write occurs; the only side effect is reading the specified file. The function does not catch IO or decoding exceptions raised while opening or reading the file.
    """
    from robert.report_utils import get_metrics
    return get_metrics(file, suffix, spacing)


################################################################################
# Source: robert.report_utils.get_outliers
# File: robert/report_utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def robert_report_utils_get_outliers(file: str, suffix: str, spacing: str):
    """robert.report_utils.get_outliers: Retrieve and format the outliers summary section for a ROBERT PREDICT/VERIFY report.
    
    This function reads a plain-text ".dat" report file produced by ROBERT PREDICT/VERIFY workflows, locates the textual region that documents saved outlier plots, extracts the train and test outlier lines via locate_outliers, and builds an HTML-formatted column fragment that can be embedded in the program's PDF/HTML report. In the ROBERT domain (chemistry-focused regression model reporting), this is used to present which samples were flagged as outliers and to include the corresponding outliers plot reference in the final report layout. The function does not modify the input file; it only reads and formats content for downstream report generation.
    
    Args:
        file (str): Path to the input ".dat" file created by PREDICT or VERIFY. The file is opened for reading with UTF-8 encoding. The file must be readable by the process; if the path does not exist or permissions prevent reading, a built-in IOError/OSError will be raised by open and propagate to the caller.
        suffix (str): A string that determines which outliers-plot entries to consider when scanning the file. Expected values in current code are exactly 'No PFI' or 'PFI'. When suffix == 'No PFI', the function looks for lines containing both 'o  Outliers plot saved' and 'No_PFI.png'; when suffix == 'PFI', it looks for 'o  Outliers plot saved' entries that do not reference 'No_PFI.png'. If an unexpected suffix value is provided, the function will fail when attempting to select the report column title (UnboundLocalError or NameError), so callers should use one of the documented suffix strings.
        spacing (str): A string used to indent and join summary lines in the generated HTML fragment. The function uses spacing*2 to prefix the outliers header and to join the summary list; callers should supply the spacing sequence that matches the rest of the report layout (for example, a number of non-breaking spaces or a specific HTML spacing token used elsewhere in the ROBERT report generator).
    
    Behavior and side effects:
        1. The function opens the given file with encoding='utf-8' and reads all lines into memory. Large files will therefore consume memory proportional to file size.
        2. It scans each line for the literal marker 'o  Outliers plot saved'. For each matching line it uses the provided suffix logic (see Args) to decide whether to call locate_outliers for that line index. locate_outliers is expected to be defined in the same module and to accept (index, lines) returning two lists: train_outliers and test_outliers (lists of strings). If locate_outliers or the expected title variables title_no_pfi / title_pfi are not defined in the module namespace, a NameError will be raised.
        3. The function constructs a textual summary beginning with the header "<u>Outliers (max. 10 shown)</u>" (this header text is literal in the summary and indicates the intended display policy; the actual number of items returned depends on locate_outliers output). It concatenates train_outliers and test_outliers into the summary and then builds an HTML snippet that includes a bolded title (from the module-level variables title_no_pfi or title_pfi selected according to suffix) followed by a <pre> block containing the preformatted summary. The spacing argument is used to control indentation in both the title and the <pre> content.
        4. No files are written or modified by this function; it only returns a formatted string for inclusion in reports.
    
    Failure modes and prerequisites:
        - File access errors (FileNotFoundError, PermissionError) will be raised by open and propagate.
        - If suffix is not 'No PFI' or 'PFI', title_col will not be set and the function will raise an UnboundLocalError or NameError when building the HTML column.
        - The function depends on locate_outliers being implemented in the same module and returning two lists of strings; if locate_outliers raises or returns unexpected types, behavior will be undefined or result in runtime errors during string concatenation.
        - The function also depends on module-level variables title_no_pfi and title_pfi being defined; if they are absent, a NameError will be raised.
        - The header text contains the literal phrase "max. 10 shown" but the function does not itself truncate lists to 10 items; truncation, if required, must be performed by locate_outliers or by earlier processing.
    
    Returns:
        str: An HTML fragment (string) representing a single "column" for the ROBERT report that contains the outliers section. The returned string contains a paragraph element with a bolded title (selected from module-level title_no_pfi or title_pfi based on suffix) followed by a <pre> block with the indented, preformatted summary lines. This string is intended to be embedded into the larger report generation pipeline (e.g., combined into PDF/HTML output).
    """
    from robert.report_utils import get_outliers
    return get_outliers(file, suffix, spacing)


################################################################################
# Source: robert.report_utils.get_predict_scores
# File: robert/report_utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def robert_report_utils_get_predict_scores(
    dat_predict: list,
    suffix: str,
    pred_type: str,
    data_score: dict
):
    """robert.report_utils.get_predict_scores calculates and aggregates numeric scores parsed from the textual output produced by the PREDICT module of the ROBERT package. It parses lines in dat_predict (a list of strings) to extract model type, cross-validation and test results (R2/RMSE for regression or MCC for classification), datapoints:descriptors ratio and outlier/proportion text, and computes derived metrics used by ROBERT for model quality assessment and report generation (scaled RMSE, combined scores, penalties, CV SD coverage). This function is used by the ROBERT reporting pipeline to convert raw PREDICT output into the standardized set of score fields stored in data_score for downstream PDF report generation and model comparison.
    
    Args:
        dat_predict (list): The raw PREDICT output as a list of text lines (strings). Each element is one printed line from the PREDICT module output. The function searches for specific substrings and line offsets (for example '------- ', '(No PFI)', 'with PFI', '   - Model:', 'o  Summary of results', '-fold CV : R2 =', 'Test : R2 =', '-  y range of dataset', '5-fold', '-  Test :', and '-  Average SD in test set') to identify and extract numeric values. The caller must supply the exact sequence of lines produced by PREDICT; malformed, truncated or reordered lines can produce missing keys or conversion errors (ValueError) when numeric parsing is attempted.
        suffix (str): A short label string used to select which block of PREDICT output to parse and to namespace keys written into data_score. Typical values used in ROBERT are 'No PFI' and 'PFI' to indicate whether permutation feature importance was used. The function uses suffix both to determine when to start parsing blocks (it looks for markers that include '(No PFI)' or 'with PFI' together with '------- ') and to compose output keys such as 'rmse_score_{suffix}', 'r2_cv_{suffix}', 'cv_score_combined_{suffix}', etc. The suffix is required and is directly embedded in resulting data_score keys, so use consistent suffix values when calling this function.
        pred_type (str): Prediction task type, exactly 'reg' for regression or 'clas' for classification. For 'reg', the function extracts R2 and RMSE (from CV and Test lines) and computes scaled RMSE (RMSE divided by y range of the dataset times 100), combined scores that include RMSE-based scores and R2 penalties, and a stability/difference score between CV and Test RMSE. For 'clas', the function extracts MCC values from CV and Test lines and computes scores using the same score_rmse_mcc helper (which interprets MCC when pred_type is 'clas'); combined scores for classification are taken equal to the MCC-based scores (no additional penalty). Passing any other pred_type will prevent the function from executing the documented parsing branches and may lead to incomplete keys in data_score.
        data_score (dict): Mutable mapping that will be updated in place with parsed values and derived scores. The function initializes or sets a number of keys namespaced by suffix (for example 'rmse_score_{suffix}', 'cv_type_{suffix}', 'ML_model', 'proportion_ratio_print', 'points_descp_ratio_{suffix}', 'r2_cv_{suffix}', 'r2_test_{suffix}', 'rmse_cv_{suffix}', 'rmse_test_{suffix}', 'y_range_{suffix}', 'scaled_rmse_cv_{suffix}', 'scaled_rmse_test_{suffix}', 'cv_score_rmse_{suffix}', 'test_score_rmse_{suffix}', 'cv_penalty_r2_{suffix}', 'test_penalty_r2_{suffix}', 'cv_score_combined_{suffix}', 'test_score_combined_{suffix}', 'factor_scaled_rmse_{suffix}', 'diff_scaled_rmse_score_{suffix}', 'cv_4sd_{suffix}', 'cv_range_cov_{suffix}', 'cv_sd_score_{suffix}'). The function both reads and writes data_score; it expects the caller to provide a dict (possibly pre-populated) and will mutate it. The function relies on external helper functions score_rmse_mcc(pred_type, value) and calc_penalty_r2(value) being available in the same module; if these helpers are missing a NameError will be raised.
    
    Returns:
        dict: The same data_score dictionary passed as input, returned after in-place updates. The returned dictionary contains the parsed numeric values, derived metrics, and score fields described above. If parsing fails for some expected numeric field, the function may raise ValueError (from float conversion), KeyError (if dependent keys such as y_range_{suffix} are missing when computing ratios), or ZeroDivisionError (for example if y_range_{suffix} is zero). Additionally, if the expected PREDICT output markers are not present in dat_predict, some keys will not be set; callers should validate the presence and numeric types of required keys after the call. The function also sets sensible defaults where present in the code (for example data_score[f'rmse_score_{suffix}'] is initialized to 0 and data_score[f'cv_type_{suffix}'] is initialized to "10x 5-fold CV"), and it enforces non-negative combined scores by clamping to zero when necessary.
    
    Behavioral notes and failure modes:
        - The function scans dat_predict line-by-line and uses a start_data flag controlled by suffix and marker lines to determine which block to parse. For suffix 'No PFI' it starts when a line contains both '------- ' and '(No PFI)' and stops when encountering a '------- ' line with 'with PFI'. For suffix 'PFI' it starts when '------- ' and 'with PFI' are present.
        - For regression (pred_type == 'reg'), the function expects specific line offsets following the 'o  Summary of results' marker to contain datapoints:descriptors ratio, a points:descriptors printed value, CV and Test R2 and RMSE lines and a y range line; these are parsed using string splitting and converted to float. It then computes scaled RMSE as (rmse / y_range) * 100 rounded to two decimals, looks up penalty via calc_penalty_r2, computes combined scores (RMSE-based score plus R2 penalty) and clamps negative combined scores to zero. It also computes factor_scaled_rmse and assigns a diff score using thresholds (<=1.25 -> +2, <=1.5 -> +1) to quantify stability between CV and Test RMSEs.
        - For classification (pred_type == 'clas'), the function extracts MCC values from the CV and Test result lines by searching comma-separated parts for 'MCC=...' and stores them under r2_cv_{suffix} and r2_test_{suffix} for consistency with downstream naming; score_rmse_mcc is then applied to these MCC values to obtain CV and Test scores and the combined scores are set equal to those values (no additional R2 penalty).
        - The function also handles average SD in test set lines for regression: it reads the SD value, multiplies by four to obtain 4*SD, computes the fraction of y-range covered, and assigns a small integer cv_sd_score using thresholds (<=0.25 -> +2, <=0.50 -> +1). These fields are written into namespaced keys in data_score.
        - Because parsing relies on fixed textual patterns and positional offsets relative to markers, changes in the textual format of the PREDICT output will break parsing. Callers should ensure dat_predict matches the expected format produced by the ROBERT PREDICT module and should catch and handle ValueError, KeyError, and ZeroDivisionError when invoking this function.
    """
    from robert.report_utils import get_predict_scores
    return get_predict_scores(dat_predict, suffix, pred_type, data_score)


################################################################################
# Source: robert.report_utils.get_spacing_col
# File: robert/report_utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def robert_report_utils_get_spacing_col(suffix: str, spacing_PFI: str):
    """Assign spacing string for a report column based on whether a "PFI" column is present.
    
    Args:
        suffix (str): A string flag indicating which column type is being formatted in the ROBERT report generation pipeline. In the codebase this function is used when composing PDF/report table columns and is expected to be exactly either 'No PFI' or 'PFI'. When suffix == 'No PFI' the function returns an empty string so no extra spacing is applied for the column; when suffix == 'PFI' the function returns the spacing provided by spacing_PFI so the report includes the spacing required for a Permutation Feature Importance (PFI) column.
        spacing_PFI (str): A string that encodes the spacing to apply when a PFI column is present in the report. In the context of ROBERT report utilities this typically contains whitespace or template/markup spacing used when constructing PDF or text report columns. This value is returned verbatim when suffix == 'PFI'.
    
    Returns:
        str: The spacing string to use for the requested column. Explicit behaviors:
            - If suffix == 'No PFI', returns the empty string '' (no spacing).
            - If suffix == 'PFI', returns spacing_PFI (apply PFI-specific spacing).
            - If suffix has any other value, the function does not assign a return value in the current implementation and will raise an UnboundLocalError at runtime; callers in the ROBERT report generation code must ensure suffix is exactly 'No PFI' or 'PFI' to avoid this failure mode.
    
    Side effects:
        None. This function performs no I/O and does not mutate inputs or global state; it simply returns a string used by higher-level report formatting routines.
    """
    from robert.report_utils import get_spacing_col
    return get_spacing_col(suffix, spacing_PFI)


################################################################################
# Source: robert.report_utils.get_verify_scores
# File: robert/report_utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def robert_report_utils_get_verify_scores(
    dat_verify: list,
    suffix: str,
    pred_type: str,
    data_score: dict
):
    """get_verify_scores calculates and stores verification scores extracted from the VERIFY module output lines for a ROBERT model report.
    
    This function is used in ROBERT's report utilities to parse the textual output produced by the VERIFY tests (part of the model validation workflow described in the README) and to derive numeric and categorical quality indicators for either regression or classification predictors. It inspects a sequence of lines (dat_verify) produced by VERIFY, selects the block corresponding to the requested suffix ('No PFI' or 'PFI'), evaluates whether specific subtests are flagged as UNCLEAR or FAILED, parses sorted cross-validation error/metric results, scales regression errors relative to a stored y_range, classifies each sorted result as pass/fail/min/max according to the VERIFY rules implemented here, and updates the provided data_score dictionary with computed fields used later in report generation and scoring summaries.
    
    Args:
        dat_verify (list): The raw VERIFY output as a list of strings, where each string is one line of the VERIFY report. The function scans these lines to find markers such as "------- " combined with "(No PFI)" or "with PFI" to select which block to parse, looks for "Original RMSE (" (for regression) or "Original MCC (" (for classification) and then reads subsequent lines for UNCLEAR/FAILED indicators and a "- Sorted " line containing a Python literal list of numeric sorted results. This argument is required and must preserve the VERIFY textual structure produced by ROBERT's VERIFY module.
        suffix (str): A suffix string that determines which VERIFY block to process. Accepted values used by the code are exactly 'No PFI' and 'PFI'. When suffix == 'No PFI', the function will begin collecting data when it finds a line containing both "------- " and "(No PFI)" and will stop if it later encounters the "with PFI" separator. When suffix == 'PFI', the function will begin collecting data when it finds a "------- " line containing "with PFI". The suffix is used to construct keys stored in data_score (for example scaled_rmse_sorted_No PFI).
        pred_type (str): A short string indicating predictor type; the implementation treats 'reg' (case-insensitive) as regression and any other value as classification. For regression, the function searches for "Original RMSE (" and scales sorted RMSE values by the precomputed y_range entry in data_score (key: f"y_range_{suffix}"), multiplying by 100 and rounding to two decimals. For classification, the function searches for "Original MCC (" and does not perform scaling on the parsed sorted metric values (MCC).
        data_score (dict): A mutable dictionary used to both read required auxiliary values and store computed verification scores. For regression only, the function expects a numeric entry keyed by f"y_range_{suffix}" to scale RMSE values. The function will add or overwrite the following keys in data_score (where {suffix} is the literal suffix argument and {error_keyword} is 'rmse' for regression or 'mcc' for classification):
            scaled_{error_keyword}_sorted_{suffix} (list): For regression, a list of floats equal to each parsed sorted RMSE value divided by data_score[f"y_range_{suffix}"], multiplied by 100 and rounded to 2 decimals. For classification, the list equals the parsed sorted MCC numeric results unchanged.
            min_scaled_{error_keyword}_{suffix} (float/int): The minimum value of scaled_{error_keyword}_sorted_{suffix}.
            max_scaled_{error_keyword}_{suffix} (float/int): The maximum value of scaled_{error_keyword}_sorted_{suffix}.
            scaled_{error_keyword}_results_sorted_{suffix} (list): A list of classification strings for each sorted value: for regression entries are 'min' (index of min), 'pass' (value <= min*1.25), or 'fail' (otherwise); for classification entries are 'max' (index of max), 'pass' (value >= max*0.75), or 'fail' (otherwise).
            flawed_mod_score_{suffix} (int): Aggregate flawed-test penalty computed by decrementing for 'UNCLEAR' (-1) and 'FAILED' (-2) occurrences in the three VERIFY subtests immediately after the "Original ..." line; the final stored value is capped at a maximum of 1 (so any aggregated negative penalty larger than 1 is set to 1).
            failed_tests_{suffix} (int): Count of 'FAILED' subtest occurrences found in the scanned VERIFY block.
            sorted_cv_score_{suffix} (int): An integer score computed as int(number_of_'pass'_entries_in scaled_{error_keyword}_results_sorted_{suffix} divided by 2). This value is intended for downstream scoring summaries where each pair of pass entries contributes one point.
    The function mutates data_score in place and also returns it.
    
    Returns:
        dict: The same data_score dictionary passed as input, updated with the computed verification fields described above. The returned dictionary contains the new keys added by this function for the given suffix and error keyword. If the function cannot find the expected "- Sorted " line or the "Original ..." line for the requested suffix, the corresponding scaled and result keys will not be created and downstream code expecting them may raise KeyError.
    
    Behavior, side effects, defaults, and failure modes:
        The function scans dat_verify sequentially and uses a start_data boolean determined by suffix-specific separators in the VERIFY text; only lines encountered while start_data is True are considered. It determines which error keyword to search for by treating pred_type.lower() == 'reg' as regression (error_keyword = 'rmse') and any other value as classification (error_keyword = 'mcc').
        Immediately after detecting the "Original RMSE (" or "Original MCC (" line the function inspects the next three lines for the tokens 'UNCLEAR' and 'FAILED'. Each 'UNCLEAR' decrements an internal flawed_score by 1; each 'FAILED' decrements flawed_score by 2 and increments failed_tests by 1. After these checks the function expects a "- Sorted " line four lines after the "Original ..." line that contains a Python-list literal of numeric sorted results; that literal is parsed with ast.literal_eval to produce sorted_cv_results. For regression, the code divides each numeric element of sorted_cv_results by data_score[f"y_range_{suffix}"], multiplies by 100 and rounds to two decimals to produce scaled values.
        The function sets flawed_mod_score_{suffix} to the flawed_score (after applying a cap that sets any value greater than 1 to 1), sets failed_tests_{suffix} to failed_tests, and sets sorted_cv_score_{suffix} to sorted_cv_score computed as described above. It mutates and returns data_score for use by upstream report generation and scoring functions.
        Potential failure modes include KeyError when a required key such as data_score[f"y_range_{suffix}"] is missing (regression case), ValueError or SyntaxError from ast.literal_eval if the "- Sorted " line does not contain a valid Python list literal, and IndexError if the expected lines around the "Original ..." marker are missing. The caller should ensure dat_verify contains the VERIFY output block corresponding to the requested suffix and that data_score provides required auxiliary entries for scaling before calling this function.
    """
    from robert.report_utils import get_verify_scores
    return get_verify_scores(dat_verify, suffix, pred_type, data_score)


################################################################################
# Source: robert.report_utils.locate_outliers
# File: robert/report_utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def robert_report_utils_locate_outliers(i: int, lines: list):
    """Locate and extract the Train and Test outlier lines from a PREDICT summary section of a .dat file.
    
    This function is used in the ROBERT report generation pipeline to parse the "PREDICT" summary block produced by model evaluation tools and to collect the textual lines that list training and testing outliers for later inclusion in the PDF or text report. Starting from the line index i (typically the index where the "PREDICT" header was found), the function scans subsequent lines to find the "Train:" and "Test:" subsections and extracts up to the reported outlier entries. Extracted lines have the first six characters removed (commonly line numbering or fixed indentation in the .dat output) and long entries are wrapped at 54 characters (so an entry may contain an embedded newline). The function does not modify the input list and returns two new lists containing the extracted strings for Train and Test respectively.
    
    Args:
        i (int): Zero-based integer index in lines that identifies the position immediately before the PREDICT summary scan should start. In ROBERT usage this is typically the index of a header line (for example the line containing "PREDICT") so the function begins scanning at i+1. If i is out of range such that the scan has no subsequent lines, the function returns two empty lists.
        lines (list): The file content presented as a list of lines (each element expected to be a string, as produced by file.readlines() for a .dat report). The function scans this list sequentially from index i+1 to locate the 'Train:' and 'Test:' markers and to collect the outlier description lines that follow each marker.
    
    Returns:
        tuple: A pair (train_outliers, test_outliers) where each element is a list of strings extracted from the PREDICT summary:
            train_outliers (list): List of extracted Train subsection lines (up to 11 entries: up to 10 outlier lines plus the line with the percentage of outliers). Each string has the original first six characters removed and may contain an embedded newline if the original text exceeded the internal wrap length (54 characters).
            test_outliers (list): List of extracted Test subsection lines (up to 11 entries, analogous to train_outliers) with the same slicing and wrapping behavior.
    
    Behavior and failure modes:
        The function scans forward from i+1 until it encounters a blank line (a line whose split() is empty) or the end of the lines list. When it finds 'Train:' it collects subsequent non-empty lines until a 'Test:' marker is reached. When it finds 'Test:' it collects lines until a blank line ends the block. For each collected line the function slices off the first six characters (lines[k][6:]) and, if the remaining text exceeds 54 characters, inserts a newline after the first 54 characters to keep display width consistent in reports. The append condition uses <= 10 so at most 11 entries per list may be returned (intended to capture up to 10 outliers plus the summary percentage line). The function has no side effects on the input list. If elements of lines are not strings, slicing operations may raise runtime errors; if i is not an integer the caller will encounter a type-related error before or during the range computation. If the expected 'Train:' or 'Test:' markers are absent the corresponding returned list will be empty.
    """
    from robert.report_utils import locate_outliers
    return locate_outliers(i, lines)


################################################################################
# Source: robert.report_utils.remove_quot
# File: robert/report_utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def robert_report_utils_remove_quot(name: str):
    """Remove a single leading and/or trailing quotation mark from a string used as an identifier or label.
    
    This function is a small utility used by ROBERT's report generation utilities (robert.report_utils) to sanitize names that may have been wrapped in single (') or double (") quotation marks. In the context of ROBERT, this helps ensure chemical names, column headers, filenames, or other labels appear in generated reports (PDF/CSV/HTML) without surrounding quotes that would be visually distracting or semantically misleading. The function only examines the first and last character of the input string and removes one leading quote and/or one trailing quote if present; it does not alter interior characters, remove multiple quotes, or trim whitespace.
    
    Args:
        name (str): The input text representing a name or label to sanitize. Must be a Python str containing the characters to inspect. Typical values are chemical identifiers, column names, or filenames originating from upstream parsing or file formats that sometimes add surrounding quotes. The function checks name[0] and name[-1] for a single or double quote character and removes each matching quote exactly once.
    
    Returns:
        str: A new string derived from the input with at most one leading quotation mark and at most one trailing quotation mark removed. If the input had no leading or trailing single or double quotes, the returned string is identical to the input. The function performs no other normalization (for example, it does not strip whitespace or remove interior quotes).
    
    Raises:
        IndexError: If the input string is empty or becomes empty during processing (for example, if the input is a single-character string that is a quote, the function will remove the first character and then attempt to inspect the last character, leading to IndexError). Callers should ensure the input string has sufficient length when such inputs are possible.
        TypeError: If a non-str value is passed (the function indexes the object and expects str semantics), a TypeError or other indexing-related exception may be raised; callers should pass a str as documented.
    
    Side effects:
        None. The function is pure and returns a new string without modifying the original object.
    """
    from robert.report_utils import remove_quot
    return remove_quot(name)


################################################################################
# Source: robert.report_utils.repro_info
# File: robert/report_utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def robert_report_utils_repro_info(modules: list):
    """robert.report_utils.repro_info retrieves and aggregates reproducibility metadata for a list of ROBERT modules by reading each module's "<module>_data.dat" file in the current working directory. This function is used by the ROBERT reporting utilities to populate the "Reproducibility" section of generated reports (for example the PDF report described in the README) and to support reproducibility tests by collecting version, citation, command-line, runtime and environment information recorded by individual modules.
    
    Args:
        modules (list): A list of module directory identifiers to inspect. Each element is expected to correspond to a subdirectory name in the current working directory that may contain a file named "<module>_data.dat" (for example "generate", "verify", "predict" or other ROBERT module names). The function will attempt to open and parse "<cwd>/<module>/<module>_data.dat" for every entry in this list. Missing or nonexistent dat files are ignored (no entry is added to the returned dat_files for that module).
    
    Returns:
        tuple: A 7-tuple with the following elements in this exact order and types:
            version_n_date (str): The first encountered "ROBERT v..." line from any parsed dat file, typically containing the program version and date recorded by a module. If no such line is found this will be an empty string. This value is used to show which ROBERT version produced the recorded data in the reproducibility section.
            citation (str): A citation string extracted from lines containing "How to cite: " in the dat files. If multiple modules provide citations, the last matching line read overwrites the previous value; if none are found this will be an empty string. This is used to populate the recommended citation in reports.
            command_line (str): The command-line string extracted from a line beginning with "Command line used in ROBERT: " in the dat files. The function will avoid overwriting a previously stored command line if that stored string already contains the "--csv_name" token (this preserves the command line that includes the produced CSV name when multiple modules report command lines). If no command-line line is found this will be an empty string.
            python_version (str): The running Python interpreter version determined via platform.python_version(). If the platform import or version lookup fails, the value will be "(version could not be determined)". This documents the Python runtime used when the reports were generated.
            intelex_version (str): The version of the "scikit-learn-intelex" accelerator determined via pkg_resources.get_distribution("scikit-learn-intelex").version if the dat files do not indicate that the accelerator is absent. If the dat files explicitly contain the line "The scikit-learn-intelex accelerator is not installed" the function returns the literal "not installed". If the package is present but its version cannot be determined, the string "(version could not be determined)" is returned. This value is used to document whether the scikit-learn-intelex accelerator was used and which version.
            total_time (float): The cumulative runtime in seconds found by summing numeric values from lines that contain both the words "Time" and "seconds" across all parsed dat files. The sum is rounded to two decimal places before being returned. If no such lines are found the value is 0.0. This represents the total elapsed time recorded by the modules and is used in reproducibility summaries.
            dat_files (dict): A dictionary mapping each module name (as provided in the modules argument) to a list of strings representing the full contents (lines) of that module's "<module>_data.dat" file that was successfully opened and read. Modules without an existing or readable dat file will not have an entry in this dictionary. The file reading uses UTF-8 encoding with errors="replace", so malformed bytes will be replaced rather than causing a UnicodeDecodeError.
    
    Behavior and side effects:
        The function reads files from disk using the current working directory (os.getcwd()) and Path to construct file paths of the form "<cwd>/<module>/<module>_data.dat". Files are opened for reading with encoding="utf-8" and errors="replace"; this prevents Unicode decoding errors but may alter invalid byte sequences. While reading, the function inspects each line for specific markers to extract version, citation, command line, scikit-learn-intelex presence, and per-line timing data. total_time is built by converting the third whitespace-separated token on lines that contain both "Time" and "seconds" into a float; if such lines are not well-formed (missing tokens or non-numeric data) this conversion may raise an exception (IndexError or ValueError) because parsing is not protected by an inner try/except. The function does not write to disk or modify existing files.
    
    Failure modes and defaults:
        If a dat file is missing or unreadable, that module is simply omitted from dat_files and no exception is raised for the absence itself. However, malformed timing lines or unexpected line formats inside an existing dat file may raise parsing exceptions (IndexError, ValueError) because individual line parsing is not fully guarded. Determination of python_version and scikit-learn-intelex version is attempted inside try/except blocks: failure to import platform yields "(version could not be determined)" for python_version; failure to obtain the scikit-learn-intelex distribution yields "(version could not be determined)" unless the dat files explicitly indicated the accelerator was not installed, in which case intelex_version is set to "not installed". The function assumes that module names in modules are valid directory names and does not validate their contents beyond attempting to open the expected dat file.
    """
    from robert.report_utils import repro_info
    return repro_info(modules)


################################################################################
# Source: robert.report_utils.revert_list
# File: robert/report_utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def robert_report_utils_revert_list(list_tuple: list):
    """robert.report_utils.revert_list swaps the order of a two-element list when the second element contains the literal substring 'No_PFI'. This helper is used by ROBERT report generation utilities to enforce a consistent ordering for two-component entries (for example, entries that may indicate presence/absence of permutation feature importance data) so downstream report formatting and comparisons behave predictably.
    
    This function implements a targeted conditional swap: it only changes the ordering when the input is a list of exactly two items and the second item contains 'No_PFI'. The implementation avoids in-place mutation with list.reverse() (a previously observed issue) by constructing and returning a new two-element list when swapping is needed; otherwise it returns the original list object unchanged.
    
    Args:
        list_tuple (list): A list object expected to contain two components. In the ROBERT reporting context this represents a paired entry (for example, a label and a status or data placeholder). The function inspects the second element (index 1) for the literal substring 'No_PFI' and, if found and the list has length exactly 2, returns a new list with the two elements swapped. The parameter must be a Python list; if it is not a list the behavior is not guaranteed by this function. If the second element is not an iterable or does not support the membership test ('No_PFI' in element), a TypeError may be raised.
    
    Returns:
        list: The resulting list after conditional reordering. If the input list has length exactly 2 and the second element contains the substring 'No_PFI', a new list with the elements in reversed order is returned (first element becomes second, second becomes first). For all other inputs (length not equal to 2 or second element does not contain 'No_PFI'), the original list object is returned unchanged. No other side effects occur.
    """
    from robert.report_utils import revert_list
    return revert_list(list_tuple)


################################################################################
# Source: robert.report_utils.score_rmse_mcc
# File: robert/report_utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def robert_report_utils_score_rmse_mcc(pred_type: str, scaledrmse_mcc_val: float):
    """Compute a discrete performance score from either a scaled RMSE value (regression) or an MCC value (classification) using fixed thresholds employed in ROBERT report generation and model evaluation.
    
    This function is used in the ROBERT reporting utilities to convert a single numeric performance metric into a small integer score that contributes to overall model quality summaries in automated reports. For regression models (pred_type == 'reg') the input scaledrmse_mcc_val is interpreted as a scaled root-mean-square error where smaller values indicate better performance; for classification models (any pred_type other than 'reg') the input is interpreted as Matthew's Correlation Coefficient (MCC) where larger values indicate better performance. The mapping of metric values to integer scores follows hard-coded thresholds so the function is deterministic and has no side effects.
    
    Args:
        pred_type (str): Indicator of the prediction task type. If pred_type is the exact string 'reg', the function treats scaledrmse_mcc_val as a scaled RMSE for regression and applies the regression thresholds. For any other string value, the function treats scaledrmse_mcc_val as an MCC for classification and applies the classification thresholds. This parameter controls which branch of the scoring logic is executed and therefore directly affects interpretation of scaledrmse_mcc_val in the context of ROBERT model reporting.
        scaledrmse_mcc_val (float): The numeric performance metric to score. When pred_type == 'reg', this is interpreted as a scaled RMSE (lower is better) and compared against the regression thresholds in units consistent with the caller. When pred_type != 'reg', this is interpreted as an MCC (higher is better) and compared against the classification thresholds. The function performs numeric comparisons with the provided float value; passing non-numeric types will typically result in a TypeError from the underlying comparisons.
    
    Behavior and thresholds:
        For regression (pred_type == 'reg'), the function awards 0 to 2 points:
            - If scaledrmse_mcc_val <= 10: returns 2 points.
            - Else if scaledrmse_mcc_val <= 20: returns 1 point.
            - Else: returns 0 points.
        For classification (pred_type != 'reg'), the function awards 0 to 3 points:
            - If scaledrmse_mcc_val > 0.75: returns 3 points.
            - Else if scaledrmse_mcc_val > 0.5: returns 2 points.
            - Else if scaledrmse_mcc_val > 0.3: returns 1 point.
            - Else: returns 0 points.
        Note that regression thresholds use inclusive comparisons (<=) while classification thresholds use strict greater-than (>) comparisons. Equality at the boundary values is therefore handled according to these operators.
    
    Side effects:
        This function is pure and has no side effects; it does not modify inputs, global state, or perform I/O. It only computes and returns an integer score.
    
    Failure modes and edge cases:
        - If pred_type is not a string, the branch selection may fail or behave unexpectedly; callers should pass a str.
        - If scaledrmse_mcc_val is not a float (or a numeric type comparable to float), Python comparison operators may raise a TypeError.
        - If scaledrmse_mcc_val is NaN (not-a-number), all numeric comparisons with NaN evaluate to False, and the function will return 0 for either branch.
        - The function does not validate physical units or statistical validity of the metric; it assumes the caller supplies a metric value consistent with the intended interpretation (scaled RMSE for regression, MCC for classification).
    
    Returns:
        int: Discrete score computed from scaledrmse_mcc_val according to the thresholds described above. The returned score range is 0–2 for regression inputs (pred_type == 'reg') and 0–3 for classification inputs (pred_type != 'reg'). The integer is intended for aggregation into higher-level model performance summaries in ROBERT reports.
    """
    from robert.report_utils import score_rmse_mcc
    return score_rmse_mcc(pred_type, scaledrmse_mcc_val)


################################################################################
# Source: robert.utils.Xy_split
# File: robert/utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def robert_utils_Xy_split(
    csv_df: dict,
    csv_X: dict,
    X_scaled_df: dict,
    csv_y: numpy.ndarray,
    csv_external_df: dict,
    csv_X_external: dict,
    X_scaled_external_df: dict,
    csv_y_external: numpy.ndarray,
    test_points: list,
    column_names: list
):
    """robert.utils.Xy_split returns a dictionary containing training, optional test, and optional external sets extracted from provided dataset containers. This helper is used in ROBERT's machine-learning workflows (regression tasks in chemistry) to prepare inputs for model training, validation, and external testing by slicing feature matrices, scaled feature matrices, target arrays, and name columns according to a provided list of test indices. It organizes these subsets into keys that downstream functions in the pipeline expect (for example, X_train, y_train, X_test, y_test, X_external, y_external, and associated scaled variants and name lists).
    
    Args:
        csv_df (dict): Primary database container for full-record metadata and identifiers. In the source this object is indexed by column names and sliced by row indices (the implementation calls csv_df[column_names], csv_df.drop(test_points) and csv_df.iloc[test_points]); therefore csv_df must behave like a table (e.g., a pandas DataFrame-like object) even though the annotated type here is dict. The function uses column_names to extract human-readable names for training/test/external subsets used in reports and downstream analysis.
        csv_X (dict): Feature matrix for all samples (unscaled). The code uses .drop(test_points) and .iloc[test_points] on this object, so it must support those operations (i.e., behave like a pandas DataFrame). This argument becomes X_train (or X_test when test_points is non-empty) in the returned dictionary and is meant to feed regression algorithms that expect raw features.
        X_scaled_df (dict): Scaled version of csv_X (features after preprocessing/standardization). Must support .drop and .iloc similarly. This is returned as X_train_scaled and X_test_scaled where applicable and is typically passed to ML models that require pre-scaled inputs.
        csv_y (numpy.ndarray): Target variable array for all samples (e.g., experimental values in a chemistry regression task). When test_points is non-empty the function calls .drop and .iloc on csv_y as in the source; therefore the provided object must support these operations in practice (the annotated type here is numpy.ndarray as given in the signature). The resulting y_train and y_test are intended for model fitting and evaluation.
        csv_external_df (dict): External (hold-out) dataset container analogous to csv_df for additional out-of-sample samples. If X_scaled_external_df is not None, csv_external_df is used to populate names_external by selecting column_names from this object. It must support column indexing by column_names as used by the implementation.
        csv_X_external (dict): External feature matrix (unscaled) corresponding to csv_external_df. When X_scaled_external_df is provided, csv_X_external is returned as X_external to enable external validation of trained models. It should be structured similarly to csv_X.
        X_scaled_external_df (dict): Scaled external feature matrix. If this argument is not None the function will include external data in the return dictionary by setting X_external_scaled (and, conditionally, y_external). Passing None disables inclusion of external data. The object must be compatible with downstream consumers that expect scaled external features.
        csv_y_external (numpy.ndarray): Optional target values for the external dataset. If provided (not None) the function includes y_external in the returned dictionary. If None, no y_external key is added. The annotated type is numpy.ndarray as in the function signature.
        test_points (list): List of integer row indices (relative to csv_X/csv_df ordering) that identify which rows should be reserved as an internal test/validation set. If test_points is empty (len == 0), the function places all data into training keys and does not create X_test/y_test/names_test keys. When non-empty, the function uses .drop(test_points) to form training subsets and .iloc[test_points] to form test subsets; therefore indices must be valid for the provided table-like inputs.
        column_names (list): List specifying which column(s) to extract from csv_df/csv_external_df to produce human-readable name fields (names_train, names_test, names_external). The implementation uses csv_df[column_names] and csv_external_df[column_names] exactly, so column_names must match the indexing semantics of the provided csv_df-like objects.
    
    Returns:
        dict: A dictionary (named Xy_data in the implementation) that contains the split datasets and metadata for downstream model training and evaluation. Keys produced by the function are:
            - 'X_train': value taken from csv_X or csv_X.drop(test_points) depending on test_points; intended for training features.
            - 'X_train_scaled': value taken from X_scaled_df or X_scaled_df.drop(test_points); intended for scaled training features.
            - 'y_train': value taken from csv_y or csv_y.drop(test_points); intended for training targets.
            - 'names_train': value taken from csv_df[column_names] or csv_df.drop(test_points)[column_names]; human-readable identifiers for training samples.
            - 'X_test', 'X_test_scaled', 'y_test', 'names_test': present only if test_points is non-empty and populated from csv_X.iloc[test_points], X_scaled_df.iloc[test_points], csv_y.iloc[test_points], and csv_df.iloc[test_points][column_names] respectively; intended for internal validation.
            - 'test_points': echoes the provided test_points list.
            - 'X_external', 'X_external_scaled', 'y_external', 'names_external': present only if X_scaled_external_df is not None (and y_external only if csv_y_external is not None); populated from csv_X_external, X_scaled_external_df, csv_y_external, and csv_external_df[column_names] respectively for external hold-out validation.
        The returned dictionary is newly created by the function; inputs are not overwritten by design. Failure modes: if provided objects do not support the table-like operations used in the source (for example .drop or .iloc), AttributeError or TypeError will be raised. If test_points contains indices that are out of range or invalid for the provided objects, IndexError or KeyError may be raised by the underlying indexing operations.
    """
    from robert.utils import Xy_split
    return Xy_split(
        csv_df,
        csv_X,
        X_scaled_df,
        csv_y,
        csv_external_df,
        csv_X_external,
        X_scaled_external_df,
        csv_y_external,
        test_points,
        column_names
    )


################################################################################
# Source: robert.utils.command_line_args
# File: robert/utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def robert_utils_command_line_args(exe_type: str, sys_args: dict):
    """robert.utils.command_line_args loads default and user-defined command-line arguments for the ROBERT package, parses them according to the program's expected argument types, performs type conversions, and returns the processed configuration object used by downstream ROBERT workflows (CURATE, GENERATE, VERIFY, PREDICT, AQME, REPORT, etc.). This function is used by the ROBERT command-line entry points to combine a small set of programmatic overrides (sys_args) with the full set of available command-line options and defaults defined in the module, and to produce the final "args" object consumed by the rest of the application.
    
    Args:
        exe_type (str): Indicator of the execution environment. If exe_type equals the literal string 'exe', the function simulates an executable environment by overwriting the global sys.argv list with a synthetic argv that begins with "launcher.exe" and then appends each key from sys_args and, when a value is not None, the stringified value. This parameter determines whether sys_args are injected into sys.argv for environments that do not naturally populate sys.argv (for example, when the package is launched as a bundled executable). Passing any other string leaves sys.argv unchanged and the function parses the actual process arguments.
        sys_args (dict): Dictionary of programmatic command-line overrides intended for injection when exe_type == 'exe'. Keys are expected to be the command-line flags (for example, "--y" or "--csv_name"); values are the corresponding argument values and are converted to strings when appended to sys.argv. If a value in sys_args is None, only the key (flag) is appended to sys.argv so the option is treated as a flag without an explicit value. This parameter allows callers to provide CLI-like options from code or from an executable wrapper instead of relying on a user-typed command line.
    
    Behavior and side effects: The function first optionally mutates the global sys.argv when exe_type == 'exe' to simulate command-line input. It constructs the list of accepted long options from a module-level variable collection (it iterates over var_dict to produce available_args) and combines that with hard-coded categorizations of recognized boolean, list, integer, and floating-point options. It then invokes getopt.getopt to parse sys.argv[1:] using the assembled available_args. For each parsed option, the function maps the raw option name to an internal argument name, handles a special-case help trigger (-h or --help) by printing an extensive help message (including robert_version and robert_ref) and calling sys.exit to terminate the process, and otherwise converts option values according to the predefined categories: boolean flags in bool_args are set to True when present; options in list_args are processed via format_lists(value) to produce a list-like object; options in int_args are converted with int(value); options in float_args are converted with float(value); literal string values "None", "False", and "True" are mapped to Python None, False, and True respectively; and the special option name 'files' with a value containing a glob pattern ('*') is expanded with glob.glob into a list of matching filesystem paths. Parsed and converted options are accumulated into a kwargs dictionary. Finally, the function calls load_variables(kwargs, "command") to merge kwargs with the program's default configuration and returns that merged args object.
    
    Failure modes and exceptions: If getopt.getopt raises getopt.GetoptError due to unsupported or malformed options, the function prints the error message and calls sys.exit, terminating the process. If any of the explicit type conversions (int(value) or float(value)) fail because the provided value cannot be parsed, a ValueError will propagate unless caught by the caller or by surrounding code. The function also performs global side effects that callers must be aware of: it may overwrite sys.argv and may call sys.exit (both in the getopt error path and when printing help). The function relies on module-level symbols such as var_dict, format_lists, load_variables, robert_version, and robert_ref; if these are missing or misconfigured, NameError or other exceptions may be raised during execution.
    
    Returns:
        object: The configuration object returned by load_variables(kwargs, "command"). This object represents the full set of processed command-line options merged with the package defaults and is the canonical "args" used by downstream ROBERT routines (for example, to control CURATE, GENERATE, VERIFY, PREDICT, AQME, and REPORT workflows). The exact runtime type of this object is the value returned by load_variables in the module; callers should treat it as the program configuration rather than relying on a specific Python type.
    """
    from robert.utils import command_line_args
    return command_line_args(exe_type, sys_args)


################################################################################
# Source: robert.utils.correct_hidden_layers
# File: robert/utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def robert_utils_correct_hidden_layers(params: dict):
    """robert.utils.correct_hidden_layers corrects and normalizes the 'hidden_layer_sizes' entry inside a parameter dictionary that is typically loaded from JSON for use with neural-network-based regression tools in ROBERT. The function ensures the value stored under the 'hidden_layer_sizes' key becomes an explicit Python list of integers representing the number of neurons per hidden layer (the same semantic used by scikit-learn MLP estimators), and mutates the input dictionary in place for downstream model construction and hyperparameter handling in the ROBERT regression pipeline.
    
    This function is used when JSON-serialized parameter sets store hidden layer sizes as strings (for example "[64,32]" or "64,32") or as lists of numeric/string elements; it attempts to parse those representations and produce a canonical list[int] that downstream code (e.g., model builders or hyperparameter evaluators in ROBERT) expects. The function implements the following concrete behavior from the source:
    - If params['hidden_layer_sizes'] is not an int, it treats the value as either a string or a list. If the value is a string that begins with '[' or ends with ']', the function strips the leading/trailing bracket characters. If the (possibly bracket-stripped) value is a string, it splits on commas and converts non-empty segments to ints. If the value is a list, it iterates over elements and converts non-empty elements to ints. Empty strings (''), if present in the split/list, are skipped. Parsed integers are collected into layer_arrays and then assigned back into params['hidden_layer_sizes'].
    - If params['hidden_layer_sizes'] is an int, the function reaches a code path that attempts to assign an undefined local variable to the output (this is an implementation bug in the current source) which will raise an UnboundLocalError at runtime.
    
    Behavioral notes, side effects, and failure modes:
    - The function mutates the input dictionary params in place by replacing params['hidden_layer_sizes'] with the parsed list of integers. It also returns the same dictionary.
    - Accepted input forms for params['hidden_layer_sizes'] (as handled by the current implementation) include:
      - a string representing a bracketed list, e.g. "[64,32]"
      - a comma-separated string without brackets, e.g. "64,32"
      - a list of elements (elements may be numeric or string-representations of integers)
      - an integer (handled only by the buggy branch described above; see failure modes)
    - Conversions use int(ele) for each non-empty element and therefore will raise ValueError if any non-empty element cannot be parsed as an integer (for example "64a" or "sixtyfour").
    - If the params dictionary does not contain the key 'hidden_layer_sizes', a KeyError will be raised by the implementation.
    - If params is not a dict, typical attribute or indexing errors (TypeError) will be raised.
    - If params['hidden_layer_sizes'] is already an int, the current implementation contains a bug that will raise UnboundLocalError because the code assigns layer_arrays = ele where ele is not defined in that branch. This is a known failure mode of the current source and must be handled by callers or fixed in the implementation.
    - Empty items produced by splitting strings (empty strings) are skipped and not included in the resulting list.
    
    Args:
        params (dict): A parameters dictionary used by ROBERT for regression model configuration. This dictionary is expected to include the key 'hidden_layer_sizes' whose value may be a string (e.g., "[64,32]" or "64,32"), a list of elements (strings or numbers), or an int. The function will parse and normalize that entry into a list of integers representing the number of neurons in each hidden layer (the same semantics used by scikit-learn MLP estimators). The dictionary is modified in place; callers should provide a mutable dict and be aware that parsing errors (ValueError, KeyError, UnboundLocalError when the value is int) can be raised by this routine.
    
    Returns:
        dict: The same params dictionary passed in, after mutating params['hidden_layer_sizes'] to be a list of integers parsed from the original value. If parsing fails (for example due to non-integer tokens) a ValueError will be raised; if the key is missing a KeyError will be raised; if params['hidden_layer_sizes'] is an int the current implementation will raise an UnboundLocalError due to a bug.
    """
    from robert.utils import correct_hidden_layers
    return correct_hidden_layers(params)


################################################################################
# Source: robert.utils.dict_formating
# File: robert/utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def robert_utils_dict_formating(dict_csv: dict):
    """robert.utils.dict_formating converts string-encoded Python literals that originate from CSV-loaded pandas DataFrames into native Python objects for use in the ROBERT cheminformatics and machine-learning workflows. This function is intended to be used after reading rows from CSV files (for example, exported DataFrame rows where complex objects were saved as strings) so that downstream modules in ROBERT (feature handling, model construction, reporting) receive Python lists/dicts instead of their textual representations.
    
    The function looks for the keys 'X_descriptors' and 'params' in the input dictionary and, when present, replaces their string values with Python objects using ast.literal_eval. In the ROBERT domain, 'X_descriptors' typically contains molecular descriptor lists or nested structures representing features used for regression or classification models, and 'params' typically contains model hyperparameters or parameter dictionaries saved as strings. Converting these back to their native types is necessary for correct operation of model fitting, prediction, and report generation components.
    
    Args:
        dict_csv (dict): Dictionary representing a row or record originally derived from a CSV/DataFrame in the ROBERT project. The keys may include 'X_descriptors' and/or 'params' whose values are expected to be string representations of Python literals (for example, "['desc1', 'desc2']" or "{'alpha': 0.1}"). The function mutates this dictionary in place by parsing these string literals into Python objects. If either key is absent, that key is left unchanged. If dict_csv is not a dict, the function will raise a TypeError when attempting dictionary access.
    
    Returns:
        dict: The same dictionary object passed as dict_csv, potentially mutated so that the values under 'X_descriptors' and 'params' are converted from string representations to native Python objects (lists, dicts, numbers, etc.). This returned dictionary is suitable for subsequent ROBERT processing steps that expect concrete Python types rather than textual encodings.
    
    Behavior and failure modes:
        The conversion uses ast.literal_eval to safely evaluate string literals; this is safer than eval because it only evaluates Python literal structures. However, if a value under 'X_descriptors' or 'params' is not a valid Python literal string, ast.literal_eval will raise a ValueError or SyntaxError. Such exceptions will propagate to the caller unless caught externally. The function does not perform schema validation of the parsed objects (for example, it does not check that parsed 'X_descriptors' is a list of floats or that 'params' contains required hyperparameters); callers in ROBERT should perform any domain-specific validation after calling this function.
    """
    from robert.utils import dict_formating
    return dict_formating(dict_csv)


################################################################################
# Source: robert.utils.format_lists
# File: robert/utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def robert_utils_format_lists(value: str):
    """robert.utils.format_lists transforms a string representation of a Python sequence into a normalized Python list suitable for downstream use in ROBERT workflows. In the ROBERT project (a toolkit bridging machine learning and chemistry), this function is used to normalize user-provided list-like inputs (for example: lists of descriptor names, feature names, SMILES strings, or hyperparameter entries supplied as configuration strings or command-line arguments) into a predictable Python list so downstream regression, feature-selection, and reporting code can operate on a uniform container.
    
    Args:
        value (str): Input to convert into a Python list. This is typically a string representing a Python literal sequence (for example "['a', 'b']" or "['SMILES1','SMILES2']"), but the implementation also accepts an already-built Python list and will return it after normalization. The function first attempts to parse the string safely using ast.literal_eval to produce a Python object. If ast.literal_eval raises a SyntaxError or ValueError (for example when the input uses nonstandard quoting or bracket styles such as "[X]" or ["X"] instead of "['X']"), the function falls back to a heuristic text-splitting strategy: it replaces occurrences of '[' , ',' and "'" with ']' and then splits on ']' to extract tokens, removing empty tokens. After parsing, all string elements are stripped of leading and trailing whitespace. Note that literal_eval can produce sequence types other than list (for example tuples); in that case the result will be iterated and converted into a list of elements. This parameter is required and no additional parameters are supported.
    
    Returns:
        list: A Python list containing the parsed elements. If the input was already a list, a new list with the same elements (with string elements trimmed of surrounding whitespace) is returned. If the input string is a literal sequence, elements parsed by ast.literal_eval are returned (and non-string elements are preserved as their evaluated types). If ast.literal_eval fails, the fallback heuristic returns a list of string tokens produced by splitting and trimming; this fallback is tolerant of some malformed list-like strings but may produce incorrect tokenization for nested structures or elements that contain commas or brackets. The function has no side effects on external state and does not modify files or global variables.
    """
    from robert.utils import format_lists
    return format_lists(value)


################################################################################
# Source: robert.utils.get_prediction_results
# File: robert/utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def robert_utils_get_prediction_results(
    model_data: dict,
    y: numpy.ndarray,
    y_pred_all: numpy.ndarray
):
    """robert.utils.get_prediction_results calculates standard evaluation metrics for a fitted model's predictions. This function is used in the ROBERT package (Refiner and Optimizer of a Bunch of Existing Regression Tools) to compute summary performance metrics for either regression or classification models, based on the model type declared in model_data['type']. The function chooses the metric set and minor numeric handling rules according to the 'type' field and returns a tuple of three floats that summarize predictive performance.
    
    Args:
        model_data (dict): Metadata describing the model and intended task. This dictionary must include the key 'type' whose value is a string indicating the task: 'reg' (regression) or 'clas' (classification). The comparison is case-insensitive (the code calls .lower()). The function reads model_data['type'] to select which metrics to compute; if the key is missing a KeyError will be raised. model_data is used solely to branch behavior between regression and classification metrics and does not need to contain model weights or other objects.
        y (numpy.ndarray): Ground-truth target values corresponding to the predictions. For regression tasks these are continuous target values; for classification tasks these are integer class labels (e.g., 0, 1, ...). y is expected to be a numpy array whose length matches y_pred_all; if lengths or shapes are incompatible, underlying metric functions (from numpy/scipy/scikit-learn) will raise an exception (e.g., ValueError).
        y_pred_all (numpy.ndarray): Predicted values produced by the model for the same samples as y. For regression, y_pred_all is typically continuous predicted values. For classification, y_pred_all may be probabilities, scores, or class-label-like values; the function will round y_pred_all with numpy.round and cast to int before computing classification metrics. y_pred_all must be a numpy array and must align (same number of elements) with y.
    
    Behavior and details:
    - If model_data['type'].lower() == 'reg', the function computes three regression metrics and returns them in the order (r2, mae, rmse):
        - mae: mean absolute error computed with sklearn.metrics.mean_absolute_error between y and y_pred_all.
        - rmse: root mean squared error computed as sqrt(mean_squared_error(y, y_pred_all)).
        - r2: coefficient of determination estimated from scipy.stats.linregress(y, y_pred_all) as the square of the returned Pearson correlation coefficient (rvalue**2). If either y or y_pred_all is constant (fewer than 2 unique values), the function sets r2 = 0.0 to avoid invalid regression results.
      These computations have no side effects and do not modify inputs.
    - If model_data['type'].lower() == 'clas', the function treats predictions as class-like values by rounding and casting to int: np.round(y_pred_all).astype(int). It then computes and returns three classification metrics in the order (accuracy, f1_score, mcc):
        - accuracy (acc): accuracy_score on the rounded integer predictions.
        - f1_score_val: f1_score on the rounded integer predictions. The function attempts to call f1_score with default settings (binary average). If scikit-learn raises a ValueError because the default binary setting is inappropriate for the labels present (for example, multi-class labels), the function catches the exception and recomputes f1_score using average='micro'. This fallback ensures a meaningful F1 value for multi-class scenarios.
        - mcc: Matthews correlation coefficient computed via sklearn.metrics.matthews_corrcoef on the rounded integer predictions.
      The rounding and cast behavior is important: continuous model outputs (probabilities or scores) are converted to integer class labels prior to metric computation.
    
    Returns:
        tuple: A 3-tuple of floats. For regression (model_data['type'] == 'reg'), returns (r2, mae, rmse) where r2 is the squared Pearson correlation between y and y_pred_all (or 0.0 for constant arrays), mae is mean absolute error, and rmse is root mean squared error. For classification (model_data['type'] == 'clas'), returns (accuracy, f1_score_val, mcc) computed on np.round(y_pred_all).astype(int). All returned values are Python float objects (or numpy floats castable to float).
    
    Failure modes and exceptions:
    - KeyError if model_data does not contain the 'type' key.
    - If model_data['type'] is neither 'reg' nor 'clas' (case-insensitive), the function will not match either branch and will implicitly return None (this is a code-path issue in the implementation); callers should ensure model_data['type'] is one of the supported strings.
    - ValueError, TypeError, or other exceptions may be raised by underlying metric functions (scikit-learn, scipy, numpy) if y and y_pred_all have incompatible shapes, non-numeric contents, or otherwise invalid values.
    - For classification, stable behavior is provided by rounding+casting predictions and the try/except fallback for f1_score; however, users should supply integer labels in y and sensible prediction values in y_pred_all to obtain meaningful metrics.
    
    Practical significance in ROBERT:
    - This function is used by the ROBERT framework to summarize model performance during training, validation, or reporting pipelines for chemical and molecular property prediction tasks. For regression tasks (e.g., predicted energies, pKa, spectral properties) it provides standard regression diagnostics; for classification tasks (e.g., active/inactive, categorical labels) it provides common classification summaries. The returned metrics are suitable for logging, model selection, or inclusion in ROBERT-generated reports.
    """
    from robert.utils import get_prediction_results
    return get_prediction_results(model_data, y, y_pred_all)


################################################################################
# Source: robert.utils.get_scoring_key
# File: robert/utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def robert_utils_get_scoring_key(problem_type: str, error_type: str):
    """robert.utils.get_scoring_key returns the appropriate scikit-learn scoring identifier or a custom scorer callable for use when evaluating machine learning models within ROBERT. The function maps a short, human-friendly error_type string (for example 'rmse', 'mae', 'r2', 'f1', 'acc', 'mcc') together with a problem_type indicator to the exact scoring key or scorer that downstream ROBERT model-evaluation utilities (such as cross-validation, GridSearchCV, or custom evaluation pipelines used in cheminformatics property- and reaction-prediction tasks) expect.
    
    Args:
        problem_type (str): Problem type indicator. The function treats the lowercase value 'reg' as regression and any other value as classification. In the ROBERT project this distinguishes tasks such as continuous property prediction (regression) from categorical label prediction (classification). The function calls .lower() on this argument, so case variations are accepted, but values other than 'reg' are interpreted as classification.
        error_type (str): Short identifier for the performance metric to use. For regression (problem_type == 'reg') supported identifiers are 'rmse' (maps to scikit-learn's 'neg_root_mean_squared_error'), 'mae' (maps to 'neg_median_absolute_error'), and 'r2' (maps to 'r2'). For classification, supported identifiers are 'f1' (maps to 'f1'), 'acc' (maps to 'accuracy'), and the special case 'mcc' which returns a custom Matthews correlation coefficient scorer (make_scorer(mcc_scorer_clf)) used by ROBERT to ensure integer predictions when computing MCC. The argument must be a string; unsupported identifiers will not raise inside this function but will cause the function to return None (see failure modes).
    
    Returns:
        object: Returns the scoring key or scorer object to be passed to scikit-learn evaluation routines. Specifically, for many standard metrics the function returns a string matching the scikit-learn scoring name (for example 'neg_root_mean_squared_error', 'r2', 'f1', 'accuracy'), and for the special classification metric 'mcc' it returns the result of make_scorer(mcc_scorer_clf) (a callable scorer object). If error_type is not recognized for the given problem_type, the function returns None. No exceptions are raised by this function for unknown error_type values; callers should check for None and handle it (for example by raising a clear error, selecting a default metric, or informing the user) because passing None to scikit-learn as a scoring argument will cause downstream errors.
    
    Behavior, side effects, defaults, and failure modes:
        This function is pure (no side effects) and performs only a deterministic mapping based on the provided strings. It lowercases problem_type internally to accept case-insensitive inputs for the regression switch. It does not validate that error_type is non-empty or belongs to a predefined set beyond the implemented mappings. If an unsupported error_type is provided, scoring will be None which will typically lead to a runtime error when the return value is used as the scoring parameter in scikit-learn functions; therefore callers in ROBERT should validate the return and provide appropriate error messages. The function relies on the existence of mcc_scorer_clf and make_scorer in the calling module scope when error_type == 'mcc'; if those are unavailable at import/runtime, using the returned value (or attempting to construct it) may raise a NameError or ImportError outside this function. There are no default metrics chosen by this function when inputs are unrecognized.
    """
    from robert.utils import get_scoring_key
    return get_scoring_key(problem_type, error_type)


################################################################################
# Source: robert.utils.graph_vars
# File: robert/utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def robert_utils_graph_vars(
    Xy_data: dict,
    set_types: list,
    csv_test: bool,
    path_n_suffix: str,
    sd_graph: bool
):
    """robert.utils.graph_vars sets numeric axis limits for regression result plots and composes the filesystem path and a reduced display path for the resulting PNG graph files used by ROBERT (the Refiner and Optimizer of a Bunch of Existing Regression Tools). This function is used in model evaluation and reporting workflows (for example, producing predicted vs observed plots in cheminformatics and regression benchmarks) to ensure consistent axis padding, to choose whether the plot shows single external results or multiple training/test sets, and to standardize filenames and relative paths for inclusion in reports.
    
    This function computes a symmetric padding "size_space" equal to 10% of the range of the relevant y values and returns the lower and upper axis limits after applying that padding. It also composes a full filesystem path for the PNG file (reg_plot_file) and a shortened, report-friendly path segment (path_reduced) that preserves the last components of the path. The behavior differs depending on whether csv_test is True (single external set plotting) or False (multiple sets including train and optionally test). The function does not write any files or create directories; it only inspects numeric values in Xy_data and formats path strings using path_n_suffix.
    
    Args:
        Xy_data (dict): Mapping of string keys to iterables of numeric values that represent true and predicted targets used to determine plot limits. When csv_test is False, Xy_data must contain keys 'y_train', 'y_pred_train', 'y_test', and 'y_pred_test' (or at least the keys referenced by set_types) whose values are sequences (for example, lists or numpy arrays) of numeric values; these are used to compute the global minimum and maximum for axis limits. When csv_test is True, Xy_data must contain keys 'y_external' and 'y_pred_external' (accessed as f'y_{set_type}' and f'y_pred_{set_type}' where set_type is 'external'). Missing keys or non-iterable / empty iterables will raise the underlying exceptions from min()/max() (KeyError, TypeError, or ValueError). The numeric magnitudes correspond to experimental and predicted properties in ROBERT regression workflows (e.g., molecular properties or reaction outcomes).
        set_types (list): List of dataset identifiers included in the multi-set plotting context. The function checks membership of the string 'test' in this list: if 'test' is present and csv_test is False, the test set values are included when computing axis limits. Do not modify this list in-place while calling the function. Valid entries are inferred from caller conventions (commonly 'train' and 'test'); other values are ignored by this function except for membership testing of 'test'.
        csv_test (bool): Flag indicating whether the plot corresponds to a single external CSV test set (True) or to multiple internal sets (False). If True, the function treats the plotted set as 'external' and uses Xy_data keys f'y_external' and f'y_pred_external' to compute limits and constructs filenames under a csv_test subfolder. If False, the function uses the training and (optionally) test arrays from Xy_data to compute limits and places the result at the directory corresponding to path_n_suffix. This flag controls both which Xy_data keys are read and how path_reduced is computed (last three path components for csv_test True, last two for csv_test False).
        path_n_suffix (str): Filesystem path string used as the base for the output PNG filename. The function uses os.path.dirname(path_n_suffix) as the target folder and os.path.basename(path_n_suffix) to create filenames. If csv_test is False, the file is placed at dirname(path_n_suffix)/Results_basename.png or dirname(path_n_suffix)/CV_variability_basename.png depending on sd_graph. If csv_test is True, the file is placed under dirname(path_n_suffix)/csv_test and the filename includes the set_type suffix (external). The function normalizes path separators when composing path_reduced so it is robust to Windows backslashes; it does not check for or create the target directory and will not write any files.
        sd_graph (bool): Flag that selects the graph filename prefix to indicate standard-deviation / cross-validation variability plots versus regular results plots. If False the filename prefix is 'Results_'; if True the filename prefix is 'CV_variability_'. This choice affects only the reg_plot_file string returned (and path_reduced), not the numeric axis computation.
    
    Returns:
        tuple: A 4-tuple containing:
            min_value_graph (float or int): The computed lower axis limit for the plot after subtracting the padding (size_space). This value is obtained by taking the minimum across the relevant true and predicted arrays in Xy_data and subtracting 0.1 times the absolute range of the primary y array used.
            max_value_graph (float or int): The computed upper axis limit for the plot after adding the padding (size_space). This value is obtained by taking the maximum across the relevant true and predicted arrays in Xy_data and adding 0.1 times the absolute range of the primary y array used.
            reg_plot_file (str): Full filesystem path (string) where the plot PNG is intended to be saved, following ROBERT naming conventions ('Results_' or 'CV_variability_' prefix, with an added csv_test subfolder when csv_test is True). The function does not create or validate this path; callers should ensure directories exist before attempting to save files.
            path_reduced (str): Shortened path fragment suitable for embedding in reports. When csv_test is False, this preserves the last two path components; when csv_test is True, it preserves the last three components. Path separators are normalized to forward slashes.
    
    Failure modes and notes:
        The function assumes numeric, non-empty iterables in Xy_data for the referenced keys; invoking min()/max() on empty sequences will raise ValueError. If all values for the primary y array are identical, the computed size_space will be zero and min_value_graph and max_value_graph may be equal (resulting in zero plotting range unless the caller adjusts it). Missing expected keys in Xy_data will raise KeyError. The function performs no I/O: it only computes values and formats path strings.
    """
    from robert.utils import graph_vars
    return graph_vars(Xy_data, set_types, csv_test, path_n_suffix, sd_graph)


################################################################################
# Source: robert.utils.load_minimal_model
# File: robert/utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def robert_utils_load_minimal_model(model: str):
    """Load and return the predefined minimal hyperparameter set for a named estimator used by REFCV within the ROBERT package.
    
    Args:
        model (str): Identifier of the minimalist model whose parameter set will be returned. Accepted exact values (case-sensitive) are 'RF', 'GB', 'NN', 'ADAB', 'GP', and 'MVL'. This argument selects one of the small, predefined hyperparameter dictionaries embedded in ROBERT that are intended for use by the REFCV routine (the package's reference cross-validation / low-resource benchmarking workflow used in the ROBERT machine-learning-for-chemistry tooling). The choice of model controls which set of hyperparameters is returned; the caller should pass this string verbatim.
    
    Returns:
        dict: A dictionary mapping hyperparameter names (strings) to their minimalist default values (ints, floats, or None) for the requested model. Each returned dictionary contains the exact hyperparameter keys and values used by ROBERT's REFCV to instantiate or configure the corresponding estimator with conservative, low-complexity settings. For example, 'RF' returns a dictionary with keys such as 'n_estimators', 'max_depth', etc.; 'MVL' returns an empty dictionary. The caller can pass this dictionary directly to estimator constructors or to set_params for reproducible, low-resource experiments.
    
    Raises:
        KeyError: If model is not one of the predefined keys ('RF', 'GB', 'NN', 'ADAB', 'GP', 'MVL'), a KeyError will be raised by the underlying lookup. No other validation or side effects are performed by this function; it is a pure lookup that returns a reference to the stored dictionary for the requested model.
    """
    from robert.utils import load_minimal_model
    return load_minimal_model(model)


################################################################################
# Source: robert.utils.load_variables
# File: robert/utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def robert_utils_load_variables(kwargs: dict, robert_module: str):
    """Load default and user-defined variables for a ROBERT module and apply module-specific initialization.
    
    This function is used throughout the ROBERT package to convert a user-provided options dictionary into a prepared internal options object (hereafter "self"), to merge defaults, to import variables from a YAML file when requested, to normalize file names and model identifiers, to create destination folders and log files, and to perform initial sanity checks required before executing a specific ROBERT module (for example, GENERATE, CURATE, PREDICT, VERIFY, REPORT, AQME, AQME_TEST). It is the canonical entry point to prepare runtime configuration when launching ROBERT functionality described in the project README (machine-learning workflows and cheminformatics utilities).
    
    Args:
        kwargs (dict): Dictionary of user-supplied configuration values and command-line-style options. This dictionary is passed unchanged to set_options(kwargs) and is used to build an internal options object ("self"). Typical entries in this dict correspond to flags and parameters documented for ROBERT modules (for example, keys such as "csv_name", "csv_test", "varfile", "names", "ignore", "y", "type", "model", "params_dir", "command_line", "extra_cmd", "pfi_filter", etc.). The practical role of kwargs is to carry user preferences and overrides so that set_options can instantiate an options object containing both defaults and overrides.
        robert_module (str): Name of the ROBERT submodule being prepared (case-insensitive). Expected values appearing in the code include but are not limited to: "GENERATE", "CURATE", "PREDICT", "VERIFY", "REPORT", "AQME", "AQME_TEST", and "command". This string controls module-specific behavior such as whether to create destination folders, which additional initializations to perform (for example, creating GENERATE folder structure, adjusting error metrics for classification, enforcing required CSV columns for AQME), and whether to set report resource paths instead of normal logging.
    
    Returns:
        object: The internal "self" options object returned by set_options(kwargs) and modified in place by this function. The returned object contains updated and normalized attributes required by downstream ROBERT modules. Attributes that are set or potentially modified by this function (based on the provided kwargs and module) include:
            csv_name (str): normalized CSV filename (the function will append ".csv" if a file with that extension exists but the provided name did not include it).
            csv_test (str or False): normalized CSV test filename (same behavior as csv_name when provided).
            varfile (str or None): if present, a YAML file path used to load additional parameters via load_from_yaml; its contents may modify "self" and produce a text summary captured in txt_yaml.
            initial_dir (pathlib.Path): the current working directory captured at initialization (set for modules other than "command").
            ignore (list): list of features/column names to ignore; this function may append the "names" attribute to ignore if necessary.
            names (str): feature/identifier column name used across modules; may be loaded from previous CURATE job if missing.
            y (str): target column name; may be loaded from previous CURATE job or from saved model parameters for PREDICT/VERIFY.
            log (Logger-like object): logger created and initialized to write to destination/<MODULE>; the function writes header information, executed command line, YAML import text (if any), and module-specific start messages.
            destination (pathlib.Path): output folder for module results (created via destination_folder when applicable).
            path_icons (pathlib.Path): for REPORT module, path to bundled report icons/resources.
            params_dir (str): path to model parameter folders; default is set for PREDICT/VERIFY/REPORT when empty.
            model (list of str): list of model identifiers normalized to uppercase and with specific substitutions (for classification, 'MVL' is replaced by 'AdaB' as performed in CURATE/GENERATE contexts).
            error_type (str): error metric used; this function enforces a default for classification ('mcc' if not one of ['acc','mcc','f1'] in GENERATE/VERIFY contexts).
            args (object): when loading saved model metadata for PREDICT/VERIFY, self.args may be set to self to facilitate loading behavior.
            type (str): problem type (e.g., regression or classification); may be read from saved model metadata if absent.
        The returned object is intended for direct use by the calling code and downstream functions (for example, create_folders, load_dfs, sanity_checks).
    
    Behavior, side effects, defaults, and failure modes:
        - The function first calls set_options(kwargs) to build the internal options object ("self") from defaults and provided kwargs.
        - If self.varfile is not None, load_from_yaml(self) is invoked to import options from a YAML file. The function captures textual feedback from that operation in txt_yaml. If the YAML import returns unexpected text (specific messages checked in the code), the function writes that text to a new Logger and then exits the process (sys.exit()) to avoid running with invalid YAML parameters.
        - File-name normalization: If the provided csv_name or csv_test refer to existing files only when ".csv" is appended, the function appends ".csv" to those attributes so downstream code reads the correct file.
        - For robert_module values other than "command", the function records the current working directory in self.initial_dir and may create destination folders via destination_folder(self, robert_module). When a destination is created, a Logger is instantiated at destination/ROBERT (or destination/<MODULE>) and the function writes a header with ROBERT version, runtime, and citation information. If self.command_line is True, the exact command line used to launch ROBERT (including special parsing to preserve list arguments such as those passed to --qdescp_atoms) is reconstructed and written to the log. Special handling: arguments containing --qdescp_atoms are parsed so that quoted SMARTS strings and comma-separated lists are preserved as reproducible strings in the log.
        - For modules GENERATE/VERIFY/PREDICT, the function attempts to import scikit-learn-intelex (sklearnex.patch_sklearn). If the import fails, the function writes a warning to the log indicating that execution times and results may differ without the accelerator. This warning does not abort execution.
        - For GENERATE and VERIFY modules, if the problem type self.type is classification (self.type.lower() == 'clas'), the function enforces a default error_type of 'mcc' when the current error_type is not one of 'acc', 'mcc', or 'f1'.
        - For PREDICT/VERIFY/REPORT modules, if self.params_dir is empty, the function sets it to 'GENERATE/Best_model' as a sensible default for where trained model parameters are stored.
        - For CURATE/GENERATE modules, the function normalizes model names to uppercase and substitutes 'MVL' (or 'mvl') with 'AdaB' for classification pipelines where applicable.
        - For GENERATE, the function creates a set of subfolders needed for saving best models and raw data (including separate PFI and No_PFI subfolders when self.pfi_filter is True) by calling create_folders. It also attempts to backfill missing options (y, names, ignore, csv_name) from a previous CURATE run by reading GENERATE/CURATE_options.csv when present; when backfilling ignore, format_lists is applied to convert stored representations into in-memory lists.
        - For PREDICT and VERIFY, if essential options (names, y, csv_name) are missing, the function attempts to load them from saved model metadata using load_dfs on the params_dir. When model metadata is loaded, attributes such as names, y, csv_name, error_type, and type are restored into self so that prediction/verification can proceed without explicit user re-specification.
        - For AQME and AQME_TEST, the function performs CSV content checks required by the AQME descriptor generator: it reads the CSV header to ensure no duplicate column names exist (exits via sys.exit() with a message if duplicates are present), checks that at least one column name starts with "smiles" (case-insensitive) and exits with a warning if not found, and checks that the "code_name" column contains unique entries (exits with a warning otherwise). If all checks pass, a start message for AQME generation is written to the log.
        - The function always calls sanity_checks(self, 'initial', robert_module, None) for modules other than REPORT to perform additional validation; sanity_checks may modify self or abort execution depending on detected issues.
        - Side effects include: creating or modifying destination directories and files, instantiating and writing to log files (Logger), creating GENERATE folder structure, reading CSV and YAML files, and calling sys.exit() when fatal configuration problems are encountered (for example, invalid YAML import, malformed CSV required by AQME, duplicate code_name entries, or duplicated CSV headers). The function writes human-readable warnings and errors to the logger or to stdout before exiting in these cases.
        - The function does not return None; it returns the fully prepared internal options object ("self") that should be used by the caller to run the requested ROBERT submodule.
    
    Practical significance in the ROBERT domain:
        - This function centralizes configuration handling for the ROBERT package (which integrates ML workflows and cheminformatics tools as described in the README). By merging defaults, YAML imports, previously saved CURATE/GESTATE metadata, and command-line reconstruction into a single prepared object, it ensures reproducibility (via logged commands and parameters), consistent folder/layout conventions for model artifacts, and required validations (CSV structure for AQME, proper model parameter locations for PREDICT/VERIFY). Downstream modules rely on the returned "self" object to have canonical, validated settings so model training, evaluation, prediction, and descriptor generation proceed reliably.
    """
    from robert.utils import load_variables
    return load_variables(kwargs, robert_module)


################################################################################
# Source: robert.utils.mcc_scorer_clf
# File: robert/utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def robert_utils_mcc_scorer_clf(y_true: numpy.ndarray, y_pred: numpy.ndarray):
    """robert.utils.mcc_scorer_clf computes the Matthews correlation coefficient (MCC) between true class labels and predicted labels after coercing predictions to integer class labels. This function is intended for use in the ROBERT machine-learning utilities where classifier outputs sometimes arrive as floating-point values (for example, when a model's .predict() returns floats or when a regressor output is reused as a classifier). MCC is a robust single-value summary of binary or multiclass classification quality commonly used in cheminformatics workflows implemented in ROBERT, and it is especially informative for imbalanced class problems.
    
    This function forces the provided predicted values to integer labels by applying numpy.round followed by astype(int) and then delegates to sklearn.metrics.matthews_corrcoef to compute the coefficient. The rounding strategy is simple and deterministic: values with fractional part >= 0.5 are rounded up, others rounded down. The function does not modify the caller's y_pred array (it constructs a new integer array for scoring).
    
    Args:
        y_true (numpy.ndarray): Ground-truth class labels as a 1-D numeric array. In the ROBERT context these represent the experimentally measured or reference class assignments for chemical samples or model evaluation sets. y_true must have the same length as y_pred and should contain discrete class identifiers (integers or values that sensibly compare to the rounded y_pred). If y_true contains non-discrete values or labels that do not match the rounded predictions, the resulting MCC will not be meaningful.
        y_pred (numpy.ndarray): Predicted class values as a 1-D numeric array, potentially containing floats. This parameter represents model outputs from classifiers or other predictors within the ROBERT pipelines. The function will coerce these predictions to integer class labels by applying numpy.round(y_pred).astype(int) before scoring. If y_pred already contains integer labels, rounding has no effect beyond casting to int.
    
    Returns:
        float: The Matthews correlation coefficient computed between y_true and the coerced integer y_pred. The MCC is a single scalar in the range [-1, 1] where 1 indicates perfect prediction, 0 indicates no better than random prediction, and -1 indicates total disagreement. The returned float is produced by sklearn.metrics.matthews_corrcoef and can be used as a scalar performance metric in ROBERT model selection, cross-validation, or reporting.
    
    Behavior, side effects, and failure modes:
        - The function rounds predictions using numpy.round and casts to int; this deterministic rounding may be inappropriate for probability outputs where thresholding (e.g., >= 0.5) or argmax on class probabilities would be more suitable.
        - The original y_pred passed by the caller is not modified; a new integer array is created internally.
        - Both inputs must be one-dimensional arrays of equal length. If shapes differ, or if inputs are of unsupported shapes, sklearn.metrics.matthews_corrcoef will raise an exception (typically ValueError).
        - If y_true contains labels that are not compatible with the integer labels produced from y_pred (for example different label encodings), the MCC value will not reflect meaningful agreement; ensure consistent label encoding prior to calling this function.
        - No device or dtype guarantees beyond using numpy operations are provided; the function relies on numpy and sklearn behavior for numeric conversions and error reporting.
    """
    from robert.utils import mcc_scorer_clf
    return mcc_scorer_clf(y_true, y_pred)


################################################################################
# Source: robert.utils.outlier_analysis
# File: robert/utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def robert_utils_outlier_analysis(
    print_outliers: str,
    outliers_data: dict,
    outliers_set: str
):
    """robert.utils.outlier_analysis: Build a human-readable summary of outlier counts and per-instance outlier magnitudes for a given dataset split used in ROBERT regression workflows (chemistry-focused machine learning and regression model evaluation). This function is used by reporting utilities in ROBERT to produce text sections that quantify how many outliers were detected in a dataset split (train, validation, or test), compute the percentage of outliers relative to the total scaled points in that split, and append one-line entries for each outlier showing its identifier and standardized-deviation magnitude. The produced string is suitable for inclusion in console output, log files, or text reports (for example the ROBERT PDF report generator), enabling model developers and chemists to inspect and document problematic datapoints.
    
    Args:
        print_outliers (str): Initial report text to which the function will append the outlier summary lines. This parameter supplies the current accumulated string (possibly empty) and is not modified in place because Python strings are immutable; the function returns a new string with the appended information. In the ROBERT workflow this typically contains previously generated report sections for other datasets or metrics.
        outliers_data (dict): Dictionary containing lists (or other sequence types) of detected outlier magnitudes, corresponding sample identifiers, and scaled dataset points for the split being analyzed. For each allowed outliers_set value the function expects these exact keys to exist in this dictionary: if outliers_set == 'train' then keys 'outliers_train', 'train_scaled', and 'names_train' must be present; if outliers_set == 'valid' then keys 'outliers_valid', 'valid_scaled', and 'names_valid' must be present; if outliers_set == 'test' then keys 'outliers_test', 'test_scaled', and 'names_test' must be present. The values under the outliers_* and names_* keys are iterated in parallel (zip) so they should have matching ordering and compatible lengths. The function reads these collections but does not modify them.
        outliers_set (str): Identifier of the dataset split to analyze. Accepted literal values are 'train', 'valid', or 'test'. This string determines which key names are looked up in outliers_data and which human-readable label is used in the appended summary ('Train', 'Validation', or 'Test'). Passing any other value will cause the function to raise an exception (UnboundLocalError during execution) because internal labels are not set for unknown splits.
    
    Returns:
        str: The updated report string containing the original print_outliers content followed by a summary line and one line per detected outlier for the specified split. The summary line has the form "      <Label>: <n_outliers> outliers out of <n_points> datapoints (<percentage>%)" where <percentage> is computed as 100 * (number of outliers) / (number of scaled datapoints) and formatted with one decimal place. Each outlier line has the form "      -  <name> (<value> SDs)" where <value> is the outlier magnitude formatted to two significant digits as produced by the code's formatting. No printing to stdout is performed; the function only returns the new string.
    
    Behavior, defaults, and failure modes:
        The function selects keys and a printable label based on outliers_set and then computes per_cent = (len(outliers) / len(scaled_points)) * 100. If the scaled-points collection is empty, a ZeroDivisionError will be raised. If the expected keys are absent in outliers_data a KeyError will be raised. If outliers_set is not one of 'train', 'valid', or 'test', internal variables used to build keys remain undefined and an UnboundLocalError or NameError may occur. The function assumes that the collections under the selected outliers and names keys are iterable and that zip(outliers, names) yields the correct pairing; mismatched lengths result in omission of surplus elements (zip truncates to the shortest). The function performs no side effects other than returning the constructed string and does not alter outliers_data or any external state.
    """
    from robert.utils import outlier_analysis
    return outlier_analysis(print_outliers, outliers_data, outliers_set)


################################################################################
# Source: robert.utils.plot_metrics
# File: robert/utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def robert_utils_plot_metrics(
    model_data: dict,
    suffix_title: str,
    verify_metrics: dict,
    verify_results: dict
):
    """robert.utils.plot_metrics creates and saves a bar plot that visualizes the results of VERIFY tests for a given regression model. This function is part of the ROBERT toolkit (Refiner and Optimizer of a Bunch of Existing Regression Tools) and is used in the VERIFY workflow to inspect which automated checks (tests) for a trained model passed, failed, or are unclear, and to record the figure for reproducibility and reporting.
    
    Args:
        model_data (dict): Dictionary describing the model file and used to build the output path and filenames. The function expects model_data to contain at least the key 'model' whose value is a string path to a CSV-style model file. The basename of model_data['model'] is split on the suffix '_db.csv' and used to form a VERIFY subpath (base name -> "VERIFY/{csv_name}"). This argument is used to assemble the saved PNG filename and to label the plot; missing or non-string values for model_data['model'] will raise a KeyError or TypeError when the function attempts to construct the output path.
        suffix_title (str): Short string appended to the base VERIFY path to distinguish the saved figure (used to form path_n_suffix and the output PNG filename). This string is concatenated with the base VERIFY path (derived from model_data['model']) to create the final output file path and the text shown in the plot title. The value should be a filesystem-safe string; invalid characters may cause file-saving errors.
        verify_metrics (dict): Dictionary with numeric metrics, thresholds, names and colors for the VERIFY tests. Required keys (used by the implementation) are:
            'metrics' (iterable of numbers): numeric test values for each bar plotted (the function computes min/max and uses these values directly).
            'test_names' (iterable of str): labels for each bar on the x axis, in the same order as 'metrics'.
            'colors' (iterable of str): color hex codes for each bar (e.g. '#1f77b4' for pass, '#cd5c5c' for fail, '#c5c57d' for unclear). The mapping from hex codes to pass/fail/unclear text is performed exactly as in the code; differing color codes will not produce the intended text annotations.
            'higher_thres' or 'lower_thres' (number): pass threshold used for plotting a dashed horizontal threshold line. Which key is required depends on verify_results['error_type'] (see verify_results). For error types 'mae' or 'rmse' the function uses 'higher_thres'; for other error types it uses 'lower_thres'.
            'unclear_higher_thres' or 'unclear_lower_thres' (number): secondary threshold drawn as a dashed line indicating the "unclear" region; used alongside the primary threshold and selected based on verify_results['error_type'] similarly.
        verify_results (dict): Dictionary describing verification context and algorithmic assumptions. The implementation requires at least the key 'error_type' whose value is a string naming the error metric (e.g., 'mae', 'rmse', or other metrics). verify_results['error_type'] determines whether lower or higher numeric values are considered better and therefore how axis limits, threshold selection, and the direction of the pass-indicating arrow are computed. Supplying an unexpected or missing 'error_type' value will cause KeyError or logic that assumes "higher is better" behavior.
    
    Behavior and side effects:
        This function builds a matplotlib/seaborn figure (it reloads matplotlib.pyplot to avoid threading issues and resets seaborn defaults). It creates a bar chart (bar width 0.55) showing each provided test metric with colors specified in verify_metrics['colors'], annotates bars (except the one named 'Model') with pass/fail/unclear text based on exact color hex codes, formats the y axis to two decimal places, draws dashed horizontal threshold lines and an arrow indicating the direction in which tests pass, and adds a legend and gridlines. The plot title includes the VERIFY path and the supplied suffix_title. The y-axis limits are chosen from the min/max of verify_metrics['metrics'] and adjusted differently when verify_results['error_type'] is 'mae' or 'rmse' (these are treated as error metrics where lower values are better and the lower limit is set to zero) versus other error types (where higher values may be better). The function saves the figure as a PNG file into a VERIFY subpath built from model_data['model'] and suffix_title (filename pattern: VERIFY_tests_{basename}_{suffix_title}.png). The function prints and returns a short string containing the reduced path to the saved PNG.
    
    Defaults and formatting choices:
        The function uses seaborn style "ticks", a figure size of (7.45, 6), two-decimal y-axis formatting, and fixed bar edge color and line width. The mapping of exact color hex codes to textual annotations ('pass', 'fail', 'unclear') is hard-coded and must be honored by the caller if textual annotation is required.
    
    Failure modes and exceptions:
        If required dictionary keys are missing or of the wrong type, the function will raise KeyError or TypeError when accessing model_data['model'], verify_metrics[...] or verify_results['error_type']. If metrics are non-numeric, operations computing min/max and ranges will raise TypeError or ValueError. File system errors (OSError) can occur when saving the PNG if the current working directory is not writable or if the VERIFY subdirectory cannot be created. If matplotlib or seaborn are not available or fail to initialize, ImportError or runtime errors may be raised. The mapping from color hex codes to pass/fail/unclear text relies on exact string equality to '#1f77b4', '#cd5c5c', and '#c5c57d'; other color values will not produce the intended text annotations and will still plot the bars but without the textual pass/fail indicators.
    
    Returns:
        str: A printable status string that contains a shortened path to the saved PNG file (the function saves the figure to disk as a side effect and returns this status message). The returned string is intended for logging or printing by the caller and identifies where the VERIFY plot PNG was written.
    """
    from robert.utils import plot_metrics
    return plot_metrics(model_data, suffix_title, verify_metrics, verify_results)


################################################################################
# Source: robert.utils.scale_df
# File: robert/utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def robert_utils_scale_df(csv_X: numpy.ndarray, csv_X_external: numpy.ndarray):
    """Scale the feature matrix for training and (optionally) an external test set using
    scikit-learn's StandardScaler so that downstream regression models in the ROBERT
    pipeline receive mean-centered, unit-variance features.
    
    This function is used in ROBERT's preprocessing steps for quantitative structure-
    property/activity regression workflows: it fits a StandardScaler on the training
    descriptor matrix (csv_X) and applies the same scaling parameters to the
    external descriptor matrix (csv_X_external) when provided. The scaler is fitted
    only on csv_X to avoid data leakage from the test/external set.
    
    Args:
        csv_X (numpy.ndarray): The training feature matrix used to fit the scaler and
            produce the scaled training set. In practice within the ROBERT codebase
            this object is expected to behave like a 2D numeric table of shape
            (n_samples, n_features) and to provide a .columns attribute (for example,
            a pandas.DataFrame) because the function constructs the return value as
            a pandas.DataFrame with the same column names. The function fits a
            sklearn.preprocessing.StandardScaler on csv_X (computing per-feature
            mean and standard deviation). If csv_X contains non-numeric entries,
            missing values, or an unexpected shape, the scaler will raise a
            TypeError/ValueError during fitting.
        csv_X_external (numpy.ndarray): Optional external test feature matrix to be
            transformed using the scaler fitted on csv_X. If provided, it must have
            the same number of columns/features as csv_X and preferably the same
            feature order; otherwise sklearn's transform will raise a ValueError.
            As with csv_X, this argument is expected to be a 2D numeric table and
            may need a .columns attribute (e.g., a pandas.DataFrame) so that the
            returned scaled external set preserves column names. If csv_X_external
            is None, no external transformation is performed and the corresponding
            return value is None.
    
    Returns:
        X_scaled_df (pandas.DataFrame): A pandas DataFrame containing the training
            features after StandardScaler transformation (zero mean and unit
            variance per feature as computed from csv_X). Columns are set to
            csv_X.columns to preserve feature names used later in ROBERT's
            regression, explanation, or reporting modules.
        X_scaled_external_df (pandas.DataFrame or None): If csv_X_external was
            provided, this is a pandas DataFrame with the external features scaled
            using the scaler fitted on csv_X and with columns set to
            csv_X_external.columns. If csv_X_external was None, this value is None.
    
    Raises/Failure modes:
        The function relies on sklearn.preprocessing.StandardScaler and pandas.DataFrame:
        - AttributeError will occur when attempting to access .columns if the input
          objects do not provide that attribute.
        - ValueError is raised by StandardScaler.transform if csv_X_external has a
          different number of features than csv_X.
        - TypeError/ValueError may be raised during fit/transform if inputs contain
          non-numeric types or NaNs that the scaler cannot handle.
        - No in-place modification of csv_X or csv_X_external is performed; the
          function returns new pandas.DataFrame objects (or None for the external
          return) and leaves the input objects unchanged.
    
    Side effects and notes:
        - The scaler is fitted on csv_X only to prevent information leakage from the
          external set into model training.
        - Returned objects are pandas.DataFrame instances constructed inside the
          function; callers should ensure pandas is available in their environment.
        - This preprocessing is a required step before fitting many machine learning
          models in ROBERT to ensure features are on comparable scales for regression
          algorithms and for producing consistent model explanations and reports.
    """
    from robert.utils import scale_df
    return scale_df(csv_X, csv_X_external)


################################################################################
# Source: robert.utils.sort_n_load
# File: robert/utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def robert_utils_sort_n_load(Xy_data: dict):
    """robert.utils.sort_n_load sorts the training feature and target arrays contained in Xy_data to produce a reproducible, stable ordering of rows for downstream regression model training and evaluation in the ROBERT workflow.
    
    This function is used in ROBERT's machine-learning/regression pipelines to ensure that when the same database is loaded with different row orders (for example because of file or OS-dependent ordering), the resulting X and y arrays are reordered deterministically so that model training, cross-validation splits, and results are reproducible across runs and platforms. The implementation converts inputs to numpy arrays, computes a stable argsort on the target values, and reindexes the feature matrix to preserve the original feature-target pairing.
    
    Args:
        Xy_data (dict): A dictionary expected to contain at least the keys 'X_train_scaled' and 'y_train' used by ROBERT. 'X_train_scaled' should be the training feature matrix after any scaling step (for example, a 2D array-like object where rows correspond to samples and columns to scaled features). 'y_train' should be the corresponding training target values for regression (typically a 1D array-like of numeric target values). The function will convert these values to numpy.ndarray internally. The practical role of Xy_data is to carry the scaled features and target for model fitting; this function enforces a stable, reproducible row order for that data. Passing a non-dict will raise a TypeError; omission of the required keys will raise a KeyError.
    
    Returns:
        tuple[numpy.ndarray, numpy.ndarray]: A pair (sorted_X_train_scaled, sorted_y_train).
        sorted_X_train_scaled is a numpy.ndarray containing the rows of X_train_scaled reordered to match the sorting of y_train. sorted_y_train is a numpy.ndarray containing the elements of y_train sorted in ascending order using numpy.argsort with kind='stable'. Both arrays are newly created objects; the input Xy_data dictionary is not modified in place. Typical use is to feed these returned arrays into subsequent training or evaluation steps in ROBERT to ensure consistent behavior across runs.
    
    Behavior and failure modes:
        The function converts the provided X_train_scaled and y_train to numpy arrays, computes sorted_indices = np.argsort(y_train, kind='stable'), and applies those indices to reorder X_train_scaled and y_train. If y_train is not one-dimensional or not numeric, the sorting semantics may be unexpected for regression tasks; callers should provide a 1D numeric target array. If X_train_scaled and y_train have incompatible lengths (different numbers of samples), indexing X_train_scaled[sorted_indices] may raise an IndexError or produce an inconsistent pairing; callers must ensure the number of rows in X_train_scaled matches the length of y_train. The function does not perform imputation or NaN handling; behavior with NaNs follows numpy.argsort semantics. There are no other side effects beyond returning the two reordered numpy arrays.
    """
    from robert.utils import sort_n_load
    return sort_n_load(Xy_data)


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
