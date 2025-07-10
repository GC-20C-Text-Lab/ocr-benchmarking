"""
Benchmarking LLM vs. IMG JSON extraction,
generating THREE tables now:
 - Exact matching
 - Normalized matching
 - Fuzzy matching (Jaro-Winkler, threshold=0.90)
 
Tables are saved under project-root/benchmarking-results/json-accuracy/
 - gt-txt2json   # Ground truth text to JSON via LLM
 - ocr-txt2json  # OCR text to JSON via LLM
 - llm-img2json  # LLM image to JSON via LLM
 - llm-txt2json  # LLM text to JSON via LLM

Matching rules:
 - EXACT: cell1 == cell2 (after basic lowercasing/whitespace trimming)
 - NORMALIZED: ASCII-only, punctuation removed, then lower-cased, compare exact
 - FUZZY: Jaro-Winkler similarity >= 0.90 => matched

If row counts differ, we proceed with one of the two following options:
 1. label dimension mismatch (and skip that doc in the aggregated tables).
 2. WIP: search next rows based on row count difference, take higher similarity

Original authors: Niclas Griesshaber, Gavin Greif, Robin Greif
New authors: Tim Yu, Muhammad Khalid
"""

# ----------------- Imports -----------------
import json
import os
import re
import glob
import argparse
import logging
from rapidfuzz.distance import JaroWinkler
import pandas as pd
import numpy as np
import math
from datetime import datetime
import sys
import jiwer

script_dir = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))

sys.path.insert(1, os.path.join(project_root, "src"))
from tools.file_retrieval import get_doc_names, get_docs, get_all_models


# ----------------- Configure Logging -----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# Helper functions
# Based on code from https://github.com/niclasgriesshaber/llm_historical_dataset_benchmarking/blob/main/src/benchmarking/txt_accuracy.py

# ----------------- Configuration -----------------

EXPECTED_COLUMNS_DEFAULTS = {
    "lastname": "",
    "firstname": "",
    "maidenname": "",
    "birthyear": 0,
    "deathyear": 0,
    "title": "",
    "city": "",
    "publisher": "",
    "publishyear": 0,
    "pagecount": 0,
    "library": "",
    "description": "",
    "index": 0
}
EXPECTED_COLUMNS = EXPECTED_COLUMNS_DEFAULTS.keys()
FUZZY_THRESHOLD = 0.90  # Adjust if you need a different “sellable” threshold

# ----------------- Cell normalization -----------------

def normalize_cell_value(x):
    """
    Basic normalization for EXACT match:
      - lowercase, strip whitespace for strings
    """
    if isinstance(x, str):
        x_str = str(x).strip().lower()
        if x_str in ("", "null"):
            return None
        return x_str


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    In-place: normalize every cell with normalize_cell_value and fill NaN values
    """
    df = df.fillna(EXPECTED_COLUMNS_DEFAULTS)
    for col in df.columns:
        df[col] = df[col].apply(normalize_cell_value)
    return df


def ascii_only_punct_removed_lower(x):
    """
    Normalized version for "normalized matching":
      1) Lowercase
      2) Remove non-ASCII characters
      3) Remove punctuation
      4) Trim whitespace

    Ignore integers.
    """
    if not isinstance(x, str):
        return x
    if not x:
        return ""
    # Lowercase
    x = x.lower()
    # Remove non-ASCII
    x = x.encode("ascii", "ignore").decode("ascii", errors="ignore")
    # Remove punctuation (anything not alphanumeric/underscore/whitespace)
    x = re.sub(r"[^\w\s]", "", x)
    # Collapse multiple spaces
    x = " ".join(x.split())
    return x


# ----------------- Helper Functions -----------------

# def load_json_safely(path):
#     """Load JSON into a DataFrame, or return None if missing/error."""
#     if not os.path.isfile(path):
#         logger.warning(f"Could not load JSON at {path}: not a file")
#         return None
#     try:
#         with open(path, 'r') as file:
#             file_json = json.loads(file.read())
#             entries = file_json['entries']
#             df = pd.DataFrame(entries)
#             return df
#     except Exception as e:
#         logger.warning(f"Could not load JSON at {path}: {e}")
#         return None


def filter_expected_columns(df):
    """
    Keep only the EXPECTED_COLUMNS, in that exact order. Add missing columns.
    If df is None, return None => mismatch.
    If df is missing any of these columns, add columns and populate with NaN values.
    Otherwise, do a basic normalization of cells for the EXACT step.
    """
    if df is None:
        return None
    # Check if all expected columns exist
    for col in EXPECTED_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    # Reindex to EXACTLY those columns => remove extras
    df = df.reindex(columns=EXPECTED_COLUMNS)
    # Normalize all cells (for EXACT matching)
    df = normalize_dataframe(df)
    return df


# ----------------- Matching Comparisons -----------------

def compare_dataframes(gt_df, pred_df, method, options={}):
    # Check if any dataframe is empty
    if gt_df is None or pred_df is None:
        return {
            "matches": np.nan,
            "total": np.nan,
            "mismatch_bool": True,
            "pred_nrows": 0,
            "gt_nrows": 0,
        }

    # Check for row mismatch
    gt_rows = gt_df.shape[0]
    pred_rows = pred_df.shape[0]

    if gt_rows != pred_rows:
        return {
            "matches": np.nan,
            "total": np.nan,
            "mismatch_bool": True,
            "pred_nrows": pred_rows,
            "gt_nrows": gt_rows,
        }
    
    if method == 'exact':
        matches = (gt_df.values == pred_df.values).sum()
        total = gt_rows * len(EXPECTED_COLUMNS)
        return {
            "matches": matches,
            "total": total,
            "mismatch_bool": False,
            "pred_nrows": pred_rows,
            "gt_nrows": gt_rows,
        }
    
    elif method == "normalized":
        total_cells = gt_rows * len(EXPECTED_COLUMNS)
        match_count = 0

        for row_idx in range(gt_rows):
            for col_idx in range(len(EXPECTED_COLUMNS)):
                gt_val = gt_df.iat[row_idx, col_idx]
                pr_val = pred_df.iat[row_idx, col_idx]
                # Apply "normalized" pipeline:
                if isinstance(gt_val, str) and isinstance(pr_val, str):
                    norm_gt = ascii_only_punct_removed_lower(gt_val)
                    norm_pr = ascii_only_punct_removed_lower(pr_val)
                else:
                    norm_gt = gt_val
                    norm_pr = pr_val
                if norm_gt == norm_pr:
                    match_count += 1

        return {
            "matches": match_count,
            "total": total_cells,
            "mismatch_bool": False,
            "pred_nrows": pred_rows,
            "gt_nrows": gt_rows,
        }
    
    elif method == "fuzzy":
        total_cells = gt_rows * len(EXPECTED_COLUMNS)
        match_count = 0
        for row_idx in range(gt_rows):
            for col_idx in range(len(EXPECTED_COLUMNS)):
                val_gt = gt_df.iat[row_idx, col_idx]
                val_pr = pred_df.iat[row_idx, col_idx]

                if isinstance(val_gt, str) and isinstance(val_pr, str):
                    sim = JaroWinkler.similarity(val_gt, val_pr)
                    if sim >= options["threshold"]:
                        match_count += 1
                else:
                    if val_gt == val_pr:
                        match_count += 1

        return {
            "matches": match_count,
            "total": total_cells,
            "mismatch_bool": False,
            "pred_nrows": pred_rows,
            "gt_nrows": gt_rows,
        }
    
    else:
        raise ValueError("Invalid method for dataframe comparison")


def compare_dataframes_exact(gt_df, pred_df):
    """
    EXACT MATCH:
      - If row counts differ => dimension mismatch
      - Otherwise => compare cell by cell (exact equality after basic normalize)
    Returns a dict with keys: matches, total_cells, mismatch_bool, pred_nrows.
    """
    return compare_dataframes(gt_df, pred_df, "exact")


def compare_dataframes_normalized(gt_df, pred_df):
    """
    NORMALIZED MATCH (ASCII-only, punctuation removed, lower-cased):
      - If row counts differ => dimension mismatch
      - Otherwise => transform each cell, then compare exact equality
    """
    return compare_dataframes(gt_df, pred_df, "normalized")
    

def compare_dataframes_fuzzy(gt_df, pred_df, threshold=FUZZY_THRESHOLD):
    """
    FUZZY MATCH (Jaro-Winkler):
      - If row counts differ => dimension mismatch
      - Otherwise => Jaro-Winkler >= threshold => 1 match (for strings); exact match (for numbers)
    """
    return compare_dataframes(gt_df, pred_df, "fuzzy", {"threshold": threshold})
    


def build_dataframe(title, doc_names, results_data):
    """
    Build a Pandas dataframe for a given results_data and doc_lengths structure.
    - results_data[model][doc] => (matches, total, mismatch_bool, pred_nrows)

    The dataframe has one row for each document and metric:
    - `docN:matches`: Number of matching cells in document if row counts match, otherwise NaN
    - `docN:total`: Total matching cells in document if row counts match, otherwise NaN
    - `docN:matches_pct`: Percent of matching cells if row counts match, otherwise NaN
    - `docN:mismatch_bool`: True if number of rows between ground truth and predicted data matches. 
    - `docN:pred_nrows`: Number of rows in the predicted data.
    - `docN:gt_nrows`: Number of rows in the ground truth data.

    The dataframe also has rows for aggregate metrics, namely:
    - `__ALL__:matches`: Number of matching cells among pages with matching row counts
    - `__ALL__:total`: Total number of cells among pages with matching row counts
    - `__ALL__:matches_pct`: `__ALL__:matches` divided by `__ALL__:total`, or 0 if `__ALL__:total` is 0.
    - `__ALL__:mismatched_dim_count`: Number of pages with mismatched row counts.
    - `__ALL__:pred_nrows`: Total number of rows in the prediction.
    - `__ALL__:gt_nrows`: Total number of rows in the ground truth.

    The dataframe has one column for each model used, like pytesseract.

    Returns the dataframe for the results data.
    """

    # One column per document
    metrics = ["matches", "total", "mismatch_bool", "pred_nrows"]

    logger.info("Building dataframe \"%s\" with documents %s", title, doc_names)

    # Create dataframe
    df = pd.DataFrame(columns=results_data.keys())

    # Populate dataframe
    for model in results_data.keys():
        model_sum_matches = 0
        model_sum_total = 0
        model_sum_mismatches = 0
        model_sum_pred_nrows = 0
        model_sum_gt_nrows = 0

        for doc in doc_names:
            cell_data = results_data[model].get(doc, None)
            if cell_data is not None:
                df.at[f"{doc}:matches", model] = cell_data["matches"]
                df.at[f"{doc}:total", model] = cell_data["total"]
                df.at[f"{doc}:matches_pct", model] = (
                    (cell_data["matches"] / cell_data["total"]) * 100
                        if (not cell_data["mismatch_bool"]) or cell_data["total"] > 0
                        else np.nan
                            if cell_data["mismatch_bool"]
                            else 0
                )
                df.at[f"{doc}:mismatch_bool", model] = cell_data["mismatch_bool"]
                df.at[f"{doc}:pred_nrows", model] = cell_data["pred_nrows"]
                df.at[f"{doc}:gt_nrows", model] = cell_data["gt_nrows"]

                model_sum_matches += cell_data["matches"] if not pd.isna(cell_data["matches"]) else 0
                model_sum_total += cell_data["total"] if not pd.isna(cell_data["total"]) else 0
                model_sum_mismatches += 1 if cell_data["mismatch_bool"] else 0
                model_sum_pred_nrows += cell_data["pred_nrows"]
                model_sum_gt_nrows += cell_data["gt_nrows"]
        
        df.at["__ALL__:matches", model] = model_sum_matches
        df.at["__ALL__:total", model] = model_sum_total
        df.at["__ALL__:matches_pct", model] = (model_sum_matches / model_sum_total) * 100 if model_sum_total > 0 else 0
        df.at["__ALL__:mismatched_dim_count", model] = model_sum_mismatches
        df.at["__ALL__:pred_nrows", model] = model_sum_pred_nrows
        df.at["__ALL__:gt_nrows", model] = model_sum_gt_nrows

    return df


# ----------------- Main -----------------


def main():
    """
    Prerequisites:
    - Ground truth JSON files located at `project_root/ground-truth/json/gt_kbaa-pXYZ.json`
    - LLM/OCR transcribed JSON files located at:
        - for ground truth text to JSON via LLM:
            - `project_root/results/gt-txt2json/<MODEL-NAME>/<MODEL-NAME>_img_kbaa-pXYZ.json`
        - for OCR text to JSON via LLM:
            - `project_root/results/ocr-txt2json/<MODEL-NAME>/<MODEL-NAME>_img_kbaa-pXYZ.json`
        - for image to JSON via LLM:
            - `project_root/results/llm-img2json/<MODEL-NAME>/<MODEL-NAME>_img_kbaa-pXYZ.json`
        - for text to JSON via LLM:
            - `project_root/results/llm-txt2json/<MODEL-NAME>/<MODEL-NAME>_img_kbaa-pXYZ.json`

    The main function will:
    - Gather all ground truth JSON files
    - For each ground truth JSON file and for each LLM/OCR model, open the JSON file's entries object as a Pandas dataframe
    - Clean all the JSON files (either basic cleaning and normalization)
    - Compute metrics for each file and model
    - Save results in two CSV files (one for normalized, one for non-normalized)
        - Results are saved in `project_root/benchmarking-results/txt-accuracy`
    """

    # =============
    # Preliminaries
    # =============

    logger.info("Script directory: %s", script_dir)
    logger.info("Project root: %s", project_root)

    # Ground truth
    ground_truth_dir = os.path.join(project_root, "data", "ground-truth", "json")
    doc_names = get_doc_names(ground_truth_dir, "json", keep_prefix=False)

    # results/ paths
    all_models = get_all_models(
        os.path.join(project_root, "results", "gt-txt2json"),
        os.path.join(project_root, "results", "ocr-txt2json"),
        os.path.join(project_root, "results", "llm-img2json"),
        os.path.join(project_root, "results", "llm-txt2json")
    )
    logger.info(f"Models found: {all_models}")

    # ===========
    # Gather files
    # ===========

    # -> Gather ground truths and put into dict:

    ground_truths_json, _ = get_docs(
        ground_truth_dir, doc_names, "json", name_has_prefix=True
    )

    logger.info("Collected ground truth results: %s", list(ground_truths_json.keys()))

    # Convert JSON to dataframe

    ground_truths_df = {
        doc_name: filter_expected_columns(pd.DataFrame(doc_json['entries'])) for doc_name, doc_json in ground_truths_json.items()
    }

    logger.info("Converted ground truths to dataframes")

    # -> Gather each transcribed document and put into dict:

    # Structure: results[model][doc]
    results_json = {}
    results_df = {}

    for model_type, model in all_models:
        logger.info("Collecting results for model: %s/%s", model_type, model)

        model_path = os.path.join(project_root, "results", model_type, model)
        results_json[model], _ = get_docs(
            model_path, doc_names, "json", name_has_prefix=True
        )

        logger.info("Collected results for model: %s", list(results_json[model].keys()))

        results_df[model] = {
            doc_name: filter_expected_columns(pd.DataFrame(doc_json['entries'])) for doc_name, doc_json in results_json[model].items()
        }

        logger.info("Converted results to dataframes")


    # ===============
    # Compute metrics
    # ===============

    normalized_results_data = {}
    nonorm_results_data = {}
    fuzzy_results_data = {}

    for model_type, model in all_models:
        normalized_results_data[model_type] = normalized_results_data.get(model_type, {})
        normalized_results_data[model_type][model] = normalized_results_data[model_type].get(model, {})

        nonorm_results_data[model_type] = nonorm_results_data.get(model_type, {})
        nonorm_results_data[model_type][model] = nonorm_results_data[model_type].get(model, {})

        fuzzy_results_data[model_type] = fuzzy_results_data.get(model_type, {})
        fuzzy_results_data[model_type][model] = fuzzy_results_data[model_type].get(model, {})
        
        logger.info("Computing metrics for model: %s", model)

        for doc in doc_names:
            logger.info("Computing metrics for document: %s", doc)

            normalized_results_data[model_type][model][doc] = compare_dataframes_normalized(
                ground_truths_df[doc], results_df[model][doc]
            )
            nonorm_results_data[model_type][model][doc] = compare_dataframes_exact(
                ground_truths_df[doc], results_df[model][doc]
            )
            fuzzy_results_data[model_type][model][doc] = compare_dataframes_fuzzy(
                ground_truths_df[doc], results_df[model][doc]
            )


    # =====================================
    # Put metrics in table and save results
    # =====================================

    time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Iterate over model types:
    for model_type in normalized_results_data.keys():
        normalized_df = build_dataframe(f"{model_type}_normalized_{time}", doc_names, normalized_results_data[model_type])
        nonorm_df = build_dataframe(f"{model_type}_nonorm_{time}", doc_names, nonorm_results_data[model_type])
        fuzzy_df = build_dataframe(f"{model_type}_fuzzy_{time}", doc_names, fuzzy_results_data[model_type])

        results_path = os.path.join(project_root, "benchmarking-results", "json-accuracy", model_type)
        if not os.path.exists(results_path):
            os.makedirs(results_path)

        normalized_df.to_csv(os.path.join(results_path, f"{model_type}_normalized_{time}.csv"))
        nonorm_df.to_csv(os.path.join(results_path, f"{model_type}_nonorm_{time}.csv"))
        fuzzy_df.to_csv(os.path.join(results_path, f"{model_type}_fuzzy_{time}.csv"))
    


if __name__ == "__main__":
    main()
