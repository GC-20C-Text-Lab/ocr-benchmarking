"""
Benchmarking LLM JSON extraction.

We open JSON output files in a tabular format (Pandas) and fill in empty values with defaults.
Then, we perform cell-by-cell matching.

The output has one row for each document and metric:
    - `docN:__COL__:<col_name>`: Number of matching cells in `col_name`. Only exists if row counts match.
    - `docN:matches`: Number of matching cells in document if row counts match, otherwise NaN
    - `docN:total`: Total matching cells in document if row counts match, otherwise NaN
    - `docN:matches_pct`: Percent of matching cells if row counts match, otherwise NaN
    - `docN:mismatch_bool`: False if number of rows between ground truth and predicted data matches or could be adjusted to match.
    - `docN:pred_nrows`: Number of rows in the predicted data.
    - `docN:gt_nrows`: Number of rows in the ground truth data.

The output also has rows for aggregate metrics, namely:
    - `__ALL__:__COL__:<col_name>`: Number of matching cells in `col_name` for all pages where row counts match.
    - `__ALL__:matches`: Number of matching cells among pages with matching row counts
    - `__ALL__:total`: Total number of cells among pages with matching row counts
    - `__ALL__:matches_pct`: Percent of `__ALL__:matches` out of `__ALL__:total`, or 0 if total is 0.
    - `__ALL__:mismatched_dim_count`: Number of pages with mismatched row counts.
    - `__ALL__:pred_nrows`: Total number of rows in the prediction.
    - `__ALL__:gt_nrows`: Total number of rows in the ground truth.
    - `__ALL__:counted_nrows`: Total number of rows from pages with matching row counts.
 
Tables are saved under ../../benchmarking-results/json-accuracy/
 - llm-img2json  *LLM image to JSON via LLM*   
    - llm-img2json_fuzzy_%Y-%m-%d_%H:%M:%S.csv       *See `fuzzy_cell_matcher`*
    - llm-img2json_nonorm_%Y-%m-%d_%H:%M:%S.csv      *See `exact_cell_matcher`*
    - llm-img2json_normalized_%Y-%m-%d_%H:%M:%S.csv  *See `normalized_cell_matcher`*
 - llm-txt2json  *LLM text to JSON via LLM*
    - llm-txt2json_fuzzy_%Y-%m-%d_%H:%M:%S.csv
    - llm-txt2json_nonorm_%Y-%m-%d_%H:%M:%S.csv
    - llm-txt2json_normalized_%Y-%m-%d_%H:%M:%S.csv

String matching rules:
 - EXACT: cell1 == cell2 (after basic lowercasing/whitespace trimming)
 - NORMALIZED: ASCII-only, punctuation removed, then lower-cased, compare exact
 - FUZZY: Jaro-Winkler similarity >= 0.90 => matched
Numeric values must exactly match.

If the ground truth has one more row than the LLM output, add an empty row to the LLM output and
proceed with matching. Otherwise, skip the document and mark `mismatch_bool` as True.

Original authors: Niclas Griesshaber, Gavin Greif, Robin Greif
New authors: Tim Yu, Muhammad Khalid
"""

# ----------------- Imports -----------------
import json
import os
import re
import logging
from typing import Callable
from rapidfuzz.distance import JaroWinkler
import pandas as pd
import numpy as np
import sys


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
        return x_str
    else:
        return x


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

def load_json_safely(path):
    """Load JSON into a DataFrame, or return None if missing/error."""
    if not os.path.isfile(path):
        logger.warning(f"Could not load JSON at {path}: not a file")
        return None
    try:
        with open(path, 'r') as file:
            file_json = json.loads(file.read())
            entries = file_json['entries']
            df = pd.DataFrame(entries)
            return df
    except Exception as e:
        logger.warning(f"Could not load JSON at {path}: {e}")
        return None


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
            df[col] = pd.NA
    # Reindex to EXACTLY those columns => remove extras
    df = df.reindex(columns=EXPECTED_COLUMNS)
    # Normalize all cells (for EXACT matching)
    df = normalize_dataframe(df)
    return df


# ----------------- Matching Comparisons -----------------

def make_match_dataframe(
    gt_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    cell_matcher: Callable[[any, any, dict[str, any]], bool],
    options: dict[str, any]
) -> pd.DataFrame:
    """
    Create a boolean dataframe whose cell is True if the two corresponding cells in gt_df and
    pred_df match using the cell_matcher predicate function.

    gt_df and pred_df must have the same dimensions.
    """
    match_df = pd.DataFrame().reindex_like(gt_df).astype(bool)

    nrows = gt_df.shape[0]
    ncols = len(EXPECTED_COLUMNS)

    if nrows != pred_df.shape[0]:
        raise ValueError("gt_df and pred_df do not have correct dimensions")

    for row in range(nrows):
        for col in range(ncols):
            match_df.iat[row, col] = cell_matcher(gt_df.iat[row, col], pred_df.iat[row, col], options)
    
    return match_df


def exact_cell_matcher(gt_cell, pred_cell, options):
    return gt_cell == pred_cell


def normalized_cell_matcher(gt_cell, pred_cell, options):
    """
    If either the ground-truth and/or prediction are strings, remove non-ASCII/punctuation; make lower-case.
    Then check for exact match.
    """
    norm_gt = ascii_only_punct_removed_lower(gt_cell) if isinstance(gt_cell, str) else gt_cell
    norm_pr = ascii_only_punct_removed_lower(pred_cell) if isinstance(pred_cell, str) else pred_cell
    return norm_gt == norm_pr


def fuzzy_cell_matcher(gt_cell, pred_cell, options):
    """
    Use the Jaro-Winkler similarity if both the ground-truth and prediction are strings.
    Use options["threshold"] for threshold.
    Otherwise, check for exact match.
    """
    if isinstance(gt_cell, str) and isinstance(pred_cell, str):
        return JaroWinkler.similarity(gt_cell, pred_cell) >= options["threshold"]
    else:
        return gt_cell == pred_cell


def compare_dataframes_core(gt_df: pd.DataFrame, pred_df: pd.DataFrame, method: str, options={"threshold": FUZZY_THRESHOLD}):
    """
    Compares the two dataframes, gt_df and pred_df, using the method parameter (exact, normalized, fuzzy).
    If gt_df has one more row than pred_df, then add one empty row to the top of pred_df to normalize dimensions.
    
    Returns a dictionary containing the keys and corresponding values:
    - gt_df_adj: A copy of the ground truth dataframe after dimensional adjustments
    - pred_df_adj: A copy of the predicted dataframe after dimensional adjustments
    - match_df: A boolean dataframe where a cell is true if and only if the corresponding cell in gt_df_adj and pred_df_adj match.
    - results: A dictionary with selected results. See the top of document for details.
    """
    # Check if any dataframe is empty
    if gt_df is None or pred_df is None:
        results = {
            "matches": np.nan,
            "total": np.nan,
            "mismatch_bool": True,
            "pred_nrows": 0,
            "pred_adj_nrows": 0,
            "gt_nrows": 0,
        }
        return {
            "gt_df_adj": None,
            "pred_df_adj": None,
            "match_df": None,
            "results": results
        }
    
    # Copy dataframes since we are potentially modifying them
    gt_df_temp = gt_df.copy()
    pred_df_temp = pred_df.copy()

    # Check for row mismatch
    gt_rows = gt_df_temp.shape[0]
    pred_rows = pred_df_temp.shape[0]

    # If gt_row is one more than pred_rows, add empty row to beginning of pred_df due to observation
    # of missing empty rows.
    #
    # For other row number mismatches, skip comparison and return a mismatch.
    if gt_rows == pred_rows + 1:
        pred_df_temp.loc[-1] = np.nan
        pred_df_temp = pred_df_temp.fillna(EXPECTED_COLUMNS_DEFAULTS)
        pred_df_temp.index = pred_df_temp.index + 1
        pred_df_temp.sort_index(inplace=True)
    elif gt_rows != pred_rows:
        results = {
            "matches": np.nan,
            "total": np.nan,
            "mismatch_bool": True,
            "pred_nrows": pred_rows,
            "pred_adj_nrows": pred_rows,
            "gt_nrows": gt_rows,
        }
        return {
            "gt_df_adj": gt_df_temp,
            "pred_df_adj": pred_df_temp,
            "match_df": None,
            "results": results
        }
        
    
    # Determine adjusted row count
    pred_adj_rows = pred_df_temp.shape[0]

    # Select matching method
    matcher = {
        "exact": exact_cell_matcher,
        "normalized": normalized_cell_matcher,
        "fuzzy": fuzzy_cell_matcher
    }.get(method, "exact")

    # Create dataframe for matches and results dict
    match_df = make_match_dataframe(gt_df_temp, pred_df_temp, matcher, options)
    results = {}

    # Count number of True values in each column and add to returned results
    total_matches = 0
    for col in match_df.columns:
        col_matches = match_df[col].sum()
        results[f"__COL__:{col}"] = col_matches
        total_matches += col_matches
    
    results["matches"] = total_matches
    results["total"] = gt_rows * len(EXPECTED_COLUMNS)
    results["mismatch_bool"] = False
    results["pred_nrows"] = pred_rows
    results["pred_adj_nrows"] = pred_adj_rows
    results["gt_nrows"] = gt_rows

    return {
        "gt_df_adj": gt_df_temp,
        "pred_df_adj": pred_df_temp,
        "match_df": match_df,
        "results": results
    }


def compare_dataframes(gt_df: pd.DataFrame, pred_df: pd.DataFrame, method: str, options={"threshold": FUZZY_THRESHOLD}):
    """
    Compares the two dataframes, gt_df and pred_df, using the method parameter (exact, normalized, fuzzy).
    If gt_df has one more row than pred_df, then add one empty row to the top of pred_df to match dimensions.
    Otherwise, no comparison is made and "mismatch_bool" in the returned dictionary is True.
    """

    compare_output = compare_dataframes_core(gt_df, pred_df, method, options)
    return compare_output["results"]



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
    Build a Pandas dataframe for a given results_data and doc_names structure.
    - results_data[model][doc] => (matches, total, mismatch_bool, pred_nrows)

    See the top of the file for information about metrics.

    The dataframe has one column for each model used and one row for each metric (per page).

    Returns the dataframe for the results data.
    """

    logger.info("Building dataframe \"%s\" with documents %s", title, doc_names)

    # Create dataframe
    df = pd.DataFrame(columns=results_data.keys())

    # Populate dataframe
    for model in results_data.keys():
        model_sum_matches = 0
        model_sum_total = 0
        model_sum_mismatches = 0
        model_sum_pred_nrows = 0
        model_sum_pred_adj_nrows = 0
        model_sum_gt_nrows = 0
        model_sum_counted_nrows = 0
        model_col_results = {}

        for doc in doc_names:
            cell_data = results_data[model].get(doc, None)

            if cell_data is not None:
                # Move all results from cell_data to returned dataframe
                for key in cell_data.keys():
                    df.at[f"{doc}:{key}", model] = cell_data[key]
                    if key.startswith("__COL__:"):
                        model_col_results[key] = model_col_results.get(key, 0) + cell_data[key]

                df.at[f"{doc}:matches_pct", model] = (
                    (cell_data["matches"] / cell_data["total"]) * 100
                        if (not cell_data["mismatch_bool"]) or cell_data["total"] > 0
                        else np.nan
                            if cell_data["mismatch_bool"]
                            else 0
                )

                model_sum_matches += cell_data["matches"] if not pd.isna(cell_data["matches"]) else 0
                model_sum_total += cell_data["total"] if not pd.isna(cell_data["total"]) else 0
                model_sum_mismatches += 1 if cell_data["mismatch_bool"] else 0
                model_sum_pred_nrows += cell_data["pred_nrows"]
                model_sum_pred_adj_nrows += cell_data["pred_adj_nrows"]
                model_sum_gt_nrows += cell_data["gt_nrows"]
                model_sum_counted_nrows += cell_data["gt_nrows"] if not cell_data["mismatch_bool"] else 0
        
        for key in model_col_results.keys():
            df.at[f"__ALL__:{key}", model] = model_col_results[key]
        df.at["__ALL__:matches", model] = model_sum_matches
        df.at["__ALL__:total", model] = model_sum_total
        df.at["__ALL__:matches_pct", model] = (model_sum_matches / model_sum_total) * 100 if model_sum_total > 0 else 0
        df.at["__ALL__:mismatched_dim_count", model] = model_sum_mismatches
        df.at["__ALL__:pred_nrows", model] = model_sum_pred_nrows
        df.at["__ALL__:pred_adj_nrows", model] = model_sum_pred_adj_nrows
        df.at["__ALL__:gt_nrows", model] = model_sum_gt_nrows
        df.at["__ALL__:counted_nrows", model] = model_sum_counted_nrows

    return df
