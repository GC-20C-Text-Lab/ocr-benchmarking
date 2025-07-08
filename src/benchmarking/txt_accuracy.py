"""
Benchmarking OCR vs. LLM for text extraction, parallelized with joblib + rapidfuzz.

Generates TWO Pandas dataframes in TWO .csv files:

1) Normalized Results
   - Non-ASCII removed entirely
   - Lowercase
   - Remove punctuation => only [a-z0-9] plus spaces
   - Collapse multiple spaces
   - Strip leading/trailing
   - Remove line breaks/tabs

2) Non-normalized Results
   - Preserve punctuation, casing, accented letters
   - Remove line breaks/tabs
   - Collapse multiple spaces
   - Strip leading/trailing

Each type of results has 4 rows for each document and for all documents:
   1) Levenshtein distance ({doc}:dist_char)
   2) ground-truth doc length (for that table's version) ({doc}:gt_length)
   3) CER% (distance / length_of_that_version) ({doc}:cer_pct)
   4) WER% ({doc}:wer_pct)

Each model is on a separate column.

Original authors: Niclas Griesshaber, Gavin Greif, Robin Greif
New authors: Tim Yu, Muhammad Khalid
"""

# ----------------- Imports -----------------
import os
import re
import glob
import argparse
import logging
from rapidfuzz import distance, fuzz
import pandas as pd
from datetime import datetime
import sys
import jiwer

script_dir = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))

sys.path.insert(1, os.path.join(project_root, "src"))
from tools.file_retrieval import get_doc_names, get_docs


# ----------------- Configure Logging -----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# Helper functions
# Based on code from https://github.com/niclasgriesshaber/llm_historical_dataset_benchmarking/blob/main/src/benchmarking/txt_accuracy.py


def clean_text_nonorm(text, index_numbers=True):
    """
    Minimal cleaning:
      - Remove index numbers (if specified)
      - Remove linebreaks/tabs (replace with space)
      - Remove all instances of \"- \" (dash space; word separated by line break)
      - Remove extra spaces of number intervals separated by line break
      - Collapse multiple spaces
      - Strip leading/trailing
      - Preserve punctuation, casing, accented letters
    """

    # If index_numbers == False, remove index numbers
    text = re.sub(r" *\[ *[0-9]+ *\] *", " ", text) if not index_numbers else text

    # Replace various forms of whitespace with space
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")

    # Replace multiple spaces with single space
    text = re.sub(r"\s+", " ", text)

    # Remove instances of "- " for words separated by line break.
    text = re.sub(r"([A-Za-z]+)- ([a-z]+)", r"\1\2", text)

    # Replace spaces in "- " for number ranges and abbreviations separated by line break.
    text = re.sub(r"([0-9A-Z]+)- ([0-9A-Z]+)", r"\1-\2", text)

    return text.strip()


def clean_json_nonorm(data, index_numbers=True):
    """
    Minimal cleaning:
      - Remove index numbers (if specified)
      - Remove linebreaks/tabs (replace with space)
      - Remove all instances of \"- \" (dash space; word separated by line break)
      - Remove extra spaces of number intervals separated by line break
      - Collapse multiple spaces
      - Strip leading/trailing
      - Preserve punctuation, casing, accented letters
    """
    for entry in data["entries"]:

        for key, text in entry.items():
            # Skips over integers since we'll compare them directly
            if isinstance(text, str):

                # If index_numbers == False, remove index numbers
                text = (
                    re.sub(r" *\[ *[0-9]+ *\] *", " ", text)
                    if not index_numbers
                    else text
                )

                # Replace various forms of whitespace with space
                text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")

                # Replace multiple spaces with single space
                text = re.sub(r"\s+", " ", text)

                # Remove instances of "- " for words separated by line break.
                text = re.sub(r"([A-Za-z]+)- ([a-z]+)", r"\1\2", text)

                # Replace spaces in "- " for number ranges and abbreviations separated by line break.
                text = re.sub(r"([0-9A-Z]+)- ([0-9A-Z]+)", r"\1-\2", text)
                entry[key] = text.strip()
    return data


def clean_text_normalized(text, index_numbers=True):
    """
    Fully normalized:
      - Remove linebreaks/tabs
      - Remove all instances of \"- \" (dash space; word separated by line break)
      - Remove extra spaces of number intervals separated by line break
      - Remove all non-ASCII (accented letters are dropped)
      - Convert to lowercase
      - Remove punctuation => keep only [a-z0-9] plus spaces
      - Collapse multiple spaces
      - Strip leading/trailing
    """
    # Remove linebreaks/tabs
    text = clean_text_nonorm(text, index_numbers)

    # Remove all non-ASCII
    text = text.encode("ascii", errors="ignore").decode("ascii")

    # Lowercase
    text = text.lower()

    # Replace periods with a space before removing other punctuation
    text = re.sub(r"\.", " ", text)

    # Keep only [a-z0-9] + space
    text = re.sub(r"[^a-z0-9 ]+", "", text)

    # Collapse multiple spaces again
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_json_normalized(data, index_numbers=True):
    """
    Fully normalized:
      - Remove linebreaks/tabs
      - Remove all instances of \"- \" (dash space; word separated by line break)
      - Remove extra spaces of number intervals separated by line break
      - Remove all non-ASCII (accented letters are dropped)
      - Convert to lowercase
      - Remove punctuation => keep only [a-z0-9] plus spaces
      - Collapse multiple spaces
      - Strip leading/trailing

    Returns:
        - Cleaned JSON object
    """
    for entry in data["entries"]:
        for key, text in entry.items():

            # Skips over integers since we'll compare them directly
            if isinstance(text, str):
                # Remove linebreaks/tabs
                text = clean_text_nonorm(text, index_numbers)

                # Remove all non-ASCII
                text = text.encode("ascii", errors="ignore").decode("ascii")

                # Lowercase
                text = text.lower()

                # Replace periods with a space before removing other punctuation
                text = re.sub(r"\.", " ", text)

                # Keep only [a-z0-9] + space
                text = re.sub(r"[^a-z0-9 ]+", "", text)

                # Collapse multiple spaces again
                text = re.sub(r"\s+", " ", text)
                entry[key] = text.strip()
    return data


def flatten(json_text):
    lines = []
    for entry in json_text["entries"]:
        for val in entry.values():
            if isinstance(val, int):
                val = str(val)
            lines.append(val)
    return " ".join(lines)


def compute_metrics(
    ref_text, hyp_text, doc_format, normalized=False, index_numbers=True
):
    """
    Compute Levenshtein distance, CER, WER.
    If normalized=True => use clean_text_normalized,
    else => use clean_text_nonorm.
    If index_numbers=True => keep index numbers
    else => remove index numbers
    """
    if normalized:
        ref_clean = (
            clean_text_normalized(ref_text, index_numbers)
            if doc_format == "txt"
            else clean_json_normalized(ref_text, index_numbers)
        )
        hyp_clean = (
            clean_text_normalized(hyp_text, index_numbers)
            if doc_format == "txt"
            else clean_json_normalized(hyp_text, index_numbers)
        )
    else:
        ref_clean = (
            clean_text_nonorm(ref_text, index_numbers)
            if doc_format == "txt"
            else clean_json_nonorm(ref_text, index_numbers)
        )
        hyp_clean = (
            clean_text_nonorm(hyp_text, index_numbers)
            if doc_format == "txt"
            else clean_json_nonorm(hyp_text, index_numbers)
        )
    if doc_format == "json":
        ref_clean = flatten(ref_clean)
        hyp_clean = flatten(hyp_clean)
    dist_char = distance.Levenshtein.distance(ref_clean, hyp_clean)
    ref_len = len(ref_clean)

    cer = dist_char / ref_len if ref_len > 0 else 0.0

    # For WER, split by whitespace and full stops
    ref_delimited = re.sub(r"\.", " ", ref_clean)
    hyp_delimited = re.sub(r"\.", " ", hyp_clean)
    processed_words = jiwer.process_words(ref_clean, hyp_clean)
    dist_word = processed_words.insertions + processed_words.deletions + processed_words.substitutions
    wer = processed_words.wer if len(ref_words) > 0 else 0.0

    token_sort_ratio = (
        fuzz.token_sort_ratio(ref_delimited, hyp_delimited) if ref_len > 0 else 0.0
    )

    return {
        "dist_char": dist_char,
        "cer": cer,
        "dist_word": dist_word,
        "wer": wer,
        "token_sort_ratio": token_sort_ratio,
    }


def build_dataframe(title, doc_names, results_data, doc_lengths, total_doc_len):
    """
    Build a Pandas dataframe for a given results_data and doc_lengths structure.
    - results_data[model][doc] => (dist_char, cer, wer)
    - doc_lengths[doc] => length of that doc in the relevant cleaning
    - total_doc_len => sum of all doc lengths in that cleaning

    The dataframe has one row for each document and metric, for example:
    - doc1:dist_char, doc1:doc_len, doc1:cer_pct, doc1:wer_pct, doc2:dist_char, ..., __ALL__:dist_char, ...

    The dataframe has one column for each model used, like pytesseract.

    Returns the dataframe for the results data.
    """

    # One column per document
    metrics = ["dist_char", "doc_len", "cer_pct", "wer_pct", "token_sort_ratio"]
    # df_columns = [f'{doc}:{metric}' for doc in doc_names + ['__ALL__'] for metric in metrics]

    # Create dataframe
    df = pd.DataFrame(columns=results_data.keys())

    # Populate dataframe
    for model in results_data.keys():
        for doc in doc_names:
            cell_data = results_data[model].get(doc, None)
            if cell_data is not None:
                dist_char = cell_data["dist_char"]
                doc_len = doc_lengths.get(doc, 0)
                cer_pct = cell_data["cer"] * 100
                wer_pct = cell_data["wer"] * 100
                token_sort_ratio = cell_data["token_sort_ratio"]

                df.at[f"{doc}:dist_char", model] = dist_char
                df.at[f"{doc}:doc_len", model] = doc_len
                df.at[f"{doc}:cer_pct", model] = cer_pct
                df.at[f"{doc}:wer_pct", model] = wer_pct
                df.at[f"{doc}:token_sort_ratio", model] = token_sort_ratio

        all_data = results_data[model].get("__ALL__", None)
        if all_data is not None:
            dist_char = all_data["dist_char"]
            doc_len = total_doc_len
            #         cer_pct = all_data["cer"] * 100
            #         wer_pct = all_data["wer"] * 100

            #         df.at[model, f"__ALL__:dist_char"] = dist_char
            #         df.at[model, f"__ALL__:doc_len"] = doc_len
            #         df.at[model, f"__ALL__:cer_pct"] = cer_pct
            #         df.at[model, f"__ALL__:wer_pct"] = wer_pct

            # return df
            cer_pct = all_data["cer"] * 100
            wer_pct = all_data["wer"] * 100
            token_sort_ratio = all_data["token_sort_ratio"]

            df.at[f"__ALL__:dist_char", model] = dist_char
            df.at[f"__ALL__:doc_len", model] = doc_len
            df.at[f"__ALL__:cer_pct", model] = cer_pct
            df.at[f"__ALL__:wer_pct", model] = wer_pct
            df.at[f"__ALL__:token_sort_ratio", model] = token_sort_ratio

    return df


def get_doc_names(dir):
    """
    Return a list of txt document names from `dir` without the .txt extensions.
    """

    gt_paths = glob.glob(os.path.join(dir, "*.txt"))
    logger.info("Found ground-truth txt files: %s", gt_paths)

    doc_names = [os.path.splitext(os.path.basename(p))[0] for p in gt_paths]
    logger.info("Found file names: %s", doc_names)
    return doc_names


def get_all_models(llm_root, ocr_root, ocr_llm_root):
    """
    llm_root is the directory where LLM model folders are located.
    ocr_root is the directory where OCR model folders are located.
    ocr_llm_root is the directory where OCR-LLM model folders are located.

    Example file structure:
    - llm_root
        - gpt-4o
            - doc1.txt
            - doc2.txt
        - gemini-2.0
            - doc1.txt
            - doc2.txt
    - ocr_root
        - pytesseract
            - doc1.txt
            - doc2.txt

    Returns a list of 2-tuples with
    - the model type:
        - "llm_img2txt" for LLM models
        - "ocr_img2txt" for OCR models
    - the model name (found using the directory structure)
    """

    llm_models = []
    if os.path.isdir(llm_root):
        llm_models = [
            m for m in os.listdir(llm_root) if os.path.isdir(os.path.join(llm_root, m))
        ]
    ocr_models = []
    if os.path.isdir(ocr_root):
        ocr_models = [
            m for m in os.listdir(ocr_root) if os.path.isdir(os.path.join(ocr_root, m))
        ]

    ocr_llm_models = []
    if os.path.isdir(ocr_llm_root):
        ocr_llm_models = [
            m
            for m in os.listdir(ocr_llm_root)
            if os.path.isdir(os.path.join(ocr_llm_root, m))
        ]

    all_models = (
        [("llm-img2txt", m) for m in llm_models]
        + [("ocr-img2txt", m) for m in ocr_models]
        + [("ocr-llm-img2txt", m) for m in ocr_llm_models]
    )
    # sort by model name
    all_models.sort(key=lambda x: x[1].lower())

    return all_models


def get_docs(dir, doc_names):
    """
    Returns a 2-tuple containing
    - a dict with
        - `doc_names` as the keys
        - the contents of `dir/{key}.txt` for each key as the values.
    - a string containing the content of all the values in the dict
        - in the order given by doc_names
        - each value is separated by a newline

    Since doc_names is a list, all_docs preserves order between different directories.
    """

    docs = {}
    all_docs = ""

    for doc in doc_names:
        path = os.path.join(dir, f"{doc}.txt")
        with open(path, "r", encoding="utf-8") as f:
            txt = f.read()
        docs[doc] = txt
        all_docs += txt + "\n"

    return docs, all_docs


def main():
    """
    Prerequisites:
    - Ground truth text files located at `project_root/ground-truth/txt/gt_kbaa-pXYZ.txt`
    - LLM/OCR transcribed files located at:
        - for LLM transcriptions: `project_root/results/llm_img2txt/<MODEL-NAME>/<MODEL-NAME>_img_kbaa-pXYZ.txt`
        - for OCR transcriptions: `project_root/results/ocr_img2txt/<MODEL-NAME>/<MODEL-NAME>_img_kbaa-pXYZ.txt`

    The main function will:
    - Gather all ground truth text files
    - For each ground truth text file and for each LLM/OCR model, gather the corresponding transcription
    - Clean all the text files (normalized and not normalized)
    - Compute metrics for each file and model
    - Save results in two CSV files (one for normalized, one for non-normalized)
        - Results are saved in `project_root/benchmarking-results/txt-accuracy`
    """

    # =============
    # Preliminaries
    # =============

    # args = parse_arguments()

    logger.info("Script directory: %s", script_dir)
    logger.info("Project root: %s", project_root)

    # Ground truth
    ground_truth_dir = os.path.join(project_root, "data", "ground-truth", "txt")
    doc_names = get_doc_names(ground_truth_dir, keep_prefix=False)

    # results/ paths
    all_models = get_all_models(
        os.path.join(project_root, "results", "llm-img2txt"),
        os.path.join(project_root, "results", "ocr-img2txt"),
        os.path.join(project_root, "results", "ocr-llm-img2txt"),
    )
    logger.info(f"Models found: {all_models}")

    # ===========
    # Gather files
    # ===========

    # -> Gather ground truths and put into dict:

    ground_truths, ground_truths["__ALL__"] = get_docs(
        ground_truth_dir, doc_names, name_has_prefix=True
    )
    doc_lengths_normalized = {
        doc: len(clean_text_normalized(text)) for doc, text in ground_truths.items()
    }
    doc_lengths_nonorm = {
        doc: len(clean_text_nonorm(text)) for doc, text in ground_truths.items()
    }
    total_doc_len_normalized = len(clean_text_normalized(ground_truths["__ALL__"]))
    total_doc_len_nonorm = len(clean_text_nonorm(ground_truths["__ALL__"]))

    # -> Gather each transcribed document and put into dict:

    # Structure: results[model][doc]
    results = {}

    for model_type, model in all_models:
        logger.info("Collecting results for model: %s", model)
        model_path = os.path.join(project_root, "results", model_type, model)
        results[model], results[model]["__ALL__"] = get_docs(
            model_path, doc_names, name_has_prefix=True
        )
        logger.info("Collected results for model: %s", list(results[model].keys()))

    # ===============
    # Compute metrics
    # ===============

    normalized_results_data = {}
    nonorm_results_data = {}

    for _, model in all_models:
        normalized_results_data[model] = {}
        nonorm_results_data[model] = {}

        logger.info("Computing metrics for model: %s", model)
        for doc in doc_names:
            logger.info("Computing metrics for document: %s", doc)
            normalized_results_data[model][doc] = compute_metrics(
                ground_truths[doc], results[model][doc], normalized=True
            )
            nonorm_results_data[model][doc] = compute_metrics(
                ground_truths[doc], results[model][doc], normalized=False
            )

        normalized_results_data[model]["__ALL__"] = compute_metrics(
            ground_truths["__ALL__"], results[model]["__ALL__"], normalized=True
        )
        nonorm_results_data[model]["__ALL__"] = compute_metrics(
            ground_truths["__ALL__"], results[model]["__ALL__"], normalized=False
        )

    # Compute metrics separately for __ALL__]

    # ====================
    # Put metrics in table
    # ====================

    time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

    normalized_df = build_dataframe(
        f"normalized_{time}",
        doc_names,
        normalized_results_data,
        doc_lengths_normalized,
        total_doc_len_normalized,
    )
    nonorm_df = build_dataframe(
        f"nonorm_{time}",
        doc_names,
        nonorm_results_data,
        doc_lengths_nonorm,
        total_doc_len_nonorm,
    )

    # ============
    # Save results
    # ============

    # Default save to project_root/benchmarking-results/txt-accuracy
    results_path = os.path.join(project_root, "benchmarking-results", "txt-accuracy")
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    normalized_df.to_csv(os.path.join(results_path, f"normalized_{time}.csv"))
    nonorm_df.to_csv(os.path.join(results_path, f"nonorm_{time}.csv"))


if __name__ == "__main__":
    main()
