"""
Benchmarking OCR vs. LLM for text extraction. Uses RapidFuzz (for CER, TSR) and JiWER (for WER).

Each type of results has 4 rows for each document and for all (__ALL__) documents:
   1) Levenshtein distance ({doc}:dist_char)
   2) ground-truth doc length (for that table's version) ({doc}:doc_len)
   3) CER% (distance / length_of_that_version) ({doc}:cer_pct)
   4) WER% ({doc}:wer_pct)
   5) Token Sort Ratio ({doc}:token_sort_ratio)

Output structure:
- ../../benchmarking-results/txt-accuracy/
    - llm-img2txt       *(Image transcription to text using LLMs)*
    - ocr-img2txt       *(Image transcription to text using conventional OCR)*
    - ocr-llm-img2txt   *(Image + OCR post-correction using LLMs)*
        - nonorm_%Y-%m-%d_%H:%M:%S.csv      *See `clean_text_nonorm`*
        - normalized_%Y-%m-%d_%H:%M:%S.csv  *See `clean_text_normalized`*

Original authors: Niclas Griesshaber, Gavin Greif, Robin Greif
New authors: Tim Yu, Muhammad Khalid
"""

# ----------------- Imports -----------------
import os
import re
import logging
from rapidfuzz import distance, fuzz
import pandas as pd
import jiwer

script_dir = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))

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


def clean_text_normalized(text, index_numbers=True):
    """
    Fully normalized:
      - All minimal cleaning steps (see `clean_text_nonorm`), and then:
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


def compute_metrics(
    ref_text, hyp_text, doc_format="txt", normalized=False, index_numbers=True
):
    """
    Compute Levenshtein distance, CER, WER.
    If normalized=True => use clean_text_normalized,
    else => use clean_text_nonorm.
    If index_numbers=True => keep index numbers
    else => remove index numbers

    doc_format is deprecated; only "txt" should be allowed.

    Returns a dictionary with metrics for a single page.
    """
    if normalized:
        ref_clean = (
            clean_text_normalized(ref_text, index_numbers)
        )
        hyp_clean = (
            clean_text_normalized(hyp_text, index_numbers)
        )
    else:
        ref_clean = (
            clean_text_nonorm(ref_text, index_numbers)
        )
        hyp_clean = (
            clean_text_nonorm(hyp_text, index_numbers)
        )

    dist_char = distance.Levenshtein.distance(ref_clean, hyp_clean)
    ref_len = len(ref_clean)

    cer = dist_char / ref_len if ref_len > 0 else 0.0

    # For WER, split by whitespace and full stops
    ref_delimited = re.sub(r"\.", " ", ref_clean)
    hyp_delimited = re.sub(r"\.", " ", hyp_clean)
    processed_words = jiwer.process_words(ref_clean, hyp_clean)
    dist_word = processed_words.insertions + processed_words.deletions + processed_words.substitutions
    wer = processed_words.wer if len(ref_clean) > 0 else 0.0

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
    See top of this file for more details about metrics

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