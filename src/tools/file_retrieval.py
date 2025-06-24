"""
Functions used for file retrieval.
Script to get files for running accuracy tests.
"""

import os
import glob
import logging

# ----------------- Configure Logging -----------------
logging.basicConfig(
    level=logging.INFO,
    format="[file retrieval] %(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def get_doc_names(dir, keep_prefix=True, prefix_delimiter='_'):
    """
    Return a list of txt document names from `dir` without the .txt extensions.

    If `keep_prefix` is True, return original document names.
    Otherwise, strip prefix names (everything before the final prefix delimiter).
    - For example: gt_kbaa-123.txt -> kbaa-123.txt (prefix_delimiter='_')
    """

    gt_paths = glob.glob(os.path.join(dir, "*.txt"))
    logger.info("Found ground-truth txt files: %s", gt_paths)

    doc_names = [os.path.splitext(os.path.basename(p))[0] for p in gt_paths]
    logger.info("Found file names: %s", doc_names)

    # Strip prefix names if requested
    if not keep_prefix:
        doc_names = list(map(lambda name: str.split(name, prefix_delimiter)[-1], doc_names))

    return doc_names


def get_all_models(llm_root, ocr_root):
    """
    llm_root is the directory where LLM model folders are located.
    ocr_root is the directory where OCR model folders are located.

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
        llm_models = [m for m in os.listdir(llm_root) if os.path.isdir(os.path.join(llm_root, m))]
    ocr_models = []
    if os.path.isdir(ocr_root):
        ocr_models = [m for m in os.listdir(ocr_root) if os.path.isdir(os.path.join(ocr_root, m))]

    all_models = [("llm_img2txt", m) for m in llm_models] + [("ocr_img2txt", m) for m in ocr_models]
    # sort by model name
    all_models.sort(key=lambda x: x[1].lower())

    return all_models


def get_docs(dir, doc_names, name_has_prefix=False):
    """
    Returns a 2-tuple containing
    - a dict with
        - `doc_names` as the keys
        - The contents of `dir/{doc}` as the corresponding values
          (assuming `doc` is a key of `doc_name`).
            - If `name_has_prefix` is True, `doc` may be preceded by any prefix.
    - a string containing the content of all the values in the dict
        - in the order given by doc_names
        - each value is separated by a newline

    Since doc_names is a list, all_docs preserves order between different directories.
    """

    docs = {}
    all_docs = ''

    for doc in doc_names:
        doc_pattern = f'*{doc}.txt' if name_has_prefix else f'{doc}.txt'
        paths = glob.glob(os.path.join(dir, doc_pattern))
        path = paths[0] # find first path; assume that one page exists once in each folder

        with open(path, "r", encoding="utf-8") as f:
            txt = f.read()
        docs[doc] = txt
        all_docs += txt + '\n'

    return docs, all_docs