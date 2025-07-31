# ocr-benchmarking

Testing methods for digitizing print bibliography as structured data

# 🚀 Installation

See `src/workflow/pipeline.ipynb` for prerequisites.

# 🔧 Usage

Before usage, ensure that the `data` directory is populated with the correct files.

Then, run each cell in `src/workflow/pipeline.ipynb` following its instructions to generate and benchmark LLM outputs.

To obtain visualizations, move benchmark results from `benchmarking-results` to `benchmarking-results-for-visualizations` and then run `src/workflow/visualizations.ipynb`.

# Directory overview

- `benchmarking-results`: Benchmarking results CSV files generated from the pipeline.
- `benchmarking-results-for-visualizations`: Visualizations from our benchmark results.
- `config`: Setup files.
- `data`: Ground truth text and JSON files, as well as images.
- `project-notes`: Descriptions of workflow, scratchwork, etc.
- `results`: LLM and OCR output. Created automatically by the pipeline.
- `src`: Source code

# File naming scheme

- `XYZ` refers to a three-digit page number (with padded zeroes as necessary).
- `{A,B}` refers to either A or B.
- `<NAME>` means to replace with the correct name.

The base file naming scheme is `kbaa-pXYZ`, followed by the file extension.
This is preceded by prefixes as described below:

### data

```
data/ground-truth/{txt,json}/gt_kbaa-pXYZ.{txt,json}
data/{pngs,tiffs}/kbaa-pXYZ.{png,tif}
```

### results

```
results/{llm,ocr}{img,txt}2{txt,json}/
  <MODEL_NAME>/<MODEL_NAME>_{txt,img}_kbaa-pXYZ.{txt,json}
                            ^ input format      ^ output format
```

# Other information

Other information about our project can be found in the `project-notes` directory.

# 📋 Credits

Greif et al., [Multimodal LLMs for OCR, OCR-Post-Correction, and Named Entity Recognition in Historical Documents](https://github.com/niclasgriesshaber/llm_historical_dataset_benchmarking)