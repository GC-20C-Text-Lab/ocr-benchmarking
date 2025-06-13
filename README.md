# ocr-benchmarking

Testing methods for digitizing print bibliography as structured data

# 🚀 Installation

## Prerequisites

- Anaconda or Miniconda

## Install packages:

Use Anaconda and install all necessary packages before running commands.

```bash
# Create and activate Conda environment
conda env create --file=config/environment.yaml
conda activate ocr-benchmarking

# To install additional packages, add them to config/environment.yaml.
# Then, enter:
conda env update --file config/environment.yaml
```

## Set up API keys

Set up your API keys in `config/.env`:

```
OPENAI_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here
TRANSKRIBUS_USERNAME=your_username
TRANSKRIBUS_PASSWORD=your_password
```

# 📁 Directory Structure

The directory structure is adapted from [Greif et al.](#-credits)

```
.
├── benchmarking-results/ # Benchmarking results
│   └── txt-accuracy/     # Image-to-text accuracy
├── config/                # Configuration files
│   ├── environment.yml   # Conda environment specification
│   └── .env              # API keys and credentials
├── data/
│   ├── tiffs/            # Input PDFs (type-1.pdf to type-10.pdf)
│   ├── ground-truth/     # Ground truth files
│   │   └── txt/          
│   └── pngs/             # Intermediate image files as single PNGs
├── results/              # Output directory for all models
│   ├── llm-img2csv/      # CSV files from images using multimodal LLMs
│   │   └── <MODEL_NAME>/ # One folder per model
│   ├── llm-txt2csv/      # CSV files from transcribed text using multimodal LLMs
│   │   └── <MODEL_NAME>/
│   ├── llm-img2txt/      # Text transcribed from images using multimodal LLMs
│   │   └── <MODEL_NAME>/
│   ├── ocr-img2txt/      # Text transcribed from images using OCR software
│   │   └── <MODEL_NAME>/
├── src/                  # Source code
│   ├── benchmarking/     # Benchmarking tools
│   ├── llm-img2csv/      # Image to CSV converters using multimodal LLMs
│   ├── llm-txt2csv/      # Text to CSV converters
│   ├── llm-img2txt/      # Image to text converters using multimodal LLMs
│   ├── ocr-img2txt/      # OCR processors
│   └── scripts/          # Utility scripts
└── logs/                 # Log files
```

## File naming scheme

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

# 🔧 Usage

Before usage, ensure that the `data` and `results` directories are populated with the correct files.

```bash
# Perform text accuracy analysis using ground truth and transcribed text files
python src/benchmarking/txt_accuracy.py
```

## Input Data Format

The pipeline expects:
- TIFF files in `/tiffs`. Named `kbaa-p#ABC.tif`, where `ABC` is the three-digit page number (with leading zeros if necessary)
- Ground truth text files in `data/ground-truth/txt`. Named `kbaa-p#ABC.txt`
  - See formatting guidelines for ground truth text files in the [Ground Truth Guidelines](./ground-truth-guidelines.md)

# 📊 Benchmarking

# 📋 Credits

Greif et al., [Multimodal LLMs for OCR, OCR-Post-Correction, and Named Entity Recognition in Historical Documents](https://github.com/niclasgriesshaber/llm_historical_dataset_benchmarking)