# ocr-benchmarking

Testing methods for digitizing print bibliography as structured data

# 🚀 Installation

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
├── config/                # Configuration files
│   ├── environment.yml   # Conda environment specification
│   └── .env              # API keys and credentials
├── data/
│   ├── tiffs/            # Input PDFs (type-1.pdf to type-10.pdf)
│   ├── ground-truth/
│   │   └── txt/          # Ground truth text files
│   └── pngs/             # Intermediate image files as single PNGs
├── results/              # Output directory for all models
├── src/                  # Source code
│   ├── benchmarking/     # Benchmarking tools
│   ├── llm-img2csv/      # Image to CSV converters using multimodal LLMs
│   ├── llm-txt2csv/      # Text to CSV converters
│   ├── ocr-img2txt/      # OCR processors
│   └── scripts/          # Utility scripts
└── logs/                 # Log files
```

# 🔧 Usage

## Input Data Format

The pipeline expects:
- TIFF files in `/tiffs`. Named `kbaa-p#ABC.tif`, where `ABC` is the three-digit page number (with leading zeros if necessary)
- Ground truth text files in `data/ground-truth/txt`. Named `kbaa-p#ABC.txt`
  - See formatting guidelines for ground truth text files in the [Ground Truth Guidelines](./ground-truth-guidelines.md)

# 📊 Benchmarking

# 📋 Credits

Greif et al., [Multimodal LLMs for OCR, OCR-Post-Correction, and Named Entity Recognition in Historical Documents](https://github.com/niclasgriesshaber/llm_historical_dataset_benchmarking)