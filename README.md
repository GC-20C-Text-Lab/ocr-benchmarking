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
│   │   └── <MODEL>/      # One folder per model
│   ├── llm-txt2csv/      # CSV files from transcribed text using multimodal LLMs
│   │   └── <MODEL>/
│   ├── llm-img2txt/      # Text transcribed from images using multimodal LLMs
│   │   └── <MODEL>/
│   ├── ocr-img2txt/      # Text transcribed from images using OCR software
│   │   └── <MODEL>/
├── src/                  # Source code
│   ├── benchmarking/     # Benchmarking tools
│   ├── llm-img2csv/      # Image to CSV converters using multimodal LLMs
│   ├── llm-txt2csv/      # Text to CSV converters
│   ├── llm-img2txt/      # Image to text converters using multimodal LLMs
│   ├── ocr-img2txt/      # OCR processors
│   └── scripts/          # Utility scripts
└── logs/                 # Log files
```

# 🔧 Usage

Before usage, ensure that the `data` and `results` directories are populated with the correct files.

```bash
# Perform text accuracy analysis using ground truth and transcribed text files
python src/benchmarking/txt_accuracy.py
```

Note: use Anaconda and install all necessary packages before running commands.

## Input Data Format

The pipeline expects:
- TIFF files in `/tiffs`. Named `kbaa-p#ABC.tif`, where `ABC` is the three-digit page number (with leading zeros if necessary)
- Ground truth text files in `data/ground-truth/txt`. Named `kbaa-p#ABC.txt`
  - See formatting guidelines for ground truth text files in the [Ground Truth Guidelines](./ground-truth-guidelines.md)

# 📊 Benchmarking

# 📋 Credits

Greif et al., [Multimodal LLMs for OCR, OCR-Post-Correction, and Named Entity Recognition in Historical Documents](https://github.com/niclasgriesshaber/llm_historical_dataset_benchmarking)