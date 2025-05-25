# Kreuzberg

[![PyPI version](https://badge.fury.io/py/kreuzberg.svg)](https://badge.fury.io/py/kreuzberg)
[![Documentation](https://img.shields.io/badge/docs-GitHub_Pages-blue)](https://goldziher.github.io/kreuzberg/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Kreuzberg is a Python library for text extraction from documents. It provides a unified interface for extracting text from PDFs, images, office documents, and more, with both async and sync APIs.

## Why Kreuzberg?

- **Simple and Hassle-Free**: Clean API that just works, without complex configuration
- **Local Processing**: No external API calls or cloud dependencies required
- **Resource Efficient**: Lightweight processing without GPU requirements
- **Format Support**: Comprehensive support for documents, images, and text formats
- **Multiple OCR Engines**: Support for Tesseract, EasyOCR, and PaddleOCR
- **Metadata Extraction**: Get document metadata alongside text content
- **Table Extraction**: Extract tables from documents using the excellent GMFT library
- **Modern Python**: Built with async/await, type hints, and a functional-first approach
- **Permissive OSS**: MIT licensed with permissively licensed dependencies

## Quick Start

```bash
pip install kreuzberg
```

Install pandoc:

```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr pandoc

# macOS
brew install tesseract pandoc

# Windows
choco install -y tesseract pandoc
```

The tesseract OCR engine is the default OCR engine. You can decide not to use it - and then either use one of the two alternative OCR engines, or have no OCR at all.

### Alternative OCR engines

```bash
# Install with EasyOCR support
pip install "kreuzberg[easyocr]"

# Install with PaddleOCR support
pip install "kreuzberg[paddleocr]"
```

## Quick Example

```python
import asyncio
from kreuzberg import extract_file

async def main():
    # Extract text from a PDF
    result = await extract_file("document.pdf")
    print(result.content)

    # Extract text from an image
    result = await extract_file("scan.jpg")
    print(result.content)

    # Extract text from a Word document
    result = await extract_file("report.docx")
    print(result.content)

asyncio.run(main())
```

## Documentation

For comprehensive documentation, visit our [GitHub Pages](https://goldziher.github.io/kreuzberg/):

- [Getting Started](https://goldziher.github.io/kreuzberg/getting-started/) - Installation and basic usage
- [User Guide](https://goldziher.github.io/kreuzberg/user-guide/) - In-depth usage information
- [API Reference](https://goldziher.github.io/kreuzberg/api-reference/) - Detailed API documentation
- [Examples](https://goldziher.github.io/kreuzberg/examples/) - Code examples for common use cases
- [OCR Configuration](https://goldziher.github.io/kreuzberg/user-guide/ocr-configuration/) - Configure OCR engines
- [OCR Backends](https://goldziher.github.io/kreuzberg/user-guide/ocr-backends/) - Choose the right OCR engine

## Supported Formats

Kreuzberg supports a wide range of document formats:

- **Documents**: PDF, DOCX, RTF, TXT, EPUB, etc.
- **Images**: JPG, PNG, TIFF, BMP, GIF, etc.
- **Spreadsheets**: XLSX, XLS, CSV, etc.
- **Presentations**: PPTX, PPT, etc.
- **Web Content**: HTML, XML, etc.

## OCR Engines

Kreuzberg supports multiple OCR engines:

- **Tesseract** (Default): Lightweight, fast startup, requires system installation
- **EasyOCR**: Good for many languages, pure Python, but downloads models on first use
- **PaddleOCR**: Excellent for Asian languages, pure Python, but downloads models on first use

For comparison and selection guidance, see the [OCR Backends](https://goldziher.github.io/kreuzberg/user-guide/ocr-backends/) documentation.

Sure! Here's your cleaned-up version with all headers reduced to `#` and formatting kept concise for a smooth README experience:


# Contribution

This library is open to contribution. Feel free to open issues or submit PRs. It's better to discuss issues before submitting PRs to avoid disappointment.

If you're familiar with what to do, that’s great. But if you're lost in the myriad of tools and libraries, here’s a quick head-start to help you get familiar with the repo


### 🔧 Basics

`attrs`, `typing-extensions`, `decorator` – Simplify class creation and type handling.
`setuptools`, `virtualenv`, `filelock`, `distlib` – Manage packaging and environments.
`pathspec`, `pyyaml`, `pyyaml-env-tag`, `jsonpointer`, `jsonpatch`, `orjson`, `ujson` – Handle config and data formats.
`zipp`, `regex`, `six` – Utilities for zip, regex, and cross-version support.

### 🕸️ Web Requests & Networking

`aiohttp`, `requests`, `aiohappyeyeballs`, `async-timeout`, `idna`, `charset-normalizer`, `certifi`, `urllib3`, `requests-toolbelt`, `yarl` – Handle HTTP, URLs, and async requests.

### 🖼️ Image Processing & Vision

`opencv-python-headless`, `Pillow` – Core image handling.
`scikit-image`, `numpy`, `scipy` – Image math and analysis.
`albumentations`, `albucore` – Data augmentation for ML.
`easyocr`, `pyclipper`, `tifffile` – OCR, shape ops, and TIFF support.

### 📄 Document & Office Files

`python-docx`, `python-pptx` – Work with Word and PowerPoint.
`openpyxl`, `xlsxwriter`, `et-xmlfile` – Handle Excel files.

### 📊 Data Handling

`pandas`, `tabulate`, `prettytable` – Tables, dataframes, pretty outputs.
`pyparsing`, `jsonpointer`, `more-itertools`, `sortedcontainers` – Parsing, data structures, iteration helpers.

### 🤖 Machine Learning

`torch`, `torchvision`, `triton`, `einops` – Deep learning tools.
`huggingface-hub`, `safetensors`, `tiktoken` – LLMs and tokenization.
`pydantic`, `pydantic-core`, `dataclasses-json` – Model schemas and validation.
`opt-einsum`, `stringzilla`, `threadpoolctl` – Speed up math and strings.

### 🔍 Text & Semantic Analysis

`semantic-text-splitter`, `rapidfuzz` – Text splitting and fuzzy matching.
`markdown`, `markupsafe`, `pymdown-extensions`, `babel` – Markdown and i18n tools.

### 🎨 Visualization & Rendering

`matplotlib`, `cycler`, `fonttools`, `colorama`, `termcolor` – Charts, fonts, and color outputs.
`cairocffi`, `cairosvg` – SVG to image conversion.
`csscompressor`, `cssselect`, `cssselect2`, `cssutils`, `tinycss2` – CSS handling.
`webencodings`, `wcwidth` – Web-style text display.

### 📖 Documentation

`mkdocs`, `mkdocs-material`, `mkdocstrings`, `mkdocs-autorefs`, `mkdocs-minify-plugin`, `mkdocs-git-revision-date-localized-plugin`, `ghp-import` – Build beautiful Markdown docs.
`griffe` – Extract docstrings from code for docs.

### 🧠 Advanced & Optional

`nvidia-*` – GPU support for CUDA training.
`cython`, `jiter` – Speed up Python with compiled code.
`lmdb`, `sqlalchemy` – Fast storage and DB interaction.
`defusedxml`, `astor`, `mergedeep`, `propcache`, `typing-inspect` – Advanced parsing and safe tools.




### Local Development

- Clone the repo
- Install the system dependencies
- Install the full dependencies with `uv sync`
- Install the pre-commit hooks with: `pre-commit install && pre-commit install --hook-type commit-msg`
- Make your changes and submit a PR

## License

This library is released under the MIT license.
