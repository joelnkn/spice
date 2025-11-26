# SPICE — Synthetic Polyglot Injection for Cross-lingual Evaluation

A project for improving multilingual model capability through fine-tuning on synthetic language data.

Colab: https://colab.research.google.com/drive/1ig1AQZ26Q-1xXjZMbJi1cLRfKrghR2Lf#scrollTo=NsYEJe4Umz_h

## Project Structure

```
spice/
├── scripts/             # Utility scripts
│   ├── setup.sh         # Environment setup
│   └── generate.sh      # Language generation
├── synthetic/           # Synthetic language generation module
│   ├── config.py        # Configuration management
│   ├── conglanger.py    # Conglanger import wrapper
│   ├── data/            # Data directories (raw, processed)
│   ├── generation/      # Language generation
│   │   └── api.py       # High-level API
│   ├── typology/        # Typological analysis
│   │   ├── extraction.py  # Extract WALS features
│   │   ├── features.py    # Parse/vectorize features
│   │   └── similarity.py  # Language distance metrics
│   └── utils/           # Utility functions
├── third_party/         # Third-party code/submodules
│   └── conglanger/      # Conglanger submodule
├── requirements.txt     # Python dependencies
├── setup.py            # Package installation
└── README.md           # This file
```

## Setup

1. **Initialize environment:**
   ```bash
   bash scripts/setup.sh
   ```

2. **Install package (optional, for development):**
   ```bash
   pip install -e .
   ```

3. **Configure API keys:**
   Edit `.env` and add your API keys:
   ```bash
   GOOGLE_API_KEY=your_key_here
   OPENAI_API_KEY=your_key_here
   TOGETHER_API_KEY=your_key_here
   ```

## Usage

### Generate a Consistent Language

```python
from synthetic.generation.api import generate_consistent_language, translate
from synthetic.conglanger import create_llm_client
from synthetic.typology.extraction import extract_features

# Generate a language trained on a corpus
corpus = ["Hello world!", "The quick brown fox"]
language_id = generate_consistent_language(corpus)

# Translate new sentences
translate("How are you today?", language_id=language_id)

# Extract typological features
llm_client = create_llm_client(model="gemini-2.5-pro")
extract_features(llm_client, "consistent", language_id)
```

### Run from Command Line

```bash
python -m synthetic.generation.api
```

## Module Overview

- **`synthetic.generation`** - Language generation pipeline
  - `api.py` - High-level functions for generating and using languages
  
- **`synthetic.typology`** - Typological analysis
  - `extraction.py` - Extract WALS-style features using LLM
  - `features.py` - Parse and vectorize extracted features
  - `similarity.py` - Compute typological distances
  
- **`synthetic.conglanger`** - Wrapper for Conglanger functionality
  - Provides clean imports of Conglanger functions
  - Includes `run_conglanger()` for CLI execution

## Citation

If you use or build on this work, please cite:

```bibtex
@article{conlangcrafter2025,
  title={ConlangCrafter: Constructing Languages with a Multi-Hop LLM Pipeline},
  author={Morris Alper and Moran Yanuka and Raja Giryes and Ga{\v{s}}per Begu{\v{s}}},
  year={2025},
  eprint={2508.06094},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2508.06094}
}
```
