# Synthetic Language Generation

This module handles the generation and analysis of synthetic languages for multilingual model evaluation.

## Directory Structure

```
synthetic/
├── __init__.py           # Module initialization
├── config.py             # Configuration management
├── conglanger.py         # Conglanger import wrapper
├── data/                 # Data directories
│   ├── raw/             # Raw input data
│   └── processed/       # Processed/generated data
├── generation/          # Language generation
│   └── api.py          # High-level API
├── typology/            # Typological analysis
│   ├── extraction.py   # Extract WALS features using LLM
│   ├── features.py     # Parse and vectorize features
│   └── similarity.py   # Language distance metrics
├── utils/              # Utility functions
│   ├── __init__.py
│   └── logger.py      # Logging utilities
└── README.md          # This file
```

## Quick Start

### Generate a Consistent Language

```python
from synthetic.generation.api import generate_consistent_language, translate

# Generate a language stabilized on a corpus
corpus = ["Hello world!", "The quick brown fox jumps over the lazy dog."]
language_id = generate_consistent_language(corpus, run_name="my_language")

# Translate new sentences using the stabilized language
translate("How are you today?", language_id=language_id, run_name="my_language")
translate("The sun is shining.", language_id=language_id, run_name="my_language")
```

### Extract Typological Features

```python
from synthetic.conglanger import create_llm_client
from synthetic.typology.extraction import extract_features

# Create LLM client
llm_client = create_llm_client(model="gemini-2.5-pro")

# Extract WALS-style typological features
extract_features(llm_client, run_name="my_language", language_id=language_id)
```

### Analyze Language Similarity

```python
from synthetic.typology.similarity import compute_language_distance
from synthetic.typology.features import load_feature_vector

# Load feature vectors for two languages
features_1 = load_feature_vector("my_language", "lang_id_1")
features_2 = load_feature_vector("my_language", "lang_id_2")

# Compute typological distance
distance = compute_language_distance(features_1, features_2)
```

## Output Structure

Generated languages are saved to `synthetic/outputs/<run_name>/languages/<language_id>/`:

```
outputs/
└── <run_name>/
    └── languages/
        └── <language_id>/
            ├── memory/              # Current language state
            │   ├── phonology.txt
            │   ├── grammar.txt
            │   ├── lexicon.csv
            │   └── translation.json
            ├── logs/                # Execution logs
            ├── analysis/            # Typological features
            │   └── features.txt
            ├── iter_0/              # Initial snapshot
            ├── iter_1/              # After 1st adaptation
            └── ...
```

## Running from Command Line

```bash
# Run example workflow
python -m synthetic.generation.api

# Or directly
python synthetic/generation/api.py
```

## Configuration

Edit `synthetic/config.py` or set environment variables:

- `OUTPUT_DIR` - Base output directory (default: `synthetic/outputs`)
- `PROMPT_DIR` - Conglanger prompts directory
- `GOOGLE_API_KEY` - For Gemini models
- `OPENAI_API_KEY` - For OpenAI models
- `TOGETHER_API_KEY` - For DeepSeek models

## Module Documentation

### `synthetic.generation.api`

High-level API for language generation:

- `generate_consistent_language(corpus, language_id=None, output_dir=OUTPUT_DIR, run_name="consistent")` 
  - Generates a language stabilized on a corpus
  - Returns the language_id
  
- `translate(sentence, language_id, output_dir=OUTPUT_DIR, run_name="consistent")`
  - Translates a sentence using an existing language
  - Appends new vocabulary only (no grammar changes)

### `synthetic.typology.extraction`

Extract typological features:

- `extract_features(llm_client, run_name, language_id, prompt_dir=None, output_dir=None)`
  - Extracts WALS-style features using LLM analysis
  - Saves to `<run_name>/languages/<language_id>/analysis/features.txt`

### `synthetic.conglanger`

Clean wrapper for Conglanger functionality:

- `create_llm_client(model, max_tokens, temperature, ...)`
  - Factory for creating LLM clients (Gemini, OpenAI, DeepSeek)
  
- `run_conglanger(...)`
  - Executes Conglanger pipeline as subprocess
  - Handles all CLI argument construction

- Direct imports: `PromptManager`, `load_required_files`, etc.


