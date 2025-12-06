# Synthetic Language Generation

This module handles the generation and analysis of synthetic languages for multilingual model evaluation.

## Directory Structure

```
synthetic/
├── __init__.py           # Module initialization
├── config.py             # Configuration management
├── generation/           # Language generation
│   ├── __init__.py
│   └── api.py           # High-level API
├── utils/               # Utility functions
│   ├── __init__.py
│   └── logger.py        # Logging utilities
├── data/                # Data directories
│   ├── __init__.py
│   ├── raw/             # Raw input data
│   └── processed/       # Processed/generated data
└── README.md            # This file
```

## Output Structure

Generated languages are saved to `synthetic/outputs/<run_name>/languages/<language_id>/`:

```
outputs/
└── <run_name>/
    └── languages/
        └── <language_id>/
            ├── metadata.json         # Tracks creation, continuations, QA metrics
            └── memory/               # Language state (persistent across iterations)
                ├── lexicon/
                │   └── lexicon.csv   # Working lexicon, updated with new words
                ├── valid_translations.json  # Accumulated valid translations with iteration tracking
                ├── iter_0/
                │   ├── translation/
                │   │   ├── translation.json
                │   │   └── translation_qa.json
                │   ├── new_words.json
                │   └── required_lexicon.csv
                ├── iter_1/
                │   ├── translation/
                │   │   ├── translation.json
                │   │   └── translation_qa.json
                │   ├── new_words.json
                │   └── required_lexicon.csv
                └── iter_N/
                    ├── translation/
                    │   ├── translation.json
                    │   └── translation_qa.json
                    ├── new_words.json
                    └── required_lexicon.csv
```

**Key structure:**
- `metadata.json` - Top-level metadata tracking creation, continuations, and QA metrics
- `memory/` - Persistent language state directory
- `memory/lexicon/lexicon.csv` - Working lexicon, updated with new words across iterations
- `memory/valid_translations.json` - Accumulated valid translations with iteration tracking
- `memory/iter_N/` - Each iteration stored within memory
- `memory/iter_N/translation/` - Translation outputs and QA results for iteration N
- `memory/iter_N/new_words.json` - New words extracted in iteration N
- `memory/iter_N/required_lexicon.csv` - Subset of lexicon used for iteration N

## Running from Command Line

```bash
# Run example workflow
python -m synthetic.generation.api
```

## Configuration

Set environment variables:
- `GOOGLE_API_KEY` - For Gemini models

### `synthetic.generation.api`

High-level API for language translation and dataset processing:

**Directory Management:**
- `get_target_directory(target_lang)` - Returns the output directory path where all translation attempts for a target language are stored
  - Languages are organized as `outputs/<target_lang>/languages/<lang_id>/` where `lang_id` follows the pattern `attempt_0`, `attempt_1`, etc.
  
- `get_random_directory(group, index)` - Returns the output directory path for a random language group
  - `group`: "low" or "high" complexity
  - `index`: Which language within the group (0, 1, etc.)
  - Languages are organized as `outputs/random/languages/<lang_id>/`

**Language ID Management:**
- `get_new_target_id(target_lang)` - Generates the next attempt number for a new target language translation
  - Automatically increments from existing attempts (e.g., if `attempt_0` and `attempt_1` exist, returns `attempt_2`)
  - Returns `attempt_0` if no attempts exist yet
  
- `get_new_random_id(average_hamming_dist, num_in_group)` - Generates the next attempt number for a random language
  - `average_hamming_dist`: "low" or "high" complexity
  - `num_in_group`: Which language within the group (0, 1, etc.)
  
- `get_latest_target_id(target_lang)` - Gets the most recent language ID for a target language
  - Useful for continuing from the last attempt instead of starting fresh
  - Returns `None` if no attempts exist

- `get_latest_random_id(average_hamming_dist, num_in_group)` - Gets the most recent language ID for a random language

**Iteration Management:**
- `get_latest_target_iteration(target_lang, lang_id)` - Gets the highest iteration number completed for a target language
  - Returns 0 if no iterations exist
  - Use `get_latest_target_iteration(...) + 1` to start the next iteration

- `get_latest_random_iteration(average_hamming_dist, num_in_group, lang_id)` - Gets the highest iteration for a random language

**Example: Start a New Translation Attempt**
```python
from synthetic.generation.api import (
    translate_dataset_for_target, 
    get_new_target_id
)

# Generate a new attempt ID (attempt_0, attempt_1, etc.)
lang_id = get_new_target_id("swahili")

# Start translating from iteration 0
translate_dataset_for_target(
    corpus=corpus,
    lang_id=lang_id,
    target_lang="swahili",
    num_batches=Nine, # set to None to run every batch
    iteration=0
)
```

**Example: Continue from Last Attempt on Next Iteration**
```python
from synthetic.generation.api import (
    translate_dataset_for_target,
    get_latest_target_id,
    get_latest_target_iteration
)

# Get the most recent attempt
lang_id = get_latest_target_id("swahili")

# Get the next iteration to run (after the last completed one)
# IMPORTANT: each iteration corresponds to batch_idx + 1 (since it starts at 1)
# therefore, if next_iteration = 2, num_batches must be > 2 for at least 1 batch to run
if lang_id:
    next_iteration = get_latest_target_iteration("swahili", lang_id) + 1
    
    # Continue translating on the next iteration
    translate_dataset_for_target(
        corpus=corpus,
        lang_id=lang_id,
        target_lang="swahili",
        num_batches=2,
        iteration=next_iteration 
    )
```

**Example: Start a New Random Language Translation**
```python
from synthetic.generation.api import (
    translate_dataset_using_random,
    get_new_random_id
)

# Generate a new attempt ID for this random language (low complexity, first language)
lang_id = get_new_random_id("low", 0)

# Start translating the corpus with the random language
translate_dataset_using_random(
    corpus=corpus,
    lang_id=lang_id,
    average_hamming_dist="low",
    num_in_group=0,
    num_batches=2,
    iteration=0
)
```

## Error Recovery & Debugging

### Pipeline Interruptions

If the pipeline stops running (due to an error or interruption):

Retry this batch, run again with the **same iteration number**

### Understanding the Output

**Execution Logs:**
- Location: `logs/pipeline.log`
- Contains detailed logs of every step executed during all iterations
- Shows LLM prompts, responses, QA critiques, and any errors encountered
- Useful for debugging which specific iteration failed and why

**Language Metadata:**
- Location: `metadata.json` (at `attempt_N/`)
- Tracks overall language statistics:
  - `running_qa`: Sum of all QA scores across iterations
  - `running_qa_count`: Total number of QA evaluations
  - `average_qa`: Overall average QA score across all iterations
  - `num_valid_batches`: Count of successfully translated batches (no conflicts)
  - `num_invalid_batches`: Count of batches with word conflicts
  - `invalid_batches`: List of iteration numbers that had conflicts

**Valid Translations:**
- Location: `memory/valid_translations.json`
- Contains all translations from successful iterations
- Each sentence includes metadata: `iteration`, `index_in_iteration`, `global_index` assuming batch_size = 8
- Only translations without conflicts are included here

**Per-Iteration New Words:**
- Location: `memory/iter_N/new_words.json`
- Shows what new vocabulary this iteration required
- If iteration succeeded, these are added to `memory/lexicon/lexicon.csv`

**Per-Iteration Required Lexicon:**
- Location: `memory/iter_N/required_lexicon.csv`
- An LLM extracts from the full lexicon which entries are necessary for translating the specific sentences
- Shows which lexicon entries were actually used in this iteration

**Iteration QA Results:**
- Location: `memory/iter_N/translation/translation_qa.json`
- `final_qa` object: Evaluation of the final translations
  - Includes `overall_score` and any issues found
  - `conflicts` key: Lists new words that conflict with existing lexicon entries
    - Conflict = same form but different part-of-speech or translation
    - If conflicts exist, the translation is NOT added to `valid_translations.json`
    - Conflicting new words are NOT added to the lexicon
- `all_iterations` array: Full history of critic→amend cycles for this batch







