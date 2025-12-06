# ConlangCrafter: Constructing Languages with a Multi-Hop LLM Pipeline

**Project Page:** [http://conlangcrafter.github.io](http://conlangcrafter.github.io)  
**Paper:** [ConlangCrafter: Constructing Languages with a Multi-Hop LLM Pipeline](https://arxiv.org/abs/2508.06094)

We introduce a fully automated system for constructing languages (conlangs) using large language models. Our multi-stage pipeline creates coherent, diverse artificial languages with their own phonology, grammar, lexicon, and translation capabilities.

## Code

### Supported Models

You can pass any valid model string for the provider you choose:

- Google Gemini (e.g., gemini-2.5-pro, gemini-1.5-flash)
- OpenAI (e.g., o4-mini, gpt-4o, gpt-5, gpt-4.1-mini)
- DeepSeek via Together (e.g., deepseek-ai/DeepSeek-R1)

### Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up API keys:**
   ```bash
   cp .env.example .env
   # Edit .env and add your API keys
   ```

3. **Generate a language:**
   ```bash
   python src/run_pipeline.py --model gemini-2.5-pro
   ```

   Or with OpenAI models (choose the model you prefer):
   ```bash
   # Reasoning model (o-series)
   python src/run_pipeline.py --model o4-mini --reasoning-effort medium

   # GPT-family examples
   python src/run_pipeline.py --model gpt-4o
   python src/run_pipeline.py --model gpt-5
   python src/run_pipeline.py --model gpt-4.1-mini
   ```

Notes for OpenAI:
- o-series (o1/o3/o4) reasoning models ignore temperature/top_p; use `--reasoning-effort`.
- GPT-family models (e.g., gpt-4o) respect temperature/top_p.

### Directory Structure

```
src/                    # Core source code
├── run_pipeline.py     # Main pipeline script
├── llm_client.py       # LLM API clients
├── pipeline_steps.py   # Language generation steps
├── cleanup.py          # Post-processing for iteration mode
└── utils.py            # Utility functions

prompts/                # Prompt templates
├── phonology/          # Phonology generation prompts
├── grammar/            # Grammar generation prompts
├── lexicon/            # Lexicon building prompts
├── translation/        # Translation prompts
└── qa/                 # QA/refinement prompts

base_specifications/    # Pre-generated language specifications
├── target/             # Target language group
│   ├── affixes.json
│   ├── feature_vectors.json
│   ├── lexicons.csv
│   └── orthography.py
└── random/             # Random language groups
    ├── low/            # Low complexity
    └── high/           # High complexity
```

### Output Directory Structure

When you generate a language, the output is organized as follows:

```
output_dir/
└── run_name/
    └── languages/
        └── language_id/              # Unique 8-character ID
            ├── metadata.json         # Tracks creation, continuations, QA metrics
            ├── logs/
            │   └── pipeline.log
            └── memory/               # Persistent language state
                ├── lexicon/
                │   └── lexicon.csv   # Working lexicon, updated with new words
                ├── valid_translations.json    # Accumulated valid translations with iteration tracking
                ├── iter_0/           # Iteration 0 translations (--iteration mode)
                │   ├── translation/
                │   │   ├── translation.json
                │   │   └── translation_qa.json
                │   ├── new_words.json
                │   └── required_lexicon.csv
                ├── iter_1/           # Iteration 1 translations
                │   ├── translation/
                │   │   ├── translation.json
                │   │   └── translation_qa.json
                │   ├── new_words.json
                │   └── required_lexicon.csv
                └── iter_N/           # Subsequent iterations
                    ├── translation/
                    │   ├── translation.json
                    │   └── translation_qa.json
                    ├── new_words.json
                    └── required_lexicon.csv
```

**Key directories:**
- `metadata.json`: Tracks creation time, continued runs, and QA metrics
- `memory/`: Persistent language state (modified during iteration mode)
- `memory/lexicon/lexicon.csv`: Working lexicon, updated with new words across iterations
- `memory/valid_translations.json`: Accumulated valid translations with iteration metadata
- `memory/iter_N/`: Each iteration's translation results (only in `--iteration` mode)
  - `translation/`: Contains `translation.json` and `translation_qa.json`
  - `new_words.json`: New words extracted in this iteration
  - `required_lexicon.csv`: Subset of lexicon used for this iteration
- `logs/`: Pipeline execution logs

**Iteration Mode (`--iteration` flag):**
- **First run** (iteration=0): Generates initial language, saves to `memory/iter_0/`
- **Subsequent runs** (iteration≥1): Runs translation, extracts new words, updates `memory/lexicon/`, saves to `memory/iter_N/`
- Each iteration preserves language evolution; translations tracked with iteration metadata in `valid_translations.json`

### Configuration

The system supports various parameters for customizing language generation:

```bash
python src/run_pipeline.py \
    --model gemini-2.5-pro \
    --steps phonology,grammar,lexicon,translation \
    --custom-constraints "Use only 3 vowels" \
    --translation-sentence "Hello, world!"
```

#### Model-specific parameter guide

- reasoning-effort: Applies to OpenAI o-series reasoning models only (o1, o3, o4, including o4-mini). Ignored by GPT-family models like gpt-4o and gpt-5.
- thinking-budget: Applies to Google Gemini models that support thinking output. Supported: gemini-2.5-pro. Not supported/ignored: gemini-1.5-flash and OpenAI models in this project.
- DeepSeek: DeepSeek-R1 automatically emits a <think> section; thinking-budget isn’t used here. Use temperature/top_p as usual.

### API Keys

You'll need API keys for the language models:

- **Google Gemini**: Get from [Google AI Studio](https://aistudio.google.com/app/apikey) → set `GOOGLE_API_KEY`
- **OpenAI**: Get from [OpenAI API Keys](https://platform.openai.com/api-keys) → set `OPENAI_API_KEY`
- **DeepSeek (via Together)**: Get from [Together AI](https://api.together.xyz/settings/api-keys) → set `TOGETHER_API_KEY`

Add these to your `.env` file (copy from `.env.example`).

### Self-Refinement (Critic / Amend Loop)

Enable an optional QA loop that critiques and amends intermediate artifacts (affix, lexicon, translation).

Scoring scale used by prompts:
- 10: Completely consistent / excellent
- 9: Consistent, only clarity or minor style ambiguities
- 8: Very minor issues (default acceptance threshold)
- 7: Some moderate issues – needs revision
- 6 or below: Significant inconsistencies or errors

Run with QA enabled (global threshold & custom self-refine cycles):
```bash
python src/run_pipeline.py --model gemini-2.5-pro \
   --qa-enabled \
   --self-refine-steps 4 \
   --qa-threshold 8
```

Flags:
- `--qa-enabled`: Activate QA self-refine loop.
- `--self-refine-steps`: Number of critic→amend cycles (default 3).
- `--qa-threshold`: Global acceptance threshold (1–10 scale) overriding per-step thresholds when set.
- `--qa-threshold-<step>`: Per-step acceptance threshold (default 8.0) used only if `--qa-threshold` not supplied.
- `--continue-qa`: Append new QA iterations onto existing `<step>_qa.json`.

Each QA cycle:
1. Critic prompt returns JSON: `overall_score` (1–10) + issues list
2. If score < threshold and cycles remain, amend prompt applies corrections
3. Loop stops early if threshold met; otherwise after self-refine budget exhausted
4. Iterations logged in `<step>_qa.json` (before/after snapshots + iteration metadata)

QA results are persisted at:
- `memory/iter_N/translation/translation_qa.json` - QA results for translation at iteration N
- `memory/valid_translations.json` - Accumulated valid translations with QA metadata per iteration

Note: A score of 9 typically indicates only clarity/ambiguity issues; 8 allows very minor contradictions. Adjust thresholds if you want stricter acceptance.

### Base Specifications

The system uses pre-generated baseline language specifications stored in `base_specifications/` directory. These are manually curated or generated once per language/group and reused for all translation iterations.

```
base_specifications/
├── target/                    # Target languages (one per language)
│   ├── affixes.json          # Affix inventory
│   ├── feature_vectors.json  # Typological features
│   ├── lexicons.csv          # Base lexicon
│   └── orthography.py        # Orthography rules
└── random/                    # Random language groups
    ├── low/                   # Low complexity group
    │   ├── affixes.json
    │   ├── feature_vectors.json
    │   ├── low_0.csv          # Lexicon for low_0
    │   ├── low_1.csv          # Lexicon for low_1
    │   └── orthography.py
    └── high/                  # High complexity group
        ├── affixes.json
        ├── feature_vectors.json
        ├── high_0.csv
        ├── high_1.csv
        └── orthography.py
```

### Workflow for Setting Up Base Specifications

1. **Generate Affix & Lexicon Prompts:**
   ```bash
   # Runs affix step (prints prompts, no LLM call)
   python src/run_pipeline.py --model gemini-2.5-pro --steps affix --lang-name english
   
   # Runs lexicon step (prints prompts, no LLM call)
   python src/run_pipeline.py --model gemini-2.5-pro --steps lexicon --lang-name english
   ```
   
   Prompts are saved to:
   - `prompts/affix/all_target.txt` (for target languages)
   - `prompts/affix/all_random_low.txt` or `prompts/affix/all_random_high.txt` (for random groups)
   - `prompts/lexicon/all_target.txt` (for target languages)
   - `prompts/lexicon/all_random_low.txt` or `prompts/lexicon/all_random_high.txt` (for random groups)

2. **Generate with External LLM:**
   - Copy the prompts and ask your preferred LLM (ChatGPT, Claude, etc.) to generate affixes and lexicon
   - Ensure the output is valid JSON for affixes and CSV for lexicon

3. **Add to Base Specifications:**
   - Save the generated affixes JSON to `base_specifications/target/affixes.json` (or appropriate random group)
   - Save the generated lexicon CSV to `base_specifications/target/lexicons.csv` (or `base_specifications/random/<group>/<group>_<idx>.csv`)

4. **Run Translation Iterations:**
   ```bash
   # First translation iteration (iteration=0)
   python src/run_pipeline.py --model gemini-2.5-pro --steps translation \
       --iteration 0 --lang-name english
   
   # Subsequent iterations (reuses specifications, accumulates translations)
   python src/run_pipeline.py --model gemini-2.5-pro --steps translation \
       --iteration 1 --lang-id attempt_0 --lang-name english
   ```
   
   On iteration 0:
   - Base lexicon from `base_specifications/target/lexicons.csv` is copied to `memory/lexicon/lexicon.csv`
   - Translations are generated and new words extracted
   - `memory/lexicon/lexicon.csv` is updated with new words
   - Base specification remains untouched

   On subsequent iterations:
   - Uses the already-updated `memory/lexicon/lexicon.csv`
   - Continues accumulating new words and translations

## Citation

If you use ConlangCrafter in your research, please cite:

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

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
