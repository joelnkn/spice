# SPICE — Synthetic Polyglot Injection for Cross-lingual Evaluation

A project for improving multilingual model capability through fine-tuning on synthetic language data.

Colab: https://colab.research.google.com/drive/1ig1AQZ26Q-1xXjZMbJi1cLRfKrghR2Lf#scrollTo=NsYEJe4Umz_h

## Project Structure

```
spice/
├── scripts/            # Utility scripts
├── synthetic/          # Synthetic language generation module
│   ├── config.py       # Configuration management
│   ├── data/           # Data directories
│   ├── generation/     # Language generation logic
│   └── utils/          # Utility functions
└── third_party/          # Third-party code/submodules
│   └── conglanger/      # Conglanger submodule
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Setup

Run the setup script to initialize the environment:

```bash
bash scripts/setup.sh
```

## Synthetic Language Generation

The `synthetic/` module contains the infrastructure for generating synthetic language data. See `synthetic/README.md` for details.

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
