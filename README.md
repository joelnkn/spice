# SPICE — Synthetic Polyglot Injection for Cross-lingual Evaluation

A project for improving multilingual model capability through fine-tuning on synthetic language data.

Colab: https://colab.research.google.com/drive/1ig1AQZ26Q-1xXjZMbJi1cLRfKrghR2Lf#scrollTo=NsYEJe4Umz_h

## Project Structure

```
spice/
├── synthetic/           # Synthetic language generation module
│   ├── config.py       # Configuration management
│   ├── data/           # Data directories
│   ├── generation/     # Language generation logic
│   └── utils/          # Utility functions
├── scripts/            # Utility scripts
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
