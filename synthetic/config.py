"""
Configuration management for synthetic language generation
"""
import os

# 🔹 Base project directory (auto-detected)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 🔹 Paths
CONLANGER_DIR = os.path.join(PROJECT_ROOT, "conlanger")
CONLANGER_PATH = os.path.join(CONLANGER_DIR, "src", "run_pipeline.py")

# 🔹 Default data and prompt directories
DATA_DIR = os.path.join(PROJECT_ROOT, "synthetic", "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "synthetic", "outputs")
PROMPT_DIR = os.path.join(CONLANGER_DIR, "prompts")

# 🔹 Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)

class SyntheticConfig:
    """Configuration for synthetic language generation"""

    pass
