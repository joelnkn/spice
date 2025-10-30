"""
Configuration management for synthetic language generation
"""
import os

# 🔹 Base project directory (auto-detected)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 🔹 Paths
THIRD_PARTY_DIR = os.path.join(PROJECT_ROOT, "third_party")
CONGLANGER_DIR = os.path.join(THIRD_PARTY_DIR, "conglanger")
CONGLANGER_PATH = os.path.join(CONGLANGER_DIR, "src", "run_pipeline.py")

# 🔹 Default data and prompt directories
DATA_DIR = os.path.join(PROJECT_ROOT, "synthetic", "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "synthetic", "outputs")
PROMPT_DIR = os.path.join(CONGLANGER_DIR, "prompts")

# 🔹 Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)

class SyntheticConfig:
    """Configuration for synthetic language generation"""

    pass
