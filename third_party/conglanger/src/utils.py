import os
import json
import logging
import re
import shutil

logger = logging.getLogger(__name__)


def create_llm_client(model='gemini-2.5-pro', max_tokens=8192, temperature=0.7, 
                     sleep_between_calls=30, debug=False, thinking_budget=1000, 
                     reasoning_effort='medium'):
    """Create an LLM client based on model name.
    
    Args:
        model: Model identifier (e.g., gemini-2.5-pro, gpt-4o, deepseek-ai/DeepSeek-R1)
        max_tokens: Maximum tokens for generation
        temperature: Temperature for sampling
        sleep_between_calls: Sleep time between API calls (seconds)
        debug: Enable debug mode with dummy responses
        thinking_budget: Thinking budget for models that support it (Gemini)
        reasoning_effort: Reasoning effort for OpenAI o-series models
        
    Returns:
        Initialized LLM client instance
    """
    from llm_client import LLMClientGemini, LLMClientDeepseek, LLMClientOpenAI
    
    if model.startswith('gemini'):
        return LLMClientGemini(
            model_checkpoint=model,
            max_tokens=max_tokens,
            temperature=temperature,
            sleep_between_calls=sleep_between_calls,
            debug=debug,
            thinking_budget=thinking_budget
        )
    elif model.startswith('deepseek'):
        return LLMClientDeepseek(
            model_checkpoint=model,
            max_tokens=max_tokens,
            temperature=temperature,
            sleep_between_calls=sleep_between_calls,
            debug=debug
        )
    elif model.startswith('o') or model.startswith('gpt-'):
        return LLMClientOpenAI(
            model_checkpoint=model,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            temperature=temperature,
            sleep_between_calls=sleep_between_calls,
            debug=debug
        )
    else:
        raise ValueError(f"Unsupported model: {model}")


def clean_response(response: str, response_type: str = "text") -> str:
    """Clean LLM response by removing markdown formatting and extracting content."""
    if response_type == "csv":
        # Extract CSV content from markdown code blocks
        csv_pattern = r"```(?:csv)?\s*\n(.*?)\n```"
        matches = re.findall(csv_pattern, response, re.DOTALL)
        if matches:
            return matches[0].strip()
        return response.strip()

    elif response_type == "json":
        # Extract JSON content from markdown code blocks
        json_pattern = r"```(?:json)?\s*\n(.*?)\n```"
        matches = re.findall(json_pattern, response, re.DOTALL)
        if matches:
            return matches[0].strip()
        return response.strip()

    return response.strip()


def alphabetize_csv_text(csv_text: str) -> str:
    """Alphabetize CSV entries while preserving the header."""
    lines = csv_text.strip().split("\n")
    if len(lines) <= 1:
        return csv_text

    header = lines[0]
    data_lines = [line for line in lines[1:] if line.strip()]

    # Sort data lines alphabetically
    data_lines.sort()

    return "\n".join([header] + data_lines)


def get_csv_text_n_entries(csv_text: str) -> int:
    """Count the number of data entries in CSV text (excluding header)."""
    lines = csv_text.strip().split("\n")
    # Count non-empty lines excluding the header
    return len([line for line in lines[1:] if line.strip()]) if len(lines) > 1 else 0


def load_required_files(memory_dir: str, required_files: dict) -> dict:
    """Load required files from memory directory."""
    files = {}

    for key, filename in required_files.items():
        # Try to find the file in the appropriate subdirectory
        file_path = os.path.join(memory_dir, key, filename)

        if not os.path.exists(file_path):
            logger.error(f"Required file not found: {file_path}")
            return None

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                files[key] = f.read()
            logger.info(f"Loaded {key}: {file_path}")
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return None

    return files


def save_memory(content: str, memory_dir: str, filename: str, metadata: dict):
    """Save content and metadata to memory directory."""
    os.makedirs(memory_dir, exist_ok=True)

    # Save content
    content_path = os.path.join(memory_dir, filename)
    with open(content_path, "w", encoding="utf-8") as f:
        f.write(content)

    # Save metadata
    metadata_path = os.path.join(memory_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved {filename} and metadata to {memory_dir}")


# === Additional helper utilities for QA-enabled pipeline ===


def load_files_with_optional(
    memory_dir: str, required_files: dict, optional_files: dict
) -> dict:
    """Load required files plus optional files if they exist.

    Returns a dict containing all loaded file contents, or None if a required file is missing.
    """
    files = load_required_files(memory_dir, required_files)
    if files is None:
        return None
    for key, filename in optional_files.items():
        path = os.path.join(memory_dir, key, filename)
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    files[key] = f.read()
                logger.info(f"Loaded optional {key}: {path}")
            except Exception as e:
                logger.warning(f"Failed loading optional {key} ({path}): {e}")
    return files


def save_memory_without_metadata(content: str, memory_dir: str, filename: str):
    """Save content file only (do not create/modify metadata.json)."""
    os.makedirs(memory_dir, exist_ok=True)
    path = os.path.join(memory_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    logger.info(f"Saved {filename} (no metadata) to {memory_dir}")


def save_individual_metadata(metadata: dict, memory_dir: str, filename: str):
    """Save a standalone metadata JSON (used for per-item outputs like individual translations)."""
    os.makedirs(memory_dir, exist_ok=True)
    path = os.path.join(memory_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved individual metadata {filename} to {memory_dir}")


def copy_folders(src_dir: str, dst_dir: str, folder_names: list) -> None:
    """Copy specified folders from source directory to destination directory.

    Args:
        src_dir: Source directory path
        dst_dir: Destination directory path
        folder_names: List of folder names to copy

    Raises:
        FileNotFoundError: If source directory doesn't exist
    """
    if not os.path.exists(src_dir):
        raise FileNotFoundError(f"Source directory not found: {src_dir}")

    os.makedirs(dst_dir, exist_ok=True)

    for folder in folder_names:
        src = os.path.join(src_dir, folder)
        dst = os.path.join(dst_dir, folder)

        if os.path.exists(src):
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            logger.info(f"Copied folder '{folder}' from {src_dir} to {dst_dir}")
        else:
            logger.warning(f"Folder '{folder}' not found in source directory: {src}")

