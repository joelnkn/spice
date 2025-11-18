import os
import json
import logging
import re
import shutil

logger = logging.getLogger(__name__)


def clean_response(response: str, response_type: str = "text") -> str:
    """Clean LLM response by removing markdown formatting and extracting content."""
    if response_type == "csv":
        # Extract CSV content from markdown code blocks
        csv_pattern = r'```(?:csv)?\s*\n(.*?)\n```'
        matches = re.findall(csv_pattern, response, re.DOTALL)
        if matches:
            return matches[0].strip()
        return response.strip()
    
    elif response_type == "json":
        # Extract JSON content from markdown code blocks
        json_pattern = r'```(?:json)?\s*\n(.*?)\n```'
        matches = re.findall(json_pattern, response, re.DOTALL)
        if matches:
            return matches[0].strip()
        return response.strip()
    
    return response.strip()


def alphabetize_csv_text(csv_text: str) -> str:
    """Alphabetize CSV entries while preserving the header."""
    lines = csv_text.strip().split('\n')
    if len(lines) <= 1:
        return csv_text
    
    header = lines[0]
    data_lines = [line for line in lines[1:] if line.strip()]
    
    # Sort data lines alphabetically
    data_lines.sort()
    
    return '\n'.join([header] + data_lines)


def get_csv_text_n_entries(csv_text: str) -> int:
    """Count the number of data entries in CSV text (excluding header)."""
    lines = csv_text.strip().split('\n')
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
            with open(file_path, 'r', encoding='utf-8') as f:
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
    with open(content_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    # Save metadata
    metadata_path = os.path.join(memory_dir, 'metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved {filename} and metadata to {memory_dir}")


# === Additional helper utilities for QA-enabled pipeline ===

def load_files_with_optional(memory_dir: str, required_files: dict, optional_files: dict) -> dict:
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
                with open(path, 'r', encoding='utf-8') as f:
                    files[key] = f.read()
                logger.info(f"Loaded optional {key}: {path}")
            except Exception as e:
                logger.warning(f"Failed loading optional {key} ({path}): {e}")
    return files


def save_memory_without_metadata(content: str, memory_dir: str, filename: str):
    """Save content file only (do not create/modify metadata.json)."""
    os.makedirs(memory_dir, exist_ok=True)
    path = os.path.join(memory_dir, filename)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    logger.info(f"Saved {filename} (no metadata) to {memory_dir}")


def save_individual_metadata(metadata: dict, memory_dir: str, filename: str):
    """Save a standalone metadata JSON (used for per-item outputs like individual translations)."""
    os.makedirs(memory_dir, exist_ok=True)
    path = os.path.join(memory_dir, filename)
    with open(path, 'w', encoding='utf-8') as f:
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


def find_last_created_folder(parent_dir: str) -> str:
    """Find the folder with the maximum iteration number in a directory.
    
    Args:
        parent_dir: Directory to search in
    
    Returns:
        Path to the folder with the highest iter_N number
    """
    if not os.path.exists(parent_dir):
        raise FileNotFoundError(f"Directory not found: {parent_dir}")
    
    # Get all directories starting with 'iter_'
    entries = [e for e in os.listdir(parent_dir) 
               if os.path.isdir(os.path.join(parent_dir, e)) and e.startswith('iter_')]
    
    if not entries:
        raise FileNotFoundError(f"No iter_ folders found in {parent_dir}")
    
    # Extract iteration numbers and find the maximum
    iter_nums = []
    for entry in entries:
        parts = entry.split('_')
        if len(parts) == 2 and parts[1].isdigit():
            iter_nums.append((int(parts[1]), entry))
    
    if not iter_nums:
        raise FileNotFoundError(f"No valid iter_ folders found in {parent_dir}")
    
    # Get the folder with the max iteration number
    _, max_folder_name = max(iter_nums, key=lambda x: x[0])
    max_folder_path = os.path.join(parent_dir, max_folder_name)
    
    logger.info(f"Found folder with max iteration: {max_folder_name}")
    return max_folder_path