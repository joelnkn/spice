"""
Language similarity metrics based on typological feature vectors.

This module provides distance and similarity functions for comparing
constructed languages based on their WALS-style feature vectors.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Tuple
from dataclasses import dataclass

@dataclass
class LanguageComparison:
    distance: int          # number of differing comparable features
    similarity: float      # matching / comparable
    valid_features: int    # comparable features

def load_feature_dict(lang_name: str) -> Dict[str, str]:
    """Load feature vector for a language.
    
    Args:
        lang_name: Language name. Can be:
            - "low_0", "low_1", etc. → loads from base_specifications/random/low.json, gets entry i
            - "high_0", "high_1", etc. → loads from base_specifications/random/high.json, gets entry i
            - Any other name → loads from base_specifications/target/feature_vectors.json, gets lang_name entry
    
    Returns:
        Dictionary containing feature vector for the language
    """
    # Find the project root by looking for conlanger
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = current_dir
    while project_root != os.path.dirname(project_root):  # Until we reach filesystem root
        if os.path.exists(os.path.join(project_root, 'conlanger')):
            break
        project_root = os.path.dirname(project_root)
    
    base_spec_dir = os.path.join(project_root, 'conlanger', 'base_specifications')
    
    # Check if lang_name matches pattern: low_N or high_N
    parts = lang_name.split('_')
    if len(parts) == 2 and parts[0] in ['low', 'high']:
        try:
            index = int(parts[1])
            group = parts[0]
            feature_path = os.path.join(base_spec_dir, 'random', f'{group}', 'feature_vectors.json')
            
            with open(feature_path, 'r', encoding='utf-8') as f:
                all_features = json.load(f)
            
            # all_features should be a list, get the ith entry
            if isinstance(all_features, list) and index < len(all_features):
                return all_features[index]
            else:
                return {}
        except (ValueError, FileNotFoundError):
            pass
    
    # Otherwise, load from target/feature_vectors.json
    feature_path = os.path.join(base_spec_dir, 'target', 'feature_vectors.json')
    
    with open(feature_path, 'r', encoding='utf-8') as f:
        all_features = json.load(f)
    
    return all_features.get(lang_name, {})

def compare_languages(lang1_name: str, lang2_name: str) -> LanguageComparison:
    """Compare two languages based on their feature vectors."""
    
    features1 = load_feature_dict(lang1_name)
    features2 = load_feature_dict(lang2_name)
    valid_features = 0
    matching_features = 0

    for feature_name in features1.keys():
        val1 = features1[feature_name]
        val2 = features2.get(feature_name, "null")

        if val1 == "null" or val2 == "null":
            continue

        valid_features += 1
        if val1 == val2:
            matching_features += 1

    if valid_features == 0:
        return LanguageComparison(distance=0, similarity=1.0, valid_features=0)

    distance = valid_features - matching_features
    similarity = matching_features / valid_features

    return LanguageComparison(
        distance=distance,
        similarity=similarity,
        valid_features=valid_features
    )

def average_pairwise_distance(lang_names: list[str]) -> Tuple[float, int]:
    """
    Compute the average *normalized* pairwise distance over all languages.
    
    Args:
        lang_names: List of language names
    
    Returns:
        Tuple of (mean_distance, num_pairs)
    """
    n = len(lang_names)
    if n < 2:
        return 0.0, 0

    pair_dists = []
    for i in range(n):
        for j in range(i + 1, n):
            comp = compare_languages(lang_names[i], lang_names[j])
            # skip pairs that had no overlapping features
            if comp.valid_features == 0:
                continue
            norm_dist = comp.distance / comp.valid_features  # in [0,1]
            pair_dists.append(norm_dist)

    if not pair_dists:
        return 0.0, 0

    mean_dist = sum(pair_dists) / len(pair_dists)
    return mean_dist, len(pair_dists)

