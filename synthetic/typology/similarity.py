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
from synthetic.config import OUTPUT_DIR

@dataclass
class LanguageComparison:
    distance: int          # number of differing comparable features
    similarity: float      # matching / comparable
    valid_features: int    # comparable features

def get_synthetic_feature_path(run_name: str, language_id: str) -> str:
    """Get the path to the feature_analysis.json file for a language.
    
    Args:
        run_name: Name of the run
        language_id: Unique ID of the language
        
    Returns:
        Path to feature_analysis.json file
    """
    feature_path = os.path.join(OUTPUT_DIR, run_name, 'languages', language_id, 'analysis', 'features.json')
    return feature_path

def load_feature_dict(feature_path: str) -> Dict[str, str]:
    """Load feature dictionary from feature_analysis.json.
    
    Args:
        feature_path: Path to feature_analysis.json file
    Returns:
        Dictionary mapping feature names to their values
    """
    with open(feature_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def compare_languages(lang1_feature_path: str, lang2_feature_path: str) -> LanguageComparison:
    """Compare two languages based on their feature vectors."""
    
    features1 = load_feature_dict(lang1_feature_path)
    features2 = load_feature_dict(lang2_feature_path)

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

def average_pairwise_distance(feature_paths: list[str]) -> Tuple[float, int]:
    """
    Compute the average *normalized* pairwise distance over all languages.
    
    Args:
        feature_paths: List of paths to feature_analysis.json files
    
    Returns:
        Tuple of (mean_distance, num_pairs)
    """
    n = len(feature_paths)
    if n < 2:
        return 0.0, 0

    pair_dists = []
    for i in range(n):
        for j in range(i + 1, n):
            comp = compare_languages(feature_paths[i], feature_paths[j])
            # skip pairs that had no overlapping features
            if comp.valid_features == 0:
                continue
            norm_dist = comp.distance / comp.valid_features  # in [0,1]
            pair_dists.append(norm_dist)

    if not pair_dists:
        return 0.0, 0

    mean_dist = sum(pair_dists) / len(pair_dists)
    return mean_dist, len(pair_dists)

