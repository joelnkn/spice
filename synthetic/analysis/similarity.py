"""
Language similarity metrics based on typological feature vectors.

This module provides distance and similarity functions for comparing
constructed languages based on their WALS-style feature vectors.
"""

from __future__ import annotations

import json
import os
from typing import Optional, Dict, Any, NamedTuple
from dataclasses import dataclass

import numpy as np


@dataclass
class LanguageComparison:
    distance: int          # number of differing comparable features
    similarity: float      # matching / comparable
    valid_features: int    # comparable features

def get_synthetic_feature_path(language_id: str, run_name: Optional[str] = None) -> str:
    """Get the path to the synthetic feature_analysis.json file for a language."""
    base_dir = "synthetic_data" if not run_name else os.path.join("synthetic_data", run_name)
    base_dir = os.path.join(base_dir, "languages", language_id)

    feature_path = os.path.join(base_dir, 'memory', 'analysis', 'feature_analysis.json')
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


from typing import List, Tuple, Optional

def average_pairwise_distance(path_to_languages: str) -> Tuple[float, int]:
    """
    Compute the average *normalized* pairwise distance over all languages in <path_to_languages>.
    """
    lang_dirs = [os.path.join(path_to_languages, d) for d in os.listdir(path_to_languages)
                 if os.path.isdir(os.path.join(path_to_languages, d))]
    
    n = len(lang_dirs)
    if n < 2:
        return 0.0, 0

    pair_dists = []
    for i in range(n):
        for j in range(i + 1, n):
            path_1 = os.path.join(lang_dirs[i], 'memory', 'analysis', 'feature_analysis.json')
            path_2 = os.path.join(lang_dirs[j], 'memory', 'analysis', 'feature_analysis.json')
            comp = compare_languages(path_1, path_2)
            # skip pairs that had no overlapping features
            if comp.valid_features == 0:
                continue
            norm_dist = comp.distance / comp.valid_features  # in [0,1]
            pair_dists.append(norm_dist)

    if not pair_dists:
        return 0.0, 0

    mean_dist = sum(pair_dists) / len(pair_dists)
    return mean_dist, len(pair_dists)

