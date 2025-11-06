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


def load_feature_dict(language_dir: str) -> Dict[str, str]:
    """Load feature dictionary from feature_analysis.json.
    
    Args:
        language_dir: Path to language directory
        
    Returns:
        Dictionary mapping feature names to their values
    """
    feature_path = os.path.join(language_dir, 'memory', 'analysis', 'feature_analysis.json')
    with open(feature_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def compare_languages(lang1_dir: str, lang2_dir: str) -> LanguageComparison:
    features1 = load_feature_dict(lang1_dir)
    features2 = load_feature_dict(lang2_dir)

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

def average_pairwise_distance(lang_dirs: List[str]) -> Tuple[float, int]:
    """
    Compute the average *normalized* pairwise distance over a list of languages.
    """
    n = len(lang_dirs)
    if n < 2:
        return 0.0, 0

    pair_dists = []
    for i in range(n):
        for j in range(i + 1, n):
            comp = compare_languages(lang_dirs[i], lang_dirs[j])
            # skip pairs that had no overlapping features
            if comp.valid_features == 0:
                continue
            norm_dist = comp.distance / comp.valid_features  # in [0,1]
            pair_dists.append(norm_dist)

    if not pair_dists:
        return 0.0, 0

    mean_dist = sum(pair_dists) / len(pair_dists)
    return mean_dist, len(pair_dists)

