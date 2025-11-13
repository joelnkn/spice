"""
Feature extraction and parsing for language analyses.

This module parses the WALS-style JSON output produced by the analysis prompt
and converts it into a canonical feature dict. It saves per-language artifacts 
under the language memory folder:

- memory/analysis/feature_analysis.json  (canonical judgments + metadata)

The schema used here corresponds to the 18 typological features described in
the updated evaluation prompt, including separate flags for causative,
passive, and applicative morphology. Each feature includes an explicit "null"
category so that unknown values are represented consistently.
"""


from __future__ import annotations

import json
import os
import re
import logging
from typing import Dict, List, Tuple, Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Canonical feature schema (each feature includes explicit 'null')
# Order here defines vector layout.
# ---------------------------------------------------------------------------
FEATURE_SCHEMA: Dict[str, List[str]] = {
    "consonant_inventory": ["small", "large", "null"],
    "vowel_inventory": ["small", "large", "null"],
    "tone": ["tonal", "non-tonal", "null"],
    "morphological_fusion": ["isolating", "agglutinating", "fusional", "polysynthetic", "null"],
    "affixation_balance": ["prefixing", "suffixing", "null"],
    "gender_inventory": ["none", "few", "moderate", "many", "null"],
    "case_inventory": ["minimal", "moderate", "extensive", "null"],
    "numeral_classifiers": ["classifier", "non-classifier", "null"],
    "basic_word_order": ["SVO", "SOV", "VSO", "VOS", "OVS", "OSV", "null"],
    "adposition_type": ["prepositional", "postpositional", "null"],
    "genitive_noun_order": ["genitive-before-noun", "genitive-after-noun", "null"],
    "adjective_noun_order": ["adjective-before-noun", "adjective-after-noun", "null"],
    "relative_clause_order": ["head-initial", "head-final", "null"],
    "alignment_typology": ["nominative–accusative", "active–stative", "null"],
    "causative_morphology": ["present", "absent", "null"],
    "passive_morphology": ["present", "absent", "null"],
    "applicative_morphology": ["present", "absent", "null"],
    "question_marking_strategy": ["interrogative-particle", "inversion", "null"],
}

# Flattened index map will be computed when needed

# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _extract_json_block(text: str) -> Optional[str]:
    """Try to extract a JSON block from analysis text (code fences or plain)."""
    if not text:
        return None
    # Look for ```json ... ``` or ``` ... ```
    m = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
    if m:
        return m.group(1)
    # Fallback: look for first { ... } block
    m2 = re.search(r"(\{[\s\S]*\})", text)
    if m2:
        return m2.group(1)
    return None


def parse_analysis_text_to_feature_dict(analysis_text: str) -> Dict[str, Any]:
    """Parse analysis text (expected JSON) into a canonical feature dict.

    If parsing fails, attempt some heuristics. Unknown or missing features are
    returned as the string 'null'.
    """
    if not analysis_text:
        return {k: "null" for k in FEATURE_SCHEMA.keys()}

    # Try direct JSON
    try:
        data = json.loads(analysis_text)
        logger.debug("Parsed analysis_text as JSON")
    except Exception:
        # Try extracting JSON block
        block = _extract_json_block(analysis_text)
        if block:
            try:
                data = json.loads(block)
                logger.debug("Parsed extracted JSON block from analysis_text")
            except Exception:
                data = None
        else:
            data = None

    if data is None or not isinstance(data, dict):
        # As a fallback, attempt to parse simple `key: value` lines
        data = {}
        for line in analysis_text.splitlines():
            if ':' not in line:
                continue
            key, val = line.split(':', 1)
            key = key.strip().strip('\"\'')
            val = val.strip().strip('\"\'')
            if key:
                # map a few likely key variants to canonical
                canonical_key = _map_key_to_canonical(key)
                data[canonical_key] = val

    # Normalize and ensure all keys exist
    feature_dict = {}
    for key in FEATURE_SCHEMA.keys():
        raw = data.get(key, data.get(_map_key_to_canonical(key), None))
        if raw is None:
            feature_dict[key] = "null"
        else:
            feature_dict[key] = _normalize_value_for_feature(key, raw)
    return feature_dict


def _map_key_to_canonical(k: str) -> str:
    k2 = k.strip().lower().replace(' ', '_').replace('-', '_').replace('–', '-').replace('\u2013', '-')
    # some specific aliases
    aliases = {
        'genitive–noun_order': 'genitive_noun_order',
    }
    return aliases.get(k2, k2)


def _normalize_value_for_feature(feature: str, raw_value: Any) -> str:
    """Normalize common synonyms/case differences into canonical categories.

    raw_value may be a list, dict, number, or string.
    """
    if raw_value is None:
        return "null"
    if isinstance(raw_value, (list, dict)):
        # Not expected for most features; convert to JSON string
        raw = json.dumps(raw_value)
    else:
        raw = str(raw_value).strip()

    if raw == '':
        return 'null'

    low = raw.lower()

    # Feature-specific normalization
    if feature in ('consonant_inventory', 'vowel_inventory'):
        if 'large' in low or '>' in low or 'more' in low or '≥' in low:
            return 'large'
        if 'small' in low or '<=' in low or '≤' in low or 'fewer' in low:
            return 'small'
        # numeric guesses
        m = re.search(r'(\d+)', low)
        if m:
            n = int(m.group(1))
            if feature == 'consonant_inventory':
                return 'small' if n <= 20 else 'large'
            if feature == 'vowel_inventory':
                return 'small' if n < 9 else 'large'
        return 'null'

    if feature == 'tone':
        if 'non' in low or 'no tone' in low or 'not tonal' in low:
            return 'non-tonal'
        if 'ton' in low:
            return 'tonal'
        return 'null'

    if feature == 'morphological_fusion':
        for opt in FEATURE_SCHEMA[feature]:
            if opt in low:
                return opt
        # some heuristics
        if 'isolat' in low or 'analytic' in low:
            return 'isolating'
        if 'agglut' in low:
            return 'agglutinating'
        if 'fus' in low:
            return 'fusional'
        if 'poly' in low:
            return 'polysynthetic'
        return 'null'

    if feature == 'affixation_balance':
        if 'prefix' in low:
            return 'prefixing'
        if 'suffix' in low:
            return 'suffixing'
        return 'null'

    if feature == 'gender_inventory':
        if 'none' in low or 'no gender' in low:
            return 'none'
        m = re.search(r'(\d+)', low)
        if m:
            return m.group(1)
        if 'multiple' in low or 'many' in low:
            return 'many'
        return 'null'

    if feature == 'case_inventory':
        if 'minimal' in low or 'none' in low or 'little' in low:
            return 'minimal'
        if 'moderate' in low or 'some' in low:
            return 'moderate'
        if 'extensive' in low or 'many' in low:
            return 'extensive'
        return 'null'

    if feature == 'numeral_classifiers':
        if 'classif' in low:
            return 'classifier'
        if 'no' in low or 'non' in low:
            return 'non-classifier'
        return 'null'

    if feature == 'basic_word_order':
        for opt in FEATURE_SCHEMA[feature]:
            if opt.lower() in low:
                return opt
        # try to capture common words
        for opt in ['SOV', 'SVO', 'VSO', 'VOS', 'OVS', 'OSV']:
            if opt.lower() in low:
                return opt
        return 'null'

    if feature == 'adposition_type':
        if 'post' in low:
            return 'postpositional' if 'postpos' not in low else 'postpositional'
        if 'pre' in low:
            return 'prepositional' if 'prepos' not in low else 'prepositional'
        # normalise to keys used above
        if 'postposition' in low or 'postpos' in low:
            return 'postpositional'
        if 'preposition' in low or 'prepos' in low:
            return 'prepositional'
        return 'null'

    if feature in ('genitive_noun_order', 'adjective_noun_order'):
        if 'before' in low or 'genitive before' in low or 'adjective before' in low:
            if feature == 'genitive_noun_order':
                return 'genitive-before-noun'
            return 'adjective-before-noun'
        if 'after' in low or 'genitive after' in low or 'adjective after' in low:
            if feature == 'genitive_noun_order':
                return 'genitive-after-noun'
            return 'adjective-after-noun'
        return 'null'

    if feature == 'relative_clause_order':
        if 'head-initial' in low or 'head initial' in low or 'head-initial' in raw:
            return 'head-initial'
        if 'head-final' in low or 'head final' in low:
            return 'head-final'
        # heuristics: if adjective_noun is after-noun, relative may be head-final
        return 'null'

    if feature == 'alignment_typology':
        if 'nomin' in low:
            return 'nominative–accusative'
        if 'active' in low or 'stativ' in low:
            return 'active–stative'
        return 'null'

    # New schema uses separate flags for valence morphology.
    if feature in ('causative_morphology', 'passive_morphology', 'applicative_morphology'):
        # Decide present/absent/null for each specific morphology
        if feature == 'causative_morphology':
            if 'caus' in low or 'causative' in low:
                return 'present'
            if 'no caus' in low or 'no causative' in low or 'absent' in low or 'none' in low:
                return 'absent'
            return 'null'

        if feature == 'passive_morphology':
            if 'pass' in low or 'passive' in low:
                return 'present'
            if 'no pass' in low or 'no passive' in low or 'absent' in low or 'none' in low:
                return 'absent'
            return 'null'

        if feature == 'applicative_morphology':
            if 'applic' in low or 'applicative' in low:
                return 'present'
            if 'no applic' in low or 'no applicative' in low or 'absent' in low or 'none' in low:
                return 'absent'
            return 'null'

    if feature == 'question_marking_strategy':
        if 'inversion' in low:
            return 'inversion'
        if 'particle' in low or 'interrogative' in low:
            return 'interrogative-particle'
        return 'null'

    # Default fallback: return raw lowercased trimmed value if it matches one of the choices
    for opt in FEATURE_SCHEMA.get(feature, []):
        if opt in low:
            return opt
    return 'null'


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_feature_files(language_dir: str, feature_dict: Dict[str, Any]) -> Dict[str, str]:
    """Save canonical feature JSON into language memory.

    Returns a dict with the saved file path.
    """
    analysis_dir = os.path.join(language_dir, 'memory', 'analysis')
    os.makedirs(analysis_dir, exist_ok=True)

    feature_json_path = os.path.join(analysis_dir, 'feature_analysis.json')

    with open(feature_json_path, 'w', encoding='utf-8') as f:
        json.dump(feature_dict, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved features to {feature_json_path}")
    return {'feature_json': feature_json_path}


def extract_and_save_from_analysis(language_dir: str) -> Optional[Dict[str, str]]:
    """Read `memory/analysis/features.txt` or `analysis.json`, parse and save features.

    Returns dict of saved file path on success, else None.
    """
    analysis_txt_path = os.path.join(language_dir, 'memory', 'analysis', 'features.txt')
    analysis_json_path = os.path.join(language_dir, 'memory', 'analysis', 'analysis.json')
    raw = None
    if os.path.exists(analysis_json_path):
        try:
            with open(analysis_json_path, 'r', encoding='utf-8') as f:
                raw = f.read()
        except Exception:
            raw = None
    if raw is None and os.path.exists(analysis_txt_path):
        with open(analysis_txt_path, 'r', encoding='utf-8') as f:
            raw = f.read()
    if not raw:
        logger.warning(f"No analysis text found in {language_dir}/memory/analysis")
        return None

    # Parse
    feature_dict = parse_analysis_text_to_feature_dict(raw)
    saved = save_feature_files(language_dir, feature_dict)
    return saved
