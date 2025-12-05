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
import random
import re
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

# Canonical typological feature schema.
# - Each key corresponds to a single discrete feature.
# - Each list contains all allowed values + a "null" fallback.
# - "null" is ONLY for "cannot determine"; it should not be chosen for synthetic specs.

FEATURE_SCHEMA: Dict[str, List[str]] = {
    # Overall morphological type: how morphemes bundle meaning.
    # isolating: little/no inflection; agglutinating: clear stacked morphemes;
    # fusional: endings bundle features; polysynthetic: very many morphemes per word.
    "morphological_fusion": ["isolating", "agglutinating", "fusional", "polysynthetic", "null"],

    # Where inflectional/derivational morphology mainly sits on the word.
    # prefixing: mostly on the left; suffixing: mostly on the right.
    "affixation_balance": ["prefixing", "suffixing", "null"],

    # Size of grammatical gender/noun-class system (if any).
    # none: no gender; few: e.g. 2–3; moderate: several; many: large/complex.
    "gender_inventory": ["none", "few", "moderate", "many", "null"],

    # How often nouns (or agreement) mark number morphologically.
    # none: no number marking; optional: only in some contexts; obligatory: regularly marked.
    "number_marking": ["none", "optional", "obligatory", "null"],

    # Richness of morphological case on nouns.
    # none: no case; minimal: 2–3 cases; moderate: ~4–6; extensive: many.
    "case_inventory": ["none", "minimal", "moderate", "extensive", "null"],

    # Whether numerals require classifiers (like "three CLF-dog") or not.
    "numeral_classifiers": ["classifier", "non-classifier", "null"],

    # Default / neutral clause word order (single dominant pattern only).
    "basic_word_order": ["SVO", "SOV", "VSO", "VOS", "OVS", "OSV", "null"],

    # Position of adpositions relative to the noun phrase.
    "adposition_type": ["prepositional", "postpositional", "null"],

    # Order of possessor (genitive) and possessed noun.
    "genitive_noun_order": ["genitive-before-noun", "genitive-after-noun", "null"],

    # Order of adjectives and head noun.
    "adjective_noun_order": ["adjective-before-noun", "adjective-after-noun", "null"],

    # Whether the head noun comes before or after its relative clause.
    "relative_clause_order": ["head-initial", "head-final", "null"],

    # How relative clauses are morphosyntactically marked.
    # particle: special marker at clause edge; relative-pronoun: who/which-like;
    # gap: largely unmarked, structure alone signals relation.
    "relative_clause_marking": ["particle", "relative-pronoun", "gap", "null"],

    # Alignment of core arguments.
    # nominative–accusative vs ergative–absolutive vs active–stative.
    # We intentionally EXCLUDE "split" to force a single dominant alignment type.
    "alignment_typology": ["nominative–accusative", "ergative–absolutive", "active–stative", "null"],

    # Which arguments verbs agree with (if any).
    "agreement_pattern": ["no-agreement", "subject-only", "subject-object", "null"],

    # How tense is grammatically encoded.
    # none: no tense morphology; binary-past-nonpast; three-way-past-present-future; rich: more distinctions.
    "tense_system": ["none", "binary-past-nonpast", "three-way-past-present-future", "rich", "null"],

    # How aspect (ongoing/completed/habitual etc.) is encoded.
    # simple: e.g. perf vs imperf; moderate: adds progressive/habitual; rich: many distinctions.
    "aspect_system": ["none", "simple", "moderate", "rich", "null"],

    # How many grammatical moods (indicative/imperative/conditional/subjunctive/etc.) are distinguished.
    "mood_system": ["none", "simple", "moderate", "rich", "null"],

    # Main strategy for clausal negation.
    # preverbal-particle: "not" before verb; verbal-affix: negative prefix/suffix on verb;
    # negative-auxiliary: separate neg-verb; clause-final-particle: neg marker at end.
    "negation_strategy": ["preverbal-particle", "verbal-affix", "negative-auxiliary", "clause-final-particle", "null"],

    # Presence/absence of productive morphological causatives.
    "causative_morphology": ["present", "absent", "null"],

    # Presence/absence of productive morphological passives.
    "passive_morphology": ["present", "absent", "null"],

    # Presence/absence of productive applicatives (promoting oblique roles to core arguments).
    "applicative_morphology": ["present", "absent", "null"],

    # Primary strategy for marking interrogatives.
    # interrogative-particle: question particle; inversion: subject–verb inversion;
    # intonation-only: questions mostly prosodic; wh-fronting: dedicated wh-movement.
    "question_marking_strategy": ["interrogative-particle", "inversion", "intonation-only", "wh-fronting", "null"]
}

def generate_random_feature_vector(
    schema: Optional[Dict[str, List[str]]] = None,
    seed: Optional[int] = None,
) -> Dict[str, str]:
    """
    Generate a random typological feature vector for a synthetic language.

    - Picks one non-"null" value uniformly at random for each feature.
    - Uses FEATURE_SCHEMA by default, but a custom schema can be passed.
    - Optionally accepts a seed for reproducible generation.

    Returns:
        The generated feature vector as a dict[str, str].
    """
    if schema is None:
        schema = FEATURE_SCHEMA

    rng = random.Random(seed) if seed is not None else random

    feature_vector: Dict[str, str] = {}

    for feature_name, options in schema.items():
        # Filter out "null" – we never want to generate null for random languages
        non_null_options = [opt for opt in options if opt != "null"]
        choice = rng.choice(non_null_options)
        feature_vector[feature_name] = choice

    return feature_vector

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

def _check_which_option_in_value(value: str, feature: str) -> str | None:
    """Check if normalized value matches one of the options (substring, case-insensitive)."""
    v = value.lower()
    for opt in FEATURE_SCHEMA[feature]:
        if opt == "null":
            continue
        if opt.lower() in v:
            return opt
    return None

def _check_which_option_in_value(value: str, feature: str) -> str | None:
    """Check if normalized value matches one of the options (substring, case-insensitive)."""
    v = value.lower()
    for opt in FEATURE_SCHEMA[feature]:
        if opt == "null":
            continue
        if opt.lower() in v:
            return opt
    return None


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

    # First try a direct match against option strings for this feature
    matched_option = _check_which_option_in_value(low, feature)
    if matched_option is not None:
        return matched_option

    # ------------------------------------------------------------------
    # Feature-specific heuristics
    # ------------------------------------------------------------------

    if feature == 'morphological_fusion':
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
        if 'no gender' in low or 'no grammatical gender' in low or 'none' in low:
            return 'none'
        if 'few' in low or 'two gender' in low or 'three gender' in low:
            return 'few'
        if 'moderate' in low or 'some genders' in low:
            return 'moderate'
        if 'multiple' in low or 'many' in low or 'rich' in low:
            return 'many'
        return 'null'

    if feature == 'number_marking':
        if 'no number' in low or 'no grammatical number' in low or 'none' in low:
            return 'none'
        if 'optional' in low or 'sometimes marked' in low:
            return 'optional'
        if 'obligat' in low or 'always' in low or 'regularly' in low:
            return 'obligatory'
        return 'null'

    if feature == 'case_inventory':
        if 'no case' in low or 'no morphological case' in low or 'none' in low:
            return 'none'
        if 'minimal' in low or 'little' in low or 'few cases' in low:
            return 'minimal'
        if 'moderate' in low or 'some cases' in low:
            return 'moderate'
        if 'extensive' in low or 'many cases' in low or 'rich case' in low:
            return 'extensive'
        return 'null'

    if feature == 'numeral_classifiers':
        if 'classif' in low:
            return 'classifier'
        if 'no classifier' in low or 'no classif' in low or 'non-classifier' in low:
            return 'non-classifier'
        return 'null'

    if feature == 'basic_word_order':
        for opt in FEATURE_SCHEMA[feature]:
            if opt != "null" and opt.lower() in low:
                return opt
        return 'null'

    if feature == 'adposition_type':
        if 'postposition' in low or 'postpos' in low or 'post' in low:
            return 'postpositional'
        if 'preposition' in low or 'prepos' in low or 'pre ' in low:
            return 'prepositional'
        return 'null'

    if feature in ('genitive_noun_order', 'adjective_noun_order'):
        if 'before' in low:
            if feature == 'genitive_noun_order':
                return 'genitive-before-noun'
            return 'adjective-before-noun'
        if 'after' in low:
            if feature == 'genitive_noun_order':
                return 'genitive-after-noun'
            return 'adjective-after-noun'
        return 'null'

    if feature == 'relative_clause_order':
        if 'head-initial' in low or 'head initial' in low:
            return 'head-initial'
        if 'head-final' in low or 'head final' in low:
            return 'head-final'
        return 'null'

    if feature == 'relative_clause_marking':
        if 'particle' in low:
            return 'particle'
        if 'relative pronoun' in low or 'relative-pronoun' in low or 'rel pronoun' in low:
            return 'relative-pronoun'
        if 'gap' in low or 'unmarked' in low or 'zero marking' in low:
            return 'gap'
        return 'null'

    if feature == 'alignment_typology':
        if 'nomin' in low:
            return 'nominative–accusative'
        if 'ergat' in low:
            return 'ergative–absolutive'
        if 'active' in low or 'stativ' in low:
            return 'active–stative'
        return 'null'

    if feature == 'agreement_pattern':
        if 'no agreement' in low or 'without agreement' in low:
            return 'no-agreement'
        if 'subject only' in low or 'only subject' in low or 'subject-agreement' in low:
            return 'subject-only'
        if 'object agreement' in low or 'subject and object' in low:
            return 'subject-object'
        return 'null'

    if feature == 'tense_system':
        if 'no tense' in low or 'no grammatical tense' in low:
            return 'none'
        if 'binary' in low or ('past' in low and 'nonpast' in low):
            return 'binary-past-nonpast'
        if 'three' in low and 'past' in low and 'present' in low and 'future' in low:
            return 'three-way-past-present-future'
        if 'rich' in low or 'complex' in low or 'multiple past' in low or 'several past' in low:
            return 'rich'
        return 'null'

    if feature == 'aspect_system':
        if 'no aspect' in low or 'no grammatical aspect' in low or 'none' in low:
            return 'none'
        if 'perfective' in low and 'imperfective' in low and 'progressive' not in low and 'habitual' not in low:
            return 'simple'
        if 'progressive' in low or 'habitual' in low or 'continuous' in low:
            return 'moderate'
        if 'rich' in low or 'complex' in low or 'several aspects' in low:
            return 'rich'
        return 'null'

    if feature == 'mood_system':
        if 'no mood' in low or 'no grammatical mood' in low or 'none' in low:
            return 'none'
        if 'imperative' in low and 'subjunctive' not in low and 'conditional' not in low:
            return 'simple'
        if 'subjunctive' in low or 'conditional' in low:
            return 'moderate'
        if 'rich' in low or 'complex' in low or 'multiple moods' in low:
            return 'rich'
        return 'null'

    if feature == 'negation_strategy':
        if 'preverbal' in low or 'before the verb' in low:
            return 'preverbal-particle'
        if 'suffix' in low or 'prefix' in low or 'affix' in low:
            return 'verbal-affix'
        if 'auxiliary' in low or 'aux' in low:
            return 'negative-auxiliary'
        if 'clause-final' in low or 'sentence-final' in low or 'final particle' in low:
            return 'clause-final-particle'
        return 'null'

    if feature in ('causative_morphology', 'passive_morphology', 'applicative_morphology'):
        if feature == 'causative_morphology':
            if 'caus' in low:
                return 'present'
            if 'no caus' in low or 'no causative' in low or 'absent' in low or 'none' in low:
                return 'absent'
            return 'null'

        if feature == 'passive_morphology':
            if 'pass' in low:
                return 'present'
            if 'no pass' in low or 'no passive' in low or 'absent' in low or 'none' in low:
                return 'absent'
            return 'null'

        if feature == 'applicative_morphology':
            if 'applic' in low:
                return 'present'
            if 'no applic' in low or 'no applicative' in low or 'absent' in low or 'none' in low:
                return 'absent'
            return 'null'

    if feature == 'question_marking_strategy':
        if 'inversion' in low:
            return 'inversion'
        if 'particle' in low or 'interrogative particle' in low:
            return 'interrogative-particle'
        if 'intonation' in low:
            return 'intonation-only'
        if 'wh-front' in low or 'wh front' in low or 'wh-movement' in low or 'wh movement' in low:
            return 'wh-fronting'
        return 'null'

    return 'null'


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_feature_files(language_dir: str, feature_dict: Dict[str, Any]) -> Dict[str, str]:
    """Save canonical feature JSON into language analysis directory.

    Returns a dict with the saved file path.
    """
    analysis_dir = os.path.join(language_dir, 'analysis')
    os.makedirs(analysis_dir, exist_ok=True)

    feature_json_path = os.path.join(analysis_dir, 'feature_analysis.json')

    with open(feature_json_path, 'w', encoding='utf-8') as f:
        json.dump(feature_dict, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved features to {feature_json_path}")
    return {'feature_json': feature_json_path}
