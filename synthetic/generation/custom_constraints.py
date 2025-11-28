BASE = """
When you introduce or describe any rule, follow these priorities:
- Prefer SIMPLE, GENERAL rules that can apply to MANY different words or constructions.
- Avoid HYPER-SPECIFIC rules that only exist to explain a single example or a very narrow phrase.
- CRITICALLY IMPORTANT: Be STRONGLY dissimilar from rules in English. Actively avoid English-like patterns, word orders, grammatical structures, and morphological processes. The language should feel distinctly non-English in its grammar, syntax, and word formation.
- A good rule should:
  - be reusable across multiple roots or contexts,
  - fit naturally with the rest of the system,
  - differ substantially from how English handles similar concepts.
"""


def format_wals_features(wals_features: dict) -> str:
    """Format WALS features dictionary into a readable constraint string.
    
    Args:
        wals_features: Dictionary mapping feature names to values (e.g., 
                      {"basic_word_order": "SOV", "morphological_fusion": "agglutinating"})
    
    Returns:
        Formatted string describing the typological constraints
    """
    if not wals_features:
        return ""
    
    # Feature descriptions for better readability
    feature_descriptions = {
        "morphological_fusion": "Morphological fusion type",
        "affixation_balance": "Affixation balance",
        "gender_inventory": "Gender inventory",
        "case_inventory": "Case inventory",
        "numeral_classifiers": "Numeral classifiers",
        "basic_word_order": "Basic word order",
        "adposition_type": "Adposition type",
        "genitive_noun_order": "Genitive-noun order",
        "adjective_noun_order": "Adjective-noun order",
        "relative_clause_order": "Relative clause order",
        "alignment_typology": "Alignment typology",
        "causative_morphology": "Causative morphology",
        "passive_morphology": "Passive morphology",
        "applicative_morphology": "Applicative morphology",
        "question_marking_strategy": "Question marking strategy",
    }
    
    constraints = []
    constraints.append("The language MUST follow these typological features:")
    
    for feature, value in wals_features.items():
        if value and value != "null":
            description = feature_descriptions.get(feature, feature.replace("_", " ").title())
            # Format value for readability
            formatted_value = value.replace("-", " ").replace("–", "-").title()
            constraints.append(f"- {description}: {formatted_value}")
    
    return "\n".join(constraints)


def format_language_features(language_features: str) -> str:
    """Format language feature description text into a constraint string.
    
    This function takes descriptive text about language features (e.g., describing
    characteristics similar to a particular language) and formats it as constraints.
    The text should describe features without explicitly naming the target language
    or including words from that language.
    
    Args:
        language_features: Multi-line string describing language features to emulate
    
    Returns:
        Formatted constraint string
    """
    if not language_features or not language_features.strip():
        return ""
    
    constraints = []
    constraints.append("The language should exhibit the following typological characteristics:")
    constraints.append("")
    # Split into lines and format each as a bullet point
    for line in language_features.strip().split("\n"):
        line = line.strip()
        if line:
            # If line doesn't start with a bullet, add one
            if not line.startswith("-") and not line.startswith("*"):
                constraints.append(f"- {line}")
            else:
                constraints.append(line)
    
    return "\n".join(constraints)


def combine_constraints(
    base_constraints: str = BASE, 
    wals_features: dict = None,
    language_features: str = None
) -> str:
    """Combine BASE constraints with WALS feature constraints and language feature descriptions.
    
    Args:
        base_constraints: Base constraint string (defaults to BASE)
        wals_features: Optional dictionary of WALS features to enforce
        language_features: Optional multi-line string describing language features to emulate
                          (e.g., characteristics similar to a particular language without
                          naming it or using words from that language)
    
    Returns:
        Combined constraint string ready to pass to run_conglanger
    """
    constraints = [base_constraints.strip()]
    
    if wals_features:
        wals_constraints = format_wals_features(wals_features)
        if wals_constraints:
            constraints.append(wals_constraints)
    
    if language_features:
        lang_constraints = format_language_features(language_features)
        if lang_constraints:
            constraints.append(lang_constraints)
    
    return "\n\n".join(constraints)


# Example language feature descriptions
# These describe characteristics without naming the language or using words from it

URDU_LIKE_FEATURES = """
Verbs agree with their subjects in person, number, and gender, with gender distinctions being marked on both nouns and verbs.
The language has a two-gender system (masculine and feminine) that affects noun agreement throughout the grammar, not just in isolated contexts.
Verbs can be marked for aspect (perfective/imperfective) and tense, often through morphological changes or auxiliary verbs.
The language has a rich system of honorifics and politeness markers that affect verb forms and pronouns, with different levels of formality encoded in the grammar.
Verb roots can be modified through various derivational processes to create related meanings (causative, passive, intensive, etc.).
The language has a system of compound verbs where a main verb combines with an auxiliary verb to express aspectual or modal meanings.
"""