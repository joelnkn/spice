from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from synthetic.config import OUTPUT_DIR
from synthetic.conglanger import run_conglanger, create_llm_client
from synthetic.typology.extraction import extract_features
from synthetic.generation.custom_constraints import BASE, combine_constraints, URDU_LIKE_FEATURES
import uuid
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader

import os
import json

def load_metadata(lang_dir):
    """Load metadata.json from a language directory, or return {} if missing."""
    metadata_path = os.path.join(lang_dir, "metadata.json")
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: could not load metadata.json: {e}")
    return {}

class NLISentenceOnlyDataset(Dataset):
    def __init__(self, hf_dataset):
        """
        hf_dataset: a HuggingFace split like ds['train'] with fields 'premise' and 'label'
        """
        self.data = hf_dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]
        return ex["premise"]


def generate_from_snli(
    language_id=None,
    output_dir=OUTPUT_DIR,
    run_name="consistent",
    max_stabilize_steps=32,
    wals_features=None,
    language_features=None,
):
    """Generate a language from SNLI dataset.
    
    Args:
        language_id: Optional language ID to reuse
        output_dir: Base output directory
        run_name: Name for this language generation run
        max_stabilize_steps: Maximum number of stabilization steps
        wals_features: Optional dict of WALS features to enforce (e.g., 
                      {"basic_word_order": "SOV", "morphological_fusion": "agglutinating"})
        language_features: Optional multi-line string describing language features to emulate
                          (e.g., characteristics similar to a particular language without
                          naming it or using words from that language)
    
    Returns:
        str: The language_id for the generated language
    """
    snli = load_dataset("snli", split="train")
    ds = NLISentenceOnlyDataset(snli)
    loader = DataLoader(
        ds, batch_size=8, shuffle=True, collate_fn=lambda batch: "\n".join(batch)
    )
    return generate_consistent_language(
        loader, language_id, output_dir, run_name, max_stabilize_steps, wals_features, language_features
    )


def generate_consistent_language(
    corpus,
    language_id=None,
    output_dir=OUTPUT_DIR,
    run_name="consistent",
    max_stabilize_steps=32,
    wals_features=None,
    language_features=None,
):
    """Generate a consistent language by training on a corpus of sentences.

    Args:
        corpus: List of sentences to use for language stabilization
        language_id: Optional language ID to reuse (default: generate new UUID)
        output_dir: Base output directory (default: OUTPUT_DIR from config)
        run_name: Name for this language generation run (default: "consistent")
        max_stabilize_steps: Maximum number of stabilization steps
        wals_features: Optional dict of WALS features to enforce (e.g., 
                      {"basic_word_order": "SOV", "morphological_fusion": "agglutinating"})
        language_features: Optional multi-line string describing language features to emulate
                          (e.g., characteristics similar to a particular language without
                          naming it or using words from that language)

    Returns:
        str: The language_id for the generated language
    """
    # Generate language ID if not provided
    if language_id is None:
        language_id = str(uuid.uuid4())[:8]

    print(f"Generating consistent language with ID: {language_id}")
    
    # Combine BASE constraints with WALS features and language features if provided
    constraints = combine_constraints(BASE, wals_features, language_features)
    if wals_features:
        print(f"Enforcing WALS features: {list(wals_features.keys())}")
    if language_features:
        print("Enforcing custom language feature descriptions")

    # Generate base language
    run_conglanger(
        steps=("grammar", "lexicon"),
        qa_enabled=False,
        output_dir=output_dir,
        run_name=run_name,
        reasoning_effort="low",
        iteration=True,
        lang_id=language_id,
        custom_constraints=constraints,
    )
    
    lang_dir = os.path.join(output_dir, run_name, "languages", language_id)
    memory_dir = os.path.join(lang_dir, "memory")
    
    # Verify that required files were created
    grammar_file = os.path.join(memory_dir, "grammar", "grammar.txt")
    if not os.path.exists(grammar_file):
        raise RuntimeError(
            f"Grammar file was not created at {grammar_file}. "
            f"The grammar step may have failed. Please check the logs and try again."
        )
    
    # Track previous totals so we can compute deltas per step
    prev_words = 0
    prev_rules = 0

    # Make consistent using corpus
    print("Stabilizing language with corpus...")
    for i, sample in enumerate(corpus, 1):
        print(f"Processing sample {i}/{len(corpus)}: {sample[:50]}...")
        run_conglanger(
            steps=("translation",),
            translation_sentence=sample,
            output_dir=output_dir,
            lang_id=language_id,
            run_name=run_name,
            iteration=True,
            custom_constraints=constraints,
        )
        
        metadata = load_metadata(lang_dir)
        total_rules = metadata.get("num_new_grammar_rules", 0)
        total_words = metadata.get("num_new_words", 0)

        delta_rules = total_rules - prev_rules
        delta_words = total_words - prev_words

        print(
            f"New words this step: {delta_words} (total: {total_words}) | "
            f"New rules this step: {delta_rules} (total: {total_rules})"
        )

        prev_rules = total_rules
        prev_words = total_words
        
        if i >= max_stabilize_steps:
            break

    print(f"Language stabilization complete! ID: {language_id}")
    return language_id


def translate(sentence, language_id, output_dir=OUTPUT_DIR, run_name="consistent", wals_features=None, language_features=None):
    """Translate a sentence using an already stabilized language.

    Args:
        sentence: The sentence to translate
        language_id: ID of the stabilized language
        output_dir: Base output directory
        run_name: Name of the run containing the language
        wals_features: Optional dict of WALS features (usually not needed for translation)
        language_features: Optional multi-line string describing language features (usually not needed for translation)

    Returns:
        Result from run_conglanger
    """
    print(f"Translating with language {language_id}: {sentence[:50]}...")
    
    # Use constraints if provided, otherwise just BASE
    if wals_features or language_features:
        constraints = combine_constraints(BASE, wals_features, language_features)
    else:
        constraints = BASE

    result = run_conglanger(
        steps=("translation",),
        translation_sentence=sentence,
        output_dir=output_dir,
        run_name=run_name,
        lang_id=language_id,
        iteration=False,  # Not iteration mode - just append new words
        custom_constraints=constraints,
    )

    return result


if __name__ == "__main__":
    # Toggle: Set to True to use Urdu typological features, False to generate without constraints
    USE_URDU_FEATURES = False
    USE_URDU_LIKE_DESCRIPTIONS = False
    
    # Urdu typological features based on WALS
    urdu_features = {
        "morphological_fusion": "fusional",
        "affixation_balance": "suffixing",
        "gender_inventory": "moderate",
        "case_inventory": "extensive",
        "numeral_classifiers": "non-classifier",
        "basic_word_order": "SOV",
        "adposition_type": "postpositional",
        "genitive_noun_order": "genitive-before-noun",
        "adjective_noun_order": "adjective-before-noun",
        "relative_clause_order": "head-final",
        "alignment_typology": "nominative–accusative",
        "causative_morphology": "present",
        "passive_morphology": "present",
        "applicative_morphology": "null",
        "question_marking_strategy": "interrogative-particle",
    }
    
    # Example 1: Generate a consistent language from a corpus
    wals_features = urdu_features if USE_URDU_FEATURES else None
    language_features = URDU_LIKE_FEATURES if USE_URDU_LIKE_DESCRIPTIONS else None
    
    if USE_URDU_FEATURES:
        print("Generating language with Urdu typological features...")
    if USE_URDU_LIKE_DESCRIPTIONS:
        print("Generating language with Urdu-like feature descriptions...")
    if not USE_URDU_FEATURES and not USE_URDU_LIKE_DESCRIPTIONS:
        print("Generating language without feature constraints...")
    
    language_id = generate_from_snli(
        max_stabilize_steps=4, 
        wals_features=wals_features,
        language_features=language_features
    )

    # Example 2: Use the stabilized language to translate new sentences
    translate("How are you today?", run_name="consistent", language_id=language_id)
    translate("The sun is shining.", run_name="consistent", language_id=language_id)

    # Example 3: Analyze the language to extract WALS-style features
    print(f"Analyzing language {language_id}...")
    llm_client = create_llm_client(model="gemini-2.5-pro")
    extract_features(llm_client, "consistent", language_id)
