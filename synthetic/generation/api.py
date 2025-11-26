from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from synthetic.config import OUTPUT_DIR
from synthetic.conglanger import run_conglanger, create_llm_client
from synthetic.typology.extraction import extract_features
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
):
    snli = load_dataset("snli", split="train")
    ds = NLISentenceOnlyDataset(snli)
    loader = DataLoader(
        ds, batch_size=8, shuffle=True, collate_fn=lambda batch: "\n".join(batch)
    )
    return generate_consistent_language(
        loader, language_id, output_dir, run_name, max_stabilize_steps
    )


def generate_consistent_language(
    corpus,
    language_id=None,
    output_dir=OUTPUT_DIR,
    run_name="consistent",
    max_stabilize_steps=32,
):
    """Generate a consistent language by training on a corpus of sentences.

    Args:
        corpus: List of sentences to use for language stabilization
        language_id: Optional language ID to reuse (default: generate new UUID)
        output_dir: Base output directory (default: OUTPUT_DIR from config)
        run_name: Name for this language generation run (default: "consistent")

    Returns:
        str: The language_id for the generated language
    """
    # Generate language ID if not provided
    if language_id is None:
        language_id = str(uuid.uuid4())[:8]

    print(f"Generating consistent language with ID: {language_id}")

    # Generate base language
    run_conglanger(
        steps=("grammar", "lexicon"),
        qa_enabled=False,
        output_dir=output_dir,
        run_name=run_name,
        reasoning_effort="low",
        iteration=True,
        lang_id=language_id,
    )
    
    lang_dir = os.path.join(output_dir, run_name, "languages", language_id)
    
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


def translate(sentence, language_id, output_dir=OUTPUT_DIR, run_name="consistent"):
    """Translate a sentence using an already stabilized language.

    Args:
        sentence: The sentence to translate
        language_id: ID of the stabilized language
        output_dir: Base output directory
        run_name: Name of the run containing the language

    Returns:
        Result from run_conglanger
    """
    print(f"Translating with language {language_id}: {sentence[:50]}...")

    result = run_conglanger(
        steps=("translation",),
        translation_sentence=sentence,
        output_dir=output_dir,
        run_name=run_name,
        lang_id=language_id,
        iteration=False,  # Not iteration mode - just append new words
    )

    return result


if __name__ == "__main__":
    # Example 1: Generate a consistent language from a corpus
    language_id = generate_from_snli(max_stabilize_steps=4)

    # Example 2: Use the stabilized language to translate new sentences
    translate("How are you today?", run_name="consistent", language_id=language_id)
    translate("The sun is shining.", run_name="consistent", language_id=language_id)

    # Example 3: Analyze the language to extract WALS-style features
    print(f"Analyzing language {language_id}...")
    llm_client = create_llm_client(model="gemini-2.5-pro")
    extract_features(llm_client, "consistent", language_id)
