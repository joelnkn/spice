from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from synthetic.conglanger import run_conglanger, create_llm_client
from synthetic.typology.extraction import extract_features
import uuid
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from synthetic.utils import load_metadata

import os
import re

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


def get_snli_batches():
    snli = load_dataset("snli", split="train")
    ds = NLISentenceOnlyDataset(snli)
    loader = DataLoader(
        ds, batch_size=8, shuffle=True, collate_fn=lambda batch: "\n".join(batch)
    )
    return loader
    
def translate_dataset(corpus, language_id, run_name, num_batches=None):
    """
    Translate an entire corpus using an already stabilized language, where each item in corpus is a batch (e.g., a string of joined sentences).
    Args:
        corpus: Iterable of batches (each batch is a string of sentences)
        language_id: ID of the stabilized language
        run_name: Name of the run containing the language
        num_batches: If given, only process this many batches (default: all)
    Returns:
        List of results from run_conglanger for each batch
    """
    results = []
    for batch_idx, batch in enumerate(corpus, 1):
        if num_batches is not None and batch_idx > num_batches:
            break
        print(f"Translating (batch {batch_idx}): {batch[:20]}... with language {language_id}")
        result = run_conglanger(
            steps=("translation",),
            translation_sentence=batch,
            run_name=run_name,
            lang_id=language_id,
            iteration=False,
        )
        results.append(result)
    return results

# TODO: make sure its randomly sampling 
def generate_consistent_language(
    corpus,
    language_id,
    run_name,
    max_stabilize_steps=32,
):
    # Generate base language
    run_conglanger(
        steps=("grammar", "lexicon"),
        run_name=run_name,
        iteration=True,
        lang_id=language_id,
    )
    
    lang_dir = os.path.join(run_name, "languages", language_id)
    
    # Track totals at each iteration
    iteration_stats = []

    print("Stabilizing language with corpus...")
    for i, batch in enumerate(corpus, 1):
        print(f"Processing sample {i}/{len(corpus)}: {batch[:20]}...")
        run_conglanger(
            steps=("translation",),
            translation_sentence=batch,
            lang_id=language_id,
            run_name=run_name,
            iteration=True,
        )

        metadata = load_metadata(lang_dir)
        total_rules = metadata.get("num_new_grammar_rules", 0)
        total_words = metadata.get("num_new_words", 0)

        iteration_stats.append({
            "iteration": i,
            "num_new_words": total_words,
            "num_new_grammar_rules": total_rules
        })

        if i >= max_stabilize_steps:
            break

    print(f"\nLanguage stabilization complete! ID: {language_id}")
    print("\nSummary of new words and grammar rules at each iteration:")
    for stat in iteration_stats:
        print(f"Iteration {stat['iteration']}: New words: {stat['num_new_words']}, New grammar rules: {stat['num_new_grammar_rules']}")
    return run_name, language_id


def generate_consistent_language_for_target(target_lang, corpus, max_stabilize_steps=32):
    """
    Generate a consistent language for a given target language name.
    - Sets run_name to target/target_lang
    - Sets language_id to target_<iter> where iter = 1 + max number in that directory
    - Calls generate_consistent_language with all params
    Args:
        target_lang (str): The name of the target language (e.g., 'french')
        corpus (iterable): Sentences to use for stabilization
        max_stabilize_steps (int): Max stabilization steps
    Returns:
        str: The language_id for the generated language
    """
    base_dir = os.path.join("target", target_lang)
    lang_dir = os.path.join(base_dir, "languages")
    os.makedirs(lang_dir, exist_ok=True)
    # Find all subdirs matching target_#
    max_iter = 0
    for name in os.listdir(lang_dir):
        m = re.match(r"target_(\d+)$", name)
        if m:
            num = int(m.group(1))
            if num > max_iter:
                max_iter = num
    next_iter = max_iter + 1
    language_id = f"target_{next_iter}"
    run_name = os.path.join("target", target_lang)
    return generate_consistent_language(
        corpus=corpus,
        language_id=language_id,
        run_name=run_name,
        max_stabilize_steps=max_stabilize_steps,
    )

def generate_random_consistent_language(corpus, max_stabilize_steps=32):
    """
    Generate a consistent language with a random language ID and run_name 'random'.
    Args:
        corpus (iterable): Sentences to use for stabilization
        max_stabilize_steps (int): Max stabilization steps
    Returns:
        str: The language_id for the generated language
    """
    return generate_consistent_language(
        corpus=corpus,
        language_id=str(uuid.uuid4())[:8],
        run_name="random",
        max_stabilize_steps=max_stabilize_steps,
    )

if __name__ == "__main__":
    corpus = get_snli_batches()
    
    # create stabilized language
    # run_name, language_id = generate_random_consistent_language(corpus, max_stabilize_steps=2) # change for the maximum number of times to translate a batch
    
    # OR create stabilized language for target
    run_name, language_id = generate_consistent_language_for_target("urdu", corpus, max_stabilize_steps=2)
    
    # translate dataset using stabilized language
    translate_dataset(corpus, language_id, run_name, num_batches=2) # don't include num_batches to translate all

    # Analyze the language to extract WALS-style features
    print(f"Analyzing language {language_id}...")
    llm_client = create_llm_client(model="gemini-2.5-pro")
    extract_features(llm_client, run_name, language_id)
    
    # TODO: when using target language, make sure the feature vector here is same as that in prompts/typology/features.json.py maybe add a helper in utils to call here
    # TODO: qa also checks that new words and grammar rules are actually listed in translation.json --> make sure violations actually penalize overall score (at least < 8)
    # TODO: see how much longer it takes with qa enabled & if it is that much better
    # TODO: consider changing thinking budget to low ? maybe this decreases time? Also using the pro model instead?
    # TODO: increase max stabilization steps & look at the number of new words & grammar rules at each step
    # TODO: consider adding more features to feature vector 
    # TODO: make sure features in prompts/typology/features.json.py are correct
    # TODO: make sure orthographies in prompts/typology/orthography.py are correct