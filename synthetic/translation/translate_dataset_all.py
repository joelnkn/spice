from datasets import load_dataset, Dataset
import json
import csv
import os
import random
import sys
import logging
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv(os.path.join(PROJECT_ROOT, '.env'))

CONGLANGER_SRC = os.path.join(PROJECT_ROOT, 'third_party', 'conglanger', 'src')

if CONGLANGER_SRC not in sys.path:
    sys.path.insert(0, CONGLANGER_SRC)

from llm_client import LLMClientGemini, LLMClientOpenAI, LLMClientDeepseek, PromptManager
from utils import load_files_with_optional, clean_response

def init_llm_client(
    model="gemini-2.5-pro",
    max_tokens=8192,
    temperature=0.7,
    thinking_budget=1000,
    reasoning_effort="medium",
    sleep_between_calls=30,
    debug=False,
):
    if model.startswith('gemini'):
        llm_client = LLMClientGemini(
            model_checkpoint=model,
            max_tokens=max_tokens,
            temperature=temperature,
            sleep_between_calls=sleep_between_calls,
            debug=debug,
            thinking_budget=thinking_budget
        )
    elif model.startswith('deepseek'):
        llm_client = LLMClientDeepseek(
            model_checkpoint=model,
            max_tokens=max_tokens,
            temperature=temperature,
            sleep_between_calls=sleep_between_calls,
            debug=debug
        )
    elif model.startswith('o') or model.startswith('gpt-'):
        llm_client = LLMClientOpenAI(
            model_checkpoint=model,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            temperature=temperature,
            sleep_between_calls=sleep_between_calls,
            debug=debug
        )
    else:
        raise ValueError(f"Unsupported model: {model}")
    
    return llm_client

def update_lexicon(language_dir, new_words):
    if not new_words:
        return
    
    lexicon_path = os.path.join(language_dir, 'memory', 'lexicon', 'lexicon.csv')
    
    # Read existing lexicon if it exists
    existing_words = set()
    if os.path.exists(lexicon_path):
        with open(lexicon_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            existing_words = {row['ipa'] for row in reader if 'ipa' in row}
    
    # Append new words that don't already exist
    with open(lexicon_path, 'a', encoding='utf-8', newline='') as f:
        fieldnames = ['ipa', 'pos', 'translation', 'grammar', 'derivation', 'notes']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        # Write header if file is new or empty
        if not existing_words:
            writer.writeheader()
        
        for word_entry in new_words:
            if word_entry['ipa'] not in existing_words:
                writer.writerow(word_entry)
                existing_words.add(word_entry['ipa'])

def run_translation_step(language_dir, llm_client, sentences):
    memory_dir = os.path.join(language_dir, 'memory')
    required = {'phonology': 'phonology.txt', 'grammar': 'grammar.txt'}
    optional = {'lexicon': 'lexicon.csv'}
    files = load_files_with_optional(memory_dir, required, optional)
    
    if files is None:
        raise FileNotFoundError(f"Required language files not found in {memory_dir}")

    if 'lexicon' in files:
        lex_section = f"""It has the following lexicon:\n\n=== START ===\n{files['lexicon']}\n=== END ==="""
    else:
        lex_section = """Note: No specific lexicon has been provided. You will need to create appropriate vocabulary words that follow the phonological and morphological patterns of the language."""
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_path = os.path.join(script_dir, 'translation_prompt.txt')
    raw_prompt = PromptManager.load_prompt(prompt_path)
    kwargs = {'phonology': files['phonology'], 'grammar': files['grammar'], 'lexicon_section': lex_section, 'sentences': '\n'.join(sentences)}
    prompt = PromptManager.format_prompt(raw_prompt, **kwargs)
    
    _, content = llm_client.generate_and_extract(prompt, do_sleep=False)
    content = clean_response(content, 'json')
    
    try:
        result = json.loads(content)
        
        if 'new_words' in result and result['new_words']:
            update_lexicon(language_dir, result['new_words'])
        
        translated = result.get('translated_sentences', [])
        return translated
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON response: {e}")
        return []

"""
Translates a Hugging Face Dataset into a new language.

Params:
    dataset: Hugging Face Dataset
    translate_cols: list of components to translate
    languages_dir: directory containing generated languages
    language_id: language to translate into
    batch_size: number of sentences to translate at a time
    output_dir: directory to save translated dataset

Returns:
    Translated Hugging Face Dataset format, saved in output_dir

TODO: translate only a subset of the dataset
"""
def translate_dataset(
    dataset: Dataset,
    translate_cols: list[str],
    languages_dir: str,
    language_id: str,
    batch_size: int = 100,
    output_dir: str = "./"
):
    logger.info(f"Translating {dataset.info.dataset_name} into {language_id}")
    llm_client = init_llm_client()
    translated_dataset = {col: [] if col in translate_cols else dataset[col] for col in dataset.column_names}
    
    for component in translate_cols:
        for i in range(0, len(dataset), batch_size):
            logger.info(f"Translating {component} {i} to {i + batch_size}")
            translated_sentences = run_translation_step(f"{languages_dir}/{language_id}", llm_client, dataset[component][i:i + batch_size])
            translated_dataset[component].extend(translated_sentences)
        
    # Save as HF dataset
    hf_dataset = Dataset.from_dict(translated_dataset)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{dataset.info.dataset_name}_{language_id}")
    hf_dataset.save_to_disk(output_path)
    
    # Save as JSON
    output_path = os.path.join(output_dir, f"{dataset.info.dataset_name}_{language_id}.json")
    with open(output_path, 'w') as f:
        json.dump(translated_dataset, f, indent=4)
        
def main():
    copa = load_dataset("pkavumba/balanced-copa", split="train")
    logger.info(f"Loaded dataset: {copa}")
    translate_dataset(
        dataset=copa,
        translate_cols=["premise", "choice1", "choice2"],
        languages_dir="synthetic/outputs/smoketest_gemini/languages",
        language_id="5afa96bf",
        batch_size=100,
        output_dir="synthetic/data"
    )

if __name__ == "__main__":
    main()

