from datasets import load_dataset
import json
import csv
import os
import random
import sys
import logging
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Get the project root (spice directory)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables from .env file
load_dotenv(os.path.join(PROJECT_ROOT, '.env'))

# Add third_party/conglanger/src to path for imports
CONGLANGER_SRC = os.path.join(PROJECT_ROOT, 'third_party', 'conglanger', 'src')

if CONGLANGER_SRC not in sys.path:
    sys.path.insert(0, CONGLANGER_SRC)

# Now import from the conglanger src directory
from llm_client import LLMClientGemini, LLMClientOpenAI, LLMClientDeepseek, PromptManager
from utils import load_files_with_optional, clean_response

"""
Few shot SLs
Dataset: X SL languages, N datasets (corresponding to tasks), K examples
For each dataset, for each language, select K examples from English Dataset and translate
"""

def init_llm_client(
    model="gemini-2.5-pro",
    max_tokens=8192,
    temperature=0.7,
    thinking_budget=1000,
    reasoning_effort="medium",
    sleep_between_calls=30,
    debug=False,
):
    logger.info(f"Initializing LLM client with model: {model}")
    # Initialize LLM client
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
        # OpenAI client supports both o-series reasoning models and gpt-4o style
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
    
    logger.info(f"LLM client initialized successfully")
    return llm_client

def update_lexicon(language_dir, new_words):
    """Update lexicon.csv with new words"""
    if not new_words:
        return
    
    logger.info(f"Updating lexicon with {len(new_words)} new words")
    
    # Lexicon is in memory/lexicon/ subdirectory
    lexicon_path = os.path.join(language_dir, 'memory', 'lexicon', 'lexicon.csv')
    
    # Read existing lexicon if it exists
    existing_words = set()
    if os.path.exists(lexicon_path):
        with open(lexicon_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            existing_words = {row['word'] for row in reader if 'word' in row}
    
    # Append new words that don't already exist
    with open(lexicon_path, 'a', encoding='utf-8', newline='') as f:
        fieldnames = ['word', 'meaning', 'reason']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        # Write header if file is new or empty
        if not existing_words:
            writer.writeheader()
        
        for word_entry in new_words:
            if word_entry['word'] not in existing_words:
                writer.writerow(word_entry)
                existing_words.add(word_entry['word'])

def update_grammar(language_dir, new_grammar_rules):
    """Append new grammar rules to grammar.txt"""
    if not new_grammar_rules:
        return
    
    logger.info(f"Updating grammar with {len(new_grammar_rules)} new rules")
    
    # Grammar is in memory/grammar/ subdirectory
    grammar_path = os.path.join(language_dir, 'memory', 'grammar', 'grammar.txt')
    
    with open(grammar_path, 'a', encoding='utf-8') as f:
        f.write('\n\n--- Grammar Extensions from Translation ---\n')
        for rule_entry in new_grammar_rules:
            f.write(f"\n{rule_entry['rule']}\n")
            f.write(f"Reason: {rule_entry['reason']}\n")

def run_translation_step(language_dir, llm_client, sentences):
    logger.info(f"Starting translation for {len(sentences)} sentences")
    # Language files are stored in the 'memory' subdirectory
    memory_dir = os.path.join(language_dir, 'memory')
    logger.debug(f"Loading language files from: {memory_dir}")
    required = {'phonology': 'phonology.txt', 'grammar': 'grammar.txt'}
    optional = {'lexicon': 'lexicon.csv'}
    files = load_files_with_optional(memory_dir, required, optional)
    
    if files is None:
        raise FileNotFoundError(f"Required language files not found in {memory_dir}")

    if 'lexicon' in files:
        lex_section = f"""It has the following lexicon:\n\n=== START ===\n{files['lexicon']}\n=== END ==="""
    else:
        lex_section = """Note: No specific lexicon has been provided. You will need to create appropriate vocabulary words that follow the phonological and morphological patterns of the language."""
    
    # Get prompt file path relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_path = os.path.join(script_dir, 'translation_prompt.txt')
    raw_prompt = PromptManager.load_prompt(prompt_path)
    kwargs = {'phonology': files['phonology'], 'grammar': files['grammar'], 'lexicon_section': lex_section, 'sentences': '\n'.join(sentences)}
    prompt = PromptManager.format_prompt(raw_prompt, **kwargs)
    
    logger.info("Calling LLM for translation...")
    _, content = llm_client.generate_and_extract(prompt, do_sleep=False)
    logger.info("LLM response received")
    content = clean_response(content, 'json')
    
    # Parse JSON response
    try:
        result = json.loads(content)
        
        # Update language files with new words and grammar rules
        if 'new_words' in result and result['new_words']:
            update_lexicon(language_dir, result['new_words'])
        
        if 'new_grammar_rules' in result and result['new_grammar_rules']:
            update_grammar(language_dir, result['new_grammar_rules'])
        
        # Return only the translated sentences
        translated = result.get('translated_sentences', [])
        logger.info(f"Successfully translated {len(translated)} sentences")
        return translated
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON response: {e}")
        return []

"""
Params:
    dataset_spec: list of (dataset_name, [translated_components])
    language_dir: directory containing generated languages
    languages: language ids
    k: number of examples to translate per language

Returns:
    Translated Hugging Face Dataset format, saved in output_dir
"""
def translate_datasets(
    dataset_spec: list[tuple[str, list[str]]],
    languages_dir: str,
    languages: list[str],
    k: int = 16,
    output_dir: str = "./"
):
    llm_client = init_llm_client()
    logger.info(f"Processing {len(dataset_spec)} dataset(s)")
    
    for name, translated_components in dataset_spec:
        logger.info(f"\n{'='*60}")
        logger.info(f"Loading dataset: {name}")
        dataset = load_dataset(name, "en")
        
        # Sample indices for all languages
        dataset_size = len(dataset['train'])
        logger.info(f"Dataset size: {dataset_size} examples")
        logger.info(f"Sampling {k} examples per language for {len(languages)} language(s)")
        sample_indices = random.sample(range(dataset_size), k * len(languages))
        
        # Initialize combined dataset with all columns from original
        combined_data = {col: [] for col in dataset['train'].features}
        combined_data['language'] = []  # Add language column
        
        # Process each language
        for i, language_id in enumerate(languages):
            logger.info(f"\n--- Processing language {i+1}/{len(languages)}: {language_id} ---")
            language_sample_indices = sample_indices[i * k:(i + 1) * k]
            language_dir = os.path.join(languages_dir, language_id)
            logger.info(f"Language directory: {language_dir}")
            
            # Translate each component for this language
            language_translations = {}
            logger.info(f"Translating {len(translated_components)} component(s): {translated_components}")
            for component in translated_components:
                logger.info(f"  Translating component: '{component}'")
                sentences = [dataset['train'][j][component] for j in language_sample_indices]
                try:
                    translated_sentences = run_translation_step(language_dir, llm_client, sentences)
                    language_translations[component] = translated_sentences
                except FileNotFoundError as e:
                    print(f"Error: {e}")
                    raise
            
            # Add rows for this language to combined dataset
            logger.info(f"Adding {len(language_sample_indices)} rows to combined dataset")
            for idx, sample_idx in enumerate(language_sample_indices):
                # Copy all original columns
                for col in dataset['train'].features:
                    if col in translated_components:
                        # Use translated version
                        combined_data[col].append(language_translations[col][idx])
                    else:
                        # Keep original value
                        combined_data[col].append(dataset['train'][sample_idx][col])
                
                # Add language ID
                combined_data['language'].append(language_id)
        
        # Save combined dataset
        from datasets import Dataset
        logger.info(f"\n{'='*60}")
        logger.info(f"Creating final dataset with {len(combined_data['language'])} total rows")
        hf_dataset = Dataset.from_dict(combined_data)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{name}_translated")
        logger.info(f"Saving dataset to: {output_path}")
        hf_dataset.save_to_disk(output_path)
        logger.info(f"✓ Successfully saved translated dataset!")
        logger.info(f"  Total rows: {len(combined_data['language'])} ({k} per language × {len(languages)} languages)")
        logger.info(f"  Columns: {list(combined_data.keys())}")
        
def main():
    logger.info("Starting translation pipeline...")
    translate_datasets(
        dataset_spec=[("xnli", ["premise", "hypothesis"])],
        languages_dir="synthetic/outputs/smoketest_gemini/languages",
        languages=["f36621ea"],
        k=16,
        output_dir="synthetic/data"
    )

if __name__ == "__main__":
    main()

