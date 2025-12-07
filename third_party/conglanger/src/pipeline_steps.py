import logging
import os
import json
from llm_client import PromptManager
from tqdm.auto import tqdm
from cleanup import extract_new_vocabulary
from utils import (
    check_new_word_conflicts, clean_response, alphabetize_csv_text, get_csv_text_n_entries, get_specification_value, load_affix, load_feature_vector, load_lexicon, load_orthography_rules, load_required_files,
    save_memory
)

logger = logging.getLogger(__name__)


# ===================== QA SUPPORT ===================== #

def save_with_qa(args, llm_client, content, step_name, filename, metadata,
                 context=None, context_type=None):
    """Run QA and save content with QA results embedded in metadata."""
    original_content = content
    qa_passed, qa_data, final_content = run_qa_step(
        args, llm_client, step_name, content, step_name, context=context, context_type=context_type
    )

    if final_content != content:
        logger.info(f"Using amended {step_name} content from QA")
        content = final_content

    if not qa_passed:
        logger.warning(f"{step_name} QA failed (continuing). Issues: {qa_data.get('issues') if qa_data else 'N/A'}")

    qa_metadata = qa_data if qa_data else {}
    qa_metadata['content_before_qa'] = original_content
    qa_metadata['content_after_qa'] = final_content
    qa_metadata['content_changed'] = original_content != final_content
        
    metadata['qa_results'] = qa_metadata

    step_memory_dir = os.path.join(args.memory_dir, step_name)
    if args.iteration >= 0:
        # translation only no need for subfolder
        step_memory_dir = os.path.join(args.memory_dir, f"iter_{args.iteration}", step_name)
    
    os.makedirs(step_memory_dir, exist_ok=True)
    save_memory(content, step_memory_dir, filename, metadata)

    if qa_data:
        qa_filename = f"{step_name}_qa.json"
        qa_filepath = os.path.join(step_memory_dir, qa_filename)
        qa_data_with_content = qa_metadata.copy()

        if args.continue_qa and os.path.exists(qa_filepath):
            try:
                with open(qa_filepath, 'r', encoding='utf-8') as f:
                    existing = json.load(f)
                if 'all_iterations' in existing and 'all_iterations' in qa_data:
                    max_iter = max((it.get('iteration', 0) for it in existing['all_iterations']), default=0)
                    for it in qa_data['all_iterations']:
                        it['iteration'] += max_iter
                    existing['all_iterations'].extend(qa_data['all_iterations'])
                    existing['final_qa'] = qa_data.get('final_qa')
                    existing['continue_qa_run'] = True
                    existing.update({
                        'content_before_qa': original_content,
                        'content_after_qa': final_content,
                        'content_changed': original_content != final_content,
                    })
                    qa_data_with_content = existing
            except Exception as e:
                logger.warning(f"Failed extending QA file, recreating: {e}")

        save_memory(json.dumps(qa_data_with_content, indent=2, ensure_ascii=False), step_memory_dir, qa_filename, {})

    return content

def _get_lexicon_conflicts(args):
    new_words = extract_new_vocabulary(args.lang_dir, args.iteration)
    if new_words:
        return check_new_word_conflicts(new_words, args)
    return None

def run_qa_step(args, llm_client, step_name, content, content_type="translation", context=None, context_type=None):
    """Run QA critic/amend loop for a content artifact."""
    if context and not context_type:
        raise ValueError("Missing context_type")
    if context_type and not context:
        raise ValueError("Missing context")
    has_context = context is not None

    if not getattr(args, 'qa_enabled', False): 
        return True, None, content

    prompt_dir = os.path.join(args.prompt_dir, "qa")
    critic = PromptManager.load_prompt(
        os.path.join(prompt_dir, f"qa_critic_{content_type}.txt")
    )
    amend = PromptManager.load_prompt(
        os.path.join(prompt_dir, f"qa_amend_{content_type}.txt")
    )

    current = content
    all_iters = []
    final_qa = None
    max_iters = getattr(args, 'self_refine_steps', 3)

    for i in range(max_iters):
        conflicts = None
        if has_context:
            # For translation, check for new word conflicts and prepare for QA prompt
            if content_type == "translation":
                conflicts = _get_lexicon_conflicts(args)
                conflicts_input = json.dumps(conflicts, indent=2, ensure_ascii=False) if conflicts else "none"
                qa_prompt = PromptManager.format_prompt(critic, content=current, content_type=content_type, context=context, input_sentences=content, lexicon_conflicts=conflicts_input)
            else:
                qa_prompt = PromptManager.format_prompt(critic, content=current, content_type=content_type, context=context, context_type=context_type)
        else:
            qa_prompt = PromptManager.format_prompt(critic, content=current, content_type=content_type)
        logger.info(f"QA prompt ({step_name} iter {i+1}): {qa_prompt}")   
        qa_raw, _ = generate_and_parse_json_with_retries(llm_client, qa_prompt, max_retries=3, do_sleep=False)
        try:
            qa_data = json.loads(qa_raw)
            final_qa = qa_data
            overall = qa_data.get('overall_score', 0)
            iter_record = {'iteration': i+1, 'qa_data': qa_data, 'content_length': len(current), 'amended': False, 'num_conflicts': 0}
            if conflicts:
                iter_record['num_conflicts'] = len(conflicts)
                qa_data['conflicts'] = conflicts
            # Determine threshold
            if getattr(args, 'qa_threshold', None) is not None:
                threshold = args.qa_threshold
                logger.info(f"QA step {step_name} iteration {i+1}: score={overall}, threshold={threshold}, conflicts={bool(conflicts)}")
            else:
                if step_name == 'translation':
                    threshold = args.qa_threshold_translation
                elif step_name == 'lexicon':
                    threshold = args.qa_threshold_lexicon
                elif step_name == "affix":
                    threshold = args.qa_threshold_affix
                else:
                    threshold = 8.0
            if overall >= threshold and not conflicts:
                all_iters.append(iter_record)
                return True, {'final_qa': qa_data, 'all_iterations': all_iters}, current
            if i < max_iters - 1:
                iter_record['amended'] = True
                all_iters.append(iter_record)
                amend_prompt = PromptManager.format_prompt(amend, content=current, judgement=qa_raw)
                current, _ = generate_and_parse_json_with_retries(llm_client, amend_prompt, max_retries=3, do_sleep=False)
            else:
                all_iters.append(iter_record)
        except json.JSONDecodeError:
            iter_record = {'iteration': i+1, 'qa_data': None, 'error': 'json_parse_failed', 'raw_response': qa_raw, 'amended': False}
            all_iters.append(iter_record)
            if i == max_iters - 1:
                return False, {'final_qa': None, 'all_iterations': all_iters}, current

    return False, {'final_qa': final_qa, 'all_iterations': all_iters}, current


# ===================== GENERATION HELPERS ===================== #

def _generate_with_prompts(llm_client, prompts, kwargs_list, do_sleep_flags=None):
    if do_sleep_flags is None:
        do_sleep_flags = [True] * len(prompts)
    responses = []
    for i, (prompt_key, kwargs) in enumerate(zip(prompts.keys(), kwargs_list)):
        prompt = PromptManager.format_prompt(prompts[prompt_key], **kwargs)
        logger.info(f"Prompt {i+1}: {prompt}")
        full_response, extracted = llm_client.generate_and_extract(
            prompt, do_sleep=do_sleep_flags[i] if i < len(do_sleep_flags) else False
        )
        responses.append((full_response, extracted))
    return responses

def generate_and_parse_json_with_retries(llm_client, prompt, max_retries=3, do_sleep=False):
    """Generate LLM output, clean, and parse as JSON, retrying up to max_retries if parsing fails."""
    last_error = None
    for attempt in range(1, max_retries + 1):
        _, extracted = llm_client.generate_and_extract(prompt, do_sleep=do_sleep)
        logger.info(f"LLM response (attempt {attempt}): {extracted}")
        cleaned = clean_response(extracted, 'json')
        try:
            parsed = json.loads(cleaned)
            return cleaned, parsed
        except json.JSONDecodeError as e:
            last_error = e
            if logger:
                logger.warning(f"Attempt {attempt}: Failed to parse LLM response as JSON. Retrying... Error: {e}\nResponse: {cleaned}")

    raise RuntimeError(f"Failed to parse JSON from LLM after {max_retries} attempts. Last error: {last_error}")


def _csv_to_text_for_qa(csv_data: str) -> str:
    """Turn CSV into numbered plain-text list for QA readability."""
    lines = csv_data.strip().split("\n")
    if not lines:
        return ""
    header = lines[0]
    out = [f"Lexicon entries (format: {header}):"]
    for idx, line in enumerate(lines[1:], 1):
        if line.strip():
            out.append(f"{idx}. {line}")
    return "\n".join(out)


def _text_to_csv_for_qa(text_content: str) -> str:
    """Convert QA-ed text version back to CSV."""
    lines = text_content.strip().split("\n")
    header = None
    entries = []

    for line in lines:
        l = line.strip()
        if not l:
            continue

        # header line like: "Lexicon entries (format: form,pos,translation):"
        if l.lower().startswith("lexicon entries (format:"):
            try:
                fmt_part = l.split("format:", 1)[1]
                fmt_part = fmt_part.rstrip("):").strip()
                header = fmt_part
            except Exception:
                pass
            continue

        # Strip numbering "1. ..."
        if "." in l and l.split(".", 1)[0].isdigit():
            l = l.split(".", 1)[1].strip()

        # Any line with commas is treated as CSV row
        if "," in l and not l.lower().startswith("format:"):
            entries.append(l)

    # Fallback header if QA removed it
    if header is None and entries:
        col_count = len(entries[0].split(","))
        if col_count == 3:
            header = "form,pos,translation"
        else:
            # In case something weird happens
            header = ",".join([f"col{i}" for i in range(col_count)])

    if header is not None:
        csv_lines = [header] + entries
    else:
        csv_lines = entries

    return "\n".join(csv_lines)
    
def run_affix_step(args, llm_client) -> str:
    """
    Generate affix inventory JSON for this language and save to:
      memory_dir/affix/affix.json

    Returns the JSON text (string).
    """
    logger.info("Lang name", args.lang_name)
    orthography = load_orthography_rules(args.prompt_dir, args.lang_name, args.random)
    feature_vector = load_feature_vector(args.prompt_dir, args.lang_name, args.random)
    fv_text = json.dumps(feature_vector, indent=2, ensure_ascii=False)

    affix_prompt_dir = os.path.join(args.prompt_dir, "affix")
    affix_prompt = PromptManager.load_prompt(
        os.path.join(affix_prompt_dir, "affix_singular.txt")
    )
    affix_filled = PromptManager.format_prompt(
        affix_prompt,
        orthography=orthography,
        feature_vector=fv_text,
    )
    logger.info(f"Affix prompt: {affix_filled}")
    
    # TODO: use this to get the prompt to paste into chat gpt (see in prompts/affix/<all_target>)
    # change to <all_random_<low|medium|high>> for random languages
    try:
        target_file = os.path.join(args.prompt_dir, "affix", "all_random.txt")
        with open(target_file, "a", encoding="utf-8") as f:
            f.write(f"===={args.lang_name} START====\n{affix_filled}\n===={args.lang_name} END====\n\n")
        logger.info(f"Appended affix prompt to {target_file}")
    except Exception as e:
        logger.warning(f"Could not append to all_random.txt: {e}")

    # affix_clean, affix_parsed = generate_and_parse_json_with_retries(
    #     llm_client, affix_filled, max_retries=3, do_sleep=False
    # )
    # if affix_parsed is None:
    #     logger.error("Affix generation step failed: LLM did not return valid JSON.")
    #     raise RuntimeError("Affix generation failed")

    # # QA for affix
    # context = (
    #     f"ORTHOGRAPHY:\n{orthography}\n\n"
    #     f"FEATURE_VECTOR:\n{fv_text}"
    # )
    # metadata = {"feature_vector": feature_vector}

    # save_with_qa(
    #     args,
    #     llm_client,
    #     affix_clean,
    #     "affix",
    #     "affix.json",
    #     metadata,
    #     context=context,
    #     context_type="language_spec",
    # )
    # return True

def run_lexicon_step(args, llm_client):
    """
    Lexicon step:

    - Ensures affix.json exists (calling run_affix_step if necessary).
    - Generates lexicon CSV using lex_singular_step.txt.
    - Runs QA (qa_critic_singular / qa_amend_singular).
    """
    orthography = load_orthography_rules(args.prompt_dir, args.lang_name, args.random)
    feature_vector = load_feature_vector(args.prompt_dir, args.lang_name, args.random)
    fv_text = json.dumps(feature_vector, indent=2, ensure_ascii=False)

    # 1) Ensure affix inventory exists (call affix step)
    try:
        affix_text = load_affix(args.prompt_dir, args.lang_name, args.random)
        affix_text = json.dumps(affix_text, indent=2, ensure_ascii=False)
        logger.info("Found existing affix inventory; reusing it.")
    except FileNotFoundError:
        raise RuntimeError("Affix inventory not found; please run affix step before lexicon step.")

    # 2) Generate lexicon CSV
    lex_prompt_dir = os.path.join(args.prompt_dir, "lexicon")
    lex_prompt = PromptManager.load_prompt(
        os.path.join(lex_prompt_dir, "lex_singular.txt")
    )

    lex_filled = PromptManager.format_prompt(
        lex_prompt,
        orthography=orthography,
        feature_vector=fv_text,
        affixes_json=affix_text,
    )
    
    # TODO: use this to get the prompt to paste into chat gpt (see in prompts/lexicon/<all_target>)
    # change to <all_random_<low|medium|high>> for random languages
    try:
        target_file = os.path.join(args.prompt_dir, "lexicon", "all_random.txt")
        with open(target_file, "a", encoding="utf-8") as f:
            f.write(f"===={args.lang_name} START====\n{lex_filled}\n===={args.lang_name} END====\n\n")
        logger.info(f"Appended affix prompt to {target_file}")
    except Exception as e:
        logger.warning(f"Could not append to all_random.txt: {e}")

    # _, extracted = llm_client.generate_and_extract(lex_filled, do_sleep=False)
    # csv_data = clean_response(extracted, "csv")
    # csv_data = alphabetize_csv_text(csv_data)
    # entries = get_csv_text_n_entries(csv_data)
    # logger.info(f"Generated lexicon with ~{entries} entries")

    # # 3) QA for lexicon
    # text_for_qa = _csv_to_text_for_qa(csv_data)
    # context = (
    #     f"ORTHOGRAPHY:\n{orthography}\n\n"
    #     f"FEATURE_VECTOR:\n{fv_text}\n\n"
    #     f"affix:\n{affix_text}"
    # )
    # metadata = {
    #     "feature_vector": feature_vector,
    #     "lexicon_entries": entries,
    # }

    # final_text = save_with_qa(
    #     args,
    #     llm_client,
    #     text_for_qa,
    #     "lexicon",
    #     "lexicon.csv",
    #     metadata,
    #     context=context,
    #     context_type="language_spec",
    # )
    # final_csv = _text_to_csv_for_qa(final_text)

    # with open(os.path.join(lex_dir, "lexicon.csv"), "w", encoding="utf-8") as f:
    #     f.write(final_csv)

    return True



# ===================== TRANSLATION (single) ===================== #

def run_translation_step(args, llm_client):
    """
    Translation step (single batch, possibly with iteration index):

    - Load orthography, feature_vector, affix, lexicon
    - Optionally shrink lexicon via lex_extraction.txt
    - Run translation_singular.txt
    - Run QA
    - Save translation.json under memory_dir/iter_N/translation
    """
    trans_dir = os.path.join(args.memory_dir, f"iter_{args.iteration}", "translation")
    os.makedirs(trans_dir, exist_ok=True)
    
    orthography = load_orthography_rules(args.prompt_dir, args.lang_name, args.random)
    logger.info(f"Loaded orthography rules:\n{orthography}")
    feature_vector = load_feature_vector(args.prompt_dir, args.lang_name, args.random)
    logger.info(f"Loaded feature vector:\n{json.dumps(feature_vector, indent=2, ensure_ascii=False)}")
    fv_text = json.dumps(feature_vector, indent=2, ensure_ascii=False)
    affix_text = load_affix(args.prompt_dir, args.lang_name, args.random)
    affix_text = json.dumps(affix_text, indent=2, ensure_ascii=False)
    logger.info(f"Loaded affix inventory:\n{affix_text}")
    lexicon_path = os.path.join(args.memory_dir, "lexicon", "lexicon.csv")
    
    # Load lexicon from base_specifications if it doesn't exist yet
    if not os.path.exists(lexicon_path):
        load_lexicon(args)
    
    # Read lexicon from memory_dir
    with open(lexicon_path, "r", encoding="utf-8") as f:
        lexicon_csv = f.read()

    input_sentences = getattr(args, "input_sentences", "")
    if not input_sentences:
        logger.error("No input_sentences provided on args.")
        return False

    translation_prompt_dir = os.path.join(args.prompt_dir, "translation")

    # 1) Optional lexicon extraction (subset lexicon for these sentences)
    try:
        lex_extract_prompt = PromptManager.load_prompt(
            os.path.join(translation_prompt_dir, "lex_extraction.txt")
        )
        lex_extract_filled = PromptManager.format_prompt(
            lex_extract_prompt,
            lexicon_csv=lexicon_csv,
            input_sentences=input_sentences,
        )
        _, lex_extract_out = llm_client.generate_and_extract(
            lex_extract_filled, do_sleep=False
        )
        required_lex_csv = clean_response(lex_extract_out, "csv")

        # sanity: header present?
        if not required_lex_csv.strip():
            logger.warning("lex_extraction returned empty; falling back to full lexicon.")
            required_lex_csv = lexicon_csv
    except FileNotFoundError:
        logger.warning(
            "lex_extraction.txt not found; passing full lexicon to translation."
        )
        required_lex_csv = lexicon_csv

    # Save required lexicon at iteration level
    iter_dir = os.path.join(args.memory_dir, f"iter_{args.iteration}")
    os.makedirs(iter_dir, exist_ok=True)
    req_lex_path = os.path.join(iter_dir, "required_lexicon.csv")
    with open(req_lex_path, "w", encoding="utf-8") as f:
        f.write(required_lex_csv)
    logger.info(f"Saved required lexicon to {req_lex_path}")

    # 2) Translation prompt
    translation_prompt = PromptManager.load_prompt(
        os.path.join(translation_prompt_dir, "translation_singular.txt")
    )
    trans_filled = PromptManager.format_prompt(
        translation_prompt,
        orthography=orthography,
        feature_vector=fv_text,
        affixes_json=affix_text,
        lexicon_csv=required_lex_csv,
        input_sentences=input_sentences,
    )

    content, parsed_json = generate_and_parse_json_with_retries(
        llm_client, trans_filled, max_retries=3, do_sleep=False
    )
    if parsed_json is None:
        logger.error(
            "Translation step failed: LLM did not return valid JSON after 3 attempts."
        )
        return False

    # Write translation.json to trans_dir
    translation_json_path = os.path.join(trans_dir, "translation.json")
    with open(translation_json_path, "w", encoding="utf-8") as f:
        f.write(content)
    logger.info(f"Saved translation to {translation_json_path}")

    # 3) QA for translation
    context = (
        f"The language uses this orthography:\n\n"
        f"=== ORTHOGRAPHY START ===\n{orthography}\n=== ORTHOGRAPHY END ===\n\n"
        f"It has this typological profile:\n\n"
        f"=== FEATURE VECTOR START ===\n{fv_text}\n=== FEATURE VECTOR END ===\n\n"
        f"Its affix system is:\n\n"
        f"=== AFFIXES START ===\n{affix_text}\n=== AFFIXES END ===\n\n"
        f"Its lexicon is:\n\n"
        f"=== LEXICON START ===\n{required_lex_csv}\n=== LEXICON END ==="
    )
    
    metadata = {
        "input_sentences": input_sentences,
        "lexicon_provided": True,
        "iteration": args.iteration,
    }

    save_with_qa(
        args,
        llm_client,
        content,
        "translation",
        "translation.json",
        metadata,
        context=context,
        context_type="language_spec",
    )

    return True
