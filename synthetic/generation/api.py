import os
from dotenv import load_dotenv

from synthetic.utils import get_new_target_language_id

# Load environment variables from .env file
load_dotenv()

from synthetic.conglanger import run_conglanger
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader

# Target languages available for synthetic language generation
TARGET_LANGUAGES = [
    "arabic",
    "bulgarian",
    "german",
    "greek",
    "english",
    "spanish",
    "french",
    "hindi",
    "russian",
    "swahili",
    "thai",
    "turkish",
    "urdu",
    "vietnamese",
    "chinese",
]

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
        ds, batch_size=2, shuffle=True, collate_fn=lambda batch: "\n".join(batch)
    )
    return loader

def apply_step_for_random(step, lang_id, average_hamming_dist, num_in_group, iteration=-1, batch=None):
    return run_conglanger(
        lang_name=f"f{average_hamming_dist}_{num_in_group}",
        run_name='random',
        random=True,
        iteration=iteration,
        lang_id=lang_id,
        steps=(step,),
        translation_sentence=batch,
    )

def apply_step_for_target(step, lang_id, target_lang, iteration=-1, batch=None):
    return run_conglanger(
        lang_name=target_lang,
        run_name=target_lang,
        random=False,
        iteration=iteration,
        lang_id=lang_id,
        steps=(step,),
        translation_sentence=batch,
    )
    
def translate_dataset_using_random(corpus, lang_id, average_hamming_dist, num_in_group, num_batches=None):
    results = []
    dataset = corpus if corpus is not None else get_snli_batches()
    for batch_idx, batch in enumerate(dataset, 1):
        if num_batches is not None and batch_idx > num_batches:
            break
        print(f"Translating (batch {batch_idx}): {batch[:20]}... for {lang_id}")
        result = apply_step_for_random(
            step="translation",
            lang_id=lang_id,
            average_hamming_dist=average_hamming_dist,
            num_in_group=num_in_group,
            iteration=batch_idx - 1,
            batch=batch,
        )
        results.append(result)
    return results

def translate_dataset_for_target(corpus, lang_id, target_lang, num_batches=None):
    results = []
    dataset = corpus if corpus is not None else get_snli_batches()
    for batch_idx, batch in enumerate(dataset, 1):
        if num_batches is not None and batch_idx > num_batches:
            break
        print(f"Translating (batch {batch_idx}): {batch[:20]}... for{lang_id}")
        result = apply_step_for_target(
            step="translation",
            lang_id=lang_id,
            target_lang=target_lang,
            iteration=batch_idx - 1,
            batch=batch,
        )
        results.append(result)
    return results

if __name__ == "__main__":
    corpus = None  # Lazily load SNLI only if translation is requested

    lang_id = get_new_target_language_id("arabic")
    # lang_id = get_new_random_language_id(average_hamming_dist="low", num_in_group=0)
    
    # create affixes
    # for lang in TARGET_LANGUAGES:
    # apply_step_for_target(
    #     step="affix",
    #     lang_id=lang_id,
    #     target_lang="arabic",
    # )
    # apply_step_for_random(
    #     step="affix",
    #     lang_id=lang_id,
    #     average_hamming_dist="low",
    #     num_in_group=0,
    # )
    
    # create lexicon
    # apply_step_for_target(
    #     step="lexicon",
    #     lang_id=lang_id,
    #     target_lang="arabic",
    # )
    # apply_step_for_random(
    #     step="lexicon",
    #     lang_id=lang_id,
    #     average_hamming_dist="low",
    #     num_in_group=0,
    # )

    # translate corpus
    translate_dataset_for_target(
        corpus=corpus,
        lang_id=lang_id,
        target_lang="arabic",
        num_batches=1,
    )
    # translate_dataset_using_random(
    #     corpus=corpus,
    #     lang_id=lang_id,
    #     average_hamming_dist="low",
    #     num_in_group=0,
    #     num_batches=2,
    # )
    
    # TODO: check affix structure
