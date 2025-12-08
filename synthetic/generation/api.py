from dotenv import load_dotenv

from synthetic.utils import get_latest_random_id, get_latest_random_iteration, get_latest_target_id, get_latest_target_iteration, get_new_random_id, get_new_target_id

# Load environment variables from .env file
load_dotenv()

from synthetic.conglanger import run_conglanger
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader

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


def get_xnli_batches():
    xnli = load_dataset("xnli", "en", split="train")
    ds = [xnli[i//2]["premise"] if i % 2 == 0 else xnli[i//2]["hypothesis"] for i in range(len(xnli) * 2)]
    loader = DataLoader(
        ds, batch_size=10, shuffle=False, collate_fn=lambda batch: "\n".join(batch)
    )
    return loader

def apply_step_for_random(step, lang_id, average_hamming_dist, num_in_group, iteration=-1, batch=None):
    return run_conglanger(
        lang_name=f"{average_hamming_dist}_{num_in_group}",
        run_name=f'random/{average_hamming_dist}_{num_in_group}',
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
    
def translate_dataset_using_random(corpus, lang_id, average_hamming_dist, num_in_group, num_batches=None, iteration=0):
    results = []
    dataset = corpus if corpus is not None else get_xnli_batches()
    for batch_idx, batch in enumerate(dataset, iteration):
        if num_batches is not None and batch_idx > num_batches:
            break
        print(f"Translating (batch {batch_idx}): {batch[:20]}... for {lang_id}")
        result = apply_step_for_random(
            step="translation",
            lang_id=lang_id,
            average_hamming_dist=average_hamming_dist,
            num_in_group=num_in_group,
            iteration=batch_idx,
            batch=batch,
        )
        results.append(result)
    return results

def translate_dataset_for_target(corpus, lang_id, target_lang, num_batches=None, iteration=0):
    results = []
    dataset = corpus if corpus is not None else get_xnli_batches()
    for batch_idx, batch in enumerate(dataset, iteration):
        print(f"num_batches and batch_idx: {num_batches}, {batch_idx}")
        if num_batches is not None and batch_idx > num_batches:
            break
        print(f"Translating (batch {batch_idx}): {batch[:20]}... for{lang_id}")
        result = apply_step_for_target(
            step="translation",
            lang_id=lang_id,
            target_lang=target_lang,
            iteration=batch_idx,
            batch=batch,
        )
        results.append(result)
    return results

if __name__ == "__main__":
    corpus = get_xnli_batches() 
    
    # start a new attempt for translating a language
    # lang_id = get_new_target_id("swahili")
    lang_id = get_new_random_id("low", 0)
    
    # continue translating last attempt for a language after the last translation iteration (when it stopped)
    # lang_id = get_latest_target_id("swahili") 
    # iteration = get_latest_target_iteration("swahili", lang_id) + 1 # plug in this value below to start on next iteration
    # lang_id = get_latest_random_id("low", 0)
    iteration = 0
    # iteration = get_latest_random_iteration("low", 0, lang_id) + 1
    
    # translate corpus
    # translate_dataset_for_target(
    #     corpus=corpus,
    #     lang_id=lang_id,
    #     target_lang="swahili",
    #     num_batches=None, # runs all batches
    #     iteration=iteration,
    # )
    translate_dataset_using_random(
        corpus=corpus,
        lang_id=lang_id,
        average_hamming_dist="low",
        num_in_group=0,
        num_batches=None,
        iteration=iteration,
    )
    

    # IGNORE BUT LEAVE BELOW
    
    # create affixes
    # apply_step_for_target(
    #     step="affix",
    #     lang_id=lang_id,
    #     target_lang="urdu",
    # )
    # for div, i in [("low", 0), ("low", 1), ("high", 0), ("high", 1)]:
    #     lang_id = get_new_random_id(div, i)
    #     apply_step_for_random(
    #         step="lexicon",
    #         lang_id=lang_id,
    #         average_hamming_dist=div,
    #         num_in_group=i,
    #     )
    
    # create lexicon
    # apply_step_for_target(
    #     step="lexicon",
    #     lang_id=lang_id,
    #     target_lang="urdu",
    # )
    # apply_step_for_random(
    #     step="lexicon",
    #     lang_id=lang_id,
    #     average_hamming_dist="low",
    #     num_in_group=0,
    # )
