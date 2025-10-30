from synthetic.generation.conglanger_client import run_conglanger
from synthetic.utils.logger import log_info

def generate_language(**kwargs):
    log_info("Generating synthetic language...")
    result = run_conglanger(**kwargs)
    log_info("Language generation completed.")
    return result
