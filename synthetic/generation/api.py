from synthetic.generation.conglanger_client import run_conglanger

def generate_language(**kwargs):
    result = run_conglanger(**kwargs)
    return result