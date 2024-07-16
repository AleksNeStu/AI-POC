import os
import pickle
from pathlib import Path

import torch
from dotenv import find_dotenv, load_dotenv

env_file = find_dotenv()
load_dotenv(env_file, override=True)


device = "cuda" if torch.cuda.is_available() else "cpu"
# You can set compute_type to "float32" or "int8".
# Since your GPU does not support float16, you should set "int8" and not "int8_float16".
compute_type = "float16" if device == "cuda" else "int8"


OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
COHERE_API_KEY = os.environ.get("COHERE_API_KEY")
HUGGINGFACEHUB_API_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
SERPAPI_API_KEY = os.environ.get("SERPAPI_API_KEY")
LANGCHAIN_API_KEY = os.environ.get("LANGCHAIN_API_KEY")


def get_is_interactive():
    try:
        __file__
        return False
    except NameError:
        return True

is_interactive = get_is_interactive()


def dump_data(data_obj: object, data_path: Path):
    with open(data_path, "wb") as f:
        pickle.dump(data_obj, f)


def load_data(data_path: Path):
    with open(data_path, "rb") as f:
        res = pickle.load(f)
    return res


def get_func_meta(func, current_dir):
    f_name = get_func_meta.__name__
    data_path = current_dir / f'{f_name}.pkl'