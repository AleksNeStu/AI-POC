import os
import pickle
from functools import wraps
from pathlib import Path

import torch
from dotenv import find_dotenv, load_dotenv
import inspect
import openai


# Load environment variables
env_file = find_dotenv()
load_dotenv(env_file, override=True)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
COHERE_API_KEY = os.environ.get("COHERE_API_KEY")
HUGGINGFACEHUB_API_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
SERPAPI_API_KEY = os.environ.get("SERPAPI_API_KEY")
LANGCHAIN_API_KEY = os.environ.get("LANGCHAIN_API_KEY")

openai.api_key = OPENAI_API_KEY


# Set paths and cfg parts
current_dir = Path(__file__).parent.resolve()
root_dir = Path(__file__).parent.parent.resolve()
md_dir = root_dir / "data/docs/md"
txt_dir = root_dir / "data/docs/txt"
txt1_path = txt_dir / "txt_example1.txt"
pdf_dir = root_dir / "data/docs/pdf/"
pdf_ai_report_dir = root_dir/ "data/pdf_ai_report"
pdf_1_path = pdf_dir / "MachineLearning-Lecture01.pdf"
pdf_2_path = pdf_dir / "MachineLearning-Lecture02.pdf"
pdf_3_path = pdf_dir / "MachineLearning-Lecture03.pdf"
pdf_langchain = pdf_dir / "langchain.pdf"


device = "cuda" if torch.cuda.is_available() else "cpu"
# You can set compute_type to "float32" or "int8".
# Since your GPU does not support float16, you should set "int8" and not "int8_float16".
compute_type = "float16" if device == "cuda" else "int8"



# Interactive or not
def get_is_interactive():
    try:
        __file__
        return False
    except NameError:
        return True
is_interactive = get_is_interactive()


# Helpers
def dump_data(data_obj: object, data_path: Path):
    with open(data_path, "wb") as f:
        pickle.dump(data_obj, f)


def load_data(data_path: Path):
    with open(data_path, "rb") as f:
        res = pickle.load(f)
    return res


def get_func_data_path(func, current_dir):
    f_name = get_func_data_path.__name__
    data_path = current_dir / f'{f_name}.pkl'
    return data_path



def get_data(file = None, load = True, *args, **kwargs):
    caller_file = file if file else inspect.stack()[1].filename

    def decorator(func):

        def get_data_path(func):
            current_dir = Path(caller_file).parent.resolve()
            func_name = func.__name__
            data_path = current_dir / f'{func_name}.pkl'
            return data_path

        data_path = get_data_path(func)

        @wraps(func)
        def wrapper(*args, **kwargs):
            if data_path.exists() and load:
                res = load_data(data_path)
            else:
                res = func(*args, *kwargs)
                dump_data(res, data_path)
            return res

        return wrapper

    return decorator