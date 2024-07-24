from pathlib import Path

import streamlit as st
from langchain_community.llms import OpenAI
import sys
import subprocess
import os
from common.cfg import *

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


def get_openai_api_key():
    openai_api_key = st.sidebar.text_input(
        'OpenAI API Key [Optional]', type='password', value=OPENAI_API_KEY)
    if not openai_api_key.startswith('sk-'):
        st.warning('Please enter your OpenAI API key!', icon='⚠')
    return openai_api_key


st.title('GPT with langchain and streamlit')

@get_data(log=False)
def generate_response(input_text, api_key):
    llm = OpenAI(temperature=0.7, openai_api_key=api_key)
    res = llm(input_text)
    st.info(res)
    return res

def execute():
    openai_api_key = get_openai_api_key()
    with st.form('my_form'):
        text = st.text_area('Enter text:', 'What is GenAI? 3 steps to learn.')
        submitted = st.form_submit_button('Submit')
        if submitted:
            generate_response(text, openai_api_key)


if __name__ == "__main__":
    # Check if the script is being run directly or via subprocess
    if len(sys.argv) > 1 and sys.argv[1] == "run":
        execute()
    else:
        # Re-run the script with Streamlit
        filename = Path(__file__).resolve()
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(filename), "run"])