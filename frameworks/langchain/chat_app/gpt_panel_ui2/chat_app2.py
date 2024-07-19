import sys
from pathlib import Path

from langchain_text_splitters import CharacterTextSplitter

root_dir = Path(__file__).parent.resolve().parent.resolve().parent.resolve().parent.resolve().parent.resolve()
sys.path.insert(0, str(root_dir))
from common.cfg import *


import os
import panel as pn
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.llms import OpenAI
from langchain_community.vectorstores import Chroma

QN = "What is the total number of AI publications between 2010 and 2022?"
convos = []  # store all panel objects in a list

pn.extension('texteditor', template="bootstrap", sizing_mode='stretch_width')
pn.state.template.param.update(
    main_max_width="690px",
    header_background="#F08080",
)


# Row 2x1
# The pn.widgets.FileInput() widget in the Panel library doesn't provide a path to the uploaded file. This is because the file data is sent directly from the client's browser to your Python process without being saved in a temporary location.
# Web browsers (and thus web-based libraries like Panel) are designed to not have direct access to the file system
# for security reasons. This is why when you use a file upload widget, you get the file's content and name, but not the file's original path on the client's machine.
file_input_pdf = pn.widgets.FileInput(accept='.pdf', filename='HAI_AI-Index-Report-2024_Chapter1.pdf')
row_input = pn.Row(file_input_pdf, styles=dict(background='WhiteSmoke'), width=500)
text_editor_prompt = pn.widgets.TextEditor(
    value=QN, placeholder=QN, height=300, toolbar=False
)
button_run = pn.widgets.Button(name="Run")
row_2_1_prompt = pn.Column(row_input, text_editor_prompt, button_run, width=500)

# Row 2x2
pass_input_api_key = pn.widgets.PasswordInput(
    value="", placeholder="OPENAI_API_KEY"
)
rb_search_type = pn.widgets.RadioButtonGroup(
    name="Search type:", options=[
        "similarity", "similarity_score_threshold", "mmr"],
    value='similarity'
)
rb_chain_type = pn.widgets.RadioButtonGroup(
    name='Chain type:',
    options=['stuff', 'map_reduce', "refine", "map_rerank"],
    # button_type='success',
    value='map_reduce'
)
int_slider_num_chunks = pn.widgets.IntSlider(
    name="Number of relevant chunks", start=1, end=5, step=1, value=2
)
row2_1_settings_card = pn.Card(
    pn.Column(
        "Optional env var:",
        pass_input_api_key,
        "Search type:",
        rb_search_type,
        "Chain type:",
        rb_chain_type,
        int_slider_num_chunks
    ),
    title="Advanced settings",
    width=200,
    # background='LightGray'
)



def qa(file, query, chain_type, search_type, k):
    # load document
    # https://python.langchain.com/v0.2/docs/integrations/document_loaders/
    loader = PyPDFLoader(file)
    documents = loader.load()
    # split the documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    # select which embeddings we want to use
    embeddings = OpenAIEmbeddings()
    # create the vectorestore to use as the index
    db = Chroma.from_documents(texts, embeddings)
    # expose this index in a retriever interface
    retriever = db.as_retriever(search_type=search_type, search_kwargs={"k": k})
    # create a chain to answer questions 
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(), chain_type=chain_type, retriever=retriever, return_source_documents=True)
    result = qa({"query": query})
    print(result['result'])
    return result



def qa_result(_):
    if pass_input_api_key.value:
        os.environ["OPENAI_API_KEY"] = pass_input_api_key.value

    # save pdf file to a temp file
    file_data = file_input_pdf.value
    if file_data:
        if not os.path.exists('tmp'):
            os.makedirs('tmp')
        #file_content = io.BytesIO(file_data)
        file_input_pdf.save("tmp/temp.pdf")

        prompt_text = text_editor_prompt.value
        if prompt_text:
            result = qa(
                # file="temp.pdf",
                file="tmp/temp.pdf",
                query=prompt_text,
                chain_type=rb_chain_type.value,
                search_type=rb_search_type.value,
                k=int_slider_num_chunks.value
            )
            convos.extend([
                pn.Row(
                    pn.panel("\U0001F60A", width=10),
                    prompt_text,
                    width=600
                ),
                pn.Row(
                    pn.panel("\U0001F916", width=10),
                    pn.Column(
                        result["result"],
                        "Relevant source text:",
                        pn.pane.Markdown('\n--------------------------------------------------------------------\n'.join(doc.page_content for doc in result["source_documents"]))
                    )
                )
            ])
    return pn.Column(*convos, margin=15, width=575, min_height=400)




# 1st row
md_desc = pn.pane.Markdown("""
    ## Question Answering with pdf and prompt.\n
    Step 1: Upload a PDF file.
    Step 2: Enter OpenAI API key (optional). 
    Step 3: Type question
    Step 4: Click `Run`\n
    """)
row1_desc = pn.Row(
    md_desc,
    width=800
)

# 2nd row
row2_prompt_settings = pn.Row(
    row_2_1_prompt,
    row2_1_settings_card,
    # column_temp,
    width=800
)

# 3rd row
qa_interactive = pn.panel(
    pn.bind(qa_result, button_run),
    loading_indicator=True,
)
row3_output_widget_box = pn.WidgetBox(
    '*Output: *', qa_interactive,
    width=800, scroll=True
)

# Layout
layout = pn.Column(
    row1_desc,
    row2_prompt_settings,
    row3_output_widget_box
)

def execute():
    pn.extension()
    if is_interactive:
        layout.servable()
    else:
        pn.serve(layout, autoreload=True, port=5000)

if __name__ == "__main__":
    execute()