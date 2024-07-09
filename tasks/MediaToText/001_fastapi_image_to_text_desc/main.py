import io

import requests
import uvicorn
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from typing import Optional
from model import model_pipeline

app = FastAPI()

URL_IMG = "http://images.cocodataset.org/val2017/000000039769.jpg"


@app.get("/")
def read_root():
    return {"Hello": "World"}


def ask(text, image):
    result = model_pipeline(text, image)
    return {"answer": result}


@app.post("/ask_file")
def ask_file(text: str, image: UploadFile):
    content = image.file.read()
    image = Image.open(io.BytesIO(content))
    return ask(text, image)


@app.post("/ask_url")
def ask_url(text: str, url: str = URL_IMG):
    image = Image.open(requests.get(url, stream=True).raw)
    return ask(text, image)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True)
