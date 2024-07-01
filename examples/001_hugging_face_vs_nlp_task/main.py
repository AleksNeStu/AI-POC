import json
import requests
import time
import cv2
# from google.colab.patches import cv2_imshow
import os
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type


token_access = os.environ.get("API_KEY_HUGGING_FACE")
HEADERS = {
    "Authorization": f"Bearer {token_access}"
}
# https://huggingface.co/facebook/detr-resnet-50
# Detection TRansformer (DETR) model trained end-to-end on COCO 2017 object detection (118k annotated images).
API_URL = "https://api-inference.huggingface.co/models/facebook/detr-resnet-50"


# https://huggingface.co/docs/api-inference/detailed_parameters#object-detection-task


@retry(wait=wait_exponential(
    multiplier=1, min=2, max=10),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type(requests.exceptions.RequestException)
)
def query(filename):
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.request("POST", API_URL, headers=HEADERS, data=data)
    data = json.loads(response.content.decode("utf-8"))
    if isinstance(data, dict) and data.get("error"):
        estimated_time = data.get("estimated_time", 0)
        time.sleep(estimated_time)
        print(f"Error: {data['error']}, estimated time: {data['estimated_time']}")
        raise requests.exceptions.RequestException

    return data


if __name__ == "__main__":
    data = query("savana.jpg")
    print(data)