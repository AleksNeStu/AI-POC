import json
import requests
import time
import cv2
# from google.colab.patches import cv2_imshow
import os
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
import matplotlib.pyplot as plt


def _is_interactive():
    import __main__ as main
    return not hasattr(main, '__file__')

is_interactive = _is_interactive()

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
def get_cv_data(filename: str):
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

def get_image_data(filename: str):
    image = cv2.imread(filename)

    #Show image
    if is_interactive:
        pass
        # from google.colab.patches import cv2_imshow
        # cv2_imshow(image)
    else:
        # cv2.imshow('Savanna Image', image)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) # Convert the image from BGR to RGB color space
        plt.title("Savanna Image")
        plt.show()

        # Wait for a key press and close the window
        cv2.waitKey(0)
        cv2.destroyAllWindows()



if __name__ == "__main__":
    image = "savanna.jpg"
    image_data = get_image_data(image)
    cv_data = get_cv_data(image)
    print(image_data)