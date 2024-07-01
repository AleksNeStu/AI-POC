import json
import requests
import time
import cv2
# from google.colab.patches import cv2_imshow
import os
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
import matplotlib.pyplot as plt
from PIL import Image



def _is_interactive():
    import __main__ as main
    return not hasattr(main, '__file__')

is_interactive = _is_interactive()

API_KEY_HUGGING_FACE = os.environ.get("API_KEY_HUGGING_FACE")

# https://huggingface.co/docs/api-inference/detailed_parameters#object-detection-task
# https://huggingface.co/facebook/detr-resnet-50
# Detection TRansformer (DETR) model trained end-to-end on COCO 2017 object detection (118k annotated images).
API_URL_CV = "https://api-inference.huggingface.co/models/facebook/detr-resnet-50"
HEADERS_CV = {
    "Authorization": f"Bearer {API_KEY_HUGGING_FACE}"
}
# https://huggingface.co/Helsinki-NLP/opus-mt-en-es
API_URL_NLP = "https://api-inference.huggingface.co/models/Helsinki-NLP/opus-mt-en-fr"
HEADERS_NLP = {"Authorization": f"Bearer {API_KEY_HUGGING_FACE}"}



@retry(wait=wait_exponential(
    multiplier=1, min=2, max=10),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type(requests.exceptions.RequestException)
)
def get_cv_data(image_path: str):
    with open(image_path, "rb") as f:
        data = f.read()
    response = requests.request("POST", API_URL_CV, headers=HEADERS_CV, data=data)
    data = json.loads(response.content.decode("utf-8"))
    if isinstance(data, dict) and data.get("error"):
        estimated_time = data.get("estimated_time", 0)
        time.sleep(estimated_time)
        print(f"Error: {data['error']}, estimated time: {data['estimated_time']}")
        raise requests.exceptions.RequestException

    return data


@retry(wait=wait_exponential(
    multiplier=1, min=2, max=10),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type(requests.exceptions.RequestException)
)
def translate_text(text: str):
    payload = {
        "inputs": text,
    }
    data = json.dumps(payload)
    response = requests.request(
        "POST", API_URL_NLP, headers=HEADERS_NLP, data=data)
    data = json.loads(response.content.decode("utf-8"))

    if isinstance(data, dict) and data.get("error"):
        estimated_time = data.get("estimated_time", 0)
        time.sleep(estimated_time)
        print(f"Error: {data['error']}, estimated time: {data['estimated_time']}")
        raise requests.exceptions.RequestException

    return data


def get_image(image_path: str, to_show: bool = True):
    image = cv2.imread(image_path)
    return image

def show_image(image = None, image_path = None, use_pil: bool = True):
    if image_path:
        image = get_image(image_path)
    #Show image
    if is_interactive:
        pass
        # from google.colab.patches import cv2_imshow
        # cv2_imshow(image)
    else:
        if use_pil:
            # Convert the image from BGR to RGB color space
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Converting numpy array to PIL image so we can use show method of PIL Image class
            Image.fromarray(image_rgb).show()
        else:
            # Some issues with res quality
            # cv2.imshow('Savanna Image', image)
            plt.imshow(
                cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                interpolation='none') # Convert the image from BGR to RGB
            # color
            # space
            plt.title("Image repr")
            plt.show()

            # Wait for a key press and close the window
            cv2.waitKey(0)
            cv2.destroyAllWindows()


def add_cv_data_to_image(image, cv_data):
    for result in cv_data:
        box = result["box"]
        xmin = box["xmin"]
        ymin = box["ymin"]
        xmax = box["xmax"]
        ymax = box["ymax"]
        label = result["label"]

        red_color = (255, 50, 50)

        # Draw a line between the top-left and bottom-right corners of the bounding box.
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), red_color, 2)

        # Draw the label.
        cv2.putText(
            image, label,
            (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX,
            1, red_color, 2)

    return image


if __name__ == "__main__":
    image_path = "savanna2.jpg"
    res = translate_text('Hello word')


    image = get_image(image_path)
    # Show original image
    # show_image(image)
    # Get CV data
    image_cv_data = get_cv_data(image_path)
    image_labeled = add_cv_data_to_image(image, image_cv_data)
    # Show labeled image
    show_image(image_labeled)