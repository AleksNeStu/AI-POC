import json
import requests
import time
import cv2
# from google.colab.patches import cv2_imshow
import os

token_access = os.environ.get("API_KEY_HUGGING_FACE")
headers = {"Authorization": f"Bearer {token_access}"}