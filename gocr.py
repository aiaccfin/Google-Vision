import cv2
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from base64 import b64encode

ENDPOINT_URL = 'https://vision.googleapis.com/v1/images:annotate'
API_KEY = "AIz"
IMG_LOC = "rawbs_Page_2.jpg"

# Function to prepare image data for the API request
def make_image_data(imgpath):
    with open(imgpath, 'rb') as f:
        content = b64encode(f.read()).decode()
        img_req = {
            'image': {
                'content': content
            },
            'features': [{
                'type': 'DOCUMENT_TEXT_DETECTION',
                'maxResults': 1
            }]
        }
    return json.dumps({"requests": [img_req]}).encode()

# Function to send the OCR request
def request_ocr(endpoint_url, api_key, imgpath):
    img_data = make_image_data(imgpath)
    response = requests.post(
        endpoint_url,
        data=img_data,
        params={'key': api_key},
        headers={'Content-Type': 'application/json'}
    )
    return response



# Perform OCR
result = request_ocr(ENDPOINT_URL, API_KEY, IMG_LOC)

if result.status_code != 200 or result.json().get('error'):
    print("Error during OCR request.")
else:
    ocr_result = result.json()['responses'][0]['textAnnotations']

# Print detected text
for index, item in enumerate(ocr_result):
    print(ocr_result[index]["description"])

