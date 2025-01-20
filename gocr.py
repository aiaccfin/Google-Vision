import cv2
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from base64 import b64encode

# Set the figure size for plotting
plt.rcParams['figure.figsize'] = (10, 20)

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


ENDPOINT_URL = 'https://vision.googleapis.com/v1/images:annotate'
API_KEY = "AIzaSyB8jECh7Mri0-vzmnm8zSghiQvMcq6zVjA"
IMG_LOC = "rawbs_Page_2.jpg"

# Perform OCR
result = request_ocr(ENDPOINT_URL, API_KEY, IMG_LOC)

if result.status_code != 200 or result.json().get('error'):
    print("Error during OCR request.")
else:
    ocr_result = result.json()['responses'][0]['textAnnotations']

# Print detected text
for index, item in enumerate(ocr_result):
    print(ocr_result[index]["description"])

# Generate bounding box coordinates
def gen_cord(result):
    cord_df = pd.DataFrame(result['boundingPoly']['vertices'])
    x_min, y_min = np.min(cord_df["x"]), np.min(cord_df["y"])
    x_max, y_max = np.max(cord_df["x"]), np.max(cord_df["y"])
    return result["description"], x_max, x_min, y_max, y_min

# Highlight the last detected text
text, x_max, x_min, y_max, y_min = gen_cord(ocr_result[-1])
image = cv2.imread(IMG_LOC)
cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

# Display the image with the bounding box
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title(f"Text Detected: {text}")
plt.show()

print(f"Text Detected = {text}")