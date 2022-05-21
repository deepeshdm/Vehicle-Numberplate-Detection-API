# Vehicle-Numberplate-Detection-API
A Rest API Backend for Vehicle and Numberplate detection ML model

<div align="center">
  <img src="/assets/doc.png" width="95%"/>
</div>


## To Run (Locally)

1. Git clone the repository on your local system.
```
git clone https://github.com/deepeshdm/Vehicle-Numberplate-Detection-API.git
```

2. Install the required dependencies to run the app
```
pip install -r requirements.txt
```

3. Execute the "app.py" with python
```
python app.py
```


## API snippets

#### 1] Sending an local Image directly as multipart-form data.


```python

import requests
from PIL import Image
from io import BytesIO

url = 'http://ec2-3-16-25-70.us-east-2.compute.amazonaws.com:8000/detect/vehicle_base64'
image_path = r"C:\Desktop\images\Image1.jpg"

files = {'file': ("",open(image_path, 'rb')),'Content-Type': 'image/jpeg'}

response = requests.post(url, files=files)

if response.status_code!= 200:
    print(f"Request was unsucessfull with Status Code {response.status_code}")
else:
   print("Request was sucessfull, showing image...")
   image_bytes = response.content
   output_image = Image.open(BytesIO(image_bytes))
   output_image.show()

```

#### 2] Sending Image as Base64 and receiving Base64 in return.
```python

import cv2
import numpy as np
from PIL import Image
import pybase64
import base64,requests
import io

ENDPOINT = 'http://ec2-3-16-25-70.us-east-2.compute.amazonaws.com:8000/detect/vehicle_base64'
image_path = r"C:\Users\Deepesh\Desktop\Vehicle Detection\images\Image87.jpg"

# Convert Image to numpy array
img = Image.open(image_path)
img = np.array(img)

# Takes numpy array and decode's the image as base64
def encodeImageToBase64(numpy_img):
    img = Image.fromarray(numpy_img)
    im_file = io.BytesIO()
    img.save(im_file, format="JPEG")
    im_bytes = im_file.getvalue()  # im_bytes: image in binary format.
    im_base64 = pybase64.standard_b64encode(im_bytes)
    # decode bytes to string
    encoded = im_base64.decode('utf-8')
    return encoded

# Encode image to base64
image_base64 = encodeImageToBase64(img)

# Send Post request with JSON 
response = requests.post(ENDPOINT, json={ "image_base64": image_base64})

if response.status_code!=200:
    print(f"Request was failed, Status Code - {response.status_code}")
    raise Exception(f"Request was failed, Status Code - {response.status_code}")

# Decode Base64 image 
response = response.json()
image_base64 = response["image_base64"]
base64_decoded = base64.b64decode(image_base64)
pilImage = Image.open(io.BytesIO(base64_decoded))
# Convert BGR Image to RGB
img = cv2.cvtColor(np.array(pilImage), cv2.COLOR_BGR2RGB)
cv2.imshow("Result",img)
cv2.waitKey(0)
```

#### 3] Sending Image as Base64 and receiving Base64 in return (NodeJS)
```javascript
const fs = require("fs");
const axios = require("axios");

const ENDPOINT = 'http://ec2-3-16-25-70.us-east-2.compute.amazonaws.com:8000/detect/vehicle_base64'
const image_path = "C:/Users/Deepesh/Desktop/Vehicle Detection/images/Image55.jpg"

// Encode local image to base64
console.log("Converting Image to Base64...")
const image_base64 = fs.readFileSync(image_path,{encoding: "base64"});
const data = JSON.stringify({image_base64:image_base64})

// Set request headers
const config = {headers: {'Content-Type': 'application/json'}};

console.log("Sending Post Request...")
axios
  .post(ENDPOINT, data,config)
  .then((response) => {
    // handle success
    console.log("Request is Sucessful !");
    console.log(response.data);
  })
  .catch((error) => {
    // handle error
    console.log("Request is NOT Sucessful !");
    console.log(error);
  });
```








