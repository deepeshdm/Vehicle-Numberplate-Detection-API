# Vehicle-Numberplate-Detection-API
A Rest API Backend for Vehicle and Numberplate detection ML model

<div align="center">
  <img src="/assets/doc.png" width="93%"/>
</div>

```python

import requests
from PIL import Image
from io import BytesIO

url = 'http://127.0.0.1:8000/numberplate'
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
