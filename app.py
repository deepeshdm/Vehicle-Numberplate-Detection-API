import io
import cv2
import requests
from PIL import Image, ImageDraw
from requests_toolbelt.multipart.encoder import MultipartEncoder
import numpy as np
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse
import uvicorn

description = """
Just upload your Image and our Machine Learning model will detect Vehicles/Numberplats inside it 🚗

## Endpoints
 - ### /vehicle - Detects only vehicles and returns output image.
 - ### /numberplate - Detects only numberplates and returns output image.
 
"""

app = FastAPI(
    title="Vehicle & Numberplate Detection Rest-API",
    description=description,
    version="1.0",
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
    },
    swagger_ui_parameters={"defaultModelsExpandDepth": -1})


# Enable Cors
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------------------------

def load_image_into_numpy_array(data):
    npimg = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    # Convert from BGR to RGB
    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame


@app.post("/numberplate", response_class=StreamingResponse)
async def upload_file(file: UploadFile):
    upload_image = load_image_into_numpy_array(await file.read())
    pilImage = Image.fromarray(upload_image)

    # resize to (416,416)
    resized_pilImage = pilImage.resize(size=(416, 416))

    # Convert to JPEG Buffer
    buffered = io.BytesIO()

    try:
        resized_pilImage.save(buffered, quality=100, format="JPEG")

        # Build multipart form and post request
        m = MultipartEncoder(fields={'file': ("imageToUpload", buffered.getvalue(), "image/jpeg")})

        response = requests.post(
            "https://detect.roboflow.com/license-plate-detection-s3g4g/1?api_key=vGEbPRlKg27qAG4N76W0", data=m,
            headers={'Content-Type': m.content_type})

        print(response)
        preds = response.json()

        detections = preds['predictions']
        print(preds['predictions'])

        # ----------------Draw BBoxes--------------------------------

        image_with_detections = resized_pilImage
        draw = ImageDraw.Draw(image_with_detections)

        for box in detections:
            x1 = box['x'] - box['width'] / 2
            x2 = box['x'] + box['width'] / 2
            y1 = box['y'] - box['height'] / 2
            y2 = box['y'] + box['height'] / 2
            draw.rectangle([x1, y1, x2, y2], outline="#00ff00", width=2)

        # return the output image
        img = np.array(image_with_detections)
        res, im_png = cv2.imencode(".jpg", img)
        return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/png")

    except:
        raise HTTPException(status_code=500, detail="Some Error Occured, try another Image")


@app.post("/vehicle", response_class=StreamingResponse)
async def upload_file(file: UploadFile):
    upload_image = load_image_into_numpy_array(await file.read())
    pilImage = Image.fromarray(upload_image)

    # resize to (416,416)
    resized_pilImage = pilImage.resize(size=(416, 416))

    # Convert to JPEG Buffer
    buffered = io.BytesIO()

    try:
        resized_pilImage.save(buffered, quality=100, format="JPEG")

        # Build multipart form and post request
        m = MultipartEncoder(fields={'file': ("imageToUpload", buffered.getvalue(), "image/jpeg")})

        response = requests.post(
            "https://detect.roboflow.com/vehicle-detection-ptces/1?api_key=vGEbPRlKg27qAG4N76W0", data=m,
            headers={'Content-Type': m.content_type})

        print(response)
        preds = response.json()

        detections = preds['predictions']
        print(preds['predictions'])

        # ----------------Draw BBoxes--------------------------------

        image_with_detections = resized_pilImage
        draw = ImageDraw.Draw(image_with_detections)

        for box in detections:
            x1 = box['x'] - box['width'] / 2
            x2 = box['x'] + box['width'] / 2
            y1 = box['y'] - box['height'] / 2
            y2 = box['y'] + box['height'] / 2
            draw.rectangle([x1, y1, x2, y2], outline="#00ff00", width=2)

        # return the output image
        img = np.array(image_with_detections)
        res, im_png = cv2.imencode(".jpg", img)
        return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/png")

    except:
        raise HTTPException(status_code=500, detail="Some Error Occured, try another Image")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
