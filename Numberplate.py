import base64
from fastapi import APIRouter
import io
import cv2
import requests
import pybase64
from PIL import Image, ImageDraw
from requests_toolbelt.multipart.encoder import MultipartEncoder
import numpy as np
from fastapi import UploadFile, HTTPException
from starlette.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

numberplate_router = APIRouter()


def load_image_into_numpy_array(data):
    npimg = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    # Convert from BGR to RGB
    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame


# ---------------------------------------------------------------------------------

@numberplate_router.post("/numberplate", response_class=StreamingResponse)
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


# ---------------------------------------------------------------------------------


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


# create pydantic model
class userData(BaseModel):
    image_base64: str


@numberplate_router.post("/numberplate_base64", response_class=JSONResponse)
async def upload_base64(data: userData):
    # Get the base64 string
    data = data.dict()
    image_base64 = data.get("image_base64")

    # Strip extra content from base64 string
    image_base64 = image_base64.replace("data:image/jpeg;base64,", "")
    image_base64 = image_base64.replace("data:image/png;base64,", "")

    print("Decoding base64 to Image...")
    base64_decoded = base64.b64decode(image_base64)
    pilImage = Image.open(io.BytesIO(base64_decoded))

    # Convert BGR Image to RGB
    img = cv2.cvtColor(np.array(pilImage), cv2.COLOR_BGR2RGB)
    pilImage = Image.fromarray(img)

    print("Resizing image to (416x416)")
    # resize to (416,416)
    resized_pilImage = pilImage.resize(size=(416, 416))

    # Convert to JPEG Buffer
    buffered = io.BytesIO()

    print("Sending Post request....")

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

        print("Drawing BBOXES on output image....")
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

        # Convert BGR to RGB image
        img = np.array(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        print("Encoding output image to base64...")
        image_base64 = encodeImageToBase64(img)

        return {"image_base64": image_base64}

    except:
        raise HTTPException(status_code=500, detail="Some Error Occured, try another Image")
