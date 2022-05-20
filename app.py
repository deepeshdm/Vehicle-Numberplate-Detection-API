from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
import uvicorn
from Vehicle import vehicle_router
from Numberplate import numberplate_router

description = """
Just upload your Image and our Machine Learning model will detect Vehicles/Numberplats inside it 🚗

### Endpoints
 - #### /detect/vehicle - Detects only vehicles and returns output image.
 - #### /detect/numberplate - Detects only numberplates and returns output image.
  - #### /detect/vehicle_base64 - Takes Base64 of Image and return Base64 string of output Image.
 - #### /detect/numberplate_base64 - Takes Base64 of Image and return Base64 string of output Image.
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


@app.get("/", include_in_schema=False)
def root():
    # redirect to documentation
    return RedirectResponse("/docs")


# -----------------------------------------------------------

# Combine all routes
app.include_router(vehicle_router, prefix='/detect')
app.include_router(numberplate_router, prefix='/detect')

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
