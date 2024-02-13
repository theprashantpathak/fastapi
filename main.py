from typing import Union
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from keras.models import load_model
from PIL import Image
import numpy as np
import io


app = FastAPI()


# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow requests from any origin
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],  # Allow these HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Load the pre-trained MobileNetV2 model
model = load_model('test.h5')

# Define labels for classification
custom_labels = ['001', '002', '003', '004', '005', '006', '007', '008', '009', '010', '011', '012', '013', '014', '015', '016', '017', '018', '019', '020', '021', '022', '023', '024', '025', '026', '027', '028', '029', '030', '031', '032', '033', '034', '035', '036', '037', '038', '039', '040', '041', '42', '43', '44', '45', '46']
# Define a function to process the image and make predictions
def predict_image(image):
    img = Image.open(image)
    img = img.resize((224, 224))  # Resize image to match model input size
    img_array = np.expand_dims(np.array(img), axis=0)
    #img_array = img_array / 255.0  # Normalize pixel values if necessary
    predictions = model.predict(img_array)
    return predictions

# @app.get("/", response_class=HTMLresponse)
# ?def home(request: Request):
 #return  templates.TemplateResponse("default.html", {"request: request"})
# def read_root():
#     return {"Hello World"}
# Define a route to accept image uploads
@app.post("/predict/")
async def predict_endpoint(request: Request, file: UploadFile = File(...)):
    contents = await file.read()
    image = io.BytesIO(contents)
    #return JSONResponse(content={"message": "File uploaded successfully"}, status_code=200)
    predictions = predict_image(image)
    top_prediction_index = np.argmax(predictions)
    # Get the corresponding label from the custom labels list
    predicted_label = custom_labels[top_prediction_index]
    
    return {"predicted_label": predicted_label}
