import PIL
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from app.utils.model_prepare import  load_model, transform_image, class_label
import torch

model = None 
app = FastAPI()


# Create class of answer: only class name 
class ImageClass(BaseModel):
    prediction: str

# Load model at startup
@app.on_event("startup")
def startup_event():
    global model
    model = load_model()

@app.get('/')
def return_info():
    return 'Hello FastAPI'

@app.get('/about')
def return_what():
    return 'It is a FAST API service'


@app.post('/classify')
def classify(file: UploadFile = File(...)):
    image = PIL.Image.open(file.file)
    adapted_image = transform_image(image)
    pred_index = torch.sigmoid(model(adapted_image.unsqueeze(0))).round().item()
    imagenet_class = class_label(pred_index)
    response = ImageClass(
        prediction=imagenet_class
    )

    return response