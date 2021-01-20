from enum import Enum
from fastapi import FastAPI, File, UploadFile
import uvicorn
import shutil
import cv2
import os, shutil, io
from PIL import Image

class Engine(str,Enum):
    small="small"
    Mediuum="medium"
    large="large"
class SCTURBO(str,Enum):
    yes="yes"
    no="no"
class Weight(str,Enum):
    small="small"
    Mediuum="average"
    large="large"
class fuelEconomy(str,Enum):
    less="less"
    average="average"
    high="high"

app=FastAPI()

from fastapi.responses import FileResponse



@app.get("/")
async def start():
    return {"Hello World"}

@app.get("/image/")
async def main():
    return FileResponse("horse.jpg")

@app.post("/uploadfile/")
async def create_upload_file(image: UploadFile = File(...)):
    temp_file = _save_file_to_disk(image, path='original_image', save_as=image.filename)
    img = cv2.imread('original_image/'+image.filename)
    return FileResponse(image.filename)



@app.get("/models/")
def get_model(engine: Engine, sc_turbo: SCTURBO,weight: Weight, FuelEconomy: fuelEconomy):
    if engine.value=='small':
        return {0}
    if engine.value=='medium':
        return {1}
    if engine.value=="large":
        return {2}
def _save_file_to_disk(uploaded_file, path=".", save_as="default"):
    extension = os.path.splitext(uploaded_file.filename)[-1]
    temp_file = os.path.join(path, save_as)
    with open(temp_file, "wb") as buffer:
        shutil.copyfileobj(uploaded_file.file, buffer)
    return temp_file

if __name__=="__main__":
    uvicorn.run("main:app", port=5005,reload=True)
