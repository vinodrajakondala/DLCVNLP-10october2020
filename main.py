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
   
    img = cv2.imread(image.filename)
    return FileResponse(image.filename)



@app.get("/models/")
def get_model(engine: Engine, sc_turbo: SCTURBO,weight: Weight, FuelEconomy: fuelEconomy):
    if engine.value=='small':
        return {0}
    if engine.value=='medium':
        return {1}
    if engine.value=="large":
        return {2}


if __name__=="__main__":
    uvicorn.run("main:app", port=5005,reload=True)
