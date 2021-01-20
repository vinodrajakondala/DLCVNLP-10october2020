from fastapi import FastAPI, File, UploadFile, HTTPException,Response
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel
from typing import List
from starlette.responses import StreamingResponse
from starlette.responses import FileResponse




app=FastAPI()
@app.get("/")
async def start():
  return {"Hello World"}



@app.post("/uploadimage")
async def create_upload_files(image: UploadFile = File(...)):
  print(type(image))
  temp_file = _save_file_to_disk(image, path='original_image', save_as=image.filename)
  img = cv2.imread('original_image/'+image.filename)

  from detectron2 import model_zoo
  from detectron2.config import get_cfg
  from detectron2.engine import DefaultPredictor

  cfg = get_cfg()
  cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))


  cfg.DATALOADER.NUM_WORKERS = 2

  cfg.SOLVER.IMS_PER_BATCH = 2
  cfg.MODEL.ROI_HEADS.NUM_CLASSES = 12
  cfg.MODEL.DEVICE = 'cpu'

  cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set a custom testing threshold


  # cfg.MODEL.WEIGHTS = os.path.join("/var/home_", "where model is saved")
  cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "/content/drive/MyDrive/testing/Instance_segmentation/model_final.pth")


  classes_seg=['conveyor', 'Yard_ramp', 'ScissorLift', 'Platform_Trucks', 'In_Plant_Office','WirePartitions_Cages',
  'ForkLift', 'Storage_Rack', 'industrial_scale', 'Case_Sealer', 'packing table', 'stretch wrap machine']
  predictor = DefaultPredictor(cfg)
  outputs = predictor(img)
  from detectron2.utils.visualizer import ColorMode
  v = Visualizer(img[:, :, ::-1],scale=0.8)
  out = v.draw_instance_predictions(outputs["instances"].to('cpu'))
  img_out = Image.fromarray(out.get_image()[:, :, ::-1])

  prediction= outputs['instances'].pred_classes.numpy()
  dict_list= list(set(prediction))
  for name_ in dict_list:
    print('dict_ value.....', classes_seg[name_])
  cv2.imwrite(f'predicted_image/{image.filename}',out.get_image()[:, :, ::-1])
  return StreamingResponse(io.BytesIO(out.get_image()[:, :, ::-1].tobytes()), media_type="image/png")


def _save_file_to_disk(uploaded_file, path=".", save_as="default"):
    extension = os.path.splitext(uploaded_file.filename)[-1]
    temp_file = os.path.join(path, save_as)
    with open(temp_file, "wb") as buffer:
        shutil.copyfileobj(uploaded_file.file, buffer)
    return temp_file

import nest_asyncio
from pyngrok import ngrok
PORT=8008
ngrok_tunnel = ngrok.connect(PORT)
print('Public URL:', ngrok_tunnel.public_url)
nest_asyncio.apply()
uvicorn.run(app, port=PORT)
