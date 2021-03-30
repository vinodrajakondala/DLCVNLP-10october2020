from detect1 import *
from PyQt5.QtWidgets import QApplication, QDialog
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow
import os
import sys
import cv2

class My_Application(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui=Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.predict.clicked.connect(self.predict)
        self.ui.toolButton_uploadImage.clicked.connect(self.select_photo)
        self.ui.saveImage.clicked.connect(self.savingimage)
        
        
    def select_photo(self):
        photopath,ext=QtWidgets.QFileDialog.getOpenFileName(self)
        if photopath:
            self.ui.lineEdit_1.setText(photopath)
            print("photopath is ",photopath)
            
            print(self.ui.lineEdit_1.text())
            #self.predict(photopath)
        if not photopath:
            QtWidgets.QMessageBox.about(self,"Please upload an Iamge","image requird")

    def predict(self):
        #if os.path.isfile(photopath):
        #self.ui.label.setPixmap(QPixmap(self.ui.lineEdit_1.text()))
################################################################################################################
        img=self.ui.lineEdit_1.text()

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
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")


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

         
        self.ui.label.setPixmap(QPixmap(img_out))
  
#################################################################################################################

    def savingimage(self):
        img = cv2.imread(self.ui.lineEdit_1.text())
        filename=r"C:\Users\Kuber Kumar\Desktop\ObjectDetection Detectron model\saved_images\predicted.jpg"
        status=cv2.imwrite(filename, img)
        if status==True:
            QtWidgets.QMessageBox.information(self,"Image Saved","Successfully!")
            #https://doc.qt.io/qtforpython-5/PySide2/QtWidgets/QMessageBox.html
        else:
            QtWidgets.QMessageBox.warning(self,"Image Saving Status","Please Check!")

        


if __name__=='__main__':
    app=QApplication(sys.argv)
    class_instance= My_Application()
    class_instance.show()
    sys.exit(app.exec_())