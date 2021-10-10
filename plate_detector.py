from utils.torch_utils import select_device 
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.datasets import LoadImage
import numpy as np
import time
import torch
import cv2

class PlateDetector:
    def __init__(self, weights, device):
        self.device = select_device(device)
        self.half = self.device.type != 'cpu'  
        self.imgsz = 640 # ВООБЩЕ ЕБУ ЧТО ЭТО ЗА РАЗМЕР
                         # так и не нашел для чего он ( может быть окно вывода такого размера было)
        self.model = attempt_load(weights, map_location=device)

        if self.half:
            self.model.half()

    def get_boxes(self, image):
        start = time.time()
        loader = LoadImage(image, img_size=self.imgsz)
        img, im0s = loader.proc()

        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()
        img /= 255.0  
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        
        pred = self.model(img, augment=False)[0]
        pred = non_max_suppression(pred, 0.25, 0.45, classes=False, agnostic=False)
        
        boxes = []
        for det in pred:
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()
                for box in det:
                    boxes.append(np.array(box[:4]).astype('int'))
        print('time to find boxes ', time.time()-start)
        return np.array(boxes)
            
#path to weights, device type
#detector = PlateDetector('weights/best.pt', 'cpu')
#print(detector.get_boxes(cv2.imread('images/car.jpg')))