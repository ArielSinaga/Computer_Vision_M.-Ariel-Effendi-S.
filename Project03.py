import cv2 as cv
from ultralytics import YOLO

model = YOLO("yolov8s.pt")

results1 = model(source="mclaren.jpg", show = True, conf = 0.4, save = True)

results2 = model(source="Person.mp4", show = True, conf = 0.4, save = True)

results3 = model(source="Traffic.mp4", show = True, conf = 0.4, save = True)
