from ultralytics import YOLO

model = YOLO("models/yoloV5trained.pt")

results = model.predict("inputVideos/galo_2.mp4",save=True,conf=0.1)
print(results[0])
for box in results[0].boxes:
    print(box)

