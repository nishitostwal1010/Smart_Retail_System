from ultralytics import YOLO

model = YOLO('yolov12n.pt')  # load a pretrained model (recommended for training)

model.train(data='C:/Users/nishi/PycharmProjects/pythonProject/data/dataset', epochs=20, imgsz=64)
