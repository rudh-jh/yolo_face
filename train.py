from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolov8n.pt")
    model.train(data=r"D:\yolo\code\mydata.yaml",
                epochs=150,
                batch=8,
                workers=0,
                imgsz=640)