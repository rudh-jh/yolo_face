from ultralytics import YOLO

# 原版代码
# if __name__ == '__main__':
#     model = YOLO("./runs/detect/train/weights/best.pt")
#     model.val(data=r"D:\yolo\code\mydata.yaml",
#               batch=8,
#               workers=0,
#               imgsz=640)

from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("./runs/detect/train/weights/best.pt")
    model.val(
        data="data/data.yaml",
        batch=8,
        workers=0,
        imgsz=640
    )