# ----------------------------------------------------------------------------------------------------------------------
# 加载库
# ----------------------------------------------------------------------------------------------------------------------

# from ultralytics.utils.ops import non_max_suppression, scale_boxes
from ultralytics import YOLO
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPainter, QPixmap
import numpy as np
import argparse
import torch
import cv2
import sys


# ----------------------------------------------------------------------------------------------------------------------

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, ratio, (dw, dh)


# ----------------------------------------------------------------------------------------------------------------------
# 参数设置，权重位置，置信度筛选，nms
# ----------------------------------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('--weights', type=str, default='runs/detect/train/weights/best.pt', help='path to weights file')
parser.add_argument('--conf-thres', type=float, default=0.29, help='object confidence threshold')
parser.add_argument('--nms-thres', type=float, default=0.5, help='iou threshold for non-maximum suppression')
opt = parser.parse_args()
print(opt)


# ----------------------------------------------------------------------------------------------------------------------
# 检测部分
# ----------------------------------------------------------------------------------------------------------------------


class Yolo():

    # ------------------------------------------------------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self):
        self.prepare()

    # ------------------------------------------------------------------------------------------------------------------
    # select_device(device='cpu') --- 选择计算设备，GPU的话，参数改成0
    # torch.load(opt.weights, map_location=device)['model'].float() --- 加载模型
    # model.to(device).eval() --- 将模型加载到计算设备，设置测试模型
    # model.names if hasattr(model, 'names') else model.modules.names --- 获取类别信息
    # ------------------------------------------------------------------------------------------------------------------



    def prepare(self):
        global model, device, names
        device = 0 if torch.cuda.is_available() else "cpu"
        model = YOLO(opt.weights)
        names = model.names

    # ------------------------------------------------------------------------------------------------------------------
    # 检测
    # ------------------------------------------------------------------------------------------------------------------


    def detect(self, frame):
        results = model.predict(
            source=frame,
            imgsz=640,
            conf=opt.conf_thres,
            iou=opt.nms_thres,
            device=device,
            verbose=False
        )

        boxes = []
        if not results:
            return boxes

        result = results[0]
        for box in result.boxes:
            xyxy = box.xyxy[0].tolist()
            score = float(box.conf[0])
            cls = int(box.cls[0])
            label = names[cls]

            boxes.append([
                int(xyxy[0]),
                int(xyxy[1]),
                int(xyxy[2]),
                int(xyxy[3]),
                score,
                cls,
                label
            ])

        return boxes


# ----------------------------------------------------------------------------------------------------------------------
# 实例化检测接口
# ----------------------------------------------------------------------------------------------------------------------

yolo = Yolo()


# ----------------------------------------------------------------------------------------------------------------------
# 调用检测函数
# ----------------------------------------------------------------------------------------------------------------------

def recognition(image):
    boxes = yolo.detect(image)
    return boxes


# ----------------------------------------------------------------------------------------------------------------------

class UiMainWindow(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # ----------------------------------------------------------------------------------------------------------------------
        # 超参数的定义以及初始化
        # ----------------------------------------------------------------------------------------------------------------------

        self.timer_camera = QtCore.QTimer()
        self.cap = cv2.VideoCapture()
        self.CAM_NUM = 0
        self.count = 0
        self.frame_count = 0
        self.flag_recognition = False
        self.myFileStr = ''
        recognition(cv2.imread('./ultralytics/assets/bus.jpg'))
        print('---------------加载模型-----------------')
        self.set_ui()
        self.slot_init()

    # ------------------------------------------------------------------------------------------------------------------
    # 设置界面
    # ------------------------------------------------------------------------------------------------------------------

    def set_ui(self):
        self.__layout_main_1 = QtWidgets.QHBoxLayout()
        self.__layout_fun_button = QtWidgets.QHBoxLayout()
        self.__layout_data_show = QtWidgets.QVBoxLayout()
        self.__layout_main_2 = QtWidgets.QVBoxLayout()

        # --------------------------------------------------------------------------------------------------------------
        # 设置按键样式
        # --------------------------------------------------------------------------------------------------------------

        self.button_open_camera = QtWidgets.QPushButton('打开相机')
        self.button_open_camera.setFixedSize(120, 60)
        self.button_open_camera.setStyleSheet("QPushButton{color:white}"
                                              "QPushButton:hover{background-color: rgb(2,110,180);}"
                                              "QPushButton{background-color:rgb(0,0,255)}"
                                              "QPushButton{border:2px}"
                                              "QPushButton{border-radius:5px}"
                                              "QPushButton{margin:5px 5px}")

        self.button_close_video      = QtWidgets.QPushButton('清屏')
        self.button_close_video.setFixedSize(120, 60)
        self.button_close_video.setStyleSheet("QPushButton{color:white}"
                                              "QPushButton:hover{background-color: rgb(2,110,180);}"
                                              "QPushButton{background-color:rgb(0,0,255)}"
                                              "QPushButton{border:2px}"
                                              "QPushButton{border-radius:5px}"
                                              "QPushButton{padding:5px 5px}"
                                              "QPushButton{margin:5px 5px}")
        self.button_open_image       = QtWidgets.QPushButton('打开图像')
        self.button_open_image.setFixedSize(120, 60)
        self.button_open_image.setStyleSheet("QPushButton{color:white}"
                                              "QPushButton:hover{background-color: rgb(2,110,180);}"
                                              "QPushButton{background-color:rgb(0,0,255)}"
                                              "QPushButton{border:2px}"
                                              "QPushButton{border-radius:5px}"
                                              "QPushButton{padding:5px 5px}"
                                              "QPushButton{margin:5px 5px}")
        self.button_recognition      = QtWidgets.QPushButton('开始识别')
        self.button_recognition.setFixedSize(120, 60)
        self.button_recognition.setStyleSheet("QPushButton{color:white}"
                                              "QPushButton:hover{background-color: rgb(2,110,180);}"
                                              "QPushButton{background-color:rgb(0,0,255)}"
                                              "QPushButton{border:2px}"
                                              "QPushButton{border-radius:5px}"
                                              "QPushButton{padding:5px 5px}"
                                              "QPushButton{margin:5px 5px}")
        self.button_stop_recognition = QtWidgets.QPushButton('停止识别')
        self.button_stop_recognition.setFixedSize(120, 60)
        self.button_stop_recognition.setStyleSheet("QPushButton{color:white}"
                                              "QPushButton:hover{background-color: rgb(2,110,180);}"
                                              "QPushButton{background-color:rgb(0,0,255)}"
                                              "QPushButton{border:2px}"
                                              "QPushButton{border-radius:5px}"
                                              "QPushButton{padding:5px 5px}"
                                              "QPushButton{margin:5px 5px}")
        self.button_close            = QtWidgets.QPushButton('退出程序')
        self.button_close.setFixedSize(120, 60)
        self.button_close.setStyleSheet("QPushButton{color:white}"
                                              "QPushButton:hover{background-color: rgb(2,110,180);}"
                                              "QPushButton{background-color:rgb(0,0,255)}"
                                              "QPushButton{border:2px}"
                                              "QPushButton{border-radius:5px}"
                                              "QPushButton{padding:5px 5px}"
                                              "QPushButton{margin:5px 5px}")

        # ----------------------------------------------------------------------------------------------------------------------
        # 设置按键的高度
        # ----------------------------------------------------------------------------------------------------------------------

        self.button_close.setMinimumHeight(50)
        self.button_stop_recognition.setMinimumHeight(50)
        self.button_recognition.setMinimumHeight(50)
        self.button_close_video.setMinimumHeight(50)
        self.button_open_image.setMinimumHeight(50)
        self.button_open_camera.setMinimumHeight(50)

        # ----------------------------------------------------------------------------------------------------------------------
        # 显示图像区域
        # ----------------------------------------------------------------------------------------------------------------------

        self.label_show_camera = QtWidgets.QLabel()
        self.label_show_camera.setFixedSize(1000, 600)

        # ----------------------------------------------------------------------------------------------------------------------
        # 显示文本区域
        # ----------------------------------------------------------------------------------------------------------------------

        self.label_show_info = QtWidgets.QLabel()
        self.label_show_info.setFixedSize(300, 600)

        # ----------------------------------------------------------------------------------------------------------------------
        # 将前面定义的小零件添加到整体的布局上
        # ----------------------------------------------------------------------------------------------------------------------

        self.__layout_fun_button.addWidget(self.button_open_camera)
        self.__layout_fun_button.addWidget(self.button_close_video)
        self.__layout_fun_button.addWidget(self.button_open_image)
        self.__layout_fun_button.addWidget(self.button_recognition)
        self.__layout_fun_button.addWidget(self.button_stop_recognition)
        self.__layout_fun_button.addWidget(self.button_close)

        self.__layout_main_1.addWidget(self.label_show_camera)
        self.__layout_main_1.addWidget(self.label_show_info)

        self.__layout_main_2.addLayout(self.__layout_main_1)
        self.__layout_main_2.addLayout(self.__layout_fun_button)

        self.setLayout(self.__layout_main_2)
        self.setWindowTitle("基于深度学习的识别系统")

    # ----------------------------------------------------------------------------------------------------------------------
    # 按键连接对用的函数
    # ----------------------------------------------------------------------------------------------------------------------

    def slot_init(self):
        self.button_close.clicked.connect(self.close)
        self.button_recognition.clicked.connect(self.recognition)
        self.button_stop_recognition.clicked.connect(self.stop_recognition)
        self.button_close_video.clicked.connect(self.close_video)
        self.button_open_image.clicked.connect(self.open_image)
        self.timer_camera.timeout.connect(self.show_camera)
        self.button_open_camera.clicked.connect(self.button_open_camera_clicked)

    # ------------------------------------------------------------------------------------------------------------------
    # 点击'打开图像'按键调用的函数
    # ------------------------------------------------------------------------------------------------------------------

    def open_image(self):
        self.close_video()
        fname = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', './inference/images/')
        if fname[0]:
            self.label_show_info.setText('')
            self.myFileStr = fname[0]
            image = cv2.imread(self.myFileStr)
            image = cv2.resize(image, (1000, 600))
            show = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            show_image = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)
            self.label_show_camera.setPixmap(QtGui.QPixmap.fromImage(show_image))

    # ------------------------------------------------------------------------------------------------------------------
    # 点击'打开摄像头'按键调用的函数
    # ------------------------------------------------------------------------------------------------------------------

    def button_open_camera_clicked(self):
        # 已打开时，点击同一个按钮直接关闭相机
        if self.timer_camera.isActive():
            self.close_video()
            return

        # 未打开时，尝试打开相机
        self.label_show_info.setText('')
        self.myFileStr = ''
        self.flag_recognition = False
        self.CAM_NUM = 0

        flag = self.cap.open(self.CAM_NUM, cv2.CAP_DSHOW)
        if not flag:
            QtWidgets.QMessageBox.warning(
                self,
                'warning',
                "请检查相机于电脑是否连接正确",
                buttons=QtWidgets.QMessageBox.Ok
            )
            return

        self.timer_camera.start(30)
        self.button_open_camera.setText('关闭相机')

    # ----------------------------------------------------------------------------------------------------------------------
    # 识别按键调用的函数，将标识为转为True
    # ----------------------------------------------------------------------------------------------------------------------

    def recognition(self):

        # --------------------------------------------------------------------------------------------------------------
        # self.CAM_NUM 是摄像头的时候
        # --------------------------------------------------------------------------------------------------------------

        if self.CAM_NUM == 0:
            self.flag_recognition = True

        # --------------------------------------------------------------------------------------------------------------
        # self.myFileStr是图像的时候
        # --------------------------------------------------------------------------------------------------------------

        if self.myFileStr.split('.')[-1] in ['jpg', 'png', 'jpeg']:
            self.image = cv2.imread(self.myFileStr)

            # ------------------------------------------------------------------------------------------------------
            # 识别函数
            # ------------------------------------------------------------------------------------------------------

            bboxes = recognition(self.image)

            # ------------------------------------------------------------------------------------------------------
            # 识别结果可视化
            # ------------------------------------------------------------------------------------------------------

            ss = ''
            for box in bboxes:
                cv2.rectangle(self.image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 6)
                text = "{}: {:.4f}".format(box[6], box[4])
                cv2.putText(self.image, text, (box[0], box[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)
                label = box[-1]
                ss += label + ': \n' + str(box[:4]) + '\n\n'

            # ----------------------------------------------------------------------------------------------------------------------
            # 显示数据
            # ----------------------------------------------------------------------------------------------------------------------

            self.label_show_info.setText(ss)

            # ----------------------------------------------------------------------------------------------------------------------
            # 改变帧图像的一些参数后，现在在界面
            # ----------------------------------------------------------------------------------------------------------------------

            self.image = cv2.resize(self.image, (1000, 600))
            show = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            show_image = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)
            self.label_show_camera.setPixmap(QtGui.QPixmap.fromImage(show_image))

    # ----------------------------------------------------------------------------------------------------------------------
    # 停止识别按键调用的函数，将标识为转为True
    # ----------------------------------------------------------------------------------------------------------------------

    def stop_recognition(self):
        self.flag_recognition = False

    # ------------------------------------------------------------------------------------------------------------------
    # 显示窗口部分
    # ------------------------------------------------------------------------------------------------------------------

    def show_camera(self):

        # --------------------------------------------------------------------------------------------------------------
        # 读取摄像头或者视频，有信息返回时
        # --------------------------------------------------------------------------------------------------------------

        ret, self.image = self.cap.read()
        if ret:
            self.count = 0
            ss = ''
            if self.flag_recognition:

                # ------------------------------------------------------------------------------------------------------
                # 识别函数
                # ------------------------------------------------------------------------------------------------------

                bboxes = recognition(self.image)

                # ------------------------------------------------------------------------------------------------------
                # 识别结果可视化
                # ------------------------------------------------------------------------------------------------------

                for box in bboxes:
                    cv2.rectangle(self.image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 6)
                    text = "{}: {:.4f}".format(box[6], box[4])
                    cv2.putText(self.image, text, (box[0], box[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)
                    label = box[-1]
                    ss += label + ': \n' + str(box[:4]) + '\n\n'

            # ----------------------------------------------------------------------------------------------------------------------
            # 显示数据
            # ----------------------------------------------------------------------------------------------------------------------

            self.label_show_info.setText(ss)

            # ----------------------------------------------------------------------------------------------------------------------
            # 改变帧图像的一些参数后，现在在界面
            # ----------------------------------------------------------------------------------------------------------------------

            self.image = cv2.resize(self.image, (1000, 600))
            show = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)
            self.label_show_camera.setPixmap(QtGui.QPixmap.fromImage(showImage))

        # --------------------------------------------------------------------------------------------------------------
        # 没有信息返回时，释放cap，初始化
        # --------------------------------------------------------------------------------------------------------------

        else:
            self.close_video()
            return
    # ----------------------------------------------------------------------------------------------------------------------
    # 清理UI界面屏幕函数
    # ----------------------------------------------------------------------------------------------------------------------

    def close_video(self):
        if self.timer_camera.isActive():
            self.timer_camera.stop()

        if self.cap.isOpened():
            self.cap.release()

        self.label_show_camera.clear()
        self.button_open_camera.setText('打开相机')
        self.CAM_NUM = 0
        self.label_show_info.setText('')
        self.myFileStr = ''
        self.flag_recognition = False

    def closeEvent(self, event):
        self.close_video()
        event.accept()

    # ------------------------------------------------------------------------------------------------
    # 背景设置，替换背景直接换图片
    # ------------------------------------------------------------------------------------------------

    def paintEvent(self, event):
        painter = QPainter(self)
        pixmap = QPixmap('bg.jpg')
        painter.drawPixmap(self.rect(), pixmap)


# ----------------------------------------------------------------------------------------------------------------------
# 主函数，程序的入口
# ----------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    ui = UiMainWindow()
    ui.show()
    sys.exit(app.exec_())

# ----------------------------------------------------------------------------------------------------------------------































