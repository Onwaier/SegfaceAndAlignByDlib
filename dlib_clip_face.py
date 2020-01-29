"""
代码功能：
1. 用dlib人脸检测器检测出人脸，返回的人脸矩形框
2. 对检测出的人脸进行关键点检测并切割出人脸
"""
import cv2
import dlib
import numpy as np


predictor_model = 'shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()# dlib人脸检测器
predictor = dlib.shape_predictor(predictor_model)

# cv2读取图像
test_img_path = "input/Messi.jpg"
img = cv2.imread(test_img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# 人脸数rects
rects = detector(img, 0)
# faces存储full_object_detection对象
faces = dlib.full_object_detections()

for i in range(len(rects)):
    faces.append(predictor(img,rects[i]))

face_images = dlib.get_face_chips(img, faces, size=320)
for image in face_images:
    cv_bgr_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite('output/Messi_clip.png', cv_bgr_img)
