# encoding:utf-8

import dlib
import cv2
import math
import generate_mask_bylandmark as gMark

def single_face_alignment(face, landmarks):
    eye_center = ((landmarks[36, 0] + landmarks[45, 0]) * 1. / 2,  # 计算两眼的中心坐标
                  (landmarks[36, 1] + landmarks[45, 1]) * 1. / 2)
    dx = (landmarks[45, 0] - landmarks[36, 0])  # note: right - right
    dy = (landmarks[45, 1] - landmarks[36, 1])

    angle = math.atan2(dy, dx) * 180. / math.pi  # 计算角度
    RotateMatrix = cv2.getRotationMatrix2D(eye_center, angle, scale=1)  # 计算仿射矩阵
    align_face = cv2.warpAffine(face, RotateMatrix, (face.shape[0], face.shape[1]))  # 进行放射变换，即旋转
    return align_face


if __name__ == "__main__":
    face = cv2.imread('output/Messi_add_mask.png')
    landmarks = gMark.get_landmarks(face)
    img = single_face_alignment(face, landmarks)
    cv2.imwrite('output/Messi_align.png', img)

