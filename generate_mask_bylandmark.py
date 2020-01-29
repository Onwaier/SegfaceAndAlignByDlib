import os.path as osp
import os
import numpy as np
import cv2
import dlib


def get_image_hull_mask(image_shape, image_landmarks, ie_polys=None):
    # get the mask of the image
    if image_landmarks.shape[0] != 68:
        raise Exception(
            'get_image_hull_mask works only with 68 landmarks')
    int_lmrks = np.array(image_landmarks, dtype=np.int)

    #hull_mask = np.zeros(image_shape[0:2]+(1,), dtype=np.float32)
    hull_mask = np.full(image_shape[0:2] + (1,), 0, dtype=np.float32)

    cv2.fillConvexPoly(hull_mask, cv2.convexHull(
        np.concatenate((int_lmrks[0:9],
                        int_lmrks[17:18]))), (1,))

    cv2.fillConvexPoly(hull_mask, cv2.convexHull(
        np.concatenate((int_lmrks[8:17],
                        int_lmrks[26:27]))), (1,))

    cv2.fillConvexPoly(hull_mask, cv2.convexHull(
        np.concatenate((int_lmrks[17:20],
                        int_lmrks[8:9]))), (1,))

    cv2.fillConvexPoly(hull_mask, cv2.convexHull(
        np.concatenate((int_lmrks[24:27],
                        int_lmrks[8:9]))), (1,))

    cv2.fillConvexPoly(hull_mask, cv2.convexHull(
        np.concatenate((int_lmrks[19:25],
                        int_lmrks[8:9],
                        ))), (1,))

    cv2.fillConvexPoly(hull_mask, cv2.convexHull(
        np.concatenate((int_lmrks[17:22],
                        int_lmrks[27:28],
                        int_lmrks[31:36],
                        int_lmrks[8:9]
                        ))), (1,))

    cv2.fillConvexPoly(hull_mask, cv2.convexHull(
        np.concatenate((int_lmrks[22:27],
                        int_lmrks[27:28],
                        int_lmrks[31:36],
                        int_lmrks[8:9]
                        ))), (1,))

    # nose
    cv2.fillConvexPoly(
        hull_mask, cv2.convexHull(int_lmrks[27:36]), (1,))

    if ie_polys is not None:
        ie_polys.overlay_mask(hull_mask)
    print()
    return hull_mask

# 加入alpha通道 控制透明度
def merge_add_alpha(img_1, mask):
    # merge rgb and mask into a rgba image
    r_channel, g_channel, b_channel = cv2.split(img_1)
    if mask is not None:
        alpha_channel = np.ones(mask.shape, dtype=img_1.dtype)
        alpha_channel *= mask*255
    else:
        alpha_channel = np.zeros(img_1.shape[:2], dtype=img_1.dtype)
    img_BGRA = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))
    return img_BGRA

def merge_add_mask(img_1, mask):
    if mask is not None:
        height = mask.shape[0]
        width = mask.shape[1]
        channel_num = mask.shape[2]
        for row in range(height):
            for col in range(width):
                for c in range(channel_num):
                    if mask[row, col, c] == 0:
                        mask[row, col, c] = 0
                    else:
                        mask[row, col, c] = 255

        r_channel, g_channel, b_channel = cv2.split(img_1)
        r_channel = cv2.bitwise_and(r_channel, mask)
        g_channel = cv2.bitwise_and(g_channel, mask)
        b_channel = cv2.bitwise_and(b_channel, mask)
        res_img = cv2.merge((b_channel, g_channel, r_channel))
    else:
        res_img = img_1
    return res_img

def get_landmarks(image):

    predictor_model = 'shape_predictor_68_face_landmarks.dat'
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_model)
    img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    rects = detector(img_gray, 0)

    for i in range(len(rects)):
        landmarks = np.matrix([[p.x, p.y] for p in predictor(image, rects[i]).parts()])
    return landmarks


def get_seg_face(img_path):
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    landmarks = get_landmarks(image)
    mask = get_image_hull_mask(np.shape(image), landmarks).astype(np.uint8)

    image_bgra = merge_add_alpha(image, mask)
    cv2.imwrite("output/Messi_add_alpha.png", image_bgra)

    image_bgr = merge_add_mask(image, mask)
    cv2.imwrite("output/Messi_add_mask.png", image_bgr)

if __name__ == "__main__":
    get_seg_face('input/Messi.jpg')
