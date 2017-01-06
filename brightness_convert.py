# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os

def preprocess(img):
    # ガンマ定数の定義
    gamma = 0.5
    look_up_table = np.ones((256, 1), dtype = 'uint8' ) * 0
    for i in range(256):
        look_up_table[i][0] = 255 * pow(float(i) / 255, 1.0 / gamma)
    # ガンマ変換後の出力
    img = cv2.LUT(img, look_up_table)

    return img

# img = cv2.imread("test.jpg")
dire = "data_set/train/class0"

for root,dirs,files in os.walk(dire):
    for f in files:
        if os.path.splitext(f)[1] == ".jpg":
            img = cv2.imread(os.path.join(root, f))
            img = preprocess(img)
            cv2.imwrite(dire+"/"+f.rstrip(".jpg") + "bright0.jpg", img)
