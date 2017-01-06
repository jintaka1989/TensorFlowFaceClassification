# -*- coding: utf-8 -*-
import sys
import os
import commands as cmd
import cv2
import numpy as np
import tensorflow as tf
import tensorflow.python.platform
from PIL import Image
import scipy
from scipy import ndimage
from image_data_set import ImageDataSet

# read config.ini
import ConfigParser
inifile = ConfigParser.SafeConfigParser()
inifile.read('./config.ini')
NUM_CLASSES = int(inifile.get("settings", "num_classes"))
DOWNLOAD_LIMIT = int(inifile.get("settings", "download_limit"))

if __name__ == "__main__":
    # result=cmd.getstatusoutput("touch test.txt")
    app_root_path = os.getcwd() + "/"
    # 画像の種類ごとのラベルファイルの生成
    ImageDataSet.create_train_labels(app_root_path, NUM_CLASSES)
    ImageDataSet.create_test_labels(app_root_path, NUM_CLASSES)
    # ラベルファイルの統合
    ImageDataSet.joint_train_labels(app_root_path, NUM_CLASSES)
    ImageDataSet.joint_test_labels(app_root_path, NUM_CLASSES)
    # use_model.pyの実行
    os.system("python use_model.py")
