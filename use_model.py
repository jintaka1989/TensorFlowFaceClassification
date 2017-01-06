# -*- coding: utf-8 -*-
# import pdb; pdb.set_trace()
import sys
import os
import numpy as np
import tensorflow as tf
import cv2
import tensorflow.python.platform
from types import *
import time
import glob

# read config.ini
import ConfigParser
inifile = ConfigParser.SafeConfigParser()
inifile.read('./config.ini')
NUM_CLASSES = int(inifile.get("settings", "num_classes"))
DOWNLOAD_LIMIT = int(inifile.get("settings", "download_limit"))

IMAGE_SIZE = int(inifile.get("settings", "image_size"))
IMAGE_CHANNEL = int(inifile.get("settings", "image_channel"))
IMAGE_PIXELS = IMAGE_SIZE*IMAGE_SIZE*IMAGE_CHANNEL

POOL_TIMES = 2
POOL_SIZE = 2
REDUCTION = POOL_SIZE*POOL_TIMES

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('readmodels', 'models/model.ckpt', 'File name of model data')
flags.DEFINE_string('train', 'data_set/train.txt', 'File name of train data')
flags.DEFINE_string('test', 'data_set/test.txt', 'File name of test data')
flags.DEFINE_string('train_dir', '/tmp/pict_data', 'Directory to put the data_set data.')
flags.DEFINE_integer('max_steps', 201, 'Number of steps to run trainer.')
flags.DEFINE_integer('batch_size', 256, 'Batch size'
                     'Must divide evenly into the dataset sizes.')
flags.DEFINE_float('learning_rate', 1e-4, 'Initial learning rate.')
flags.DEFINE_string('class_list', 'data_set/class_list.txt', 'File name of class_list data')

with open(FLAGS.class_list, 'r') as f: # class_list.txt
    class_list = []
    for line in f:
        line = line.rstrip()
        l = line.split()
        class_list.append(l[1])

def preprocess(img):
    # # ガンマ定数の定義
    # gamma = 4.0
    # look_up_table = np.ones((256, 1), dtype = 'uint8' ) * 0
    #
    # for i in range(256):
    #     look_up_table[i][0] = 255 * pow(float(i) / 255, 1.0 / gamma)
    #
    # # ガンマ変換後の出力
    # img = cv2.LUT(img, look_up_table)

    # # RGB空間からグレースケール空間
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img

def inference(images_placeholder, keep_prob):
    def weight_variable(shape):
      initial = tf.truncated_normal(shape, stddev=0.1)
      return tf.Variable(initial)

    def bias_variable(shape):
      initial = tf.constant(0.1, shape=shape)
      return tf.Variable(initial)

    def conv2d(x, W):
      return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(x):
      return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding='SAME')

    x_image = tf.reshape(images_placeholder, [-1, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNEL])

    with tf.name_scope('conv1') as scope:
        W_conv1 = weight_variable([5, 5, IMAGE_CHANNEL, 32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    with tf.name_scope('pool1') as scope:
        h_pool1 = max_pool_2x2(h_conv1)

    with tf.name_scope('conv2') as scope:
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    with tf.name_scope('pool2') as scope:
        h_pool2 = max_pool_2x2(h_conv2)

    with tf.name_scope('fc1') as scope:
        W_fc1 = weight_variable([(IMAGE_SIZE/REDUCTION)*(IMAGE_SIZE/REDUCTION)*64, 1024])
        b_fc1 = bias_variable([1024])
        h_pool2_flat = tf.reshape(h_pool2, [-1, (IMAGE_SIZE/REDUCTION)*(IMAGE_SIZE/REDUCTION)*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    with tf.name_scope('fc2') as scope:
        W_fc2 = weight_variable([1024, NUM_CLASSES])
        b_fc2 = bias_variable([NUM_CLASSES])

    with tf.name_scope('softmax') as scope:
        y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    return y_conv

# with open(FLAGS.test, 'r') as f: # test.txt
#     test_image = []
#     test_label = []
#     test_image_name = []
#     for line in f:
#         line = line.rstrip()
#         l = line.split()
#         img = cv2.imread(l[0])
#         # 前処理
#         img = preprocess(img)
#         test_image_name.append(l[0])
#         img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
#         test_image.append(img.flatten().astype(np.float32)/255.0)
#         tmp = np.zeros(NUM_CLASSES)
#         tmp[int(l[1])] = 1
#         test_label.append(tmp)
#     test_image = np.asarray(test_image)
#     test_label = np.asarray(test_label)
#     test_len = len(test_image)

images_placeholder = tf.placeholder("float", shape=(None, IMAGE_PIXELS))
labels_placeholder = tf.placeholder("float", shape=(None, NUM_CLASSES))
keep_prob = tf.placeholder("float")

logits = inference(images_placeholder, keep_prob)
sess = tf.InteractiveSession()

saver = tf.train.Saver()
sess.run(tf.initialize_all_variables())
saver.restore(sess,FLAGS.readmodels)

# if test_len % FLAGS.batch_size is 0:
#     test_batch = test_len/FLAGS.batch_size
# else:
#     test_batch = (test_len/FLAGS.batch_size)+1
#     print "test_batch = "+str(test_batch)
#
# count = 0.0
# count_all = 0.0
#
# for i in range(len(test_image)):
#     pr = logits.eval(feed_dict={
#         images_placeholder: [test_image[i]],
#         keep_prob: 1.0 })[0]
#     pred = np.argmax(pr)
#     _max=max(pr)
#     jpgname = test_image_name[i].lstrip(os.getcwd()).lstrip("/data_set/test/class")
#     print jpgname + (":" + str(pred)).rjust(30-len(jpgname), " ") + "({:.1%})".format(_max)
#     if int(jpgname[:1])==pred:
#         count += 1.0
#         count_all += 1.0
# print (count/count_all)
print "finish"

def create_cascade():
    # サンプル顔認識特徴量ファイル
    cascade_path = "haarcascades/haarcascade_frontalface_alt.xml"
        # cascade_path = "/host/Users/YA65857/Downloads/opencv/sources/data/hogcascades/hogcascade_pedestrians.xml"
    # 分類器を作る作業
    cascade = cv2.CascadeClassifier(cascade_path)
    return cascade

def classificate_face(mirror=True, size=None):
    # これは、BGRの順になっている気がする
    color = (255, 255, 255) #白
    font_color = (0, 255, 0)
    font = cv2.FONT_HERSHEY_PLAIN
    font_size = 1

    cascade = create_cascade()
    # 保存先
    dir_path = "class"
    files = glob.glob(dir_path + "/*")
    if len(files) == 0:
        i = 0
    else:
        i = int(max(files).replace(".jpg", "").replace("tmp/", ""))

    # カメラをキャプチャする
    cap = cv2.VideoCapture(0) # 0はカメラのデバイス番号

    # 1回目の画像取得
    ret, frame = cap.read()

    # フレームをリサイズ
    # sizeは例えば(800, 600)
    if size is not None and len(size) == 2:
        frame = cv2.resize(frame, size)

    cv2.imshow("image", frame)

    while True:
        # retは画像を取得成功フラグ
        ret, frame = cap.read()

        # グレースケール変換
        gray = cv2.cvtColor(frame, cv2.cv.CV_BGR2GRAY)

        # 顔認識の実行
        facerect = cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=1, minSize=(10, 10), maxSize=(480,480))

        face_images = []

        if len(facerect) > 0:
            # 検出した顔を囲む矩形の作成
            # 検出した人物の名前をつける
            for rect in facerect:
                face_image = cut_face(frame, rect)
                # 前処理
                face_image = preprocess(face_image)
                face_image = cv2.resize(face_image, (IMAGE_SIZE, IMAGE_SIZE))
                face_images.append(face_image.flatten().astype(np.float32)/255.0)
                cv2.rectangle(frame, tuple(rect[0:2]),tuple(rect[0:2]+rect[2:4]), color, thickness=2)
                # cv2.putText(frame,"recognizing face",(10,10),font, font_size,font_color)

            face_images = np.asarray(face_images)
            face_text = ""
            for i in range(len(face_images)):
                rect = facerect[i]
                pr = logits.eval(feed_dict={
                    images_placeholder: [face_images[i]],
                    keep_prob: 1.0 })[0]
                pred = np.argmax(pr)
                # print pr
                _max=max(pr)
                # print str(pred) + "({:.1%})".format(_max)
                face_text = (class_list[pred] + "({:.1%})".format(_max))
                # cv2.putText(frame,face_text,(10,10*(i*2+1)),font, font_size,font_color)
                cv2.putText(frame,face_text,tuple(rect[0:2]-5),font, font_size,font_color)
        else:
            print("no face")

        # フレームを表示する(時間も計測する)
        cv2.imshow("image", frame)

        k = cv2.waitKey(1) # 1msec待つ
        if k == 27: # ESCキーで終了
            break
            # elif i == 800: # この枚数保存で終了
            # break

    # キャプチャを解放する
    cap.release()
    cv2.destroyAllWindows()

def cut_and_save(image, path, rect):
    # 顔だけ切り出して保存
    x = rect[0]
    y = rect[1]
    width = rect[2]
    height = rect[3]
    dst = image[y:y + height, x:x + width]
    #認識結果の保存
    cv2.imwrite(path, dst)

def cut_face(image, rect):
    # 顔だけ切り出して返す
    x = rect[0]
    y = rect[1]
    width = rect[2]
    height = rect[3]
    dst = image[y:y + height, x:x + width]
    return dst

# if __name__ == '__main__':
classificate_face()
