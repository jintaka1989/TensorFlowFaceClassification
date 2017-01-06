# -*- coding: utf-8 -*-
import sys
import cv2
import numpy as np
import tensorflow as tf
import tensorflow.python.platform

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
flags.DEFINE_string('save_model', 'models/model.ckpt', 'File name of model data')
flags.DEFINE_string('train', 'data_set/train.txt', 'File name of train data')
flags.DEFINE_string('test', 'data_set/test.txt', 'File name of test data')
flags.DEFINE_string('train_dir', '/tmp/pict_data', 'Directory to put the data_set data.')
flags.DEFINE_integer('max_steps', 201, 'Number of steps to run trainer.')
flags.DEFINE_integer('batch_size', 256, 'Batch size'
                     'Must divide evenly into the dataset sizes.')
flags.DEFINE_float('learning_rate', 1e-4, 'Initial learning rate.')

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
    ###############################################################
    #   ディープラーニングのモデルを作成する関数
    # 引数:
    #  images_placeholder: inputs()で作成した画像のplaceholder
    #  keep_prob: dropout率のplace_holder
    # 返り値:
    #  cross_entropy: モデルの計算結果
    ###############################################################
    # 重みを標準偏差0.1の正規分布で初期化
    def weight_variable(shape):
      initial = tf.truncated_normal(shape, stddev=0.1)
      return tf.Variable(initial)
    # バイアスを標準偏差0.1の正規分布で初期化
    def bias_variable(shape):
      initial = tf.constant(0.1, shape=shape)
      return tf.Variable(initial)
    # 畳み込み層の作成
    def conv2d(x, W):
      return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    # プーリング層の作成
    def max_pool_2x2(x):
      return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding='SAME')
    # 入力をIMAGE_SIZExIMAGE_SIZExIMAGE_CHANNELに変形
    x_images = tf.reshape(images_placeholder, [-1, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNEL])
    # 畳み込み層1の作成
    with tf.name_scope('conv1') as scope:
        W_conv1 = weight_variable([5, 5, IMAGE_CHANNEL, 32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(x_images, W_conv1) + b_conv1)
    # プーリング層1の作成
    with tf.name_scope('pool1') as scope:
        h_pool1 = max_pool_2x2(h_conv1)
    # 畳み込み層2の作成
    with tf.name_scope('conv2') as scope:
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    # プーリング層2の作成
    with tf.name_scope('pool2') as scope:
        h_pool2 = max_pool_2x2(h_conv2)
    # 全結合層1の作成
    with tf.name_scope('fc1') as scope:
        W_fc1 = weight_variable([(IMAGE_SIZE/REDUCTION)*(IMAGE_SIZE/REDUCTION)*64, 1024])
        b_fc1 = bias_variable([1024])
        h_pool2_flat = tf.reshape(h_pool2, [-1, (IMAGE_SIZE/REDUCTION)*(IMAGE_SIZE/REDUCTION)*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        # dropoutの設定
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    # 全結合層2の作成
    with tf.name_scope('fc2') as scope:
        W_fc2 = weight_variable([1024, NUM_CLASSES])
        b_fc2 = bias_variable([NUM_CLASSES])
    # ソフトマックス関数による正規化
    with tf.name_scope('softmax') as scope:
        y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    return y_conv

def loss(logits, labels):
    ###############################################################
    #   lossを計算する関数
    # 引数:
    #  logits: ロジットのtensor, float - [batch_size, NUM_CLASSES]
    #  labels: ラベルのtensor, int32 - [batch_size, NUM_CLASSES]
    # 返り値:
    #  cross_entropy: モデルの計算結果
    ###############################################################
    cross_entropy = -tf.reduce_sum(labels*tf.log(tf.clip_by_value(logits,1e-10,1.0)))
    tf.scalar_summary("cross_entropy", cross_entropy)
    return cross_entropy

def training(loss, learning_rate):
    ###############################################################
    #   訓練のOpを定義する関数
    # 引数:
    #  loss: 損失のtensor, loss()の結果
    #  learning_rate: 学習係数
    # 返り値:
    #  train_step: 訓練のOp
    ###############################################################
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    return train_step

def accuracy(logits, labels):
    ###############################################################
    #   正解率(accuracy)を計算する関数
    # 引数:
    #  logits: inference()の結果
    #  labels: ラベルのtensor, int32 - [batch_size, NUM_CLASSES]
    # 返り値:
    #  accuracy: 正解率(float)
    ###############################################################
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    tf.scalar_summary("accuracy", accuracy)
    return accuracy

if __name__ == '__main__':
    # ファイルを開く
    with open(FLAGS.train, 'r') as f: # train.txt
        train_image = []
        train_label = []
        for line in f:
            line = line.rstrip()
            l = line.split()
            img = cv2.imread(l[0])
            # 前処理
            img = preprocess(img)
            img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
            train_image.append(img.flatten().astype(np.float32)/255.0)
            tmp = np.zeros(NUM_CLASSES)
            tmp[int(l[1])] = 1
            train_label.append(tmp)
        train_image = np.asarray(train_image)
        train_label = np.asarray(train_label)
        train_len = len(train_image)

    with open(FLAGS.test, 'r') as f: # test.txt
        test_image = []
        test_label = []
        for line in f:
            line = line.rstrip()
            l = line.split()
            img = cv2.imread(l[0])
            # 前処理
            img = preprocess(img)
            img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
            test_image.append(img.flatten().astype(np.float32)/255.0)
            tmp = np.zeros(NUM_CLASSES)
            tmp[int(l[1])] = 1
            test_label.append(tmp)
        test_image = np.asarray(test_image)
        test_label = np.asarray(test_label)
        test_len = len(test_image)

    with tf.Graph().as_default():
        images_placeholder = tf.placeholder("float", shape=(None, IMAGE_PIXELS))
        labels_placeholder = tf.placeholder("float", shape=(None, NUM_CLASSES))
        keep_prob = tf.placeholder("float")

        logits = inference(images_placeholder, keep_prob)
        loss_value = loss(logits, labels_placeholder)
        train_op = training(loss_value, FLAGS.learning_rate)
        acc = accuracy(logits, labels_placeholder)

        saver = tf.train.Saver()
        sess = tf.Session()
        sess.run(tf.initialize_all_variables())
        summary_op = tf.merge_all_summaries()
        summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph_def)

        # 訓練の実行
        if train_len % FLAGS.batch_size is 0:
            train_batch = train_len/FLAGS.batch_size
        else:
            train_batch = (train_len/FLAGS.batch_size)+1
            print "train_batch = "+str(train_batch)
        for step in range(FLAGS.max_steps):
            for i in range(train_batch):
                batch = FLAGS.batch_size*i
                batch_plus = FLAGS.batch_size*(i+1)
                if batch_plus > train_len: batch_plus = train_len
                # feed_dictでplaceholderに入れるデータを指定する
                sess.run(train_op, feed_dict={
                  images_placeholder: train_image[batch:batch_plus],
                  labels_placeholder: train_label[batch:batch_plus],
                  keep_prob: 0.5})

            if step % 10 == 0:
                train_accuracy = 0.0
                for i in range(train_batch):
                    batch = FLAGS.batch_size*i
                    batch_plus = FLAGS.batch_size*(i+1)
                    if batch_plus > train_len: batch_plus = train_len
                    train_accuracy += sess.run(acc, feed_dict={
                        images_placeholder: train_image[batch:batch_plus],
                        labels_placeholder: train_label[batch:batch_plus],
                        keep_prob: 1.0})
                    if i is not 0: train_accuracy /= 2.0
                # 10 step終わるたびにTensorBoardに表示する値を追加する
                #summary_str = sess.run(summary_op, feed_dict={
                #    images_placeholder: train_image,
                #    labels_placeholder: train_label,
                #    keep_prob: 1.0})
                #summary_writer.add_summary(summary_str, step)
                print "step %d, training accuracy %g"%(step, train_accuracy)

    if test_len % FLAGS.batch_size is 0:
        test_batch = test_len/FLAGS.batch_size
    else:
        test_batch = (test_len/FLAGS.batch_size)+1
        print "test_batch = "+str(test_batch)
    test_accuracy = 0.0
    for i in range(test_batch):
        batch = FLAGS.batch_size*i
        batch_plus = FLAGS.batch_size*(i+1)
        if batch_plus > train_len: batch_plus = train_len
        test_accuracy += sess.run(acc, feed_dict={
                images_placeholder: test_image[batch:batch_plus],
                labels_placeholder: test_label[batch:batch_plus],
                keep_prob: 1.0})
        if i is not 0: test_accuracy /= 2.0
    print "test accuracy %g"%(test_accuracy)
    save_path = saver.save(sess, FLAGS.save_model)
