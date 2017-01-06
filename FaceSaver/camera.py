# coding: UTF-8
import pdb
import cv2
import time
import glob

def capture_camera(mirror=True, size=None):
    # カメラをキャプチャする
    cap = cv2.VideoCapture(0) # 0はカメラのデバイス番号

    while True:
        # retは画像を取得成功フラグ
        ret, frame = cap.read()

        # 鏡のように映るか否か
        if mirror is True:
            frame = frame[:,::-1]

        # フレームをリサイズ
        # sizeは例えば(800, 600)
        if size is not None and len(size) == 2:
            frame = cv2.resize(frame, size)

        # フレームを表示する
        cv2.imshow('camera capture', frame)

        k = cv2.waitKey(1) # 1msec待つ
        if k == 27: # ESCキーで終了
            break

    # キャプチャを解放する
    cap.release()
    cv2.destroyAllWindows()

def create_cascade():
    # サンプル顔認識特徴量ファイル
    cascade_path = "haarcascades/haarcascade_frontalface_alt.xml"
        # cascade_path = "/host/Users/YA65857/Downloads/opencv/sources/data/hogcascades/hogcascade_pedestrians.xml"
    # 分類器を作る作業
    cascade = cv2.CascadeClassifier(cascade_path)

    return cascade

def save_face(mirror=True, size=None):

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
        facerect = cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=1, minSize=(1, 1), maxSize=(240,240))

        if len(facerect) > 0:
            # 検出した顔を囲む矩形の作成
            for rect in facerect:
                cv2.rectangle(frame, tuple(rect[0:2]),tuple(rect[0:2]+rect[2:4]), color, thickness=2)
                cv2.putText(frame,"recognizing face",(10,10),font, font_size,font_color)
                save_path = dir_path + '/' + str(i) + '.jpg'
                cut_and_save(frame, save_path, rect)
                i += 1
        else:
            print("no face")

        # フレームを表示する(時間も計測する)
        cv2.imshow("image", frame)

        k = cv2.waitKey(1) # 1msec待つ
        if k == 27: # ESCキーで終了
            break
        elif i == 800: # この枚数保存で終了
            break

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

def recognize_face(mirror=True, size=None):

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
        facerect = cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=1, minSize=(1, 1), maxSize=(240,240))

        if len(facerect) > 0:
            # 検出した顔を囲む矩形の作成
            for rect in facerect:
                cv2.rectangle(frame, tuple(rect[0:2]),tuple(rect[0:2]+rect[2:4]), color, thickness=2)
                cv2.putText(frame,"recognizing face",(10,10),font, font_size,font_color)
                # save_path = dir_path + '/' + str(i) + '.jpg'
                # cut_and_save(frame, save_path, rect)
                # i += 1
        else:
            print("no face")

        # フレームを表示する(時間も計測する)
        cv2.imshow("image", frame)

        k = cv2.waitKey(1) # 1msec待つ
        if k == 27: # ESCキーで終了
            break
        # elif i == 800: # この枚数保存で終了
        #     break

    # キャプチャを解放する
    cap.release()
    cv2.destroyAllWindows()
