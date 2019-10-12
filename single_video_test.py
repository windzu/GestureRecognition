import cv2
import numpy as np
from PIL import Image, ImageDraw

import argparse
import configparser
from ast import literal_eval

import sys
import os
from math import ceil
import numpy as np

import lib.image as kmg
from lib.custom_callbacks import HistoryGraph
from lib.data_loader import DataLoader
from lib.utils import mkdirs
import lib.model as model
from lib.model_res import Resnet3DBuilder

from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD

import time

label_array = ['Swiping Left',
               'Swiping Right',
               'Swiping Down',
               'Swiping Up',
               'Pushing Hand Away',
               'Pulling Hand In',
               'Sliding Two Fingers Left',
               'Sliding Two Fingers Right',
               'Sliding Two Fingers Down',
               'Sliding Two Fingers Up',
               'Pushing Two Fingers Away',
               'Pulling Two Fingers In',
               'Rolling Hand Forward',
               'Rolling Hand Backward',
               'Turning Hand Clockwise',
               'Turning Hand Counterclockwise',
               'Zooming In With Full Hand',
               'Zooming Out With Full Hand',
               'Zooming In With Two Fingers',
               'Zooming Out With Two Fingers',
               'Thumb Up',
               'Thumb Down',
               'Shaking Hand',
               'Stop Sign',
               'Drumming Fingers',
               'No gesture',
               'Doing other things',
               ]


def main(args):
    # 从配置文件提取信息
    mode = config.get('general', 'mode')
    nb_frames = config.getint('general', 'nb_frames')
    skip = config.getint('general', 'skip')
    target_size = literal_eval(config.get('general', 'target_size'))
    batch_size = config.getint('general', 'batch_size')
    epochs = config.getint('general', 'epochs')
    nb_classes = config.getint('general', 'nb_classes')

    model_name = config.get('path', 'model_name')
    data_root = config.get('path', 'data_root')
    data_model = config.get('path', 'data_model')
    data_vid = config.get('path', 'data_vid')

    path_weights = config.get('path', 'path_weights')

    csv_labels = config.get('path', 'csv_labels')
    csv_train = config.get('path', 'csv_train')
    csv_val = config.get('path', 'csv_val')
    csv_test = config.get('path', 'csv_test')

    workers = config.getint('option', 'workers')
    use_multiprocessing = config.getboolean('option', 'use_multiprocessing')
    max_queue_size = config.getint('option', 'max_queue_size')

    # 路径连接
    path_vid = os.path.join(data_root, data_vid)
    path_model = os.path.join(data_root, data_model, model_name)
    path_labels = os.path.join(data_root, csv_labels)
    path_train = os.path.join(data_root, csv_train)
    path_val = os.path.join(data_root, csv_val)
    path_test = os.path.join(data_root, csv_test)

    # 张量输入shape
    #inp_shape = (None, None, None, 3)
    inp_shape = (nb_frames,) + target_size + (3,)

    # Building model
    net = Resnet3DBuilder.build_resnet_101(inp_shape, nb_classes)
    net.load_weights(path_weights)

    gen = kmg.ImageDataGenerator()

    for i in range(14):

        # 读取视频
        cap = cv2.VideoCapture('./test_videos/test'+str(i+1)+'.mp4')

        # 视频帧间隔
        timeF = 6
        # 自然帧数计数
        i = 0
        # 存储帧数计数
        j = 0
        # 存储一帧
        img_narray = np.empty([16, 64, 96, 3], dtype='float32')
        # 存储一次input
        input = np.empty([1, 16, 64, 96, 3], dtype='float32')

        temp_label = "null"
        temp_rate = 0
        while(cap.isOpened()):
            ret, frame = cap.read()
            if(ret == False):
                break

            cv2.imshow('videos', frame)
            if(i % timeF == 0 and j < 16):
                # 转为训练尺寸
                re_frame = cv2.resize(frame, (176, 100))

                # opencv 转为 PIL
                image = Image.fromarray(
                    cv2.cvtColor(re_frame, cv2.COLOR_BGR2RGB))
                image = image.resize((96, 64), Image.NEAREST)

                # PIL转为narrray，（96,64,3）
                x = np.asarray(image, dtype='float32')

                # 标准化处理
                params = gen.get_random_transform(x.shape)
                x = gen.apply_transform(x, params)
                x = gen.standardize(x)
                x = x/255

                img_narray[j] = x
                j = j+1

            if(j >= 16):
                input[0] = img_narray
                res = net.predict(input, batch_size=None,
                                  verbose=0, steps=None)
                temp_rate = np.max(res)
                res[res == np.max(res)] = 1
                res[res != np.max(res)] = 0

                where_res = np.where(res[0] == 1)
                temp_label = label_array[where_res[0][0]]
                print(temp_label)
                print(temp_rate)

                cap.release()
                cv2.destroyAllWindows()

                bk_image = cv2.imread('./back_img.jpeg')
                cv2.putText(bk_image, temp_label, (180, 233),
                            cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 255, 255), 2)
                cv2.putText(bk_image, str(temp_rate), (180, 293),
                            cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 255, 255), 2)
                cv2.imshow('results', bk_image)

                cv2.waitKey(1000)
                break

            k = cv2.waitKey(20)
            # q键退出
            if (k & 0xff == ord('q')):
                break

            i = i+1

        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="config",
                        help="Configuration file used to run the script", required=True)
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config)

    main(config)
