# -*- coding: utf-8 -*-

import os
import numpy as np
from PIL import Image
from keras import backend
from keras.layers import Input
from keras.models import load_model
from keras.utils import multi_gpu_model
from yolo3.utils import letterbox_image
from yolo3.model import yolo_eval, yolo_body

class Test:
    def __init__(self):
        print("hello")


class YOLO:
    #类的初始化
    def __init__(self):
        print("[YOLO] start init")
        self.__dict__.update(self.__defaults)
        self.__class_names = self.__GetClass()
        self.__anchors = self.__GetAnchors()
        self.__boxes,self.__scores,self.__classes = self.__Generate()
        self.sess = backend.get_session()
        print("[YOLO] finish init")

    __defaults = {
        "model_path": 'model_data/yolo.h5',
        "anchors_path": 'model_data/yolo_anchors.txt',
        "classes_path": 'model_data/coco_classes.txt',
        "score": 0.3,
        "iou": 0.45,
        "model_image_size": (416, 416),#这里必须是32的倍数
        "gpu_num": 1,
    }#这里相当于YOLO类的一个私有成员变量（字典），在init的时候update这个字典就可以初始化这些成员变量

    #读取种类名称
    def __GetClass(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    #读取anchors数据,这个应该就是估计boundingbox时用的anchor
    def __GetAnchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    # 生成boxes，scores，classes
    def __Generate(self):
        #载入模型、权重
        model_path = os.path.expanduser(self.model_path)
        num_anchors = len(self.__anchors)#anchors的数量
        num_classes = len(self.__class_names)#class的数量
        self.__yolo_model = load_model(model_path, compile=False)#读入modle，load_model是keras中的函数
        self.__yolo_model = yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)#如果是tiny版本的就导入tiny模型，否则导入正常模型
        self.__yolo_model.load_weights(model_path)#导入模型权重weights
        print('[YOLO] model, anchors, and classes is loaded from {}'.format(model_path))

        ###设定输入图片尺寸
        self.__input_image_shape = backend.placeholder(shape=(2, ))

        ###指定GPU个数
        if self.gpu_num>=2:
            self.__yolo_model = multi_gpu_model(self.__yolo_model, gpus=self.gpu_num)

        ###得到boxes, scores, classes
        boxes, scores, classes = yolo_eval(self.__yolo_model.output, self.__anchors,len(self.__class_names), self.__input_image_shape, score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def Test(self):
        print("[YOLO] bingo!")

    #检测图片
    def DetectImage(self, data):
        print("[YOLO] get data")
        #图片预处理过程：padding--转array--归一化--增加温度
        image = Image.fromarray(np.reshape(data, (1440, 1080, 3)).astype(np.uint8))
        boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))#使用填充调整大小比不变的图像
        image_data = np.array(boxed_image, dtype='float32')#将padding后的图片转为array
        image_data /= 255.#归一化处理
        image_data = np.expand_dims(image_data, 0)#增加batch的维度

        #正式预测过程
        out_boxes, out_scores, out_classes = self.sess.run(
            [self.__boxes, self.__scores, self.__classes],
            feed_dict={
                self.__yolo_model.input: image_data,
                self.__input_image_shape: [image.size[1], image.size[0]],
                backend.learning_phase(): 0
            })#这里run之后直接得到了box，scores和out_classes
        print('[YOLO] found {} boxes for the image'.format(len(out_boxes)))#打印找到了多少个boxes

        #下面对数据进行封装并return
        result = []
        for i, c in list(enumerate(out_classes)):
            class_name = self.__class_names[c]
            box = out_boxes[i]
            score = out_scores[i]
            result_data = []
            result_data.append(class_name)
            result_data.append(score)
            result_data.append(box.tolist())
            result.append(result_data)
        print('[YOLO] the result is {}'.format(result))
        return result

    #结束检测
    def CloseSession(self):
        self.sess.close()



if __name__ == '__main__':
    yolo = YOLO()
    yolo.Test()
    image = Image.open("/home/leo/Desktop/yolov3_project/src/yolov3_tf/yolov3_tf/data/girl.jpeg")
    print(type(image))





