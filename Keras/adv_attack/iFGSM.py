import numpy as np
import keras
import tensorflow as tf
from keras.datasets import mnist

tf.compat.v1.disable_eager_execution()
from .utils import fgsm

def i_fgsm_attack(lpr_model, image, epsilons = 0.05):
    ret_predict = lpr_model.predict(np.array([image]))
    ret_predict = np.argmax(ret_predict, axis=1) #进行预测 np.argmax(model.predict(testX), axis=1)
    # 获取预测结果的one-hot编码，在攻击时需要用到
    label = np.zeros([1, 10])
    label[:, ret_predict] = 1 # 视原始结果为正确结果
    img_attack = image #img_convert
    for i in range(10):
        img_attack = fgsm(lpr_model, img_attack, label, eps=epsilons)
        img_attack = img_attack[0]
        attack_res = lpr_model.predict(np.array([img_attack]))

        adv_result = np.argmax(attack_res, axis=1)
        if adv_result[0] != ret_predict[0]:
            break
    del ret_predict, label, image, adv_result
    return attack_res, img_attack

if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_train = x_train.astype('float32')
    x_train /= 255
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    x_test = x_test.astype('float32')
    x_test /= 255

    lenet_minist_path = "../models/lenet_mnist.h5"
    model = keras.models.load_model(lenet_minist_path)
    res_attack, img_attack = i_fgsm_attack(model, x_train[0], epsilons = 0.05)

    print("对抗样本标签为："+str(np.argmax(res_attack, axis=1)[0]))
    print("原始样本标签为："+str(y_train[0]))