import numpy as np
import tensorflow as tf

tf.compat.v1.disable_eager_execution()
from .utils import pgd
import gc
def pgd_attack(lpr_model, image, epsilons = 0.05, iter = 10):
    ret_predict = lpr_model.predict(np.array([image]))
    ret_predict = np.argmax(ret_predict, axis=1) #进行预测 np.argmax(model.predict(testX), axis=1)
    # 获取预测结果的one-hot编码，在攻击时需要用到
    label = np.zeros([1, 10])
    label[:, ret_predict] = 1 # 视原始结果为正确结果
    img_attack = image #img_convert
    for i in range(iter):
        img_attack = pgd(lpr_model, img_attack, label, eps=epsilons)
        img_attack = img_attack[0]
        attack_res = lpr_model.predict(np.array([img_attack]))

        adv_result = np.argmax(attack_res, axis=1)
        if adv_result[0] != ret_predict[0]:
            # print("attack sucess!")
            break
    del ret_predict, label, image, adv_result
    gc.collect()
    return attack_res, img_attack
