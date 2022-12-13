from Keras.keras_config import *
tf.compat.v1.disable_eager_execution()

def fgsm(model, input, y_true, eps=0.05):
    random_layer = np.random.normal(loc=eps*0.5, scale=eps*0.25, size=input.shape)
    input = input + random_layer
    y_pred = model.output
    loss = -losses.categorical_crossentropy(y_true, y_pred)
    gradient = backend.gradients(loss, model.input)
    gradient = gradient[0]

    adv = input + backend.sign(gradient) * eps
    sess = backend.get_session()
    adv = sess.run(adv, feed_dict={model.input: np.array([input])})
    adv = np.clip(adv, 0, 1)
    return adv

def fgsm_attack(lpr_model, image, epsilons = 0.05):
    # 加载准备攻击的模型，对要攻击的图形进行转换
    # img_convert = cv2.resize(image, (28,28)) # 这里的x/y根据要求进行修改
    ret_predict0 = lpr_model.predict(np.array([image]))
    ret_predict = np.argmax(ret_predict0, axis=1) #进行预测 np.argmax(model.predict(testX), axis=1)
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
            # print("i="+str(i))
            break
    # if adv_result[0] != ret_predict[0]:
    #     print('攻击成功，前为：',ret_predict[0],', 后为：', adv_result[0])
    # else:
    #     print('攻击失败')

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

    model = keras.models.load_model(lenet_minist_path)

    res_attack, img_attack = fgsm_attack(model, x_train[0], epsilons = 0.05)

    print("对抗样本标签为："+str(np.argmax(res_attack, axis=1)[0]))
    print("原始样本标签为："+str(y_train[0]))