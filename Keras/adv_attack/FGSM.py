from Keras.keras_config import *

def fgsm(model, input, y_true, eps=0.1):
    random_layer = np.random.normal(loc=eps*0.5, scale=eps*0.25, size=input.shape)
    print(random_layer)
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

def fgsm_attack(lpr_model, image, image_num, epsilons = 0.1):
    try:
        image = cv2.imread(image)
        if image is None:
            print(image, end=' ')
            print("图像读取失败")
            return False
    except:
        pass
    # 加载准备攻击的模型，对要攻击的图形进行转换
    img_convert = cv2.resize(image, (28,28)) # 这里的x/y根据要求进行修改
    ret_predict0 = lpr_model.predict(np.array([img_convert]))
    ret_predict = np.argmax(ret_predict0, axis=1) #进行预测 np.argmax(model.predict(testX), axis=1)
    # 获取预测结果的one-hot编码，在攻击时需要用到
    label = np.zeros([1, 10])
    label[:, 1] = 1
    img_attack = img_convert
    for i in range(10):
        img_attack = fgsm(lpr_model, img_attack, label)
        img_attack = img_attack[0]
        attack = lpr_model.predict(np.array([img_attack]))

        adv_result = np.argmax(attack, axis=1)
        if adv_result[0] != ret_predict[0]:
            print('攻击成功，前为：',ret_predict[0],', 后为：', adv_result[0])
            break
        else:
            print('攻击失败')

    return attack

if __name__ == "__main__":
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0


    model = keras.models.load_model(minist_path)

    fgsm_attack(model, x_train[0], y_train[0], epsilons = 10)

    pass

