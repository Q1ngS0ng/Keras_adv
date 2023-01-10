from Keras.keras_config import *
tf.compat.v1.disable_eager_execution()

def fgsm(model, input, y_true, eps=0.05):
    y_pred = model.output
    loss = -losses.categorical_crossentropy(y_true, y_pred)
    gradient = backend.gradients(loss, model.input)
    gradient = gradient[0]

    adv = input + backend.sign(gradient) * eps
    sess = backend.get_session()
    adv = sess.run(adv, feed_dict={model.input: np.array([input])})
    adv = np.clip(adv, 0, 1)

    del y_pred, loss, gradient, sess

    return adv


def pgd(model, input, y_true, eps=0.05):
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

    del y_pred, loss, gradient, sess, random_layer

    return adv