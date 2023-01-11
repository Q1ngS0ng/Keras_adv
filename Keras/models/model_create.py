import keras
from keras.layers import Dense, Flatten, Input, MaxPool2D, concatenate, Conv2D, LSTM, Reshape
from keras.datasets import mnist
import tensorflow as tf
from keras.utils import plot_model


def data_loader(dataset = 'mnist'):
    if dataset == 'mnist':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0
        x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
        x_train = x_train.astype('float32')
        x_train /= 255
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
        x_test = x_test.astype('float32')
        x_test /= 255
    return x_train, y_train, x_test, y_test

def keras_Net():

    input = Input(shape=(28, 28, 1), name='input')

    x = Conv2D(64, 5, activation="relu")(input)
    x = MaxPool2D(pool_size=(2, 2), padding='same')(x)
    x = Conv2D(32, 5, activation="relu")(x)
    x = MaxPool2D(pool_size=(2, 2), padding='same')(x)

    x = Flatten()(x)

    x = Dense(100, activation="relu")(x)
    output = Dense(10, activation="softmax")(x)

    model = keras.Model(inputs=[input], outputs=[output])
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    return model

def train_model(model):
    x_train, y_train, x_test, y_test = data_loader()

    predictions = model(x_train[:1]).numpy()
    tf.nn.softmax(predictions).numpy()

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    loss_fn(y_train[:1], predictions).numpy()

    model.compile(optimizer='adam',
                  loss=loss_fn,
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=5)
    return model


def rnn():
    input_ = Input(shape=(28, 28, 1,))

    x = Reshape((28, 28))(input_)
    x = LSTM(100)(x)
    x = Dense(100)(x)
    output = Dense(10, activation='softmax')(x)

    model = keras.Model(inputs=[input_], outputs=[output])
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    return model

if __name__ == "__main__":
    model = rnn()
    model = train_model(model)
    model.save("rnn.h5")
    # model = keras.models.load_model("lenet_mnist.h5")
    plot_model(model, show_shapes=True,
               to_file=('rnn.png'))
