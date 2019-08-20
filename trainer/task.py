import trainer.model as model
import trainer.util as util
import tensorflow as tf
import numpy as np


def train_and_evaluate():

    train_x, train_y = util.load_data()
    print("train_x={},train_y={}".format(train_x, train_y))

    # Create the Keras Model
    keras_model = model.create_keras_model()

    # 固定回数（データセットの反復）の試行でモデルを学習させます．
    # https://keras.io/ja/models/model/#fit
    keras_model.fit(train_x, train_y, epochs=300, validation_split=0.2)

    print("="*50)

    NUM_TEST = 50
    test_data = np.random.rand(NUM_TEST, 2)
    test_labels = (np.sum(test_data, axis=1) > 1.0) * 1
    # keras_model.predict(test_data)是怎么样进行预测的，没有看到代码不清楚
    predict = ((keras_model.predict(test_data) > 0.5) * 1).reshape(NUM_TEST)
    print("predict:\n", predict)
    print("test_labels:\n", test_labels)
    print("Accuracy:", sum(predict == test_labels) / NUM_TEST)


if __name__ == '__main__':
    tf.logging.set_verbosity("INFO")
    train_and_evaluate()


