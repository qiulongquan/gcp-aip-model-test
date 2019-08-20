import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation

data = np.random.rand(250, 5)
labels = np_utils.to_categorical((np.sum(data, axis=1) > 2.5) * 1)
# 对预测结果有直接影响的是
# Dense(200)这个参数   隐藏层神经网络节点的数量，数量越大准确率越高。
# nb_epoch=25  这个参数   这个是重复学习的次数，数量越大准确率越高，但是时间也越长。

# 下面的连接是利用Keras进行深度神经网络训练和测试的介绍，比较容易理解。
# https://qiita.com/Ishotihadus/items/c2f864c0cde3d17b7efb
model = Sequential([Dense(200, input_dim=5), Activation('relu'), Dense(2, activation='softmax')])
model.compile('rmsprop', 'categorical_crossentropy', metrics=['accuracy'])
model.fit(data, labels, nb_epoch=25, validation_split=0.2)

test = np.random.rand(200, 5)
predict = np.argmax(model.predict(test), axis=1)
real = (np.sum(test, axis=1) > 2.5) * 1
print("real:\n", real)
print("predict:\n", predict)
print("Accuracy:", sum(predict == real) / 200.0)
