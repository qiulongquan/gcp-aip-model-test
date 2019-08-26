# 無から始めるKeras 第5回
# https://qiita.com/Ishotihadus/items/b171272b954147976bfc

# EarlyStopping
# 使用EarlyStopping来防止过拟合发生。
#
# 关于batch size的解释
# batch一回放入的样本的数量。
# 下面的程序的例子
# Dropout(0.4)，所以一共处理的训练样品数量是 60000*0.4=48000
# batch size是100的话 一次放入100个样品，一共进行480回batch。
#
# batch也可以是1，数量小占用内存也小。
# 一次epoch完成后参数会更新，batch越小参数更新次数越多，收敛速度越快。但是不能太小。需要测试一下。
# 如果不指定batch size数量，默认是全数据数量。不推荐


from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.callbacks import EarlyStopping

(x_train, y_train), (x_test, y_test) = mnist.load_data(path='/Users/t-lqiu/gcp-aip-model-test/mnist.npz')
x_train = x_train.reshape(60000, 784) / 255.0
x_test = x_test.reshape(10000, 784) / 255.0
y_train = np_utils.to_categorical(y_train, num_classes=10)

model = Sequential([
    Dense(1300, input_dim=784, activation='relu'),
    Dropout(0.4),
    Dense(10, activation='softmax')
])
model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])

es = EarlyStopping(monitor='val_acc')
model.fit(x_train, y_train, batch_size=100, validation_split=0.2, callbacks=[es])

predict = model.predict_classes(x_test)
print(sum(predict == y_test) / 10000.0)
