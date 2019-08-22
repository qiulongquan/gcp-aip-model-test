# 無から始めるKeras 第2回
# 各个参数的调试和解释在下面的链接有解释。
# https://qiita.com/Ishotihadus/items/d47fc294ca568536b7f0
# 下面这三个参数 需要各种搭配测试，找到准确率最高的组合，现在这个程序已经调试完成。
# adam
# hinge
# tanh


import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation

data = np.random.rand(250,5)
labels = (np.sum(data, axis=1) > 2.5) * 2 - 1
model = Sequential([Dense(20, input_dim=5, activation='tanh'), Dense(1, activation='tanh')])
model.compile('adam', 'hinge', metrics=['accuracy'])
model.fit(data, labels, nb_epoch=150, validation_split=0.2)

test = np.random.rand(200, 5)
predict = np.sign(model.predict(test).flatten())
real = (np.sum(test, axis=1) > 2.5) * 2 - 1
print(sum(predict == real) / 200.0)
