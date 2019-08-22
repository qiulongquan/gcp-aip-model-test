# 無から始めるKeras 第3回
# https://qiita.com/Ishotihadus/items/e28dd461a8ba27a2676e
# 写法不一样，但是结果基本是一样的。结果和test1的结果基本一样。

import numpy as np
from keras.layers import Input, Dense
from keras.models import Model

data = np.random.rand(250,5)
labels = (np.sum(data, axis=1) > 2.5) * 2 - 1

inputs = Input(shape=(5,))
x = Dense(20, activation='tanh')(inputs)
predictions = Dense(1, activation='tanh')(x)

model = Model(input=inputs, output=predictions)
model.compile('adam', 'hinge', metrics=['accuracy'])
model.fit(data, labels, nb_epoch=150, validation_split=0.2)

test = np.random.rand(200, 5)
predict = np.sign(model.predict(test).flatten())
real = (np.sum(test, axis=1) > 2.5) * 2 - 1
print(sum(predict == real) / 200.0)
