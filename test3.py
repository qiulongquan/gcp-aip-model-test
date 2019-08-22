# 無から始めるKeras 第4回
# https://qiita.com/Ishotihadus/items/ca8cc2e7b5b14ed51a2f

import numpy as np
from keras.layers import Input, Dense
from keras.models import Model

data = np.random.rand(250,2)
labels = (np.sum(data, axis=1) > 0.5) * 2 - 1

input = Input(shape=(2,))
hidden = Dense(2, activation='tanh')
output = Dense(1, activation='tanh')

model = Model(input=input, output=output(hidden(input)))
model.compile('adam', 'hinge', metrics=['accuracy'])
model.fit(data, labels, nb_epoch=150, validation_split=0.2)

# 预测数值
print("sample predict", model.predict(np.array([[0.3, 0.3]])))
# 显示输出层的权重
print(output.get_weights())
# 重新设定输出层的权重
output.set_weights([np.array([[1.0], [1.0]]), np.array([-0.5])])
# 再次显示输出层权重
print(output.get_weights())
# 重新检测修改权重后的预测数值
print("fixed predict", model.predict(np.array([[0.3, 0.3]])))

test = np.random.rand(200, 2)
predict = np.sign(model.predict(test).flatten())
real = (np.sum(test, axis=1) > 0.5) * 2 - 1
print(sum(predict == real) / 200.0)
