# 【Keras入門(6)】単純なRNNモデル定義(最終出力のみ使用)
# https://qiita.com/FukuharaYohei/items/39f865bb53cdd5052179


from random import randint

import numpy as np
import matplotlib.pyplot as plt

# TensorFlowに統合されたKerasを使用
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN

NUM_RNN = 10
NUM_DATA = 200

# 空の器を作成
x_train = np.empty((0, NUM_RNN))
y_train = np.empty((0, 1))

print(x_train)
print(y_train)

for i in range(NUM_DATA):
    num_random = randint(-20, 20)
    if i % 2 == 1:  # 奇数の場合
        x_train = np.append(x_train, np.linspace(num_random, num_random+NUM_RNN-1, num=NUM_RNN).reshape(1, NUM_RNN), axis=0)
        y_train = np.append(y_train, np.zeros(1).reshape(1, 1), axis=0)
    else: # 偶数の場合
        x_train = np.append(x_train, np.linspace(num_random, num_random-NUM_RNN+1, num=NUM_RNN).reshape(1, NUM_RNN), axis=0)
        y_train = np.append(y_train, np.ones(1).reshape(1, 1), axis=0)

x_train = x_train.reshape(NUM_DATA, NUM_RNN, 1)
y_train = y_train.reshape(NUM_DATA, 1)


NUM_DIM = 16  # 中間層の次元数

model = Sequential()

# return_sequenceがFalseなので最後のRNN層のみが出力を返す
model.add(SimpleRNN(NUM_DIM, batch_input_shape=(None, NUM_RNN, 1), return_sequences=False))
model.add(Dense(1, activation='sigmoid'))  #全結合層
model.compile(loss='binary_crossentropy', optimizer='adam')

model.summary()

history = model.fit(x_train, y_train, epochs=30, batch_size=8)
loss = history.history['loss']

# np.arangeはlossの連番数列を生成(今回はepoch数の0から29)
plt.plot(np.arange(len(loss)), loss)
plt.show()

show_result = []
# データ数(10回)ループ
for i in range(10):
    y_pred = model.predict(x_train[i].reshape(1, NUM_RNN, 1))
    print(y_pred[0], ':', x_train[i].reshape(NUM_RNN))
    show_result.append(y_pred[0][0])
print("show_result", show_result)
plt.plot(np.arange(10.), show_result)
plt.show()

