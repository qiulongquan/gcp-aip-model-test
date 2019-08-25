# 【Keras入門(5)】単純なRNNモデル定義
# https://qiita.com/FukuharaYohei/items/25de4a0faf634ad34efc
# 使用plot方式画图
# 各种model add添加模式的速度比较。


import numpy as np
import matplotlib.pyplot as plt
import time

# TensorFlowに統合されたKerasを使用
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU, Bidirectional

start_time = time.time()
# -4.9から4.9までの50要素の等差数列(0.2間隔)
x_sin = np.linspace(10, 100, 3000)
y_sin = np.sin(x_sin)
plt.plot(x_sin, y_sin)
plt.show()

NUM_RNN = 10  # 1時系列のデータ数
NUM_DATA = len(x_sin) - NUM_RNN # 今回は40(=50-10)
x = []
y = []

for i in range(NUM_DATA):
    x.append(y_sin[i:i+NUM_RNN])      # 説明変数
    y.append(y_sin[i+1:i+NUM_RNN+1])  # 正解データなので1ずらした値

else:
    x_train = np.array(x).reshape(NUM_DATA, NUM_RNN, 1) # 入力を(サンプル数、時系列の数、入力層のニューロン数)にする
    y_train = np.array(y).reshape(NUM_DATA, NUM_RNN, 1) # 説明変数(x_train)と同様のshape

NUM_DIM = 8  # 中間層の次元数

model = Sequential()

# return_sequenceがTrueなので全RNN層が出力を返す(Falseだと最後のRNN層のみが出力を返す)
model.add(SimpleRNN(NUM_DIM, input_shape=(NUM_RNN, 1), return_sequences=True))

# simpleRNNではなく、実務上はLSTMを使うことが多いはずです。
# model.add(LSTM(NUM_DIM, input_shape=(NUM_RNN, 1), return_sequences=True))

# GRUだとこう。GPU向けのCuDNNGRUもあります(確かモデル間の互換性がなかったような気がします(未確認))。
# model.add(GRU(NUM_DIM, input_shape=(NUM_RNN, 1), return_sequences=True))

# 双方向にしたい場合はBidirectionalを使います。
# model.add(Bidirectional(SimpleRNN(NUM_DIM, return_sequences=True), input_shape=(NUM_RNN, 1)))

# 複数層に重ねることもできます。
# model.add(SimpleRNN(NUM_DIM, input_shape=(NUM_RNN, 1), return_sequences=True))
# model.add(SimpleRNN(NUM_DIM, input_shape=(NUM_RNN, 1), return_sequences=True))

model.add(Dense(1, activation="linear"))  #全結合層
model.compile(loss="mean_squared_error", optimizer="sgd")
model.summary()

history = model.fit(x_train, y_train, epochs=20, batch_size=8)

# Lossをグラフ表示
loss = history.history['loss']
print(loss)
plt.plot(np.arange(len(loss)), loss) # np.arangeはlossの連番数列を生成(今回はepoch数の0から19)
plt.show()

# x[0]は最初の入力(時系列10個の数)。reshape(-1)で一次元のベクトルにする。
x_test = x_train[0].reshape(-1)

# データ数(40回)ループ
for i in range(NUM_DATA):
    y_pred = model.predict(x_test[-NUM_RNN:].reshape(1, NUM_RNN, 1))  # 直近データ(最後から10要素)を使って予測
    x_test = np.append(x_test, y_pred[0][NUM_RNN-1][0])  # 出力結果をx_testに追加(n_rnn-1が10番目を意味している)

# 最初の10要素は完全に同じ
plt.plot(x_sin, y_sin, label="Training data")
plt.plot(x_sin, x_test, label="Predicted")
plt.legend()
plt.show()
end_time = time.time()
print(end_time-start_time)
