# 这是一个重要的演示程序，如何在训练的时候创建SavedModel到指定的位置。
# 或者在训练完成后，重新装载model然后生成SavedModel到指定的位置。
# 如何保存model，
# 如何装载model，
# 如何创建SavedModel，
# 到指定的位置。
#
# 【Keras入門(2)】訓練モデル保存(KerasモデルとSavedModel)
# https://qiita.com/FukuharaYohei/items/ac6333391b8abda94bdc

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
from tensorflow.keras.models import load_model
from datetime import datetime

#tf.enable_eager_execution()

# NUM_TRAIN = 128
# data = np.random.rand(NUM_TRAIN,2)
#
# labels = (np.sum(data, axis=1) > 1.0) * 1
# labels = labels.reshape(NUM_TRAIN,1)
#
# # Sequentialモデル使用(Sequentialモデルはレイヤを順に重ねたモデル)
# model = Sequential()
#创建
# # 全結合層(2層->4層)
# model.add(Dense(4, input_dim=2, activation="tanh"))
#
# # 結合層(4層->1層)：入力次元を省略すると自動的に前の層の出力次元数を引き継ぐ
# model.add(Dense(1, activation="sigmoid"))
#
# # モデルをコンパイル
# model.compile(loss="binary_crossentropy", optimizer="sgd", metrics=["accuracy"])
#
# model.summary()
#
# # Callbackを定義し、モデル保存の追加
# li_cb = []
# li_cb.append(ModelCheckpoint('./model.hdf5', save_best_only=True))
# model.fit(data, labels, epochs=300, validation_split=0.2, callbacks=li_cb)
#
#
# # SavedModelフォーマットで保存(TensorFlow2.0ではexport_saved_modelを使う
# # https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras/experimental/export_saved_model
# 这个是在程序训练的时候就保存SavedModel，到指定的位置。
# SavedModelフォーマットでの保存(訓練時)
# tf.contrib.saved_model.save_keras_model(model, './models/keras_export')

# ーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーー
# 这个是训练完成后，重新装载model然后创建SavedModel到指定的位置。
# SavedModelフォーマットでの保存(訓練後にモデル変換)
# モデルを初期化してロード
model = None
model = load_model('./model.hdf5')
path = './models/keras_export{}'.format(datetime.utcnow().strftime("%Y%m%d%H%M%S"))
tf.contrib.saved_model.save_keras_model(model, path)

# Eager Executionでないとserving_only=Trueは失敗
#tf.contrib.saved_model.save_keras_model(model, './models/keras_export', serving_only=True)

NUM_TEST = 50
test_data = np.random.rand(NUM_TEST,2)
test_labels = (np.sum(test_data, axis=1) > 1.0) * 1

predict = ((model.predict(test_data) > 0.5) * 1).reshape(NUM_TEST)
print(predict)
print(test_labels)
print("Accuracy:",sum(predict == test_labels) / NUM_TEST)
