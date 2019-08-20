from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


def create_keras_model():
    # Sequentialモデル使用(Sequentialモデルはレイヤを順に重ねたモデル)
    model = Sequential()

    # 結合層(2層->4層)
    model.add(Dense(100, input_dim=2, activation="tanh"))

    # 結合層(4層->1層)：入力次元を省略すると自動的に前の層の出力次元数を引き継ぐ
    model.add(Dense(1, activation="sigmoid"))

    # モデルをコンパイル
    model.compile(loss="binary_crossentropy", optimizer="Adadelta", metrics=["accuracy"])

    # sumamry関数を使うと下記のようにモデル定義を出力してくれます。
    model.summary()

    # _________________________________________________________________
    # Layer(type)  Output    Shape       Param  #
    # == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == =
    # dense       (Dense)    (None, 4)     12
    # _________________________________________________________________
    # dense_1     (Dense      (None, 1)     5
    # == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == =
    # Total   params: 17
    # Trainable  params: 17
    # Non - trainable   params: 0
    # _________________________________________________________________

    return model
