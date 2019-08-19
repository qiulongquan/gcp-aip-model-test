import numpy as np

NUM_TRAIN = 10

def load_data():
    data = np.random.rand(NUM_TRAIN, 2)
    print("data", data)
    # data
    # [[0.28363332 0.71070871]
    # [0.12838225 0.19633053]
    # [0.59820913 0.03147921]]
    array1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print(array1)
    # axis 0的时候是纵轴相加，1的时候是横轴相加
    print(np.sum(array1, axis=0))
    # 当相加后的结果大于1.0时，就显示1
    labels = (np.sum(data, axis=1) > 1.0) * 1
    print("labels1", labels)
    # 格式转换，由一行 变成 N行1列
    labels = labels.reshape(NUM_TRAIN, 1)
    print("labels2", labels)

    return data, labels
