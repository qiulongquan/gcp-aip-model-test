import numpy as np

NUM_TRAIN = 128


def load_data():
    data = np.random.rand(NUM_TRAIN, 2)

    labels = (np.sum(data, axis=1) > 1.0) * 1
    labels = labels.reshape(NUM_TRAIN, 1)

    return data, labels
