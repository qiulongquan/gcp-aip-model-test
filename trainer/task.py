import trainer.model as model
import trainer.util as util
import tensorflow as tf

def train_and_evaluate():

    train_x, train_y = util.load_data()
    print("train_x=%d,train_y=%d".format(train_x, train_y))

    # Create the Keras Model
    keras_model = model.create_keras_model()

    keras_model.fit(train_x, train_y, epochs=300, validation_split=0.2)


if __name__ == '__main__':
    tf.logging.set_verbosity("INFO")
    train_and_evaluate()