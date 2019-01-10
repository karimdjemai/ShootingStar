import tensorflow as tf

class ModelBuilder:

    @staticmethod
    def buildModel():
        model = tf.keras.Sequential([
            tf.keras.layers.RNN()
        ])

        return model