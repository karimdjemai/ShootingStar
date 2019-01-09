import tensorflow as tf

class Modelbuilder:

    def buildModel(self):


        model = tf.keras.Sequential([
            tf.keras.layers.RNN(),
            tf.keras.layers.dense()
        ])

        return model