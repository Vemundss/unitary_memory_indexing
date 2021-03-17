import tensorflow as tf


class InspectWeights(tf.keras.callbacks.Callback):
    def __init__(self, **kwargs):
        super(InspectWeights, self).__init__(**kwargs)
        self.weight_history = []

    def on_train_begin(self, logs=None):
        self.weight_history.append(self.model.get_weights())

    def on_epoch_end(self, epoch, logs=None):
        self.weight_history.append(self.model.get_weights())
