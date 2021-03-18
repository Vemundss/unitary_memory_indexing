import tensorflow as tf


class InspectWeights(tf.keras.callbacks.Callback):
    def __init__(self, **kwargs):
        super(InspectWeights, self).__init__(**kwargs)
        self.weight_history = []
        self.loss_history = [0]

    def on_train_begin(self, logs=None):
        self.weight_history.append(self.model.get_weights())

    def on_epoch_end(self, epoch, logs=None):
        self.weight_history.append(self.model.get_weights())
        self.loss_history.append(logs['loss'])
