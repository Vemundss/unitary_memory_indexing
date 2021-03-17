import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Input, Softmax

class MindReader(tf.keras.Model):
    def __init__(self, out_dim, **kwargs):
        super(MindReader, self).__init__(**kwargs)
        self.d1 = Dense(units=out_dim,use_bias=False)
        self.sm = Softmax()

    def call(self, inputs: tf.Tensor, softmax: bool = True) -> tf.Tensor:
        """
        NN forward pass
        Args:
            inputs: assume shape [mb_size, features]

        Returns: shape [mb_size, out_dim]
        """
        z = self.d1(inputs)
        return self.sm(z) if softmax else z

    def saliency(self,inputs,category=None):
        """
        OBS! if batch_size is larger than one, saliency
        map is computed by aggregating loss, then calculating
        gradient wrt to the input

        Args:
            inputs: to forward pass
            category: category to calculate saliency maps for
                       None => argmax class (i.e predicted class)
        """
        with tf.GradientTape() as tape:
            pred = self(inputs=inputs,softmax=False)

            if category is None:
                #score = tf.reduce_max(pred,axis=-1)
                category = tf.math.argmax(pred[0])
                score = pred[0,category]
            else:
                score = pred[:,category]
            
        return tape.gradient(score, inputs), category