import tensorflow as tf
from tensorflow.keras import layers, models

from util import ID_TO_CLASS


class MyBasicAttentiveBiGRU(models.Model):

    def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int = 128, training: bool = False):
        super(MyBasicAttentiveBiGRU, self).__init__()

        self.num_classes = len(ID_TO_CLASS) #19

        self.decoder = layers.Dense(units=self.num_classes)
        self.omegas = tf.Variable(tf.random.normal((hidden_size*2, 1))) #256*1
        self.embeddings = tf.Variable(tf.random.normal((vocab_size, embed_dim))) # [10000, 100]

        ### TODO(Students) START
        self.hs = hidden_size
        self.model = tf.keras.Sequential()
        self.model.add(layers.Bidirectional(layers.GRU(units = hidden_size,return_sequences=True)))
        # ...
        ### TODO(Students) END

    def attn(self, rnn_outputs):
        ### TODO(Students) START
        M = tf.math.tanh(rnn_outputs)
        reshaped_omega = tf.reshape(self.omegas,[1,1,2*self.hs])
        alpha_before_softmax = tf.reduce_sum(M*reshaped_omega, axis=2, keepdims=True)
        alpha = tf.nn.softmax(alpha_before_softmax,axis=1)

        r = tf.reduce_sum(alpha * rnn_outputs, axis = 1)
        output = tf.tanh(r)
        # ...
        ### TODO(Students) END

        return output

    def call(self, inputs, pos_inputs, training): #10,5 10,5 , true
        word_embed = tf.nn.embedding_lookup(self.embeddings, inputs) #[10,5,100]
        pos_embed = tf.nn.embedding_lookup(self.embeddings, pos_inputs) #[10,5,100]


        ### TODO(Students) START
        word_pos = tf.concat([word_embed, pos_embed], 2)
        tokens_mask = tf.cast(inputs != 0, tf.float32)
        sequence = self.model(word_pos, mask=tokens_mask)
        attenuated_layer = self.attn(sequence)
        decoded = self.decoder(attenuated_layer)
        logits=decoded
        # ...
        ### TODO(Students) END

        return {'logits': logits}


class MyAdvancedModel(models.Model):

    def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int = 128, training: bool = False):
        super(MyAdvancedModel, self).__init__()
        ### TODO(Students) START
        self.num_classes = len(ID_TO_CLASS)  # 19
        self.decoder = layers.Dense(units=self.num_classes)
        self.omegas = tf.Variable(tf.random.normal((hidden_size * 2, 1)))  # 256*1
        self.embeddings = tf.Variable(tf.random.normal((vocab_size, embed_dim)))  # [10000, 100]

        self.hs = hidden_size
        self.model = tf.keras.Sequential()
        self.model.add(layers.Bidirectional(layers.GRU(units=hidden_size, return_sequences=True)))
        self.model.add(layers.Bidirectional(layers.GRU(units=hidden_size, return_sequences=True)))
        ### TODO(Students END

    def pool(self, rnn_outputs):
        output = tf.reduce_max(rnn_outputs, axis=1)
        return output

    def call(self, inputs, pos_inputs, training):
        ### TODO(Students) START
        word_embed = tf.nn.embedding_lookup(self.embeddings, inputs)  # [10,5,100]
        tokens_mask = tf.cast(inputs != 0, tf.float32)
        sequence = self.model(word_embed, mask=tokens_mask)
        attenuated_layer = self.pool(sequence)
        decoded = self.decoder(attenuated_layer)
        logits = decoded
        ### TODO(Students END
        return {'logits': logits}
