import tensorflow as tf
import numpy as np
import pprint

class BasicEncoderDecoder:
    '''
    Basic Seq2Seq, BiLSTM encoder, LSTM decoder
    '''

    def __init__(self,
                 name,
                 vocab_size,
                 embedding_size=100,
                 hidden_size=150,  # encoder BiLSTM hidden size, double for decode LSTM
                 hidden_layers=1,
                 max_grad_norm=5,
                 learning_rate=0.001,
                 init_word_embs=None
                 ):
        self.name = name
        self.init_word_embs = init_word_embs
        self.encoder_hidden_size = hidden_size
        self.decoder_hidden_size = hidden_size * 2
        self.hidden_layers = hidden_layers
        self.max_grad_norm = max_grad_norm
        self.learning_rate = learning_rate
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size

        with tf.variable_scope(
                name_or_scope=self.name,
                initializer=tf.truncated_normal_initializer(0, 0.01)):
            self.get_input_data()
            self.trans_to_embs()
            self.build_encoder()
            self.build_decoder()
            self.build_loss()
            self.build_optimizer()

    def get_input_data(self):
        # padding tag <PAD> index are 0, while other words, <BOS>, <EOS>, <NUM>, <UNK> are greater than 0
        # <UNK> index is vocab_size-1, the last token in vocab dict
        self.encoder_inputs = tf.placeholder(tf.int32, shape=[None, None], name="encoder_inputs")
        self.encoder_seq_len = tf.reduce_sum(tf.sign(self.encoder_inputs), axis=1)
        self.encoder_max_len = tf.shape(self.encoder_inputs)[1]
        self.encoder_mask = tf.sequence_mask(self.encoder_seq_len, self.encoder_max_len)

        self.decoder_inputs = tf.placeholder(tf.int32, shape=[None, None], name="decoder_inputs")
        self.decoder_outputs = tf.placeholder(tf.int32, shape=[None, None], name="decoder_outputs")
        self.decoder_seq_len = tf.reduce_sum(tf.sign(self.decoder_inputs), axis=1)
        self.decoder_max_len = tf.shape(self.decoder_inputs)[1]

        self.decoder_mask = tf.sequence_mask(self.decoder_seq_len, self.decoder_max_len)
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")

        self.batch_size = tf.shape(self.encoder_inputs)[0]

    def trans_to_embs(self):
        with tf.variable_scope("Embedding"):
            if self.init_word_embs:
                self.W_emb = tf.Variable(initial_value=self.init_word_embs, name='W_emb', dtype=tf.float32)
            else:
                self.W_emb = tf.get_variable(name='W_emb', shape=[self.vocab_size, self.embedding_size], dtype=tf.float32)
            self.encoder_input_emb = tf.nn.embedding_lookup(self.W_emb, self.encoder_inputs)
            self.decoder_input_emb = tf.nn.embedding_lookup(self.W_emb, self.decoder_inputs)

    def RNN_cell(self, hidden_size):
        cell = tf.nn.rnn_cell.LSTMCell(hidden_size)
        return tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.keep_prob)

    def build_encoder(self):
        with tf.variable_scope("encoder"):
            self.encoder_cell_fw = tf.nn.rnn_cell.MultiRNNCell(
                [self.RNN_cell(self.encoder_hidden_size) for _ in range(self.hidden_layers)])
            self.encoder_cell_bw = tf.nn.rnn_cell.MultiRNNCell(
                [self.RNN_cell(self.encoder_hidden_size) for _ in range(self.hidden_layers)])


            self.encoder_init_state_fw = self.encoder_cell_fw.zero_state(
                batch_size=self.batch_size, dtype=tf.float32)
            self.encoder_init_state_bw = self.encoder_cell_bw.zero_state(
                batch_size=self.batch_size, dtype=tf.float32)
            self.bi_encoder_outputs, self.bi_encoder_state = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=self.encoder_cell_fw,
                cell_bw=self.encoder_cell_bw,
                inputs=self.encoder_input_emb,
                initial_state_fw=self.encoder_init_state_fw,
                initial_state_bw=self.encoder_init_state_bw,
                sequence_length=self.encoder_seq_len,
                swap_memory=True,
                dtype=tf.float32)
            self.encoder_outputs = tf.concat(self.bi_encoder_outputs, -1)
            self.encoder_final_state = []

            for layer_id in range(self.hidden_layers):  # layer_num
                merged_fwbw_cell = tf.nn.rnn_cell.LSTMStateTuple(
                    c=tf.concat([self.bi_encoder_state[0][layer_id][0],
                                 self.bi_encoder_state[1][layer_id][0]], 1),
                    h=tf.concat([self.bi_encoder_state[0][layer_id][1],
                                 self.bi_encoder_state[1][layer_id][1]], 1)
                )
                self.encoder_final_state.append(merged_fwbw_cell)
            self.encoder_final_state = tuple(self.encoder_final_state)  # init states for decoder

    def build_decoder(self):
        with tf.variable_scope("decoder"):
            self.decoder_cell = tf.nn.rnn_cell.MultiRNNCell(
                [self.RNN_cell(self.decoder_hidden_size) for _ in range(self.hidden_layers)])
            self.decoder_init_state = self.encoder_final_state
            self.decoder_rnn_outputs, _ = tf.nn.dynamic_rnn(
                cell=self.decoder_cell,
                initial_state=self.decoder_init_state,
                inputs=self.decoder_input_emb,
                sequence_length=self.decoder_seq_len,
                swap_memory=True,
                dtype=tf.float32)

    def build_loss(self):
        with tf.variable_scope("projection"):
            self.softmax_w = tf.get_variable('W', [self.decoder_hidden_size, self.vocab_size], dtype=tf.float32)
            self.softmax_b = tf.get_variable('b', [self.vocab_size], dtype=tf.float32)
            self.logits = tf.einsum('ijk,kl->ijl',self.decoder_rnn_outputs, self.softmax_w) + self.softmax_b
        with tf.variable_scope("loss"):
            self.crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.logits,
                labels=self.decoder_outputs)
            self.nonzeros = tf.count_nonzero(self.decoder_mask)
            self.decoder_mask = tf.cast(self.decoder_mask, dtype=tf.float32)
            self.train_loss = (tf.reduce_sum(self.crossent * self.decoder_mask) /
                               tf.cast(self.nonzeros, tf.float32))
            self.output_probs = tf.nn.softmax(self.logits)

    def build_optimizer(self):
        with tf.variable_scope("train"):
            self._lr = tf.Variable(self.learning_rate, trainable=False, name='learning_rate')
            self.tvars = tf.trainable_variables()
            # pprint.pprint(self.tvars)
            self.grads = tf.gradients(self.train_loss, self.tvars)
            self.grads, _ = tf.clip_by_global_norm(
                self.grads, self.max_grad_norm)

            self.optimizer = tf.train.AdamOptimizer(self._lr)
            # create training operation
            self.train_op = self.optimizer.apply_gradients(
                zip(self.grads, self.tvars))
            # update learining rate
            self._new_lr = tf.placeholder(tf.float32, shape=[], name='new_lr')
            self.update_lr = tf.assign(self._lr, self._new_lr)

        with tf.variable_scope("predict"):
            # predict
            self.predict = tf.argmax(self.output_probs, 2)
            self.correct_pred = tf.equal(
                tf.cast(self.predict, tf.int32),
                self.decoder_outputs)
            self.accuracy = tf.divide(
                tf.reduce_sum(
                    tf.cast(self.correct_pred, tf.float32) * self.decoder_mask
                ),
                tf.cast(self.nonzeros, tf.float32)
            )
            self.each_accuracy = tf.reduce_sum(tf.cast(self.correct_pred, tf.float32) * self.decoder_mask, 1) / \
                                 tf.reduce_sum(self.decoder_mask, 1)

    def mask_logits(self, seq_mask, scores):
        '''
        to do softmax, assign -inf value for the logits of padding tokens
        '''
        score_mask_values = -1e10 * tf.ones_like(scores, dtype=tf.float32)
        return tf.where(seq_mask, scores, score_mask_values)



class AttenNet(BasicEncoderDecoder):
    '''
    Luong's multiplicative attention style
    '''

    def build_decoder(self):
        with tf.variable_scope("decoder"):
            self.W_score = tf.get_variable(
                "W_score", shape=[self.decoder_hidden_size, self.decoder_hidden_size])
            self.W_attention = tf.get_variable(
                "W_attention", shape=[self.decoder_hidden_size*2, self.decoder_hidden_size])
            self.b_attention = tf.get_variable(
                "b_attention", shape=[self.decoder_hidden_size])

            self.encoder_outputs_flat = tf.reshape(
                self.encoder_outputs, [-1, self.decoder_hidden_size]
            )
            self.decoder_cell = tf.nn.rnn_cell.MultiRNNCell(
                [self.RNN_cell(self.decoder_hidden_size) for _ in range(self.hidden_layers)])
            self.decoder_init_state = self.decoder_cell.zero_state(self.batch_size, dtype=tf.float32)
            self.decoder_init_state = self.encoder_final_state



            def cond(time, state, max_len, decoder_rnn_outputs, cum_att_weights):
                return time < max_len

            def body(time, state, max_len, decoder_rnn_outputs, cum_att_weights):
                emb = self.decoder_input_emb[:, time, :]
                output, state = self.decoder_cell(
                    emb, state
                )
                scores = tf.einsum('ijk,kl,il->ij', self.encoder_outputs, self.W_score, output)
                mask_score = self.mask_logits(self.encoder_mask, scores)
                attention_weights = tf.nn.softmax(mask_score)
                cum_att_weights = cum_att_weights.write(time, attention_weights)
                context_vector = tf.einsum('ij,ijk->ik', attention_weights, self.encoder_outputs)
                attention_vector = tf.tanh(
                    tf.einsum('ij,jk->ik', tf.concat([context_vector, output], 1), self.W_attention)
                )
                decoder_rnn_outputs = decoder_rnn_outputs.write(time, attention_vector)
                return time + 1, state, max_len, decoder_rnn_outputs, cum_att_weights

            decoder_rnn_outputs = tf.TensorArray(
                dtype=tf.float32,
                size=self.decoder_max_len,
                name="decoder_output_list")
            cum_att_weights = tf.TensorArray(
                dtype=tf.float32,
                size=self.decoder_max_len,
                name="cum_att_weights")

            _, _, _, decoder_rnn_outputs, cum_att_weights = tf.while_loop(
                cond, body,
                loop_vars=[0,
                           self.decoder_init_state,
                           self.decoder_max_len,
                           decoder_rnn_outputs,
                           cum_att_weights])
            decoder_rnn_outputs = decoder_rnn_outputs.stack()
            cum_att_weights = cum_att_weights.stack()
            self.decoder_rnn_outputs = tf.transpose(decoder_rnn_outputs, perm=[1, 0, 2])
            self.cum_att_weights = tf.transpose(cum_att_weights, perm=[1, 0, 2])


class CopyNet(BasicEncoderDecoder):
    '''
    implementation of paper
    Incorporating Copying Mechanism in Sequence-to-Sequence Learning.
    '''

    def trans_to_embs(self):
        with tf.variable_scope("Embedding"):
            if self.init_word_embs:
                self.W_emb = tf.Variable(initial_value=self.init_word_embs, name='W_emb', dtype=tf.float32)
            else:
                self.W_emb = tf.get_variable(name='W_emb', shape=[self.vocab_size, self.embedding_size], dtype=tf.float32)

            self.encoder_input_emb = tf.nn.embedding_lookup(
                params=self.W_emb,
                ids=tf.where(condition=tf.less(self.encoder_inputs, self.vocab_size),
                             x=self.encoder_inputs,
                             y=tf.ones_like(self.encoder_inputs) * self.vocab_size-1))
            self.decoder_input_emb = tf.nn.embedding_lookup(
                params=self.W_emb,
                ids=tf.where(condition=tf.less(self.decoder_inputs, self.vocab_size),
                             x=self.decoder_inputs,
                             y=tf.ones_like(self.decoder_inputs) * self.vocab_size-1))


    def build_decoder(self):
        with tf.variable_scope("attention"):
            self.decoder_cell = tf.nn.rnn_cell.MultiRNNCell(
                [self.RNN_cell(self.decoder_hidden_size) for _ in range(self.hidden_layers)])
            self.attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                self.decoder_hidden_size, self.encoder_outputs, self.encoder_seq_len)
            self.decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
                self.decoder_cell, self.attention_mechanism, self.decoder_hidden_size)
            self.decoder_init_state = self.decoder_cell.zero_state(self.batch_size, dtype=tf.float32)
            self.decoder_init_state = self.decoder_init_state.clone(cell_state=self.encoder_final_state)

        with tf.variable_scope("copy"):
            self.weights_copy = tf.get_variable(
                "weights_copy", shape=[self.decoder_hidden_size, self.decoder_hidden_size])
            self.weights_generate = tf.get_variable(
                "weights_generate", shape=[self.decoder_hidden_size, self.vocab_size])
            self.batch_OOV_num = tf.placeholder(
                tf.int32, shape=[], name="batch_OOV_num")


            def cond(time, state, copy_pro, max_len, output_prob_list):
                return time < max_len

            def body(time, state, copy_pro, max_len, output_prob_list):
                # selective read
                this_decoder_emb = self.decoder_input_emb[:, time, :]
                this_decoder_data = self.decoder_inputs[:, time]
                selective_mask = tf.cast(tf.equal(self.encoder_inputs, tf.expand_dims(this_decoder_data, axis=1)),
                                         dtype=tf.float32)  # batch * encoder_max_len
                selective_mask_sum = tf.reduce_sum(selective_mask, axis=1)
                rou = tf.where(tf.less(selective_mask_sum, 1e-10),
                                          selective_mask, selective_mask / tf.expand_dims(selective_mask_sum, 1))

                selective_read = tf.einsum("ijk,ij->ik", self.encoder_outputs, rou)

                this_decoder_final = tf.concat([this_decoder_emb, selective_read], axis=1)
                this_decoder_output, state = self.decoder_cell(this_decoder_final, state)  # batch * hidden_dim

                # generate mode
                generate_score = tf.matmul(
                    this_decoder_output, self.weights_generate, name="generate_score")  # batch * vocab_size

                # copy mode
                copy_score = tf.einsum("ijk,km->ijm", self.encoder_outputs, self.weights_copy)
                copy_score = tf.nn.tanh(copy_score)
                copy_score = tf.einsum("ijm,im->ij", copy_score, this_decoder_output)
                copy_score = self.mask_logits(self.encoder_mask, copy_score)


                mix_score = tf.concat([generate_score, copy_score], axis=1)  # batch * (vocab_size + encoder_max_len)
                probs = tf.cast(tf.nn.softmax(mix_score), tf.float32)
                prob_g = probs[:, :self.vocab_size]
                prob_c = probs[:, self.vocab_size:]

                encoder_inputs_one_hot = tf.one_hot(
                    indices=self.encoder_inputs,
                    depth=self.vocab_size + self.batch_OOV_num)
                prob_c = tf.einsum("ijn,ij->in", encoder_inputs_one_hot, prob_c)

                # if encoder inputs has intersection words with vocab dict,
                # move copy mode probability to generate mode probability

                prob_g = prob_g + prob_c[:, :self.vocab_size]
                prob_c = prob_c[:, self.vocab_size:]
                prob_final = tf.concat([prob_g, prob_c], axis=1) + 1e-10  # batch * (vocab_size + OOV_size)

                output_prob_list = output_prob_list.write(time, prob_final)


                return time + 1, state, prob_c, max_len, output_prob_list

            self.output_prob_list = tf.TensorArray(dtype=tf.float32, size=self.decoder_max_len, name="logits_list")
            _, _, _, _, self.output_prob_list = tf.while_loop(
                cond, body,
                loop_vars=[0,
                           self.decoder_init_state,
                           tf.zeros([self.batch_size, self.batch_OOV_num], dtype=tf.float32),
                           self.decoder_max_len,
                           self.output_prob_list
                           ]
            )
            self.output_probs = self.output_prob_list.stack()  # decoder_max_len * batch * (vocab_size + OOV_size)
            self.output_probs = tf.transpose(self.output_probs, perm=[1, 0, 2])  # batch * decoder_max_len * (vocab_size + OOV_size)

    def build_loss(self):
        with tf.variable_scope("loss"):
            self.decoder_outputs_one_hot = tf.one_hot(
                self.decoder_outputs, self.vocab_size + self.batch_OOV_num)
            self.crossent = - tf.reduce_sum(
                self.decoder_outputs_one_hot * tf.log(self.output_probs), -1)
            self.nonzeros = tf.count_nonzero(self.decoder_mask)
            self.decoder_mask = tf.cast(self.decoder_mask, dtype=tf.float32)
            self.train_loss = (tf.reduce_sum(self.crossent * self.decoder_mask) /
                               tf.cast(self.nonzeros, tf.float32))


