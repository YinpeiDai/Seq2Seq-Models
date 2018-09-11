import tensorflow as tf
import numpy as np
import copy

from model import *
from input_data import ToyDataset

class TrainManager:
    def __init__(self):
        # set vocabulary dictionary
        self.all_tokens = {}
        self.all_tokens['<PAD>'] = 0
        for ch in range(97, 119 + 1):
            self.all_tokens[chr(ch)] = len(self.all_tokens)
        self.all_tokens['<BOS>'] = len(self.all_tokens)
        self.all_tokens['<EOS>'] = len(self.all_tokens)
        self.all_tokens['<UNK>'] = len(self.all_tokens)
        self.vocab_size = len(self.all_tokens)
        # print(vocab_size)
        # we treat x,y,z as OOV
        self.vocab_dict = copy.deepcopy(self.all_tokens)
        self.all_tokens['x'] = len(self.all_tokens)
        self.all_tokens['y'] = len(self.all_tokens)
        self.all_tokens['z'] = len(self.all_tokens)
        self.reverse_vocab_dict = dict(zip(self.vocab_dict.values(), self.vocab_dict.keys()))
        self.dataset = ToyDataset()
        self.tested_examples = ['circumstances', 'affirmative', 'corresponding', 'caraphernology',
                                'experimentation', 'dizziness', 'harambelover', 'terrifyingly',
                                'axbycydxexfyzxxy']


    def tok2idx(self, batch_data):
        source_sent, target_sent = batch_data
        batchsize = len(source_sent)

        encoder_max_length = 0
        for sent in source_sent:
            encoder_max_length = max(encoder_max_length, len(sent))
        encoder_inputs = np.zeros(dtype=int, shape=(batchsize, encoder_max_length))
        for i, sent in enumerate(source_sent):
            for j, word in enumerate(sent):
                if word in self.vocab_dict:
                    encoder_inputs[i][j] = self.vocab_dict[word]

                else:
                    encoder_inputs[i][j] = self.vocab_dict['<UNK>']

        decoder_max_length = 0
        for sent in target_sent:
            decoder_max_length = max(decoder_max_length, len(sent))
        decoder_inputs = np.zeros(dtype=int, shape=(batchsize, decoder_max_length + 1))
        decoder_outputs = np.zeros(dtype=int, shape=(batchsize, decoder_max_length + 1))

        for i, sent in enumerate(target_sent):
            for j, word in enumerate(sent):
                if word in self.vocab_dict:
                    decoder_inputs[i][j + 1] = self.vocab_dict[word]
                    decoder_outputs[i][j] = self.vocab_dict[word]
                else:
                    decoder_inputs[i][j + 1] = self.vocab_dict['<UNK>']
                    decoder_outputs[i][j] = self.vocab_dict['<UNK>']

            decoder_inputs[i][0] = self.vocab_dict['<BOS>']
            decoder_outputs[i][len(sent)] = self.vocab_dict['<EOS>']

        return encoder_inputs, decoder_inputs, decoder_outputs

    def run_model(self):
        pass




class Train_BasicEncoderDecoder(TrainManager):

    def run_model(self):
        model = BasicEncoderDecoder(name='BasicEncoderDecoder', vocab_size=self.vocab_size)
        plot_every_steps = 100
        print('train BasicEncoderDecoder ... ')
        with tf.Session() as sess:
            sess.run(tf.group(tf.global_variables_initializer()))
            average_loss = 0
            for step in range(1, 50000):
                encoder_inputs, decoder_inputs, decoder_outputs, \
                    = self.tok2idx(self.dataset.get_batch())

                _, training_loss = sess.run([model.train_op, model.train_loss],
                                            feed_dict={
                                                model.encoder_inputs:encoder_inputs,
                                                model.decoder_inputs:decoder_inputs,
                                                model.decoder_outputs:decoder_outputs,
                                                model.keep_prob: 0.5})
                average_loss += training_loss / plot_every_steps

                if step % plot_every_steps == 0:
                    encoder_inputs, decoder_inputs, decoder_outputs, \
                        = self.tok2idx(self.dataset.get_test_data())
                    accu = sess.run(model.accuracy,
                                    feed_dict={
                                        model.encoder_inputs:encoder_inputs,
                                        model.decoder_inputs:decoder_inputs,
                                        model.decoder_outputs:decoder_outputs,
                                        model.keep_prob: 1.0})

                    # test some difficult examples
                    encoder_inputs, decoder_inputs, decoder_outputs, \
                        = self.tok2idx((self.tested_examples, self.tested_examples))
                    pred = sess.run(model.predict,
                                    feed_dict={
                                        model.encoder_inputs: encoder_inputs,
                                        model.decoder_inputs: decoder_inputs,
                                        model.decoder_outputs: decoder_outputs,
                                        model.keep_prob: 1.0})

                    for ii in range(len(self.tested_examples)):
                        print()
                        print('true output:',
                              " ".join([self.reverse_vocab_dict[word_id] for word_id in decoder_outputs[ii]]))
                        print('pred output:', " ".join([self.reverse_vocab_dict[word_id] for word_id in pred[ii]]))
                    print("step %5d, loss=%0.4f accu=%0.4f" % (step, average_loss, accu))
                    average_loss = 0




class Train_AttenNet(TrainManager):
    def run_model(self):
        model = AttenNet(name='AttenNet', vocab_size=self.vocab_size)
        plot_every_steps = 100
        print('train EncoderDecoder with attention mechanism ... ')
        with tf.Session() as sess:
            sess.run(tf.group(tf.global_variables_initializer()))
            average_loss = 0
            for step in range(1, 50000):
                encoder_inputs, decoder_inputs, decoder_outputs, \
                    = self.tok2idx(self.dataset.get_batch())

                _, training_loss = sess.run([model.train_op, model.train_loss],
                                            feed_dict={
                                                model.encoder_inputs:encoder_inputs,
                                                model.decoder_inputs:decoder_inputs,
                                                model.decoder_outputs:decoder_outputs,
                                                model.keep_prob: 0.5})
                average_loss += training_loss / plot_every_steps

                if step % plot_every_steps == 0:
                    encoder_inputs, decoder_inputs, decoder_outputs, \
                        = self.tok2idx(self.dataset.get_test_data())
                    accu = sess.run(model.accuracy,
                                    feed_dict={
                                        model.encoder_inputs:encoder_inputs,
                                        model.decoder_inputs:decoder_inputs,
                                        model.decoder_outputs:decoder_outputs,
                                        model.keep_prob: 1.0})

                    # test some difficult examples
                    encoder_inputs, decoder_inputs, decoder_outputs, \
                        = self.tok2idx((self.tested_examples, self.tested_examples))
                    pred, cum_att_weights = sess.run([model.predict, model.cum_att_weights],
                                    feed_dict={
                                        model.encoder_inputs: encoder_inputs,
                                        model.decoder_inputs: decoder_inputs,
                                        model.decoder_outputs: decoder_outputs,
                                        model.keep_prob: 1.0})

                    for ii in range(len(self.tested_examples)):
                        print()
                        print('true output:',
                              " ".join([self.reverse_vocab_dict[word_id] for word_id in decoder_outputs[ii]]))
                        print('pred output:', " ".join([self.reverse_vocab_dict[word_id] for word_id in pred[ii]]))
                        print('    ', end=' ')
                        for word_id in encoder_inputs[ii]:
                            print('%5s' % self.reverse_vocab_dict[word_id], end=' ')
                        print()
                        for i, prob in enumerate(cum_att_weights[ii]):
                            print('%5s' % self.reverse_vocab_dict[decoder_outputs[ii][i]], end=' ')
                            for p in prob:
                                print('%0.3f' % p, end=' ')
                            else:
                                print()
                        print()
                    print("step %5d, loss=%0.4f accu=%0.4f" % (step, average_loss, accu))
                    average_loss = 0


class Train_CopyNet(TrainManager):
    def tok2idx(self, batch_data):
        source_sent, target_sent = batch_data
        batchsize = len(source_sent)

        encoder_max_length = 0
        for sent in source_sent:
            encoder_max_length = max(encoder_max_length, len(sent))
        encoder_inputs = np.zeros(dtype=int, shape=(batchsize, encoder_max_length))

        batch_OOV_tokens = []

        for i, sent in enumerate(source_sent):
            OOV_token = list(set([w for w in sent if w not in self.vocab_dict]))
            batch_OOV_tokens.append(OOV_token)
            for j, token in enumerate(sent):
                if token in self.vocab_dict:
                    encoder_inputs[i][j] = self.vocab_dict[token]
                else:
                    encoder_inputs[i][j] = self.vocab_size + OOV_token.index(token)

        decoder_max_length = 0
        for sent in target_sent:
            decoder_max_length = max(decoder_max_length, len(sent))

        decoder_inputs = np.zeros(dtype=int, shape=(batchsize, decoder_max_length + 1))
        decoder_outputs = np.zeros(dtype=int, shape=(batchsize, decoder_max_length + 1))
        for i, sent in enumerate(target_sent):
            for j, token in enumerate(sent):
                if token in self.vocab_dict:
                    decoder_inputs[i][j + 1] = self.vocab_dict[token]
                    decoder_outputs[i][j] = self.vocab_dict[token]

                elif token in batch_OOV_tokens[i]:
                    decoder_inputs[i][j + 1] = self.vocab_size + batch_OOV_tokens[i].index(token)
                    decoder_outputs[i][j] = self.vocab_size + batch_OOV_tokens[i].index(token)

                else:
                    decoder_inputs[i][j + 1] = self.vocab_dict['<UNK>']
                    decoder_outputs[i][j] = self.vocab_dict['<UNK>']

            decoder_inputs[i][0] = self.vocab_dict['<BOS>']
            decoder_outputs[i][len(sent)] = self.vocab_dict['<EOS>']

        return encoder_inputs, decoder_inputs, decoder_outputs, batch_OOV_tokens

    def run_model(self):
        model = CopyNet(name='CopyNet', vocab_size=self.vocab_size)
        plot_every_steps = 100
        print('train EncoderDecoder with copy mechanism ... ')

        with tf.Session() as sess:
            sess.run(tf.group(tf.global_variables_initializer()))
            average_loss = 0

            for step in range(1, 50000):
                encoder_inputs, decoder_inputs, decoder_outputs, batch_OOV_tokens \
                    = self.tok2idx(self.dataset.get_batch())

                _, training_loss = sess.run([model.train_op, model.train_loss],
                                            feed_dict={model.encoder_inputs: encoder_inputs,
                                                       model.decoder_inputs: decoder_inputs,
                                                       model.decoder_outputs: decoder_outputs,
                                                       model.batch_OOV_num: np.max([len(tokens) for tokens in
                                                                             batch_OOV_tokens]),
                                                       model.keep_prob: 0.5})
                average_loss += training_loss / plot_every_steps

                if step % plot_every_steps == 0:
                    encoder_inputs, decoder_inputs, decoder_outputs, batch_OOV_tokens \
                        = self.tok2idx(self.dataset.get_test_data())
                    accu = sess.run(model.accuracy,
                                    feed_dict={model.encoder_inputs: encoder_inputs,
                                               model.decoder_inputs: decoder_inputs,
                                               model.decoder_outputs: decoder_outputs,
                                               model.batch_OOV_num: np.max([len(tokens) for tokens in
                                                                             batch_OOV_tokens]),
                                               model.keep_prob: 1.0})


                    encoder_inputs, decoder_inputs, decoder_outputs, batch_OOV_tokens \
                        = self.tok2idx((self.tested_examples, self.tested_examples))
                    pred = sess.run(model.predict,
                                    feed_dict={model.encoder_inputs: encoder_inputs,
                                               model.decoder_inputs: decoder_inputs,
                                               model.decoder_outputs: decoder_outputs,
                                               model.batch_OOV_num: np.max([len(tokens) for tokens in
                                                                             batch_OOV_tokens]),
                                               model.keep_prob: 1.0})

                    for ii in range(len(self.tested_examples)):
                        print()
                        print('true output:', " ".join([self.reverse_vocab_dict[word_id]
                                                        if word_id < self.vocab_size
                                                        else batch_OOV_tokens[ii][word_id - self.vocab_size]
                                                        for word_id in decoder_outputs[ii]]
                                                       ))
                        print('pred output:', " ".join([self.reverse_vocab_dict[word_id]
                                                        if word_id < self.vocab_size
                                                        else batch_OOV_tokens[ii][word_id - self.vocab_size]
                                                        for word_id in pred[ii]]
                                                       ))
                        print()
                    print("step %5d, loss=%0.4f accu=%0.4f" % (step, average_loss, accu))
                    average_loss = 0