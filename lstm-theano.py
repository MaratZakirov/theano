# This is theano code of LSTM for simple char-rnn model

import numpy
import theano
from theano import tensor as T, printing
from theano.tensor import tanh
from theano.tensor.nnet import sigmoid as sigm
from collections import OrderedDict
import matplotlib.pyplot as plt
import random

#theano.config.optimizer='fast_compile'
#theano.config.exception_verbosity='high'
#theano.config.compute_test_value = 'warn'

def init_data(shape):
    return numpy.random.uniform(low=-1.0, high=1.0, size=(shape)).astype('float32')

def init_sh_param(shape, name):
    return theano.shared(numpy.random.uniform(low=-1.0, high=1.0, size=(shape)).astype('float32'), name=name)

def init_sh_zero(shape, name):
    temp = numpy.zeros(shape=shape, dtype='float32')
    temp.fill(1)
    return theano.shared(temp, name=name)

class lstm:
    def __init__(self, input_size=2, inner_size=3, output_size=None, batch_size = 10, lr=0.01, gamma = 0.9):
        self.bsz = batch_size

        # Forget gate matrix
        self.W_f = init_sh_param(shape=(inner_size, inner_size), name='W_f')
        self.U_f = init_sh_param(shape=(inner_size, input_size), name='U_f')
        self.b_f = init_sh_param(shape=inner_size, name='b_f')

        # Insert gate matrix
        self.W_i = init_sh_param(shape=(inner_size, inner_size), name='W_i')
        self.U_i = init_sh_param(shape=(inner_size, input_size), name='U_i')
        self.b_i = init_sh_param(shape=inner_size, name='b_i')

        # Cell gate matrix
        self.W_c = init_sh_param(shape=(inner_size, inner_size), name='W_c')
        self.U_c = init_sh_param(shape=(inner_size, input_size), name='U_c')
        self.b_c = init_sh_param(shape=inner_size, name='b_c')

        # Output gate matrix
        self.W_o = init_sh_param(shape=(inner_size, inner_size), name='W_o')
        self.U_o = init_sh_param(shape=(inner_size, input_size), name='U_o')
        self.b_o = init_sh_param(shape=inner_size, name='b_o')

        # bundle
        self.params = [ self.W_f, self.U_f, self.b_f,
                        self.W_i, self.U_i, self.b_i,
                        self.W_c, self.U_c, self.b_c,
                        self.W_o, self.U_o, self.b_o]

        self.names = [ 'W_f', 'U_f', 'b_f',
                       'W_i', 'U_i', 'b_i',
                       'W_c', 'U_c', 'b_c',
                       'W_o', 'U_o', 'b_o']

        # Softmax layer
        if output_size != None:
            self.S   = init_sh_param((output_size, inner_size), name='S_softmax')
            self.b_s = init_sh_param(output_size, name='b_s')
            self.params.append(self.S)
            self.params.append(self.b_s)
            self.names.append('S_softmax_data')
            self.names.append('b_s_data')

        # RMSProp data
        self.params_data = []
        for elem, name in zip(self.params, self.names):
            self.params_data.append(init_sh_zero(elem.get_value().shape, name=name + '_data'))

        def step(x_t, h_t_1, C_t_1):
            f_t = T.dot(self.W_f, h_t_1) + T.dot(self.U_f, x_t)
            f_t = sigm(f_t.T + self.b_f).T
            i_t = T.dot(self.W_i, h_t_1) + T.dot(self.U_i, x_t)
            i_t = sigm(i_t.T + self.b_i).T
            o_t = T.dot(self.W_o, h_t_1) + T.dot(self.U_o, x_t)
            o_t = sigm(o_t.T + self.b_o).T
            C_t_c = T.dot(self.W_c, h_t_1) + T.dot(self.U_c, x_t)
            C_t_c = tanh(C_t_c.T + self.b_c).T
            C_t = f_t * C_t_1 + i_t * C_t_c
            h_t = o_t * T.tanh(C_t)
            return h_t, C_t

        x = T.ftensor3(name='x_input')
        y = T.fmatrix(name='y_input')
        (h_t, _), _ = theano.scan(fn=step, sequences=x,
                                  outputs_info=[T.zeros(shape=(inner_size, batch_size), dtype='float32'),
                                                T.zeros(shape=(inner_size, batch_size), dtype='float32')])
        h_last = h_t[-1]

        if output_size == None:
            E = T.sum((h_last - y) ** 2)
        else:
            j = T.nnet.softmax(T.dot(self.S, h_last).T + self.b_s).T
            E = T.sum((j - y) ** 2)

        gradients = T.grad(E, self.params)
        updates = []
        for param, grad, param_data in zip(self.params, gradients, self.params_data):
            r_t = (1 - gamma) * (grad ** 2) + gamma * param_data
            v_t_1 = lr * grad/T.sqrt(r_t)
            updates.append((param, param - v_t_1))
            updates.append((param_data, r_t))

        self.train = theano.function(inputs=[x, y], outputs=E, updates=OrderedDict(updates))

        t = T.zeros(shape=(inner_size, 1), dtype='float32')
        t = T.unbroadcast(t, 1)
        (h_t_2, _), _ = theano.scan(fn=step, sequences=x, outputs_info=[t, t])
        j_test = h_t_2[-1]
        if output_size != None:
            j_test = T.nnet.softmax(T.dot(self.S, j_test).T + self.b_s).T
        self.test = theano.function(inputs=[x], outputs=j_test)

    def train_on(self, data_x, data_y, epoches=400):
        data_size     = data_x.shape[2]
        data_size_int = data_size - data_size % self.bsz
        errs          = []
        for epoch in xrange(epoches):
            I = range(data_size)
            random.shuffle(I)
            ses_data_x = data_x[:, :, I]
            ses_data_y = data_y[:,    I]
            E = 0
            for sidx in xrange(0, data_size_int, self.bsz):
                batch_x = ses_data_x[:,:, sidx : sidx + self.bsz]
                batch_y = ses_data_y[:,   sidx : sidx + self.bsz]
                E_batch = self.train(batch_x, batch_y)
                E += E_batch
                #print 'Batch: ', sidx / self.bsz, '/', data_size_int / self.bsz, ' err: ', E_batch / self.bsz
            print 'Epoch: ', epoch, ' err: ', E / data_size_int
            errs.append(E / data_size_int)
        return errs

    def predict(self, data_x):
        assert data_x.shape[2] == 1
        return self.test(data_x)

path = "/home/zakirov/proj/semantic/data2/test.txt"
text = open(path).read().lower()
text = text.decode("utf-8")

print('corpus length:', len(text))

chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

maxlen = 6
step   = 5
sentences  = []
next_chars = []

for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])

print('Vectorization...')
X = numpy.zeros((maxlen, len(chars), len(sentences)), dtype=numpy.bool)
y = numpy.zeros((len(chars), len(sentences)), dtype=numpy.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[t, char_indices[char], i] = 1
    y[char_indices[next_chars[i]], i] = 1

print('nb sequences:', len(sentences))

print('Build model...')
rnn = lstm(input_size=len(chars), inner_size=6, output_size=len(chars), batch_size=100)#batch_size=y.shape[1])
errs = rnn.train_on(X, y)
plt.plot(xrange(len(errs)), errs)
print errs
plt.show()

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    if True:
        return numpy.argmax(preds)
    preds = numpy.asarray(preds).astype('float64')
    preds = numpy.log(preds) / temperature
    exp_preds = numpy.exp(preds)
    preds = exp_preds / numpy.sum(exp_preds)
    probas = numpy.random.multinomial(1, preds, 1)
    return numpy.argmax(probas)

for s1 in xrange(40):
    start_index = random.randint(0, len(text) - maxlen - 1)
    sentence = text[start_index : start_index + maxlen]
    generated = sentence + ">>>"
    for i in range(20):
        x = numpy.zeros((maxlen, len(chars), 1)).astype('float32')
        for t, char in enumerate(sentence):
            x[t, char_indices[char], 0] = 1.

        o = rnn.predict(x)[:, 0]

        next_index = sample(o, temperature=0.6)
        next_char = indices_char[next_index]

        generated += next_char
        sentence = sentence[1:] + next_char
    print "Generated ", generated
