import theano
import numpy
import load_w2v_data
from theano import tensor as T
from theano.tensor.shared_randomstreams import RandomStreams


def dataToSigmoid(X):
    for i in xrange(X.shape[0]):
        for j in xrange(X.shape[1]):
            if X[i][j] > 0:
                X[i][j] = 1.0
            else:
                X[i][j] = 0.0
    return X

def sigmoidToData(X):
    for i in xrange(X.shape[0]):
        for j in xrange(X.shape[1]):
            if X[i][j] > 0.5:
                X[i][j] = 1.0
            else:
                X[i][j] = -1.0
    return X

class RBM(object):
    """
    Resctricted Bolzmann Machine
    """
    def __init__(self, in_size, hid_size, alpha=0.001):
        self.in_size = in_size
        self.hid_size = hid_size

        self.W = theano.shared(numpy.random.normal(loc=0.0, scale=0.1, size=(in_size, hid_size)).astype('float32'))
        self.a = theano.shared(numpy.random.normal(loc=0.0, scale=0.01, size=hid_size).astype('float32'))
        self.b = theano.shared(numpy.random.normal(loc=0.0, scale=0.01, size=in_size).astype('float32'))

        self.randgen = RandomStreams(1234)

        self.X = T.matrix()
        self.dW = self.CollectStatiscs(self.X)
        self.learn = theano.function([self.X], updates=[(self.W, self.W + alpha * self.dW)], mode='DebugMode')

        _, X_recon, _ = self.GibbsStep(self.X)
        self.sample = theano.function([self.X], X_recon)

    def GibbsStep(self, V_sample_0, steps=4):
        H_mean = T.nnet.sigmoid(theano.dot(V_sample_0, self.W) + self.a)
        H_sample_0 = self.randgen.binomial(size=H_mean.shape, n=1, p=H_mean, dtype='float32')
        V_sample = V_sample_0
        H_sample = H_sample_0
        for step in xrange(steps - 1):
            V_mean = T.nnet.sigmoid(theano.dot(H_sample, self.W.T) + self.b)
            V_sample = self.randgen.binomial(size=V_mean.shape, n=1, p=V_mean, dtype='float32')
            H_mean = T.nnet.sigmoid(theano.dot(V_sample, self.W) + self.a)
            H_sample = self.randgen.binomial(size=H_mean.shape, n=1, p=H_mean, dtype='float32')
        return H_sample_0, V_sample, H_sample

    # Collect statistics from matrix values V_sample_0
    def CollectStatiscs(self, V_sample_0):
        H_sample_0, V_sample, H_sample = self.GibbsStep(V_sample_0)
        batch_size = T.cast(V_sample_0.shape[0], 'float32')
        P_stat = T.dot(V_sample_0.T, H_sample_0) / batch_size
        N_stat = T.dot(V_sample.T, H_sample) / batch_size
        return P_stat - N_stat

    def LearnOnFile(self, vocal, vec_size, query_size, file, batch_size=30):
        X_train, _ = load_w2v_data.fromLinesToData(open(file).readlines(), 2, vocal, vec_size, query_size)
        X_train = dataToSigmoid(X_train[0 : 1000])
        print "HHHHH"
        for i in xrange(0, X_train.shape[0], batch_size):
            X_batch = X_train[i : i + batch_size]
            self.learn(X_batch)
            print "Procesed: ", i

    def Sample(self, vocal, vec_size, query_size, testlines, batch_size=100):
        X_test, _ = load_w2v_data.fromLinesToData(testlines, 2, vocal, vec_size, query_size)
        X_test = dataToSigmoid(X_test)
        X_recon = sigmoidToData(self.SampleData(X_test))
        return load_w2v_data.fromVector2Words(X_recon, vocal, vec_size)

    def TestLearn(self, X_train, batch_size=10):
        self.learn(X_train)

    def SampleData(self, X_test):
        X_recon = self.sample(X_test)
        return X_recon

def TempLearn(vocfile, trainfile, testfile, query_size=6):
    vocal = load_w2v_data.loadVocal(vocfile)
    vec_size = vocal.values()[0].shape[0]
    m = RBM(vec_size * query_size, 200)
    m.LearnOnFile(vocal, vec_size, query_size, trainfile)
    lines_ref = []
    for line in open(testfile).readlines():
        if line.rstrip().split('\t')[1] == '1':
            lines_ref.append(line)
    lines_recon = m.Sample(vocal, vec_size, query_size, lines_ref)
    assert len(lines_recon) == len(lines_ref)
    fout = open('../data/temp.txt', 'w')
    assert len(lines_recon) == len(lines_ref)
    for i in xrange(len(lines_recon)):
        fout.write(lines_recon[i] + '\t' + lines_ref[i])

if __name__ == '__main__':
    vocfile = '../data/vec.txt.reduced'
    trainfile = '../data/DPI.txt.pos.epgu'
    testfile = '../data/DPItest.txt'
    TempLearn(vocfile, trainfile, testfile)
