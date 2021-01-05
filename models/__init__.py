# python-specific libraries
import numpy as np
import math
# Scikit-learn libraries
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC


# from sklearn.model_selection import GridSearchCV
# from dask_searchcv import GridSearchCV


class XOR_PUF_MultilayerPerceptron(BaseEstimator, ClassifierMixin):
    train_time = 0.0

    def __init__(self, num_streams=2, num_stages=64, batch=200, solver='adam', n_layers=2):
        self.num_streams = num_streams
        self.num_stages = num_stages
        self.batch_size = batch
        self.weights = None
        self.intercepts = None
        self.solver = solver
        self.n_layers = n_layers

    def fit(self, C, r):

        arch = ()
        hidden = 0
        for _ in range(self.n_layers):
            if self.num_stages == 64:
                if self.num_streams == 7:
                    hidden = 64
                    self.batch_size = 10000
                elif self.num_streams == 8:
                    hidden = 128
                    self.batch_size = 1000000
                else:
                    hidden = 2 ** self.num_streams
            elif (self.num_stages == 128) and (self.num_streams == 7):
                hidden = 256 # 128 (try both and pick the best )
                self.batch_size = 1000000
            arch = arch + (hidden,)

        if self.solver == 'adam':
            print("\t\t>> Solver: ADAM | Activation: relu |Batch Size: %s |Hidden_layers:%s" % (self.batch_size, arch))
        else:
            print("\t\t>> Solver: L-BFGS |Hidden_layers= %s" % str(arch))
        self.estimator = MLPClassifier(hidden_layer_sizes=arch,
                                       activation='relu',
                                       solver=self.solver,
                                       learning_rate_init=1e-3,
                                       max_iter=100,
                                       batch_size=self.batch_size,
                                       early_stopping=True,
                                       verbose=True,
                                       weights=self.weights,
                                       intercepts=self.intercepts
                                       )
        self.estimator.fit(C, r)
        return self

    def predict(self, C):
        return self.estimator.predict(C)

    def predict_proba(self, C):
        return self.estimator.predict_proba(C)
