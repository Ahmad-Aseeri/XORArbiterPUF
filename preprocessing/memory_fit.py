import time, gc, sys
from sklearn.metrics import accuracy_score
import utilities.utils as utils
import numpy as np
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from collections import Counter
MASTER_CORE = 0


def train_test_inMemory(COM, args, gen, model, num_processors, rank, path):
    test_acc = 0.0
    train_acc = 0.0
    train_time = 0.0

    if rank is MASTER_CORE:
        print('\t[1] Memory Fit| CRP Generation ...')
        print("\t\t>>> Generating %s CRPs |" % args.challenges, end='')
        sys.stdout.flush()

    start = time.time()
    # tr_C, tr_r = utils.data_generation(COM, args.challenges, gen, rank, num_processors)
    # ts_C, ts_r = utils.data_generation(COM, int(args.challenges * 0.2), gen, rank, num_processors)
    C_, r = utils.data_generation(COM, args.challenges, gen, rank, num_processors)
    end = time.time()

    if rank is MASTER_CORE:
        time_, unit = utils.convert_time(float(end - start))
        print(' Generation elapsed time: [ %.3f %s ]' % (time_, unit))

        print('\t[2] Model Training ...')
        C = utils.transformation(C_)
        tr_C, ts_C, tr_r, ts_r = train_test_split(C, r, train_size=.8)
        print("\t>>> Train_set= %s, Test_set= %s " % (str(tr_C.shape), str(ts_C.shape)))
        start = time.time()
        model.fit(tr_C, tr_r)
        end = time.time()

        train_time = end - start
        train_acc = accuracy_score(tr_r, model.predict(tr_C)) * 100.
        test_acc = accuracy_score(ts_r, model.predict(ts_C)) * 100.

    # del tr_C, tr_r, ts_C, ts_r
    gc.collect()
    return test_acc, train_acc, train_time
