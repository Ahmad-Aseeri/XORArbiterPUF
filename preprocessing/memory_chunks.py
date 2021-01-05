import time, gc, datetime, sys, random, string, os
from sklearn.metrics import accuracy_score
import utilities.utils as utils
import numpy as np
from termcolor import colored

MASTER_CORE = 0

def handle_chunks(COM, args, gen, model, num_processors, rank, path):
    total_train_time = 0.0
    improvement_limit = 5   # how many times you want your model to iterate before terminating
    read_time = 0.0
    write_time = 0.0
    begin_offset = None
    best_round_acc = 0.0
    prev_round_acc = 0.0
    no_improvement = 0
    test_acc = 0.0
    train_time = 0.0
    num_chunks = int(round(1 + (args.challenges - 1) // args.chunk))
    _round_operation = 'w'  # Start by writing operation unless you already have the dataset
    _round = 0
    end_run = False
    if _round_operation == 'w':
        train_path = path + ''.join(random.choice(string.ascii_lowercase) for x in range(10)) + '.dat'
        test_path = path + ''.join(random.choice(string.ascii_lowercase) for x in range(10)) + '.dat'
    else:
        train_path = path + 'mpzycrmjlu.dat'
        test_path = path + 'pavcifyyzg.dat'

    while not end_run:
        if rank is MASTER_CORE:
            begin_offset = []
            print('\n\n ================ ROUND (%s) ================ ' % (_round + 1))
            sys.stdout.flush()

        round_begin_time = time.time()
        _round = _round + 1
        testing_chunk = 0
        total_io_time = 0.0

        chunk = utils.chunk_size(args.chunk, num_processors)
        for i in range(num_chunks):

            if i == (num_chunks - 1):
                training_chunk = int(args.challenges - (chunk * i))
                testing_chunk = testing_chunk + int(training_chunk * 0.2)
            else:
                training_chunk = chunk
                testing_chunk = testing_chunk + int(training_chunk * 0.2)

            # Just pretty printing!!
            if rank is MASTER_CORE:
                print('\n[%s] |[ Chunk(%s/%s) ]|:' % ((i + 1), (i + 1), num_chunks))
                if _round == 1:
                    print("\t>>> GENERATING (%sK) CRPs at (%s) |" %
                          ((training_chunk // 1000), datetime.datetime.now().strftime('%I:%M:%S %p')), end='')
                    sys.stdout.flush()

                else:
                    print("\t>>> LOADING (%sK) CRPs at {%s} .." %
                          ((training_chunk // 1000), datetime.datetime.now().strftime('%I:%M:%S %p')), end='')
                    sys.stdout.flush()
                # print('training_chunk=%s, testing_chunk=%s' % (training_chunk, testing_chunk))

            ''' ---------------------------------------------------------
                    STEP-1: Handle TRAINING
            --------------------------------------------------------- '''
            train_time, io_time = handle_train_chunk(COM, args, training_chunk, gen, model, rank, num_processors, _round_operation, train_path, i)
            if rank is MASTER_CORE:
                total_train_time = total_train_time + train_time
                total_io_time = total_io_time + io_time

            ''' ---------------------------------------------------------
                    STEP-2: Handle TESTING
            --------------------------------------------------------- '''

            testing_acc, io_time, end_run, _round_acc = handle_test_chunk(COM, args, training_chunk, testing_chunk, gen, model, rank, num_processors,
                                                                          _round_operation, test_path, i, best_round_acc, end_run)
            if rank is MASTER_CORE:
                best_round_acc = _round_acc
                test_acc = testing_acc
                total_io_time = total_io_time + io_time

            end_run = COM.bcast(end_run, root=0)
            if end_run:
                break
        # =========================== END (for i in range(num_chunks):) ===========================

        if rank is MASTER_CORE:
            print('\nROUND %s INFO:' % _round)
            # time_, unit = convert_time(float(round_io_time))
            # print('>> I/O time for round (%s)= %.3f %s' % (_round, time_, unit))
            if _round == 1:
                write_time = write_time + total_io_time
                time_, unit = utils.convert_time(float(write_time))
                print('>> Total Writing time= %.3f %s' % (time_, unit))
                sys.stdout.flush()
            else:
                read_time = read_time + total_io_time
                time_, unit = utils.convert_time(float(read_time))
                print('>> Reading time (Incremented)= %.3f %s' % (time_, unit))
                sys.stdout.flush()

        if end_run:
            # Exit while loop
            break
        else:
            if rank is MASTER_CORE:
                if prev_round_acc < best_round_acc:
                    print('>> Improvement! new Accuracy= %.2f%% ' % best_round_acc)
                    sys.stdout.flush()
                    prev_round_acc = best_round_acc
                else:
                    no_improvement = no_improvement + 1
                    print('>> NO improvement! Best Accuracy so far= %.2f%%' % best_round_acc)
                    sys.stdout.flush()
                    if no_improvement == improvement_limit:
                        # print('This run is terminated due to no improvement!!')
                        end_run = True
                del begin_offset

                time_, unit = utils.convert_time(float(time.time() - round_begin_time))
                print('>> Round Elapsed Time= %.3f %s' % (time_, unit))
                sys.stdout.flush()

        end_run = COM.bcast(end_run, root=0)
        if end_run:
            try:
                os.remove(train_path)
            except OSError or IOError:
                pass
            try:
                os.remove(test_path)
            except OSError or IOError:
                pass
            break
        else:
            _round_operation = 'r'

    # =========================== END of (while not end_run) =============================
    return test_acc, total_train_time, read_time, write_time


def handle_train_chunk(COM, args, training_chunk, gen, model, rank, num_processors, _round_operation, train_path, itr):
    train_time = 0.0
    io_time = 0.0

    if _round_operation is 'w':

        start = time.time()
        tr_C, tr_r = utils.data_generation(COM, training_chunk, gen, rank, num_processors)
        if rank is MASTER_CORE:
            time_, unit = utils.convert_time(float(time.time() - start))
            print(' Generation elapsed time: [ %.3f %s ]' % (time_, unit))
            print('\t>>> TRAINING: using %s:' % str(tr_C.shape))
            sys.stdout.flush()

            tr_C_ = utils.transformation(tr_C)
            start = time.time()
            model.fit(tr_C_, tr_r)
            end = time.time()
            del tr_C_
            gc.collect()

            train_time = end - start
            time_, unit = utils.convert_time(float(end - start))
            print('\t\t>> Train time: [ %.3f %s ]' % (time_, unit))
            sys.stdout.flush()

            # Handle writing train chunk on disk
            train_chunk_data = np.random.permutation(np.hstack((tr_C, tr_r.reshape(-1, 1))))

            start = time.time()
            utils.write_on_disk(train_path, train_chunk_data, itr)
            end = time.time()

            io_time = end - start
            time_, unit = utils.convert_time(float(end - start))
            print('\t\t>> Writing time is: [ %.3f %s ]' % (time_, unit))
            sys.stdout.flush()
            del train_chunk_data

        del tr_C, tr_r
        gc.collect()

    elif _round_operation is 'r':
        if rank is MASTER_CORE:
            r = random.randint(0, (args.challenges - training_chunk))

            start = time.time()
            start_read_offset = (r * (args.stages + 1)) * np.dtype(np.int8).itemsize
            num_to_read = training_chunk * (args.stages + 1) * np.dtype(np.int8).itemsize
            C = utils.read_from_disk(train_path, start_read_offset, num_to_read, (args.stages + 1))
            end = time.time()

            io_time = end - start
            time_, unit = utils.convert_time(float(end - start))
            print(' Reading time: [ %.3f %s ]' % (time_, unit))

            tr_r = C[:, -1]
            tr_C = np.delete(C, -1, axis=1)
            del C

            print('\t>>> TRAINING: using %s' % str(tr_C.shape))
            tr_C = utils.transformation(tr_C)
            start = time.time()
            model.fit(tr_C, tr_r)
            end = time.time()

            train_time = end - start
            time_, unit = utils.convert_time(float(end - start))
            print('\t\t>> Train time: [ %.3f %s ]' % (time_, unit))

            # train_acc = accuracy_score(tr_r, model.predict(tr_C)) * 100.
            # print('\t\t      >> Train ACC is: (%.2f%%) ' % train_acc)

            del tr_C, tr_r
            gc.collect()
    COM.Barrier()

    return train_time, io_time


def handle_test_chunk(COM, args, training_chunk, testing_chunk, gen, model, rank, num_processors, _round_operation, test_path, itr, _round_acc, end_run):
    io_time = 0.0
    test_acc = 0.0

    if _round_operation is 'w':
        if rank is MASTER_CORE:
            print('\t>>> TESTING: Generating (%sK) at {%s} |' %
                  ((int(training_chunk * 0.2) // 1000), datetime.datetime.now().strftime('%I:%M:%S %p')), end='')
            sys.stdout.flush()

        start = time.time()
        ts_C, ts_r = utils.data_generation(COM, int(training_chunk * 0.2), gen, rank, num_processors)
        end = time.time()

        if rank is MASTER_CORE:
            time_, unit = utils.convert_time(float(end - start))
            print(' Generation elapsed time= %.3f %s' % (time_, unit))
            sys.stdout.flush()

            test_chunk_data = np.hstack((ts_C, ts_r.reshape(-1, 1)))
            start = time.time()
            utils.write_on_disk(test_path, test_chunk_data, itr)

            end = time.time()
            io_time = io_time + (end - start)
            del test_chunk_data, ts_C, ts_r

            # To load the testing chunks from the beginning
            print('\t\t>> Load Test chunks (%sK) |' % testing_chunk, end='')
            start = time.time()
            start_read_offset = 0
            num_to_read = testing_chunk * (args.stages + 1) * np.dtype(np.int8).itemsize
            C = utils.read_from_disk(test_path, start_read_offset, num_to_read, (args.stages + 1))
            end = time.time()

            io_time = io_time + (end - start)
            time_, unit = utils.convert_time(float(end - start))
            print(' Loading time: [ %.3f %s ]' % (time_, unit))
            sys.stdout.flush()

            ts_r = C[:, -1]
            ts_C = np.delete(C, -1, axis=1)
            del C
            gc.collect()

            print('\t\t>> Test using %s' % str(ts_C.shape))
            ts_C = utils.transformation(ts_C)
            prd_r = model.predict(ts_C)
            test_acc = accuracy_score(ts_r, prd_r) * 100.

            if test_acc > _round_acc:
                _round_acc = test_acc
                print("\t>>> ACCURACY:", colored('%.2f%%', 'green') % test_acc)
                sys.stdout.flush()
            else:
                print('\t>>> ACCURACY: NOT improving!! Test_Acc=(%.2f%%)' % test_acc)
                sys.stdout.flush()

            # STEP: Handle writing
            if test_acc >= 98.0:
                end_run = True
            else:
                # Save the weight from this iteration to be used for the next iteration
                model.weights = model.estimator.coefs_
                model.intercepts = model.estimator.intercepts_
        del ts_C, ts_r
        gc.collect()

    elif _round_operation is 'r':
        if rank is MASTER_CORE:
            print('\t>>> TESTING: Loading (%sK) CRPs |' % (testing_chunk // 1000), end='')

            # C = np.genfromtxt(test_path, skip_header=0, max_rows=testing_chunk, dtype=np.int8)
            start = time.time()
            start_read_offset = 0
            num_to_read = (testing_chunk * (args.stages + 1)) * np.dtype(np.int8).itemsize
            C = utils.read_from_disk(test_path, start_read_offset, num_to_read, (args.stages + 1))
            print(C)
            end = time.time()

            io_time = io_time + (end - start)
            time_, unit = utils.convert_time(float(end - start))
            print(' Reading elapsed time: [ %.3f %s ]' % (time_, unit))
            sys.stdout.flush()

            ts_r = C[:, -1]
            ts_C = np.delete(C, -1, axis=1)
            del C

            print('\t>>> TESTING: using %s' % str(ts_C.shape))
            sys.stdout.flush()
            ts_C = utils.transformation(ts_C)
            prd_r = model.predict(ts_C)
            test_acc = accuracy_score(ts_r, prd_r) * 100.

            if test_acc > _round_acc:
                _round_acc = test_acc
                print("\t>>> ACCURACY:", colored('%.2f%%\n', 'green') % test_acc)
                sys.stdout.flush()
            else:
                print('\t>>> ACCURACY: NOT improving!! Test_Acc=(%.2f%%)\n' % test_acc)
                sys.stdout.flush()

            # Save the weight from this iteration to be used for the next iteration
            if test_acc >= 98.0:
                end_run = True
            else:
                model.weights = model.estimator.coefs_
                model.intercepts = model.estimator.intercepts_
            del ts_C, ts_r
            gc.collect()

    return test_acc, io_time, end_run, _round_acc
