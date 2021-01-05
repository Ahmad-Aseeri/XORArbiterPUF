# General imports
from __future__ import print_function
import os, datetime, sys
import numpy as np
from termcolor import colored

# Project imports
import preprocessing.memory_fit as mf
import preprocessing.memory_chunks as mc
import utilities.utils as utils

MASTER_CORE = 0


def XOR_Breaker(output, args, gen, model, COM, rank, num_processors, path, path_to_dir_models):
    metrics = {m: [] for m in ['acc', 'time']}
    max_acc = 0.0
    write_time = 0.0
    read_time = 0.0

    for run_index in range(args.runs):

        if rank is MASTER_CORE:
            print('                                #########################')
            print('--------------------------------# Run[%d]   {%s} # ---------------------------------' % (
            (run_index + 1), datetime.datetime.now().strftime('%m-%d-%Y')))
            print('                                #########################')
            line = '\n\n-----------| RUN ' + str(run_index + 1) + ' |-----------\n'
            output.write(line)

        if ((args.stages == 64) and (args.streams < 4)) or ((args.stages == 128) and (args.streams < 6)):

            """ ----------------------------------------------------------------------
                        This case handles training without chunks 
            ---------------------------------------------------------------------- """

            test_acc, train_acc, train_time = mf.train_test_inMemory(COM, args, gen, model, num_processors, rank, path)
            if rank is MASTER_CORE:
                if train_acc >= 98.0:
                    print("\t>>> Test ACCURACY:", colored('%.2f%%', 'green') % test_acc)
                else:
                    print("\t>>> Test ACCURACY:", colored('%.2f%%', 'red') % test_acc)

                if train_acc >= 98.0:
                    print("\t>>> Train ACCURACY:", colored('%.2f%%\n', 'green') % train_acc)
                else:
                    print("\t>>> Train ACCURACY:", colored('%.2f%%\n', 'red') % train_acc)
        else:
            """ ----------------------------------------------------------------------
                        This case handles chunking approach
            ---------------------------------------------------------------------- """
            test_acc, train_time, read_time, write_time = mc.handle_chunks(COM, args, gen, model, num_processors, rank, path)

        if rank is MASTER_CORE:
            metrics['acc'].append(test_acc)
            metrics['time'].append(train_time)

            # Save the best model
            if np.max(metrics['acc']) > max_acc:
                max_acc = np.max(metrics['acc'])
                model_name = path_to_dir_models + str(args.streams) + 'xor' + str(args.stages) + '.pkl'
                try:
                    os.remove(model_name)
                except OSError or IOError:
                    pass
                # joblib.dump(model.estimator, model_name)

                line1 = '|' + str('RESULT') + '| \n' \
                        + 'AVG_ACC= {:.2f}, '.format(np.mean(metrics['acc'])) \
                        + 'TEST_ACC= {:.2f}, '.format(test_acc) \
                        + '[ BEST_ACC= {:.2f} ], '.format(np.max(metrics['acc'])) \
                        + 'AVG_TIME= {:.4f}, '.format(np.mean(metrics['time'])) \
                        + 'TRAIN_TIME= {:.4f}'.format(train_time)

            else:
                line1 = '|' + str('RESULT') + '| \n' \
                        + 'AVG_ACC= {:.2f}, '.format(np.mean(metrics['acc'])) \
                        + 'TEST_ACC= {:.2f}, '.format(test_acc) \
                        + 'AVG_TIME= {:.4f}, '.format(np.mean(metrics['time'])) \
                        + 'TRAIN_TIME= {:.4f}'.format(train_time)
            output.write(line1)

            print("\n\n __:: Results for Run (%s) ::__" % (run_index + 1))
            if (run_index + 1) == args.runs:
                print(">> Average Test_Acc=", colored('%.2f%%', 'yellow') % float(np.mean(metrics['acc'])),
                      "|(BEST=", colored('%.2f%%', 'yellow') % np.max(metrics['acc']), ")")
            else:
                print(">> Average Test_Acc=", colored('%.2f%%', 'yellow') % float(np.mean(metrics['acc'])))

            # print(">> Training set size= %s CRPs" % training_chunk_size)
            train_time, unit = utils.convert_time(float(np.mean(metrics['time'])))
            print(">> Average Train_Time= %.3f %s" % (train_time, unit))

            time_, unit = utils.convert_time(float(read_time + write_time))
            if time_ != 0.0:
                print(">> Total I/O Time= %.3f %s" % (time_, unit))
            else:
                print(">> NO I/O Operations involved")

            print('**********************************************\n')
            output.write('\n\n*************************************\n')
            output.write('Total I/O Time= %.3f %s\n' % (time_, unit))
            sys.stdout.flush()
