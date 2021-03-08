from __future__ import print_function
import timeit, argparse, os, sys
import XORArbiterPUF
from mpi4py import MPI

import generator
import models as md
import utilities.utils as utils

'''
Implementation of XOR Arbiter PUF Mathematical Clonability stated in this paper:  
https://ieeexplore.ieee.org/abstract/document/8473439/
Developed by Ahmad O. Aseeri, 2018 (a.aseeri@psau.edu.sa)

(1) This code parallelized the generation of CRPs only using MPI4py (https://mpi4py.readthedocs.io/en/stable/).
(2) Only MASTER_CORE will gather any matrices and process them as one matrix, followed by training the model in a signal core.

*****  follow README file instructions on how to install necessary libraries for this experiment ***
'''

def get_args():
    parser = argparse.ArgumentParser(
        description="XOR PUF Experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--puf', metavar="PUF", nargs='?',
                        default="linear", help='Type of arbiter (linear, ff, xor)')
    parser.add_argument('--stages', metavar="S", type=int,
                        default=64, help='Test size (0-1)')
    parser.add_argument('--challenges', metavar="C", type=int,
                        default=30000000, help='Number of challenges')
    parser.add_argument('--streams', metavar="SS", type=int,
                        default=8, help='Number of streams in XOR Arbiters')
    parser.add_argument('--runs', metavar="R", type=int,
                        default=1, help='Number of runs')
    parser.add_argument('--train', metavar='tr_ts_cv', type=int,
                        default=0, help='Choosing 0 for tr_ts_split and 1 for CV approach')
    parser.add_argument('--hpcc', metavar="hpcc", nargs='?',
                        default='no', help='Choosing yes for hpcc')
    parser.add_argument('--load', metavar="D", nargs='?',
                        default='no', help='Pre-generated input data')
    parser.add_argument('--store', metavar="saveFile", nargs='?',
                        default="no", help='')
    parser.add_argument('--solver', metavar="ChooseSolver", nargs='?',
                        default="adam", help='')
    parser.add_argument('--batch_size', metavar='batch_size', type=int,
                        default=100000, help='')
    parser.add_argument('--chunk', metavar='chunk_size', type=int,
                        default=14000000, help='')
    parser.add_argument('--layers', metavar='n_layers', type=int,
                        default=3, help='')
    parser.add_argument('--non_blocking', metavar="IO_operation", nargs='?',
                        default="no", help='')
    return parser.parse_args()

"""
--------------------------------------------------------------------------
                        Main Function
--------------------------------------------------------------------------
"""
MASTER_CORE = 0
if __name__ == "__main__":
    experiment_start_time = timeit.default_timer()

    # MPI initialization
    COM = MPI.COMM_WORLD
    rank = COM.Get_rank()
    size = COM.Get_size()
    mode = MPI.MODE_WRONLY | MPI.MODE_CREATE
    args = get_args()
    model = None
    gen = None
    path_to_results = None
    path_to_plot = None
    output = None
    filename = None

    path = ''

    # Generate directories to hold results
    if rank is MASTER_CORE:
        path_to_results = path+ 'PUF_Results/Best_Models/'
        filename = path + 'PUF_Results/%s%d_(%d)bit.txt' % (args.puf, args.streams, args.stages)
        filename_pkl = path_to_results + str(args.streams) + 'xor' + str(args.stages) + '.pkl'

        # Remove previous results, if exists
        try:
            os.makedirs(path_to_results)
        except OSError or IOError:
            pass

        try:
            os.makedirs(path)
        except OSError or IOError:
            pass

        try:
            os.remove(filename)
        except OSError or IOError:
            pass

        try:
            os.remove(filename_pkl)
        except OSError or IOError:
            pass

        output = open(filename, "a")
        model = md.XOR_PUF_MultilayerPerceptron(num_streams=args.streams,
                                                num_stages=args.stages,
                                                batch=args.batch_size,
                                                solver=args.solver,
                                                n_layers=args.layers)
        gen = generator.XORPUFGenerator(num_stages=args.stages,
                                        num_streams=args.streams,
                                        num_challenges=args.challenges)

        line = "\n-----------------------------------------------\n" \
               + '[{}] PUF: '.format('XOR') \
               + 'XORs={:d}, '.format(int(args.streams)) \
               + 'Stages={:d}bit, '.format(int(args.stages)) \
               + 'CRPs={:d}K\n'.format(int(args.challenges/1000)) \
               + "-----------------------------------------------\n"
        print(line)
        output.write(line)

    # Start XORArbiterPUF Model Training
    gen = COM.bcast(gen, root=0)
    XORArbiterPUF.XOR_Breaker(output, args, gen, model, COM, rank, size, path, path_to_results)

    if rank is MASTER_CORE:
        elapsed = timeit.default_timer() - experiment_start_time
        time_, unit = utils.convert_time(float(elapsed))
        output.write('Experiment Total time= %.3f %s\n' % (time_, unit))
        print('Experiment Total time= %.3f %s\n\n' % (time_, unit))
        output.close()
        # if sys.platform != 'darwin':
        #     utils.stop_ec2()