from tempfile import NamedTemporaryFile

import numpy as np
import os, resource, math, multiprocessing, itertools, sys, random
import boto.ec2
import boto.utils
import boto, boto3

from generator import LinearPUFGenerator


def data_generation(COM, num_challenges, gen, rank, num_processors):
    C, r = gen.generate(num_challenges, num_processors, rank)

    gather_C = COM.gather(C, root=0)
    COM.Barrier()
    gather_r = COM.gather(r, root=0)
    COM.Barrier()

    if rank == 0:
        C = np.asarray(list(itertools.chain(*gather_C))) # , dtype=np.int8
        r = np.asarray(list(itertools.chain(*gather_r))) # , dtype=np.int8

    del gather_C, gather_r
    COM.Barrier()
    return C, r


def stop_ec2():
    conn = boto.ec2.connect_to_region('us-east-1',
                                      aws_access_key_id='AKIAJVEPNSZLFQ5LKGEQ',
                                      aws_secret_access_key='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')
    my_id = boto.utils.get_instance_metadata()['instance-id']  # Get the current instance's id
    logger.info(' stopping EC2 :' + str(my_id))
    conn.stop_instances(instance_ids=[my_id])


# def data_generation_toFile(COM, num_challenges, gen, rank, num_processors):
#
#     num_challenges = num_challenges / 5
#     for i in range(1, 6):
#         C, r = gen.generate(num_challenges, num_processors, rank)
#         gather_C = COM.gather(C, root=0)
#         COM.Barrier()
#         gather_r = COM.gather(r, root=0)
#         COM.Barrier()
#
#         if rank == 0:
#             C = np.asarray(list(itertools.chain(*gather_C)))
#             # np.set_printoptions(edgeitems=10)
#             # print(C)
#             # print("\n\n\n\n")
#
#             if i == 1:
#                 write_on_disk('/Users/Ahmad/Desktop/xor4_64_tr.dat', C, 0)
#             else:
#                 write_on_disk('/Users/Ahmad/Desktop/xor4_64_tr.dat', C, -1)
#         del gather_C, C, r
#
#     COM.Barrier()
#     sys.exit(0)
#     return C, r

# def transformation(C):
#     # Transform the 0-1 challenge to -1 and +1.
#     V = 2. * C - 1
#
#     # Compute the cumulative product of the side.
#     V = np.cumprod(V, axis=1, dtype=np.int8)
#
#     # Add the bias term.
#     # V = np.hstack((V, np.ones((C.shape[0], 1)))).astype(np.int8)
#
#     return V

def transformation(C):

    # Transform the 0-1 challenge to -1 and +1.
    V = 2 * C - 1
    # V = np.fliplr(V)

    # Compute the cumulative product (right to left)
    V = np.cumprod(V, axis=1, dtype=np.int8)
    # V = np.fliplr(V)

    return V

def chunk_size(chunk, num_processors):
    '''
    To Ensure the chunk_size is divisible by the number of processors
    :return:
    '''

    chunk_size = 0
    for i in range(chunk, (chunk + 1000)):
        if i % num_processors == 0:
            chunk_size = i
            break
    return chunk_size

def write_on_disk(path, data, i):
    if i == 0:
        fd = os.open(path, os.O_CREAT | os.O_WRONLY)
    else:
        fd = os.open(path, os.O_APPEND | os.O_WRONLY)
    os.write(fd, data.tobytes())
    os.close(fd)

def read_from_disk(path, start_read, num_to_read, dim):
    fd = os.open(path, os.O_RDONLY)
    os.lseek(fd, start_read, 0)  # Where to (start_read) from the beginning 0
    ret = os.read(fd, num_to_read)  # How many to read (num_to_read)
    C = np.frombuffer(ret, dtype=np.int8).reshape(-1, dim)
    os.close(fd)
    return C

def convert_size(size_bytes):
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes // p, 2)
    return "%s %s" % (s, size_name[i])

def memory_usage():
    # Return peak memory usage (bytes on OS X, kilobytes on Linux) for the calling process
    mem_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return convert_size(mem_usage)

def convert_time(time_):
    if time_ < 60:
        return time_, 'sec(s)'
    elif 60 < time_ < 3600:
        return (time_ / 60), 'min(s)'
    elif time_ > 3600:
        return (time_ / 3600), 'hour(s)'

def output_loss(loss_dict, loss_values, output):
    '''
    Here printing the loss values:
    (1) In the case of normal training/testing, we print the losses values of the (best) model accuracy
    (2) For the chunking, we take the loss value (model.loss_) which is s single value representing each chunk
    '''

    li = '\n\n-----|' + str('LOSS') + '|----- \n'
    output.write(li)
    if bool(loss_dict):
        output.write(str(loss_dict))
    else:
        output.write(str(loss_values))

def generate_noise(size, dim):

    X = [random.randint(0, 2 ** dim) for _ in range(size)]

    # Removes duplicates to avoid using them in training and test simulatenously
    X = list(set(X))

    # Transforms the challenges into their string representation
    # (this step could be avoided and implemented in a more efficient way)
    X = [('{0:0' + str(dim) + 'b}').format(x) for x in X]

    # Transforms each bit into an integer.
    X = np.asarray([list(map(int, list(x))) for x in X], dtype=np.int64)

    return X


# def enhanced_reading(filepath, start, chunk_size):
#     '''
#     This method performs reading from file 4x faster than classical reading via numpy
#     '''
#
#     jobs = []
#     pipe_list = []
#     offset = 10000
#     num_itr = int(chunk_size // offset)
#     # print('----> data_size=%s, offset= %s, start=%s .... num_itr= %s \n\n\n' % (chunk_size, offset, start, num_itr))
#
#     for i in range(num_itr):
#         recv_end, send_end = multiprocessing.Pipe(False)
#         p = multiprocessing.Process(target=file_read, args=(filepath, start, offset, send_end))
#         jobs.append(p)  # collect all processes so you close them later
#         pipe_list.append(recv_end)
#         p.start()
#         start = start + offset
#
#     C = []
#     for i in pipe_list:
#         C.append(i.recv())
#     C = np.vstack(C)
#
#     # close the processes
#     for proc in jobs:
#         proc.join()
#
#     return C


def test_write_read():
    """Make sure hard disk read/write operations do not change CRP data"""
    n = 64
    N = 13000
    puf = LinearPUFGenerator(n)
    C, r = puf.generate(N, 1, 0)

    with NamedTemporaryFile() as f:
        write_on_disk(f.name, np.hstack((C, r.reshape(-1, 1))), 0)
        read_len = N * (n + 1) * np.dtype(np.int8).itemsize
        disk_data = read_from_disk(f.name, 0, read_len, n + 1)
    disk_r = disk_data[:, -1]
    disk_C = np.delete(disk_data, -1, axis=1)

    assert (disk_C == C).all()
    assert (disk_r == r).all()
    assert C.dtype == disk_C.dtype == np.int8
    assert r.dtype == disk_r.dtype == np.int8
