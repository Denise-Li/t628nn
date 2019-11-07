import numpy as np
import pyopencl as cl
import datetime


def definitions(ks, vs):
    for k, v in zip(ks, vs):
        yield "-D"
        yield "%s=%s"%(k, v)


def definitions2(**kwargs):
    defs = list()
    for k in kwargs:
        key = k.upper()
        defs.append("-D")
        defs.append("%s=%s"%(key, kwargs[k]))
    return defs


def getmicro():
    now = datetime.datetime.now()
    now = now.microsecond + 1000000*now.second + 1000000*60*now.minute
    return now


def zero_buffer(ctx, shape, astype=np.float32):
    np_arr = np.zeros(shape=shape, dtype=astype)
    cl_arr = cl.Buffer(ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=np_arr)
    return np_arr, cl_arr


def arange_buffer(ctx, shape, astype=np.float32):
    total = 1
    for s in shape:
        total *= s
    np_arr = np.arange(0, total).astype(astype).reshape(shape)
    cl_arr = cl.Buffer(ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=np_arr)
    return np_arr, cl_arr

def create_buffer(ctx, shape, astype=np.float32, scaling=1.0):
    np_arr = (np.random.random_sample(size=shape).astype(astype) - 0.5)
    np_arr *= scaling
    cl_arr = cl.Buffer(ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=np_arr)
    return np_arr, cl_arr


def check_equal(a, b, epsilon=0.00001, silent=False):
    err = a - b
    err = (err ** 2).sum()

    if silent: return err < epsilon;

    if err < epsilon:
        print("CORRECT! Sum squared deviation:", err)
        return True
    else:
        print("BAD! Sum squared deviation:", err)
        return False


def get_cl(platform=0, device=0):
    """
    :param platform:
    :param device:
    :return: dev, ctx, q
    """
    dev = cl.get_platforms()[platform].get_devices()[device]
    ctx = cl.Context(devices=[dev])
    q = cl.CommandQueue(context=ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
    print("Working with device:", dev.get_info(cl.device_info.NAME))
    return dev, ctx, q


def get_stats(evt):
    end = evt.get_profiling_info(cl.profiling_info.END) / 1000
    start = evt.get_profiling_info(cl.profiling_info.START) / 1000
    return end - start


class Timer(object):
    def __init__(self, name=None, reps=1):
        self._start = None
        self._end = None
        self._reps = reps
        self._name = name + " " if name else None

    def __enter__(self):
        self._start = datetime.datetime.now()

    def __exit__(self, exc_type, exc_val, exc_tb):
        end = datetime.datetime.now()
        delta = (end - self._start)
        print("%sTime: %.2f"%(("" if self._name is None else self._name), (delta.seconds*1000000 + delta.microseconds)/self._reps))
