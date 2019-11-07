from subprocess import Popen, PIPE


def task(*args):
    args = list(map(str, args))
    process = Popen(args, stdout=PIPE)
    (output, err) = process.communicate()
    process.wait()
    return output

from .blast import *
