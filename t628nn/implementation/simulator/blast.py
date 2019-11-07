import os
import sys
from . import task


"""
This file contains functions for individually measuring the CLBlast operations that compose each neural network layer.

It makes use of the compiled executables from the clblastsimulations directory .cpp files.
"""


if "BLAST_SIMULATOR_PATH" not in os.environ:
    sys.stderr.write("Cannot run the simulator!\n")
    sys.stderr.write("BLAST_SIMULATOR_PATH environment variable should contain the path to your clblast layers!\n")
    sys.stderr.write("The layers are in the clblastsimulations directory of the repository.\n")
    sys.stderr.write("Compile the .cpp files and add a path to them to BLAST_SIMULATOR_PATH.\n")
    sys.stderr.write("The path should contain 'xcopy', 'xgemm', 'xgemv', 'xim2col' and 'xgemmbatched' executables.\n")
    exit(-1)


def get_executable(name):
    return os.path.join(os.environ["BLAST_SIMULATOR_PATH"], name)


def get_run_time(stdout):
    return float(stdout.split()[0])


def mean(l):
    return float(sum(l))/len(l)


def blast_fully_connected(m, k, n):
    return {
        'xCOPY': mean([get_run_time(task(get_executable("xcopy"), m * n)) for _ in range(3)]),
        'xGEMM': mean([get_run_time(task(get_executable("xgemm"), m, k, n)) for _ in range(3)])
    }


def blast_fully_connected_single(m, n):
    return {
        'xCOPY': mean([get_run_time(task(get_executable("xcopy"), m)) for _ in range(3)]),
        'xGEMV': mean([get_run_time(task(get_executable("xgemv"), m, n)) for _ in range(3)])
    }


def blast_conv(xs, xc, xh, xw, fs, fh, fw, sx, sy):
    yw = (xw - fw) / sx + 1
    yh = (xh - fh) / sy + 1
    return {
        'xIM2COL': mean([get_run_time(task(get_executable("xim2col"), xs * xc, xh, xw, fh, fw, sx, sy)) for _ in range(3)]),
        'xCOPY': mean([get_run_time(task(get_executable("xcopy"), xs * fs * yh * yw)) for _ in range(3)]),
        'xGEMMBATCHED': mean([get_run_time(task(get_executable("xgemmbatched"), xs, fs, fw * fh * xc, yw * yh)) for _ in range(3)])
    }


def blast_conv_single(xc, xh, xw, fs, fh, fw, sx, sy):
    yw = (xw - fw) / sx + 1
    yh = (xh - fh) / sy + 1
    return {
        'xIM2COL': mean([get_run_time(task(get_executable("xim2col"), xc, xh, xw, fh, fw, sx, sy)) for _ in range(3)]),
        'xCOPY': mean([get_run_time(task(get_executable("xcopy"), fs * yh * yw)) for _ in range(3)]),
        'xGEMM': mean([get_run_time(task(get_executable("xgemm"), fs, fw * fh * xc, yw * yh)) for _ in range(3)])
    }


def blast_subsampling(xs, xc, xh, xw, fh, fw, sx, sy):
    yw = (xw - fw) / sx + 1
    yh = (xh - fh) / sy + 1
    return {
        'xIM2COL': mean([get_run_time(task(get_executable("xim2col"), xs * xc, xh, xw, fh, fw, sx, sy)) for _ in range(3)]),
        'xCOPY': mean([get_run_time(task(get_executable("xcopy"), xs * xc * yh * yw)) for _ in range(3)]),
        'xGEMMBATCHED': mean([get_run_time(task(get_executable("xgemmbatched"), xs, xc, fw * fh * xc, yw * yh)) for _ in range(3)])
    }

