import numpy as np
import pyopencl as cl
from sanity_check import check_equal
from cl_util.tools import zero_buffer, get_cl


code = """
    __kernel void zero_out(__global float* matrix){
        matrix[get_global_id(0)] = 0;
    }
"""


class ZeroOp:
    def __init__(self, ctx, size):
        self._size = size
        self._ctx = ctx
        self._prog = cl.Program(ctx, code).build()

    def get_output_size(self):
        return self._size

    def __call__(self, q, inout=None):
        inout_np = None
        if inout is None:
            inout_np, inout = zero_buffer(self._ctx, (self._size, ))
        kernel = self._prog.zero_out
        kernel.set_args(inout)
        evt = cl.enqueue_nd_range_kernel(q, kernel, (self._size, ), None)

        return evt, inout_np, inout


def test_zero_op():
    dev, ctx, q = get_cl()
    op = ZeroOp(ctx, 5000)
    evt, zero_np, zero_cl = op(q)
    cl.enqueue_copy(q, zero_np, zero_cl)
    q.flush()
    assert check_equal(np.zeros(5000), zero_np)
    zero_np = np.random.random_sample((5000,)).astype(np.float32)
    cl.enqueue_copy(q, zero_cl, zero_np)
    q.flush()
    evt, zero_np2, zero_cl = op(q, zero_cl)
    assert zero_np2 is None
    assert not check_equal(np.zeros(5000), zero_np)
    cl.enqueue_copy(q, zero_np, zero_cl)
    q.flush()
    assert check_equal(np.zeros(5000), zero_np)
    assert 5000 == op.get_output_size()
