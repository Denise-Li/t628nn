import pyopencl as cl
from sanity_check import check_equal, insert, max_pool
from cl_operations import *
from cl_util.tools import get_cl, create_buffer


class MaxPoolLayer:
    def __init__(self, ctx, xshape, fshape, strides):
        xn, xd, xh, xw = xshape
        fh, fw = fshape
        sh, sw = strides

        self._max_pool = MaxPoolOp(ctx, xshape, fshape, strides)

        self._params = xh, xd, xh, xw, fh, fw, sh, sw

    def __call__(self, q, zeroer, buff1, buff2):
        evts = list()
        ze1, _, _ = zeroer(q, buff2)
        evts.append(ze1)
        evt1, _, _ = self._max_pool(q, buff1, buff2)
        evts.append(evt1)

        cl.wait_for_events(evts)
        return buff2, buff1

    def get_output_size(self):
        return self._max_pool.get_output_size()


def test_convolution():

    dev, ctx, q = get_cl()

    xnp, xcl = create_buffer(ctx, (1, 16, 28, 30))

    layer = MaxPoolLayer(ctx, xnp.shape, (2, 2), (2, 2))

    expected = max_pool(xnp, (2, 2), (2, 2))

    buff1np, buff1cl = create_buffer(ctx, (16*28*30,))
    buff2np, buff2cl = create_buffer(ctx, (16*28*30,))

    insert(xnp, buff1np)
    cl.wait_for_events([cl.enqueue_copy(q, buff1cl, buff1np)])

    zeroer = ZeroOp(ctx, 16*28*30)

    outcl, _ = layer(q, zeroer, buff1cl, buff2cl)

    cl.wait_for_events([cl.enqueue_copy(q, buff1np, outcl)])

    actual = buff1np[:16*14*15].reshape((1, 16, 14, 15))

    assert check_equal(expected, actual)
