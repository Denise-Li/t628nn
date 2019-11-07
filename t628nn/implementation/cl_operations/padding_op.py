import numpy as np
import pyopencl as cl
from sanity_check import check_equal
from cl_util.tools import zero_buffer, get_cl, definitions2, create_buffer


code = """
    __kernel void pad(__global const float* inputs, __global float* outputs){
    size_t x = get_global_id(0);
    size_t y = get_global_id(1);
    size_t z = get_global_id(2);
    outputs[
        (W+YWP)*(H+YHP)*z +
        (W+YWP)*YHPT +
        (W+YWP)*y + 
        YWPT + x
    ] = inputs[
        (W+XWP)*(H+XHP)*z +
        (W+XWP)*XHPT +
        (W+XWP)*y + 
        XWPT + x
    ];
}
"""


class PaddingOp:
    def __init__(self, ctx, shape, xpad, ypad):
        n, d, h, w = shape
        xhp, xhpt, xwp, xwpt = xpad
        yhp, yhpt, ywp, ywpt = ypad

        assert n == 1

        self._ctx = ctx
        self._prog = cl.Program(ctx, code).build(
            definitions2(
                n=n, d=d, h=h, w=w,
                xhp=xhp, xhpt=xhpt, xwp=xwp, xwpt=xwpt, yhp=yhp, yhpt=yhpt, ywp=ywp, ywpt=ywpt
            )
        )

        self._params = n, d, h, w, xhp, xhpt, xwp, xwpt, yhp, yhpt, ywp, ywpt

    def get_output_size(self):
        n, d, h, w, xhp, xhpt, xwp, xwpt, yhp, yhpt, ywp, ywpt = self._params
        return d*(h+yhp)*(w+ywp)

    def get_output_shape(self):
        n, d, h, w, xhp, xhpt, xwp, xwpt, yhp, yhpt, ywp, ywpt = self._params
        return n, d, h+yhp, w+ywp

    def __call__(self, q, X, Y=None):
        n, d, h, w, xhp, xhpt, xwp, xwpt, yhp, yhpt, ywp, ywpt = self._params
        ynp = None
        if Y is None:
            ynp, Y = zero_buffer(self._ctx, (n, d, h+yhp, w+ywp))

        kernel = self._prog.pad
        kernel.set_args(X, Y)
        evt = cl.enqueue_nd_range_kernel(q, kernel, (w, h, d), None)

        return evt, ynp, Y


def test_padding_op():
    dev, ctx, q = get_cl()
    xnp, xcl = create_buffer(ctx, (1, 32, 30, 30))
    ynp, ycl = zero_buffer(ctx, (1, 32, 32, 28))

    expected = np.zeros_like(ynp, np.float32)
    expected[:, :, 4:-2, 3:-1] = xnp[:, :, 1:-3, 2:-4]
    expected = expected.copy()

    op = PaddingOp(ctx, (1, 32, 26, 24), (4, 1, 6, 2), (6, 4, 4, 3))
    evt, inp, icl = op(q, xcl, ycl)
    assert inp is None
    assert icl is ycl
    cl.wait_for_events([evt])
    cl.wait_for_events([cl.enqueue_copy(q, ynp, ycl)])
    assert check_equal(expected, ynp)


def test_2d_pad():
    dev, ctx, q = get_cl()
    xnp, xcl = create_buffer(ctx, (6, 6))
    ynp, ycl = zero_buffer(ctx, (6, 4))
    xnp[:, -2:] = 0.0
    xnp = xnp.copy()
    cl.wait_for_events([cl.enqueue_copy(q, xcl, xnp)])
    expected = xnp[:, :-2]

    op = PaddingOp(ctx, (1, 1, 6, 4), (0, 0, 2, 0), (0, 0, 0, 0))
    evt, inp, icl = op(q, xcl, ycl)
    assert inp is None
    assert icl is ycl
    cl.wait_for_events([evt])
    cl.wait_for_events([cl.enqueue_copy(q, ynp, ycl)])
    assert check_equal(expected, ynp)
